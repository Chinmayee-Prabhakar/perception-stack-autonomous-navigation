from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from perception_stack import Frame, FrameRoadSegmentation
from perception_stack.road.twinlite import (
    TwinLiteRoadSegmenter, foreground_from_logits, predict_road_lane,
    preprocess_road_frame, resolve_device,
)
from perception_stack.road_assets import CHECKPOINT_SHA256, file_sha256, verify_road_checkpoint
from perception_stack.road_visualization import mask_png, render_road_overlay


def make_result(drivable: np.ndarray, lane: np.ndarray) -> FrameRoadSegmentation:
    return FrameRoadSegmentation(
        "road", (drivable.shape[1], drivable.shape[0]), drivable, lane,
        "twinlitenetplus-small-bdd100k", "small.pth", "0" * 64, "cpu", {"test": True},
    )


def test_road_result_contract_empty_masks_and_owned_arrays() -> None:
    drivable = np.zeros((3, 5), dtype=bool)
    lane = np.zeros_like(drivable)
    result = make_result(drivable, lane)
    drivable[0, 0] = True
    assert result.frame_id == "road"
    assert result.image_size == (5, 3)
    assert not result.drivable_area_mask.any() and not result.lane_marking_mask.any()
    assert not result.drivable_area_mask.flags.writeable
    assert not result.lane_marking_mask.flags.writeable
    assert result.to_summary()["lane_marking_fraction"] == 0
    assert "centerlines" in result.to_summary()["mask_semantics"]["lane_marking"]


@pytest.mark.parametrize("drivable,lane", [
    (np.zeros((2, 3), np.uint8), np.zeros((2, 3), bool)),
    (np.zeros((2, 3), bool), np.zeros((3, 2), bool)),
    (np.zeros((2, 3, 1), bool), np.zeros((2, 3), bool)),
])
def test_malformed_result_masks_rejected(drivable: np.ndarray, lane: np.ndarray) -> None:
    with pytest.raises(ValueError, match="original-resolution HxW boolean"):
        FrameRoadSegmentation("f", (3, 2), drivable, lane, "m", "c", "0" * 64, "cpu", {})


def test_result_rejects_malformed_provenance() -> None:
    mask = np.zeros((2, 3), bool)
    with pytest.raises(ValueError, match="SHA-256"):
        FrameRoadSegmentation("f", (3, 2), mask, mask, "m", "c", "bad", "cpu", {})


@pytest.mark.parametrize("shape,expected_content", [
    ((4, 8), (16, 32)),
    ((8, 4), (32, 16)),
    ((5, 7), (23, 32)),
    ((7, 5), (32, 23)),
    ((7, 7), (32, 32)),
    ((4, 16), (8, 32)),
    ((16, 4), (32, 8)),
])
def test_letterbox_non_square_geometry(shape: tuple[int, int], expected_content: tuple[int, int]) -> None:
    frame = Frame(np.zeros((*shape, 3), np.uint8), "f")
    before = frame.image_rgb.copy()
    tensor, geometry = preprocess_road_frame(frame, (32, 32))
    assert tensor.shape == (1, 3, 32, 32)
    assert geometry.resized_hw == expected_content
    assert geometry.top + expected_content[0] + geometry.bottom == 32
    assert geometry.left + expected_content[1] + geometry.right == 32
    assert np.array_equal(frame.image_rgb, before)


def test_rgb_scale_and_padding_color_without_frame_mutation() -> None:
    pixels = np.zeros((4, 8, 3), np.uint8)
    pixels[..., 0] = 255
    frame = Frame(pixels, "red")
    tensor, geometry = preprocess_road_frame(frame, (8, 8))
    assert (geometry.top, geometry.bottom, geometry.left, geometry.right) == (2, 2, 0, 0)
    assert tensor[0, :, 2, 0].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert tensor[0, :, 0, 0].tolist() == pytest.approx([114 / 255] * 3)
    assert np.array_equal(frame.image_rgb, pixels)


@pytest.mark.parametrize("shape,padding", [
    ((5, 7), (4, 5, 0, 0)),
    ((7, 5), (0, 0, 4, 5)),
])
def test_odd_asymmetric_padding_removed_without_edge_shift(
    shape: tuple[int, int], padding: tuple[int, int, int, int],
) -> None:
    frame = Frame(np.zeros((*shape, 3), np.uint8), "odd")
    _, geometry = preprocess_road_frame(frame, (32, 32))
    assert (geometry.top, geometry.bottom, geometry.left, geometry.right) == padding
    ys, xs = geometry.content_slices
    logits = torch.zeros((1, 2, 32, 32))
    logits[:, 0] = 4
    # A strong false foreground signal in padding must never enter the frame.
    logits[:, 1] = 9
    logits[:, 1, ys, xs] = 0
    assert not foreground_from_logits(logits, geometry).any()
    # Foreground touching the top-left image corner must stay at that corner.
    logits[:, 1, ys.start:ys.start + 3, xs.start:xs.start + 3] = 9
    mask = foreground_from_logits(logits, geometry)
    assert mask[0, 0] and not mask[-1, -1]
    assert not mask[-1, 0] and not mask[0, -1]


@pytest.mark.parametrize("shape", [(4, 8), (8, 4)])
def test_padding_removed_and_xy_orientation_preserved(shape: tuple[int, int]) -> None:
    frame = Frame(np.zeros((*shape, 3), np.uint8), "f")
    _, geometry = preprocess_road_frame(frame, (32, 32))
    logits = torch.zeros((1, 2, 32, 32))
    logits[:, 0] = 2
    ys, xs = geometry.content_slices
    content_h, content_w = geometry.resized_hw
    logits[:, 1, ys.start:ys.start + content_h // 2, xs.start + content_w // 2:xs.stop] = 4
    mask = foreground_from_logits(logits, geometry)
    assert mask.shape == shape and mask.dtype == np.bool_
    assert mask[0, -1] and not mask[0, 0]
    assert not mask[-1, -1] and not mask[-1, 0]


def test_class_one_is_foreground_argmax_without_activation() -> None:
    frame = Frame(np.zeros((8, 8, 3), np.uint8), "f")
    _, geometry = preprocess_road_frame(frame, (8, 8))
    logits = torch.zeros((1, 2, 8, 8))
    logits[:, 0] = 4
    logits[:, 1] = 3
    logits[:, 1, 2:6, 2:6] = 5
    foreground = foreground_from_logits(logits, geometry)
    assert foreground.sum() == 16
    assert foreground[3, 3] and not foreground[0, 0]
    logits[:, 1] = logits[:, 0]
    assert not foreground_from_logits(logits, geometry).any()


@pytest.mark.parametrize("malformed", [
    torch.zeros((1, 1, 8, 8)), torch.zeros((1, 2, 4, 8)),
    torch.zeros((1, 2, 8, 8), dtype=torch.int32),
])
def test_bad_model_head_rejected(malformed: torch.Tensor) -> None:
    frame = Frame(np.zeros((8, 8, 3), np.uint8), "f")
    _, geometry = preprocess_road_frame(frame, (8, 8))
    with pytest.raises(ValueError):
        foreground_from_logits(malformed, geometry)
    nonfinite = torch.zeros((1, 2, 8, 8))
    nonfinite[0, 0, 1, 1] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        foreground_from_logits(nonfinite, geometry)


class FakeTwinLite(torch.nn.Module):
    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = tensor.shape
        road = torch.zeros((batch, 2, height, width), device=tensor.device)
        lane = torch.zeros_like(road)
        road[:, 1] = 1
        lane[:, 1, :, width // 2:] = 1
        return road, lane


class MissingHead(torch.nn.Module):
    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.zeros((1, 2, *tensor.shape[2:])),)


def test_missing_model_prediction_rejected() -> None:
    frame = Frame(np.zeros((8, 8, 3), np.uint8), "f")
    with pytest.raises(ValueError, match="must return"):
        predict_road_lane(frame, MissingHead(), device=torch.device("cpu"),
                          checkpoint_sha256="0" * 64, model_hw=(8, 8))


def test_fake_model_result_alignment_and_metadata() -> None:
    frame = Frame(np.zeros((9, 21, 3), np.uint8), "frame-x")
    result = predict_road_lane(frame, FakeTwinLite(), device=torch.device("cpu"),
                               checkpoint_sha256="0" * 64, model_hw=(32, 32))
    assert result.drivable_area_mask.shape == (9, 21)
    assert result.drivable_area_mask.all()
    assert result.lane_marking_mask[:, -1].all()
    assert not result.lane_marking_mask[:, 0].any()
    assert result.preprocessing["model_input_hw"] == [32, 32]
    assert result.device == "cpu"


def test_visualization_and_saved_masks_round_trip(tmp_path: Path) -> None:
    frame = Frame(np.full((12, 16, 3), 64, np.uint8), "road")
    drivable = np.zeros((12, 16), bool)
    lane = np.zeros_like(drivable)
    drivable[5:, :] = True
    lane[7, 4:12] = True
    result = make_result(drivable, lane)
    overlay = render_road_overlay(frame, result)
    assert overlay.size == frame.size
    overlay.save(tmp_path / "overlay.png")
    mask_png(result.drivable_area_mask).save(tmp_path / "road.png")
    mask_png(result.lane_marking_mask).save(tmp_path / "lane.png")
    np.savez_compressed(tmp_path / "masks.npz", drivable_area_mask=result.drivable_area_mask,
                        lane_marking_mask=result.lane_marking_mask)
    assert Image.open(tmp_path / "overlay.png").size == frame.size
    assert np.array_equal(np.asarray(Image.open(tmp_path / "road.png")) > 0, drivable)
    assert np.array_equal(np.asarray(Image.open(tmp_path / "lane.png")) > 0, lane)
    with np.load(tmp_path / "masks.npz") as stored:
        assert np.array_equal(stored["drivable_area_mask"], drivable)
        assert np.array_equal(stored["lane_marking_mask"], lane)
    assert render_road_overlay(frame, make_result(np.zeros_like(drivable), np.zeros_like(lane))).size == frame.size


def test_overlay_rejects_wrong_frame() -> None:
    frame = Frame(np.zeros((3, 5, 3), np.uint8), "other")
    result = make_result(np.zeros((3, 5), bool), np.zeros((3, 5), bool))
    with pytest.raises(ValueError, match="does not belong"):
        render_road_overlay(frame, result)


def test_checkpoint_missing_corrupt_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "small.pth"
    with pytest.raises(FileNotFoundError, match="download_assets.py --road"):
        verify_road_checkpoint(path)
    path.write_bytes(b"corrupt")
    assert len(file_sha256(path)) == 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_road_checkpoint(path)


def test_wrong_architecture_state_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "wrong.pth"
    torch.save({"not_a_model": torch.zeros(1)}, path)
    monkeypatch.setattr("perception_stack.road.twinlite.verify_road_checkpoint", lambda _: CHECKPOINT_SHA256)
    with pytest.raises(ValueError, match="incompatible with TwinLiteNetPlus Small"):
        TwinLiteRoadSegmenter(path)


def test_explicit_cuda_error_auto_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        resolve_device("cuda")
