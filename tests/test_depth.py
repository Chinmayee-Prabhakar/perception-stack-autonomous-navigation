from pathlib import Path

import numpy as np
import pytest
import torch

from perception_stack import Frame, FrameDepth
from perception_stack.depth.metric import MetricDepthEstimator, predict_metric_depth, preprocess_frame, resolve_device
from perception_stack.depth_assets import CHECKPOINT_SHA256, verify_checkpoint
from perception_stack.depth_visualization import render_depth


def make_depth(values: np.ndarray, mask: np.ndarray | None = None) -> FrameDepth:
    values = np.asarray(values, dtype=np.float32)
    if mask is None:
        mask = np.isfinite(values) & (values > 0)
    return FrameDepth(
        "sample", (values.shape[1], values.shape[0]), values, mask,
        "depth-anything-v2-metric-vkitti-vits", "0" * 64, "cpu", 392,
        {"resize": "test"},
    )


def test_depth_contract_validity_saturation_and_immutability() -> None:
    values = np.array([[1, 0, -3, np.nan], [np.inf, -np.inf, 79.5, 80]], np.float32)
    result = make_depth(values)
    assert result.depth_m.dtype == np.float32
    assert result.valid_mask.dtype == np.bool_
    assert result.valid_mask.tolist() == [[True, False, False, False], [False, False, True, True]]
    assert not result.depth_m.flags.writeable and not result.valid_mask.flags.writeable
    values[0, 0] = 99
    assert result.depth_m[0, 0] == 1
    summary = result.to_summary()
    assert summary["units"] == "meters"
    assert summary["depth_convention"] == "camera_axis_z"
    assert summary["valid_pixels"] == 3
    assert summary["near_cap_pixels"] == 2
    assert summary["near_cap_percent_of_valid"] == pytest.approx(200 / 3)


def test_all_invalid_depth_has_empty_statistics() -> None:
    result = make_depth(np.array([[0, np.nan]], np.float32))
    summary = result.to_summary()
    assert summary["valid_pixels"] == 0
    assert summary["valid_depth_m"] == {"min": None, "median": None, "mean": None, "max": None}


@pytest.mark.parametrize("values,mask", [
    (np.ones((2, 3), np.float64), np.ones((2, 3), bool)),
    (np.ones((2, 3), np.float32), np.ones((3, 2), bool)),
    (np.ones((2, 3), np.float32), np.ones((2, 3), np.uint8)),
    (np.zeros((2, 3), np.float32), np.ones((2, 3), bool)),
])
def test_malformed_depth_rejected(values: np.ndarray, mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        FrameDepth("f", (3, 2), values, mask, "model", "0" * 64, "cpu", 392, {})


class CoordinateModel(torch.nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        _, _, height, width = tensor.shape
        y = torch.linspace(0, 1, height, device=tensor.device)[:, None]
        x = torch.linspace(0, 1, width, device=tensor.device)[None, :]
        return (1 + 10 * x + y)[None]


class MalformedModel(torch.nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, 2, 3), dtype=torch.int32)


def test_rectangular_alignment_orientation_and_unchanged_frame() -> None:
    pixels = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    frame = Frame(pixels, "rect")
    before = frame.image_rgb.copy()
    tensor = preprocess_frame(frame, 28)
    assert tensor.shape == (1, 3, 28, 56)
    result = predict_metric_depth(
        frame, CoordinateModel(), device=torch.device("cpu"),
        input_size=28, checkpoint_sha256="0" * 64,
    )
    assert result.depth_m.shape == result.valid_mask.shape == (4, 8)
    assert result.depth_m.dtype == np.float32
    assert result.depth_m[0, 0] == pytest.approx(1)
    assert result.depth_m[0, -1] == pytest.approx(11)
    assert result.depth_m[-1, 0] == pytest.approx(2)
    assert result.depth_m[-1, -1] == pytest.approx(12)
    assert result.valid_mask.all()
    assert np.array_equal(frame.image_rgb, before)
    assert render_depth(result).size == (98, 4)


def test_malformed_model_output_rejected() -> None:
    frame = Frame(np.zeros((4, 8, 3), np.uint8), "bad")
    with pytest.raises(ValueError, match="floating-point depth"):
        predict_metric_depth(frame, MalformedModel(), device=torch.device("cpu"),
                             input_size=28, checkpoint_sha256="0" * 64)


def test_checkpoint_missing_and_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "wrong.pth"
    with pytest.raises(FileNotFoundError, match="download_assets.py --depth"):
        verify_checkpoint(path)
    path.write_bytes(b"not the official metric checkpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checkpoint(path)


def test_wrong_architecture_state_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "wrong.pth"
    torch.save({"relative_head.weight": torch.zeros(1)}, path)
    monkeypatch.setattr("perception_stack.depth.metric.verify_checkpoint", lambda _: CHECKPOINT_SHA256)
    with pytest.raises(ValueError, match="incompatible with metric VKITTI vits architecture"):
        MetricDepthEstimator(path)


def test_explicit_cuda_error_and_auto_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        resolve_device("cuda")
