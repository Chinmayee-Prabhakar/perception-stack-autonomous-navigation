"""Official BDD100K TwinLiteNetPlus Small inference on a shared RGB Frame."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from perception_stack.frame import Frame
from perception_stack.results import FrameRoadSegmentation
from perception_stack.road_assets import (
    CHECKPOINT_ID, CHECKPOINT_PATH, MODEL_ID, SOURCE_REVISION, verify_road_checkpoint,
)


MODEL_INPUT_HW = (384, 640)  # Upstream BDD100K training/validation shape.


@dataclass(frozen=True)
class LetterboxGeometry:
    original_hw: tuple[int, int]
    model_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    top: int
    bottom: int
    left: int
    right: int

    @property
    def content_slices(self) -> tuple[slice, slice]:
        height, width = self.model_hw
        return slice(self.top, height - self.bottom), slice(self.left, width - self.right)


def resolve_device(requested: str) -> torch.device:
    if requested not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be cpu, cuda, or auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch installation; use --device cpu")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")


def preprocess_road_frame(frame: Frame, model_hw: tuple[int, int] = MODEL_INPUT_HW) -> tuple[torch.Tensor, LetterboxGeometry]:
    """RGB, aspect-preserving 114 padding, CHW float32 /255 as upstream trains."""
    model_h, model_w = model_hw
    if model_h < 8 or model_w < 8 or model_h % 8 or model_w % 8:
        raise ValueError("Model dimensions must be positive multiples of 8")
    original_h, original_w = frame.image_rgb.shape[:2]
    ratio = min(model_h / original_h, model_w / original_w)
    resized_w = min(model_w, int(round(original_w * ratio)))
    resized_h = min(model_h, int(round(original_h * ratio)))
    pad_w, pad_h = model_w - resized_w, model_h - resized_h
    left, right = int(round(pad_w / 2 - 0.1)), int(round(pad_w / 2 + 0.1))
    top, bottom = int(round(pad_h / 2 - 0.1)), int(round(pad_h / 2 + 0.1))
    resized = cv2.resize(frame.image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    if padded.shape[:2] != model_hw:
        raise RuntimeError("Letterbox dimensions do not match the model input")
    tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).float().unsqueeze(0) / 255.0
    geometry = LetterboxGeometry((original_h, original_w), model_hw, (resized_h, resized_w), top, bottom, left, right)
    return tensor, geometry


def foreground_from_logits(logits: torch.Tensor, geometry: LetterboxGeometry) -> np.ndarray:
    """Crop padding, bilinearly restore both class scores, then argmax class 1."""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[:2] != (1, 2):
        raise ValueError("Each road/lane head must be a 1x2xHxW tensor")
    if tuple(logits.shape[2:]) != geometry.model_hw or not logits.is_floating_point():
        raise ValueError("Road/lane head has wrong resolution or dtype")
    if not torch.isfinite(logits).all().item():
        raise ValueError("Road/lane head contains nonfinite class scores")
    content_y, content_x = geometry.content_slices
    cropped = logits[:, :, content_y, content_x]
    if tuple(cropped.shape[2:]) != geometry.resized_hw:
        raise ValueError("Letterbox crop is inconsistent with resized image")
    restored = F.interpolate(cropped.float(), size=geometry.original_hw, mode="bilinear", align_corners=False)
    return (restored.argmax(dim=1)[0] == 1).cpu().numpy().copy()


def predict_road_lane(
    frame: Frame, model: torch.nn.Module, *, device: torch.device,
    checkpoint_sha256: str, model_hw: tuple[int, int] = MODEL_INPUT_HW,
) -> FrameRoadSegmentation:
    tensor, geometry = preprocess_road_frame(frame, model_hw)
    with torch.inference_mode():
        output = model(tensor.to(device))
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise ValueError("TwinLiteNetPlus must return (drivable_logits, lane_logits)")
        drivable = foreground_from_logits(output[0], geometry)
        lane = foreground_from_logits(output[1], geometry)
    return FrameRoadSegmentation(
        frame_id=frame.frame_id, image_size=frame.size,
        drivable_area_mask=drivable, lane_marking_mask=lane,
        model_id=MODEL_ID, checkpoint_id=CHECKPOINT_ID,
        checkpoint_sha256=checkpoint_sha256, device=device.type,
        preprocessing={
            "source_revision": SOURCE_REVISION,
            "model_config": "small; channels=[8,16,32,64,128]; p=2; q=3",
            "input_color": "RGB",
            "pixel_scale": "uint8 / 255; no mean/std normalization",
            "resize": "aspect-preserving cv2.INTER_LINEAR to 384x640 letterbox",
            "pad_color_rgb": [114, 114, 114],
            "original_hw": list(geometry.original_hw),
            "model_input_hw": list(geometry.model_hw),
            "resized_content_hw": list(geometry.resized_hw),
            "padding_top_bottom_left_right": [geometry.top, geometry.bottom, geometry.left, geometry.right],
            "postprocess": "crop exact padding, bilinear class-score resize align_corners=False, argmax; class 1 foreground",
            "head_channels": {"0": "background", "1": "foreground"},
        },
    )


class TwinLiteRoadSegmenter:
    """Verified Small state dict with strict loading and explicit CPU/CUDA selection."""

    def __init__(self, weights: str | Path = CHECKPOINT_PATH, *, device: str = "cpu") -> None:
        self.device = resolve_device(device)
        self.checkpoint_sha256 = verify_road_checkpoint(weights)
        from perception_stack._vendor.twinlite_net_plus.model import TwinLiteNetPlus

        model = TwinLiteNetPlus(SimpleNamespace(config="small"))
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("Road/lane checkpoint must contain a tensor state dictionary")
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError("Checkpoint is incompatible with TwinLiteNetPlus Small") from error
        self._model = model.to(self.device).eval()

    def predict(self, frame: Frame) -> FrameRoadSegmentation:
        return predict_road_lane(
            frame, self._model, device=self.device,
            checkpoint_sha256=self.checkpoint_sha256,
        )
