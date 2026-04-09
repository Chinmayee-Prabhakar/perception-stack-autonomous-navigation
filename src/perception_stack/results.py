"""Stable, model-independent 2D detection and mask result structures."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class BoxXYXY:
    """Float pixel edges: x1/y1 inclusive, x2/y2 exclusive, original image space."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.x1, self.y1, self.x2, self.y2]).all():
            raise ValueError("Box coordinates must be finite")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Box must have positive width and height")

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_normalized_xywh(self, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        cx, cy = self.center
        return (cx / width, cy / height, (self.x2 - self.x1) / width, (self.y2 - self.y1) / height)

    @classmethod
    def from_normalized_xywh(
        cls, xywh: tuple[float, float, float, float], image_size: tuple[int, int]
    ) -> "BoxXYXY":
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        cx, cy, bw, bh = xywh
        return cls((cx - bw / 2) * width, (cy - bh / 2) * height,
                   (cx + bw / 2) * width, (cy + bh / 2) * height)


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: BoxXYXY
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.class_id < 0 or not self.class_name:
            raise ValueError("Detection needs a nonnegative class ID and name")
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be finite and in [0, 1]")
        if self.mask is not None:
            mask = np.asarray(self.mask)
            if mask.ndim != 2 or mask.dtype != np.bool_:
                raise ValueError("Mask must be a HxW boolean array")
            owned = np.array(mask, copy=True, order="C")
            owned.flags.writeable = False
            object.__setattr__(self, "mask", owned)


@dataclass(frozen=True)
class FrameDetections:
    frame_id: str
    image_size: tuple[int, int]
    detections: tuple[Detection, ...]
    model_id: str

    def __post_init__(self) -> None:
        width, height = self.image_size
        if width <= 0 or height <= 0 or not self.frame_id or not self.model_id:
            raise ValueError("Frame detections require valid size, frame ID and model ID")
        detections = tuple(self.detections)
        for detection in detections:
            box = detection.box
            if box.x1 < 0 or box.y1 < 0 or box.x2 > width or box.y2 > height:
                raise ValueError("Detection box lies outside the original image")
            if detection.mask is not None and detection.mask.shape != (height, width):
                raise ValueError("Mask size must match the original image")
        object.__setattr__(self, "detections", detections)

    def to_summary(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "image_size": list(self.image_size),
            "model_id": self.model_id,
            "detections": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "box_xyxy": list(d.box.xyxy),
                    "mask_pixels": int(d.mask.sum()) if d.mask is not None else None,
                }
                for d in self.detections
            ],
        }


@dataclass(frozen=True)
class FrameDepth:
    """Original-frame visible-surface estimate of forward camera-axis Z in meters."""

    frame_id: str
    image_size: tuple[int, int]
    depth_m: np.ndarray
    valid_mask: np.ndarray
    model_id: str
    checkpoint_sha256: str
    device: str
    input_size: int
    preprocessing: Mapping[str, object]
    units: str = "meters"
    depth_convention: str = "camera_axis_z"
    nominal_max_depth_m: float = 80.0

    def __post_init__(self) -> None:
        width, height = self.image_size
        depth = np.asarray(self.depth_m)
        valid = np.asarray(self.valid_mask)
        if width <= 0 or height <= 0 or not self.frame_id or not self.model_id:
            raise ValueError("Frame depth requires positive size, frame ID and model ID")
        if depth.shape != (height, width) or depth.dtype != np.float32:
            raise ValueError("depth_m must be an original-resolution HxW float32 array")
        if valid.shape != depth.shape or valid.dtype != np.bool_:
            raise ValueError("valid_mask must be an original-resolution HxW boolean array")
        if not np.array_equal(valid, np.isfinite(depth) & (depth > 0)):
            raise ValueError("valid_mask must mark exactly finite, positive depth values")
        if self.units != "meters" or self.depth_convention != "camera_axis_z":
            raise ValueError("This depth contract requires meters of camera-axis Z")
        if len(self.checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be lowercase hexadecimal SHA-256")
        if self.device not in {"cpu", "cuda"} or self.input_size < 14:
            raise ValueError("Invalid execution device or input size")
        if self.nominal_max_depth_m <= 0:
            raise ValueError("nominal_max_depth_m must be positive")
        owned_depth = np.array(depth, copy=True, order="C")
        owned_valid = np.array(valid, copy=True, order="C")
        owned_depth.flags.writeable = False
        owned_valid.flags.writeable = False
        object.__setattr__(self, "depth_m", owned_depth)
        object.__setattr__(self, "valid_mask", owned_valid)
        object.__setattr__(self, "preprocessing", dict(self.preprocessing))

    def to_summary(self) -> dict:
        values = self.depth_m[self.valid_mask]
        near_cap = values >= 0.99 * self.nominal_max_depth_m
        return {
            "frame_id": self.frame_id,
            "image_size": list(self.image_size),
            "model_id": self.model_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": self.device,
            "input_size": self.input_size,
            "units": self.units,
            "depth_convention": self.depth_convention,
            "depth_reference": "predicted_visible_surface",
            "dtype": str(self.depth_m.dtype),
            "validity_rule": "finite and > 0 meters; near-cap values remain valid",
            "nominal_max_depth_m": self.nominal_max_depth_m,
            "near_cap_threshold_m": 0.99 * self.nominal_max_depth_m,
            "preprocessing": dict(self.preprocessing),
            "valid_pixels": int(values.size),
            "valid_fraction": float(values.size / self.depth_m.size),
            "near_cap_pixels": int(near_cap.sum()),
            "near_cap_percent_of_valid": float(100 * near_cap.mean()) if values.size else 0.0,
            "valid_depth_m": {
                "min": float(values.min()) if values.size else None,
                "median": float(np.median(values)) if values.size else None,
                "mean": float(values.mean()) if values.size else None,
                "max": float(values.max()) if values.size else None,
            },
        }


@dataclass(frozen=True)
class FrameRoadSegmentation:
    """Original-frame binary drivable-area and lane-marking predictions."""

    frame_id: str
    image_size: tuple[int, int]
    drivable_area_mask: np.ndarray
    lane_marking_mask: np.ndarray
    model_id: str
    checkpoint_id: str
    checkpoint_sha256: str
    device: str
    preprocessing: Mapping[str, object]

    def __post_init__(self) -> None:
        width, height = self.image_size
        if width <= 0 or height <= 0 or not self.frame_id or not self.model_id or not self.checkpoint_id:
            raise ValueError("Road segmentation requires positive size and nonempty identifiers")
        for name in ("drivable_area_mask", "lane_marking_mask"):
            mask = np.asarray(getattr(self, name))
            if mask.shape != (height, width) or mask.dtype != np.bool_:
                raise ValueError(f"{name} must be an original-resolution HxW boolean array")
            owned = np.array(mask, copy=True, order="C")
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)
        if len(self.checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be lowercase hexadecimal SHA-256")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Road segmentation device must be cpu or cuda")
        object.__setattr__(self, "preprocessing", dict(self.preprocessing))

    def to_summary(self) -> dict:
        height, width = self.drivable_area_mask.shape
        return {
            "frame_id": self.frame_id,
            "image_size": [width, height],
            "model_id": self.model_id,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": self.device,
            "mask_semantics": {
                "drivable_area": "binary BDD100K drivable-area foreground, not all road surface",
                "lane_marking": "binary lane-marking pixels, not fitted boundaries or centerlines",
            },
            "preprocessing": dict(self.preprocessing),
            "drivable_area_pixels": int(self.drivable_area_mask.sum()),
            "drivable_area_fraction": float(self.drivable_area_mask.mean()),
            "lane_marking_pixels": int(self.lane_marking_mask.sum()),
            "lane_marking_fraction": float(self.lane_marking_mask.mean()),
        }
