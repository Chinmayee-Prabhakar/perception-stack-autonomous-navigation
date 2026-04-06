"""Stable, model-independent 2D detection and mask result structures."""

from dataclasses import dataclass

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
