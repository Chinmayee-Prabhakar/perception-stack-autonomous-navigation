"""Translate Ultralytics predictions into original-frame, model-independent results."""

from pathlib import Path
from typing import Any

import numpy as np

from perception_stack.frame import Frame
from perception_stack.results import BoxXYXY, Detection, FrameDetections


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def convert_ultralytics_result(frame: Frame, result: Any, model_id: str) -> FrameDetections:
    """Convert one Ultralytics result; tolerate zero boxes and absent masks."""
    width, height = frame.size
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return FrameDetections(frame.frame_id, frame.size, (), model_id)

    xyxy = _as_numpy(boxes.xyxy)
    confidence = _as_numpy(boxes.conf)
    classes = _as_numpy(boxes.cls)
    raw_masks = None if result.masks is None else _as_numpy(result.masks.data)
    if raw_masks is not None and len(raw_masks) != len(xyxy):
        raise ValueError("Model returned a different number of masks and boxes")

    detections: list[Detection] = []
    for index, (coords, score, class_id) in enumerate(zip(xyxy, confidence, classes)):
        x1, y1, x2, y2 = (float(value) for value in coords)
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 <= x1 or y2 <= y1:
            continue
        mask = None
        if raw_masks is not None:
            mask_array = raw_masks[index]
            if mask_array.shape != (height, width):
                raise ValueError("Model mask is not aligned to the original frame")
            mask = np.asarray(mask_array >= 0.5, dtype=np.bool_)
        label_id = int(class_id)
        name = result.names[label_id]
        detections.append(Detection(label_id, str(name), float(score),
                                    BoxXYXY(float(x1), float(y1), float(x2), float(y2)), mask))
    return FrameDetections(frame.frame_id, frame.size, tuple(detections), model_id)


class YOLOInstanceSegmenter:
    """One-image YOLO11n-seg adapter; loads only a local checkpoint."""

    def __init__(self, weights: str | Path, *, confidence: float = 0.25,
                 image_size: int = 640, device: str = "cpu") -> None:
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"Checkpoint missing: {weights}. Run `python scripts/download_assets.py` first."
            )
        if not 0 <= confidence <= 1 or image_size <= 0:
            raise ValueError("Invalid confidence or image_size")
        from ultralytics import YOLO

        self._model = YOLO(str(weights))
        if self._model.task != "segment":
            raise ValueError("Checkpoint must be an instance-segmentation model")
        self.model_id = f"ultralytics:{weights.name}"
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

    def predict(self, frame: Frame) -> FrameDetections:
        results = self._model.predict(
            # Ultralytics treats NumPy image sources as BGR (OpenCV convention).
            source=frame.image_rgb[:, :, ::-1].copy(),
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError("Expected exactly one result for one frame")
        return convert_ultralytics_result(frame, results[0], self.model_id)
