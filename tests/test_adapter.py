from types import SimpleNamespace

import numpy as np
import pytest

from perception_stack import Frame
from perception_stack.detectors.ultralytics_yolo import convert_ultralytics_result


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self):
        return len(self.xyxy)


def test_adapter_handles_no_detections_or_masks() -> None:
    frame = Frame(np.zeros((10, 20, 3), np.uint8), "empty")
    empty = SimpleNamespace(boxes=FakeBoxes(np.empty((0, 4)), np.empty(0), np.empty(0)), masks=None)
    assert convert_ultralytics_result(frame, empty, "mock").detections == ()
    raw = SimpleNamespace(
        boxes=FakeBoxes(np.array([[1, 2, 9, 8]], np.float32),
                        np.array([0.7]), np.array([2])),
        masks=None,
        names={2: "car"},
    )
    result = convert_ultralytics_result(frame, raw, "mock")
    assert result.detections[0].mask is None
    assert result.detections[0].box.xyxy == (1, 2, 9, 8)


def test_adapter_keeps_full_resolution_mask_and_rejects_misalignment() -> None:
    frame = Frame(np.zeros((10, 20, 3), np.uint8), "masked")
    mask = np.zeros((1, 10, 20), np.float32)
    mask[0, 2:8, 1:9] = 1
    raw = SimpleNamespace(
        boxes=FakeBoxes(np.array([[1, 2, 9, 8]], np.float32),
                        np.array([0.7]), np.array([2])),
        masks=SimpleNamespace(data=mask),
        names={2: "car"},
    )
    result = convert_ultralytics_result(frame, raw, "mock")
    assert result.detections[0].mask.shape == (10, 20)
    assert result.detections[0].mask.sum() == 48
    raw.masks.data = np.zeros((1, 5, 10), np.float32)
    with pytest.raises(ValueError, match="not aligned"):
        convert_ultralytics_result(frame, raw, "mock")
