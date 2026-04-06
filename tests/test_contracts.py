import numpy as np
import pytest

from perception_stack import BoxXYXY, CameraIntrinsics, Detection, Frame, FrameDetections
from perception_stack.visualization import draw_detections


def test_box_normalized_round_trip() -> None:
    box = BoxXYXY(64.0, 24.0, 320.0, 240.0)
    recovered = BoxXYXY.from_normalized_xywh(box.to_normalized_xywh((640, 480)), (640, 480))
    assert recovered.xyxy == pytest.approx(box.xyxy)
    assert box.center == (192.0, 132.0)


def test_frame_detection_mask_and_intrinsics_contract() -> None:
    intrinsics = CameraIntrinsics(8, 6, 100, 100, 4, 3)
    frame = Frame(np.zeros((6, 8, 3), dtype=np.uint8), "f0", intrinsics=intrinsics)
    mask = np.zeros((6, 8), dtype=np.bool_)
    mask[1:4, 2:6] = True
    detection = Detection(2, "car", 0.8, BoxXYXY(2, 1, 6, 4), mask)
    results = FrameDetections(frame.frame_id, frame.size, (detection,), "mock")
    assert results.to_summary()["detections"][0]["mask_pixels"] == 12
    mask[:] = False
    assert detection.mask.sum() == 12
    assert not detection.mask.flags.writeable
    assert draw_detections(frame, results).size == (8, 6)


def test_invalid_sizes_and_boxes_rejected() -> None:
    with pytest.raises(ValueError):
        BoxXYXY(3, 0, 3, 5)
    with pytest.raises(ValueError):
        Frame(np.zeros((4, 5, 3), np.uint8), "x", intrinsics=CameraIntrinsics(6, 4, 100, 100, 3, 2))
    detection = Detection(0, "person", 0.5, BoxXYXY(0, 0, 6, 2))
    with pytest.raises(ValueError):
        FrameDetections("x", (5, 4), (detection,), "mock")


def test_empty_results_can_be_visualized() -> None:
    frame = Frame(np.zeros((4, 5, 3), np.uint8), "none")
    results = FrameDetections("none", frame.size, (), "mock")
    assert results.to_summary()["detections"] == []
    assert np.array_equal(np.asarray(draw_detections(frame, results)), frame.image_rgb)
