"""Deterministic calibrated fixture, strictly synthetic and separate from real data."""

import numpy as np

from perception_stack.frame import CameraIntrinsics, Frame
from perception_stack.fusion import fuse_frame
from perception_stack.results import BoxXYXY, Detection, FrameDepth, FrameDetections, FrameRoadSegmentation
from .association import associate_road
from .results import build_scene


def synthetic_scene():
    """Two 9x9 objects: selected rays (25,35,10) and (75,35,20).

    With fx=50 and cx=50, expected X-Z points are (-5,10) and (+10,20) m.
    """
    width, height = 100, 80
    image = np.full((height, width, 3), 45, np.uint8)
    image[42:, :, 1] = 70
    intrinsics = CameraIntrinsics(width, height, 50, 50, 50, 40)
    frame = Frame(image, "synthetic_calibrated", intrinsics=intrinsics)
    boxes = (BoxXYXY(21, 31, 30, 40), BoxXYXY(71, 31, 80, 40))
    masks = []
    depth_values = np.zeros((height, width), np.float32)
    for box, z in zip(boxes, (10.0, 20.0)):
        mask = np.zeros((height, width), bool)
        mask[int(box.y1):int(box.y2), int(box.x1):int(box.x2)] = True
        depth_values[mask] = z
        masks.append(mask)
    detections = FrameDetections(frame.frame_id, frame.size,
                                 tuple(Detection(index, "synthetic_object", 0.9, box, mask)
                                       for index, (box, mask) in enumerate(zip(boxes, masks))),
                                 "synthetic_detector")
    depth = FrameDepth(frame.frame_id, frame.size, depth_values, depth_values > 0,
                       "synthetic_metric_depth", "0" * 64, "cpu", 392,
                       {"synthetic": True})
    drivable = np.zeros((height, width), bool)
    drivable[40:45] = True
    lane = np.zeros((height, width), bool)
    lane[48, 48:52] = True
    road = FrameRoadSegmentation(frame.frame_id, frame.size, drivable, lane,
                                 "synthetic_road", "synthetic", "1" * 64, "cpu", {"synthetic": True})
    fusion = fuse_frame(frame, detections, depth, calibration_provenance="synthetic known rectified pinhole",
                        rectified=True)
    associations = associate_road(frame, detections, road, fusion)
    scene = build_scene(frame, detections, road, fusion, associations)
    geometry = {
        "synthetic": True,
        "intrinsics": {"width": width, "height": height, "fx": 50, "fy": 50, "cx": 50, "cy": 40},
        "selected_pixels_and_actual_z_m": [[25, 35, 10], [75, 35, 20]],
        "expected_camera_xz_m": [[-5, 10], [10, 20]],
        "coordinate_convention": "X right, Y down, Z forward, meters",
        "reference": "synthetic predicted visible-surface points, not object centers",
    }
    return frame, detections, road, scene, geometry
