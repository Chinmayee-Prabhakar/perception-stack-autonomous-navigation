"""Synthetic paired-sampling and pinhole geometry tests; no checkpoints required."""

import json
import math

import numpy as np
import pytest

from perception_stack.frame import CameraIntrinsics, Frame
from perception_stack.fusion import FusionConfig, backproject_pixel, fuse_frame
from perception_stack.fusion.core import box_region, sample_region
from perception_stack.fusion_cli import load_rectified_intrinsics
from perception_stack.fusion_visualization import render_fusion
from perception_stack.results import BoxXYXY, Detection, FrameDepth, FrameDetections


def scene(values, *, masks=(), boxes=None, calibration=False):
    array = np.asarray(values, dtype=np.float32)
    height, width = array.shape
    intrinsics = CameraIntrinsics(width, height, 10, 10, 1, 1) if calibration else None
    frame = Frame(np.zeros((height, width, 3), np.uint8), "same", intrinsics=intrinsics)
    if boxes is None:
        boxes = [BoxXYXY(0, 0, width, height) for _ in masks]
    detections = FrameDetections("same", frame.size, tuple(
        Detection(i, "car", 0.8, box, mask)
        for i, (box, mask) in enumerate(zip(boxes, masks))
    ), "fake-detector")
    depth = FrameDepth("same", frame.size, array, np.isfinite(array) & (array > 0),
                       "fake-metric-depth", "a" * 64, "cpu", 392, {})
    return frame, detections, depth


def test_box_pixel_centers_and_clipping():
    assert box_region(BoxXYXY(0.5, 0.5, 2.5, 2.5), (4, 4)).tolist() == [
        [False]*4, [False, True, True, False], [False, True, True, False], [False]*4,
    ]
    assert box_region(BoxXYXY(-5, -5, 2, 2), (4, 4)).sum() == 4
    assert box_region(BoxXYXY(4, 0, 6, 2), (4, 4)).sum() == 0
    assert box_region(BoxXYXY(0.1, 0.1, 0.4, 0.4), (4, 4)).sum() == 0
    with pytest.raises(ValueError):
        BoxXYXY(1, 1, 1, 2)


def test_uniform_depth_agreement_counts_and_immutability():
    mask = np.ones((3, 4), bool)
    frame, detections, depth = scene(np.full((3, 4), 8), masks=(mask,))
    original_image, original_depth, original_mask = frame.image_rgb.copy(), depth.depth_m.copy(), detections.detections[0].mask.copy()
    result = fuse_frame(frame, detections, depth)
    for method in (result.objects[0].box_method, result.objects[0].mask_method):
        s = method.statistics
        assert s.status == "ok" and s.median_z_m == s.mean_z_m == 8
        assert s.min_z_m == s.max_z_m == 8
        assert s.std_z_m == s.iqr_z_m == s.mad_z_m == 0
        assert s.region_pixels == s.valid_pixels == 12 and s.valid_fraction == 1
        assert s.selected_sample.depth_z_m == 8
        assert method.surface_point is None and method.geometry_status == "unavailable_no_intrinsics"
    assert result.objects[0].to_dict()["mask_minus_box_median_z_m"] == 0
    assert np.array_equal(frame.image_rgb, original_image)
    assert np.array_equal(depth.depth_m, original_depth)
    assert np.array_equal(detections.detections[0].mask, original_mask)
    assert not frame.image_rgb.flags.writeable and not depth.depth_m.flags.writeable
    assert render_fusion(frame, result).size == frame.size


def test_foreground_mask_improves_constructed_depth_and_outlier_median():
    values = np.full((4, 4), 30, np.float32)
    values[:2, :2] = 5
    mask = np.zeros((4, 4), bool)
    mask[:2, :2] = True
    frame, det, dep = scene(values, masks=(mask,))
    obj = fuse_frame(frame, det, dep).objects[0]
    assert obj.box_method.statistics.median_z_m == 30
    assert obj.mask_method.statistics.median_z_m == 5
    assert obj.mask_method.statistics.region_pixels == 4
    assert obj.mask_method.statistics.valid_pixels == 4
    # A single outlier does not move the median of the four mask pixels much.
    values[0, 0] = 80
    frame, det, dep = scene(values, masks=(mask,))
    assert fuse_frame(frame, det, dep).objects[0].mask_method.statistics.median_z_m == 5


def test_bad_mask_can_be_worse_and_missing_mask_never_falls_back():
    values = np.full((4, 4), 5, np.float32)
    values[2:, 2:] = 30
    bad_mask = np.zeros((4, 4), bool)
    bad_mask[2:, 2:] = True
    frame, det, dep = scene(values, masks=(bad_mask, None))
    first, second = fuse_frame(frame, det, dep).objects
    assert first.box_method.statistics.median_z_m == 5
    assert first.mask_method.statistics.median_z_m == 30
    assert second.box_method.statistics.status == "ok"
    assert second.mask_method.statistics.status == "missing_mask"
    assert second.mask_method.statistics.median_z_m is None
    assert second.to_dict()["mask_minus_box_median_z_m"] is None


def test_occluding_foreground_inside_mask_can_dominate_median():
    # A mask can include a nearer occluder: the statistic is about sampled pixels,
    # not a guaranteed unobstructed object reference or center depth.
    values = np.array([[2, 2, 2], [2, 9, 9], [9, 9, 9]], np.float32)
    mask = np.zeros((3, 3), bool)
    mask[:2, :] = True
    frame, det, dep = scene(values, masks=(mask,))
    obj = fuse_frame(frame, det, dep).objects[0]
    assert obj.box_method.statistics.median_z_m == 9
    assert obj.mask_method.statistics.median_z_m == 2
    assert obj.mask_method.statistics.valid_pixels == 6


def test_invalid_sparse_and_threshold_boundaries():
    values = np.array([[0, np.nan, -1, np.inf], [5, 0, 0, 0]], np.float32)
    frame, det, dep = scene(values, masks=(np.ones((2, 4), bool),))
    obj = fuse_frame(frame, det, dep).objects[0]
    assert obj.box_method.statistics.status == "insufficient_valid_pixels"
    assert obj.box_method.statistics.valid_pixels == 1
    assert obj.box_method.statistics.valid_fraction == 0.125
    assert obj.box_method.statistics.selected_sample.depth_z_m == 5
    config = FusionConfig(min_valid_pixels=1, min_valid_fraction=0.125)
    assert fuse_frame(frame, det, dep, config=config).objects[0].box_method.statistics.status == "ok"
    config = FusionConfig(min_valid_pixels=1, min_valid_fraction=0.126)
    assert fuse_frame(frame, det, dep, config=config).objects[0].box_method.statistics.status == "low_valid_fraction"
    with pytest.raises(ValueError):
        FusionConfig(min_valid_pixels=0)


def test_no_valid_depth_empty_masks_and_tiny_box():
    frame, det, dep = scene([[0, np.nan], [-1, np.inf]], masks=(np.ones((2, 2), bool), np.zeros((2, 2), bool)))
    first, second = fuse_frame(frame, det, dep).objects
    assert first.box_method.statistics.status == first.mask_method.statistics.status == "no_valid_depth"
    assert first.box_method.statistics.median_z_m is None
    assert second.mask_method.statistics.status == "empty_mask"
    frame, det, dep = scene([[4, 4], [4, 4]], masks=(np.ones((2, 2), bool),),
                            boxes=[BoxXYXY(0.1, 0.1, 0.4, 0.4)])
    obj = fuse_frame(frame, det, dep).objects[0]
    assert obj.box_method.statistics.status == "empty_box_region"
    assert obj.mask_method.statistics.status == "empty_mask_box_intersection"


def test_mask_is_intersected_with_box_and_overlaps_are_independent():
    values = np.array([[3, 3, 20], [3, 3, 20], [20, 20, 20]], np.float32)
    mask = np.ones((3, 3), bool)
    boxes = [BoxXYXY(0, 0, 2, 2), BoxXYXY(1, 0, 3, 2)]
    frame, det, dep = scene(values, masks=(mask, mask), boxes=boxes)
    first, second = fuse_frame(frame, det, dep).objects
    assert first.mask_method.statistics.region_pixels == 4
    assert second.mask_method.statistics.region_pixels == 4
    assert first.box_method.statistics.median_z_m == 3
    assert second.box_method.statistics.median_z_m == 11.5


def test_near_cap_is_diagnostic_not_invalid():
    frame, det, dep = scene([[79.3, 80], [5, 5]], masks=(np.ones((2, 2), bool),))
    stat = fuse_frame(frame, det, dep).objects[0].box_method.statistics
    assert stat.status == "ok" and stat.near_cap_pixels == 2
    assert stat.near_cap_fraction_of_valid == 0.5


def test_actual_selected_depth_not_aggregate_median_and_tie_break():
    frame, det, dep = scene([[4, 6], [6, 4]], masks=(np.ones((2, 2), bool),), calibration=True)
    result = fuse_frame(frame, det, dep, calibration_provenance="synthetic rectified unit test", rectified=True)
    method = result.objects[0].box_method
    assert method.statistics.median_z_m == 5
    assert method.statistics.selected_sample == method.statistics.selected_sample.__class__(0, 0, 4)
    assert method.surface_point.z_m == 4
    assert method.surface_point.x_m == pytest.approx(-0.4)
    assert method.surface_point.y_m == pytest.approx(-0.4)
    assert result.calibration_status == "available_rectified_intrinsics"


def test_geometry_signs_principal_point_range_and_validation():
    k = CameraIntrinsics(5, 5, 2, 4, 1, 2)
    center = backproject_pixel(1, 2, 10, k, image_size=(5, 5), rectified=True)
    assert (center.x_m, center.y_m, center.z_m, center.radial_range_m) == (0, 0, 10, 10)
    point = backproject_pixel(3, 0, 10, k, image_size=(5, 5), rectified=True)
    assert (point.x_m, point.y_m, point.z_m) == pytest.approx((10, -5, 10))
    assert point.radial_range_m == pytest.approx(15)
    for kwargs in ({"image_size": (4, 5), "rectified": True}, {"image_size": (5, 5), "rectified": False}):
        with pytest.raises(ValueError):
            backproject_pixel(1, 2, 10, k, **kwargs)
    with pytest.raises(ValueError):
        backproject_pixel(5, 0, 10, k, image_size=(5, 5), rectified=True)
    with pytest.raises(ValueError):
        CameraIntrinsics(5, 5, math.inf, 4, 1, 2)


def test_missing_unverified_or_unrectified_calibration():
    frame, det, dep = scene(np.full((2, 2), 4), masks=(np.ones((2, 2), bool),), calibration=True)
    for kw, expected in (({}, "unavailable_unverified_provenance"),
                         ({"calibration_provenance": "synthetic"}, "unavailable_unrectified")):
        result = fuse_frame(frame, det, dep, **kw)
        assert result.objects[0].box_method.surface_point is None
        assert result.calibration_status == expected
    with pytest.raises(ValueError):
        Frame(np.zeros((2, 2, 3), np.uint8), "x", intrinsics=CameraIntrinsics(3, 2, 1, 1, 0, 0))


def test_identity_dimension_empty_frame_and_json_safety():
    frame, det, dep = scene([[1, 1], [1, 1]], masks=())
    result = fuse_frame(frame, det, dep)
    assert result.objects == () and json.loads(result.to_json())["objects"] == []
    assert result.to_json() == result.to_json()
    with pytest.raises(ValueError, match="IDs"):
        wrong = Frame(np.zeros((2, 2, 3), np.uint8), "other")
        fuse_frame(wrong, det, dep)
    with pytest.raises(ValueError, match="dimensions"):
        other = FrameDepth("same", (1, 1), np.ones((1, 1), np.float32),
                           np.ones((1, 1), bool), "model", "b" * 64, "cpu", 392, {})
        fuse_frame(frame, det, other)
    frame, det, dep = scene([[0, np.nan], [0, 0]], masks=(None,))
    payload = json.loads(fuse_frame(frame, det, dep).to_json())
    assert payload["objects"][0]["box"]["statistics"]["median_z_m"] is None
    assert payload["objects"][0]["mask"]["statistics"]["status"] == "missing_mask"


def test_region_shape_and_intrinsics_file_reject_raw_p2(tmp_path):
    frame, det, dep = scene([[1, 1], [1, 1]], masks=())
    with pytest.raises(ValueError):
        sample_region(np.ones((1, 2), bool), dep, FusionConfig())
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"P2": [1] * 12}), encoding="utf-8")
    with pytest.raises(ValueError, match="raw P2"):
        load_rectified_intrinsics(path)
    path.write_text(json.dumps({"width": 2, "height": 2, "fx": 1, "fy": 1,
                                "cx": 0, "cy": 0, "rectified": True,
                                "provenance": "synthetic unit test"}), encoding="utf-8")
    calibration, source = load_rectified_intrinsics(path)
    assert calibration.width == 2 and source == "synthetic unit test"
