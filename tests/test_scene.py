"""Synthetic scene contracts and visualization; no model checkpoints."""

from dataclasses import replace
import json

import numpy as np
import pytest

from perception_stack.frame import Frame
from perception_stack.fusion import fuse_frame
from perception_stack.fusion.results import SurfacePoint
from perception_stack.results import FrameDepth
from perception_stack.scene import associate_road, build_scene
from perception_stack.scene.synthetic import synthetic_scene
from perception_stack.scene.visualization import metric_xz_data, render_metric_xz, render_scene_overlay


def test_synthetic_geometry_and_metric_axis_convention(tmp_path):
    frame, det, road, scene, geometry = synthetic_scene()
    assert scene.metric_xz_status == "available_approximate_surface_points"
    data = metric_xz_data(scene)
    assert data["x_axis"] == "Camera X right (m)"
    assert data["z_axis"] == "Camera Z forward (m)"
    assert data["origin_xz_m"] == [0, 0] and data["aspect"] == "equal_metric"
    assert [(item["x_m"], item["z_m"]) for item in data["points"]] == [(-5, 10), (10, 20)]
    assert geometry["expected_camera_xz_m"] == [[-5, 10], [10, 20]]
    assert [obj.fusion.mask_method.statistics.selected_sample.u for obj in scene.objects] == [25, 75]
    assert [obj.fusion.mask_method.statistics.selected_sample.v for obj in scene.objects] == [35, 35]
    assert all(obj.plotted_surface[0] == "mask" for obj in scene.objects)
    image = render_metric_xz(scene)
    path = tmp_path / "xz.png"
    image.save(path)
    assert path.stat().st_size > 1000


def test_scene_preserves_m4_values_and_strict_json():
    frame, det, road, scene, _ = synthetic_scene()
    payload = json.loads(scene.to_json())
    assert scene.to_json() == scene.to_json()
    assert payload["frame_id"] == frame.frame_id and payload["image_size"] == [100, 80]
    assert [obj["detection_id"] for obj in payload["objects"]] == [0, 1]
    assert payload["objects"][0]["mask"]["statistics"]["median_z_m"] == 10
    assert payload["objects"][0]["mask"]["statistics"]["selected_sample"]["depth_z_m"] == 10
    assert payload["objects"][0]["road_association"]["region_pixels"] > 0
    assert "ground contact" in payload["geometry_semantics"]
    assert "depth_m" not in payload and "drivable_area_mask" not in payload
    assert payload["models"]["road_model_id"] == "synthetic_road"
    changed = scene.to_dict()
    changed["road_segmentation"]["preprocessing"]["synthetic"] = False
    assert scene.to_dict()["road_segmentation"]["preprocessing"]["synthetic"] is True
    assert not frame.image_rgb.flags.writeable and not road.drivable_area_mask.flags.writeable


def test_uncalibrated_scene_has_no_fabricated_coordinates():
    frame, det, road, _, _ = synthetic_scene()
    plain = Frame(frame.image_rgb, frame.frame_id)
    values = np.full((frame.size[1], frame.size[0]), 7, np.float32)
    depth = FrameDepth(frame.frame_id, frame.size, values, np.ones(values.shape, bool),
                       "metric", "a" * 64, "cpu", 392, {})
    fusion = fuse_frame(plain, det, depth)
    scene = build_scene(plain, det, road, fusion, associate_road(plain, det, road, fusion))
    payload = scene.to_dict()
    assert payload["calibration"]["status"] == "unavailable_no_intrinsics"
    assert payload["visualization"]["metric_xz"] == "unavailable_no_intrinsics"
    assert all(obj["mask"]["surface_point"] is None for obj in payload["objects"])
    with pytest.raises(ValueError, match="unavailable"):
        metric_xz_data(scene)
    with pytest.raises(ValueError, match="unavailable"):
        render_metric_xz(scene)


def test_overlay_alignment_transparency_and_empty_detections():
    frame, det, road, scene, _ = synthetic_scene()
    image = render_scene_overlay(frame, det, road, scene)
    assert image.size == frame.size
    # Untouched pixel, pavement overlay, and lane overlay are visibly distinct.
    base = np.asarray(frame.image_rgb)
    out = np.asarray(image)
    assert not np.array_equal(out[43, 5], base[43, 5])  # drivable overlay
    assert not np.array_equal(out[48, 50], out[48, 60])  # lane indication
    empty_det = replace(det, detections=())
    empty_fusion = fuse_frame(frame, empty_det, FrameDepth(frame.frame_id, frame.size,
        np.zeros((80, 100), np.float32), np.zeros((80, 100), bool), "m", "a"*64, "cpu", 392, {}),
        calibration_provenance="synthetic", rectified=True)
    empty_scene = build_scene(frame, empty_det, road, empty_fusion,
                              associate_road(frame, empty_det, road, empty_fusion))
    assert render_scene_overlay(frame, empty_det, road, empty_scene).size == frame.size
    assert empty_scene.metric_xz_status == "unavailable_no_calibrated_surface_points"
    with pytest.raises(ValueError):
        render_scene_overlay(Frame(np.zeros((80, 100, 3), np.uint8), "wrong"), det, road, scene)


def test_result_alignment_invalid_values_and_metadata_ownership():
    frame, det, road, scene, _ = synthetic_scene()
    with pytest.raises(ValueError, match="IDs/dimensions"):
        build_scene(Frame(np.zeros((80, 100, 3), np.uint8), "wrong"), det, road,
                    scene.fusion, scene.associations)
    source = {"nested": {"shape": [1, 2]}}
    hardened = replace(scene.fusion, depth_preprocessing=source)
    source["nested"]["shape"].append(3)
    assert hardened.to_dict()["depth_preprocessing"]["nested"]["shape"] == [1, 2]
    with pytest.raises(TypeError):
        hardened.depth_preprocessing["nested"]["shape"][0] = 5
    result = hardened.to_dict()
    result["depth_preprocessing"]["nested"]["shape"].append(9)
    assert hardened.to_dict()["depth_preprocessing"]["nested"]["shape"] == [1, 2]
    bad = replace(scene.fusion, depth_preprocessing={"bad": float("nan")})
    with pytest.raises(ValueError, match="Out of range"):
        build_scene(frame, det, road, bad, scene.associations)
    wrong_detector = replace(scene.fusion, detector_model_id="other")
    with pytest.raises(ValueError, match="detector provenance"):
        build_scene(frame, det, road, wrong_detector, scene.associations)
    obj = scene.fusion.objects[0]
    fabricated = replace(obj.mask_method, surface_point=SurfacePoint(999, 0, 10, 999))
    bad_object = replace(obj, mask_method=fabricated)
    wrong_geometry = replace(scene.fusion, objects=(bad_object, scene.fusion.objects[1]))
    with pytest.raises(ValueError, match="selected-ray backprojection"):
        build_scene(frame, det, road, wrong_geometry, scene.associations)
