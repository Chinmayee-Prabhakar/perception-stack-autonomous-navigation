"""Synthetic scene contracts and visualization; no model checkpoints."""

from dataclasses import replace
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from perception_stack.frame import Frame
from perception_stack.fusion import fuse_frame
from perception_stack.fusion.results import SurfacePoint
from perception_stack.results import FrameDepth
from perception_stack.scene import associate_road, build_scene
from perception_stack.scene.synthetic import synthetic_scene
from perception_stack.scene.visualization import (
    _font, metric_xz_data, render_metric_xz, render_scene_overlay,
    render_scene_presentation, scene_presentation_data, scene_presentation_layout,
)


def test_synthetic_geometry_and_metric_axis_convention(tmp_path):
    frame, det, road, scene, geometry = synthetic_scene()
    assert scene.metric_xz_status == "available_approximate_surface_points"
    data = metric_xz_data(scene)
    assert data["x_axis"] == "Camera X right (m)"
    assert data["z_axis"] == "Camera Z forward (m)"
    assert "Rectified camera-centered" in data["title"]
    camera2 = metric_xz_data(replace(scene, fusion=replace(
        scene.fusion, calibration_provenance="KITTI image_2 rectified camera-2 intrinsics")))
    assert camera2["x_axis"] == "Camera-2 X right (m)"
    assert camera2["z_axis"] == "Camera-2 Z forward (m)"
    assert "Rectified camera-2-centered" in camera2["title"]
    assert "not object centers" in data["interpretation"]
    assert "not radial range" in data["interpretation"]
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


def test_presentation_keeps_camera_pixels_and_explains_semantics():
    frame, det, road, scene, _ = synthetic_scene()
    overlay = render_scene_overlay(frame, det, road, scene)
    presentation = render_scene_presentation(frame, det, road, scene)
    assert presentation.width >= frame.size[0] and presentation.height > frame.size[1]
    assert np.array_equal(np.asarray(presentation)[:frame.size[1], :frame.size[0]], np.asarray(overlay))
    data = scene_presentation_data(scene)
    assert [row["id"] for row in data["rows"]] == list(range(len(det.detections)))
    assert "predicted drivable area" in data["mask_legend"][0]
    assert "predicted lane-marking pixels" in data["mask_legend"][1]
    assert all(word in data["adjacency_legend"] for word in (
        "Predicted image-space road adjacency", "Road-adjacent (+)",
        "Not road-adjacent (−)", "Insufficient evidence (?)"))
    assert "predicted drivable-area pixels" in data["adjacency_method"]
    assert "bounding box" in data["adjacency_method"]
    assert all(word in data["limitation"] for word in (
        "physical road contact", "safely on/off the road", "occupancy", "collision safety"))
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for line in (data["adjacency_legend"], data["adjacency_method"], data["limitation"]):
        assert measure.textbbox((0, 0), line, font=_font(15))[2] < presentation.width - 12
    assert render_scene_presentation(frame, det, road, scene).tobytes() == presentation.tobytes()


@pytest.mark.parametrize("status,label", [
    ("drivable_adjacency_detected", "Road-adjacent (+)"),
    ("drivable_adjacency_not_detected", "Not road-adjacent (−)"),
    *[(status, "Insufficient evidence (?)") for status in (
        "unknown_truncated", "unknown_occluded", "unknown_image_boundary",
        "unknown_invalid_region", "unknown_insufficient_coverage",
        "unknown_insufficient_evidence")],
])
def test_presentation_labels_keep_internal_evidence_status(status, label):
    _, _, _, scene, _ = synthetic_scene()
    changed = replace(scene, objects=(replace(scene.objects[0], road=replace(
        scene.objects[0].road, evidence_status=status)), *scene.objects[1:]))
    assert scene_presentation_data(changed)["rows"][0]["adjacency"] == label
    assert changed.objects[0].road.evidence_status == status


def test_shallow_dense_scene_layout_and_invalid_dimensions():
    layout = scene_presentation_layout((1242, 375), 12)
    assert layout["columns"] == 2 and layout["rows_per_column"] == 6
    assert layout["canvas_size"] == (1242, 691)
    # The longest status plus a two-digit ID and pedestrian depth row fits one column.
    label = "#11 pedestrian 0.99  B:79.2 M:79.2  Insufficient evidence (?)"
    text_width = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
        (0, 0), label, font=_font(15))[2]
    assert 12 + 14 + text_width < layout["canvas_size"][0] // layout["columns"]
    assert scene_presentation_layout((1242, 375), 0)["rows_per_column"] == 1
    with pytest.raises(ValueError):
        scene_presentation_layout((0, 375), 12)
    with pytest.raises(ValueError):
        scene_presentation_layout((1242, 375), -1)


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
