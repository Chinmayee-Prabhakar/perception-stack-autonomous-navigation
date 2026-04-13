"""Regression for associations paired with a different road prediction."""

from dataclasses import replace

import numpy as np
import pytest

from perception_stack.scene import ContactPolicy, associate_road, build_scene
from perception_stack.scene.association import road_source_fingerprint
from perception_stack.scene.synthetic import synthetic_scene
from perception_stack.scene.visualization import render_scene_overlay


def test_reject_stale_association_with_same_identity_but_zero_drivable_mask():
    frame, detections, road, scene, _ = synthetic_scene()
    replacement = replace(road, drivable_area_mask=np.zeros_like(road.drivable_area_mask))
    assert road.model_id == replacement.model_id
    assert road.checkpoint_id == replacement.checkpoint_id
    assert road.checkpoint_sha256 == replacement.checkpoint_sha256
    assert replacement.drivable_area_mask.sum() == 0
    assert scene.associations.objects[0].evidence_status == "drivable_adjacency_detected"
    with pytest.raises(ValueError, match="road.*(source|mask)|source.*road"):
        build_scene(frame, detections, replacement, scene.fusion, scene.associations)
    with pytest.raises(ValueError, match="road.*(source|mask)|source.*road"):
        render_scene_overlay(frame, detections, replacement, scene)


def test_same_foreground_count_different_pixels_is_stale():
    frame, detections, road, scene, _ = synthetic_scene()
    shifted = np.roll(road.drivable_area_mask, shift=5, axis=0)
    assert shifted.sum() == road.drivable_area_mask.sum()
    replacement = replace(road, drivable_area_mask=shifted)
    assert road_source_fingerprint(road) != road_source_fingerprint(replacement)
    with pytest.raises(ValueError, match="road source mask fingerprint"):
        build_scene(frame, detections, replacement, scene.fusion, scene.associations)
    # Equivalent pixel content is accepted regardless of source array layout.
    equivalent = replace(road, drivable_area_mask=np.asfortranarray(road.drivable_area_mask))
    assert road_source_fingerprint(road) == road_source_fingerprint(equivalent)
    build_scene(frame, detections, equivalent, scene.fusion, scene.associations)


@pytest.mark.parametrize("field,value", [
    ("detection_id", 9),
    ("contact_region_xyxy", (0, 0, 1, 1)),
    ("ideal_region_pixels", 999),
    ("region_pixels", 999),
    ("drivable_pixels", 0),
    ("drivable_overlap_fraction", 0.125),
    ("evidence_status", "drivable_adjacency_not_detected"),
    ("reason", "made up"),
])
def test_reject_contradictory_contact_diagnostics(field, value):
    frame, detections, road, scene, _ = synthetic_scene()
    first = replace(scene.associations.objects[0], **{field: value})
    contradictory = replace(scene.associations,
                            objects=(first, scene.associations.objects[1]))
    with pytest.raises(ValueError, match=field):
        build_scene(frame, detections, road, scene.fusion, contradictory)


def test_reject_wrong_checkpoint_policy_and_detection_identity():
    frame, detections, road, scene, _ = synthetic_scene()
    with pytest.raises(ValueError, match="checkpoint ID"):
        build_scene(frame, detections, replace(road, checkpoint_id="other"),
                    scene.fusion, scene.associations)
    with pytest.raises(ValueError, match="policy fingerprint"):
        build_scene(frame, detections, road, scene.fusion,
                    replace(scene.associations, policy=ContactPolicy(min_pixels=8)))
    with pytest.raises(ValueError, match="detection source fingerprint"):
        build_scene(frame, replace(detections, model_id="other"), road,
                    scene.fusion, scene.associations)


def test_visibility_hints_preserved_and_checked():
    frame, detections, road, scene, _ = synthetic_scene()
    hinted = associate_road(frame, detections, road, scene.fusion,
                            visibility={0: "occluded"})
    assert hinted.visibility_hints == ((0, "occluded"),)
    assert hinted.objects[0].evidence_status == "unknown_occluded"
    built = build_scene(frame, detections, road, scene.fusion, hinted)
    assert built.to_dict()["association_source"]["visibility_hints"] == [
        {"detection_id": 0, "state": "occluded"}]
    with pytest.raises(ValueError, match="evidence_status|reason"):
        build_scene(frame, detections, road, scene.fusion,
                    replace(hinted, visibility_hints=()))


def test_missing_road_pairing_and_overlay_consistency():
    frame, detections, road, scene, _ = synthetic_scene()
    missing = associate_road(frame, detections, None, scene.fusion)
    missing_scene = build_scene(frame, detections, None, scene.fusion, missing)
    assert missing.road_source_sha256 is None
    assert missing_scene.to_dict()["road_segmentation"] is None
    assert render_scene_overlay(frame, detections, None, missing_scene).size == frame.size
    with pytest.raises(ValueError, match="null provenance"):
        build_scene(frame, detections, None, scene.fusion,
                    replace(missing, road_source_sha256="a" * 64))
    with pytest.raises(ValueError, match="availability"):
        build_scene(frame, detections, road, scene.fusion, missing)
    with pytest.raises(ValueError, match="availability"):
        render_scene_overlay(frame, detections, None, scene)
    with pytest.raises(ValueError, match="availability"):
        render_scene_overlay(frame, detections, road, missing_scene)


def test_correct_source_pair_renders_and_lane_source_is_checked():
    frame, detections, road, scene, _ = synthetic_scene()
    assert render_scene_overlay(frame, detections, road, scene).size == frame.size
    different_lane = replace(road, lane_marking_mask=np.zeros_like(road.lane_marking_mask))
    assert different_lane.drivable_area_mask.sum() == road.drivable_area_mask.sum()
    with pytest.raises(ValueError, match="road source mask fingerprint"):
        render_scene_overlay(frame, detections, different_lane, scene)
