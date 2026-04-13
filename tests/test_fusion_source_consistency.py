"""Fusion must be paired with the exact detector masks used for M4 sampling."""

from dataclasses import replace

import numpy as np
import pytest

from perception_stack.fusion import fuse_frame
from perception_stack.fusion.source import DETECTOR_INPUT_SCHEMA, detector_input_fingerprint
from perception_stack.results import BoxXYXY, FrameDepth
from perception_stack.scene import associate_road, build_scene
from perception_stack.scene.synthetic import synthetic_scene
from perception_stack.scene.visualization import render_scene_overlay


def _replace_first(detections, **updates):
    return replace(detections, detections=(replace(detections.detections[0], **updates),
                                           detections.detections[1]))


def _synthetic_depth(frame, detections):
    values = np.zeros((frame.size[1], frame.size[0]), np.float32)
    for det, z in zip(detections.detections, (10.0, 20.0)):
        values[det.mask] = z
    return FrameDepth(frame.frame_id, frame.size, values, values > 0,
                      "synthetic_depth", "0" * 64, "cpu", 392, {"synthetic": True})


@pytest.mark.parametrize("replacement_kind", ["different_pixels", "missing_mask"])
def test_reject_stale_fusion_instance_mask(replacement_kind):
    frame, detections, road, scene, _ = synthetic_scene()
    original = detections.detections[0]
    replacement_mask = (np.roll(original.mask, 1, axis=1)
                        if replacement_kind == "different_pixels" else None)
    modified = _replace_first(detections, mask=replacement_mask)
    assert modified.frame_id == detections.frame_id
    assert modified.model_id == detections.model_id
    assert modified.detections[0].box == original.box
    associations = associate_road(frame, modified, road, scene.fusion)
    with pytest.raises(ValueError, match="fusion.*(source|mask)|detector.*input"):
        build_scene(frame, modified, road, scene.fusion, associations)


def test_correct_pairing_and_rendering_preserve_m4_values():
    frame, detections, road, scene, _ = synthetic_scene()
    rebuilt = build_scene(frame, detections, road, scene.fusion, scene.associations)
    assert scene.fusion.detector_input_fingerprint_sha256 == detector_input_fingerprint(detections)
    assert rebuilt.to_dict()["fusion_source"]["schema"] == DETECTOR_INPUT_SCHEMA
    assert [(obj.fusion.mask_method.statistics.median_z_m,
             obj.fusion.box_method.statistics.median_z_m) for obj in rebuilt.objects] == [(10, 10), (20, 20)]
    assert [(obj.plotted_surface[1].x_m, obj.plotted_surface[1].z_m)
            for obj in rebuilt.objects] == [(-5, 10), (10, 20)]
    assert render_scene_overlay(frame, detections, road, scene).size == frame.size


def test_same_foreground_count_different_mask_pixels_are_rejected():
    frame, detections, road, scene, _ = synthetic_scene()
    moved = np.roll(detections.detections[0].mask, 1, axis=1)
    assert moved.sum() == detections.detections[0].mask.sum()
    changed = _replace_first(detections, mask=moved)
    assert detector_input_fingerprint(changed) != scene.fusion.detector_input_fingerprint_sha256
    associations = associate_road(frame, changed, road)
    with pytest.raises(ValueError, match="fusion source fingerprint"):
        build_scene(frame, changed, road, scene.fusion, associations)
    with pytest.raises(ValueError, match="fusion source fingerprint"):
        render_scene_overlay(frame, changed, road, scene)


def test_missing_mask_replaced_by_present_and_empty_mask_distinguished():
    frame, detections, road, scene, _ = synthetic_scene()
    missing = _replace_first(detections, mask=None)
    empty = _replace_first(detections, mask=np.zeros_like(detections.detections[0].mask))
    assert detector_input_fingerprint(missing) != detector_input_fingerprint(empty)
    assert detector_input_fingerprint(missing) != detector_input_fingerprint(detections)
    missing_fusion = fuse_frame(frame, missing, _synthetic_depth(frame, detections),
                                calibration_provenance="synthetic", rectified=True)
    assert missing_fusion.objects[0].instance_mask_available is False
    assert build_scene(frame, missing, road, missing_fusion,
                       associate_road(frame, missing, road, missing_fusion)).objects[0].fusion.mask_method.statistics.status == "missing_mask"
    with pytest.raises(ValueError, match="fusion source fingerprint"):
        build_scene(frame, detections, road, missing_fusion,
                    associate_road(frame, detections, road))
    with pytest.raises(ValueError, match="fusion source fingerprint"):
        build_scene(frame, empty, road, missing_fusion,
                    associate_road(frame, empty, road))


def test_equivalent_mask_content_with_different_numpy_layout_is_accepted():
    frame, detections, road, scene, _ = synthetic_scene()
    layout = np.asfortranarray(detections.detections[0].mask)
    assert layout.flags.f_contiguous and not layout.flags.c_contiguous
    equivalent = _replace_first(detections, mask=layout)
    assert detector_input_fingerprint(equivalent) == scene.fusion.detector_input_fingerprint_sha256
    build_scene(frame, equivalent, road, scene.fusion,
                associate_road(frame, equivalent, road, scene.fusion))


@pytest.mark.parametrize("change", ["order", "class_id", "class_name", "confidence", "box", "model"])
def test_detector_metadata_and_order_change_rejected(change):
    frame, detections, road, scene, _ = synthetic_scene()
    if change == "order":
        modified = replace(detections, detections=tuple(reversed(detections.detections)))
    elif change == "model":
        modified = replace(detections, model_id="another_detector")
    else:
        value = {"class_id": 8, "class_name": "different", "confidence": 0.8,
                 "box": BoxXYXY(22, 31, 30, 40)}[change]
        modified = _replace_first(detections, **{change: value})
    assert detector_input_fingerprint(modified) != scene.fusion.detector_input_fingerprint_sha256
    with pytest.raises(ValueError):
        build_scene(frame, modified, road, scene.fusion, associate_road(frame, modified, road))


def test_missing_tampered_and_contradictory_fusion_source_rejected():
    frame, detections, road, scene, _ = synthetic_scene()
    with pytest.raises(ValueError, match="fusion source fingerprint is missing"):
        build_scene(frame, detections, road,
                    replace(scene.fusion, detector_input_fingerprint_sha256=None), scene.associations)
    with pytest.raises(ValueError, match="fusion source fingerprint"):
        build_scene(frame, detections, road,
                    replace(scene.fusion, detector_input_fingerprint_sha256="f" * 64), scene.associations)
    first = replace(scene.fusion.objects[0], instance_mask_available=False)
    with pytest.raises(ValueError, match="Scene detection IDs and predictions"):
        build_scene(frame, detections, road,
                    replace(scene.fusion, objects=(first, scene.fusion.objects[1])), scene.associations)
