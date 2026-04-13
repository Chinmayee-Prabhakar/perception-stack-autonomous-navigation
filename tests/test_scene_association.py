"""M6 contact-band evidence tests; no weights or downloads."""

import numpy as np
import pytest

from perception_stack.frame import Frame
from perception_stack.results import BoxXYXY, Detection, FrameDetections, FrameRoadSegmentation
from perception_stack.scene import ContactPolicy, associate_road


def fixture(width=12, height=8, boxes=(BoxXYXY(2, 1, 8, 4),), road_pixels=(), lane_pixels=()):
    frame = Frame(np.zeros((height, width, 3), np.uint8), "f")
    detections = FrameDetections("f", frame.size, tuple(
        Detection(index, "object", 0.8, box) for index, box in enumerate(boxes)), "detector")
    drivable = np.zeros((height, width), bool)
    lane = np.zeros((height, width), bool)
    for v, u in road_pixels:
        drivable[v, u] = True
    for v, u in lane_pixels:
        lane[v, u] = True
    result = FrameRoadSegmentation("f", frame.size, drivable, lane, "road", "checkpoint",
                                   "a" * 64, "cpu", {})
    return frame, detections, result


def test_exact_below_box_region_and_threshold_boundary():
    frame, det, road = fixture(road_pixels=[(v, u) for v in (4, 5) for u in (4, 5, 6)][:3])
    item = associate_road(frame, det, road).objects[0]
    assert item.contact_region_xyxy == (4, 4, 7, 6)
    assert item.ideal_region_pixels == item.region_pixels == 6
    assert item.drivable_pixels == 3
    assert item.drivable_overlap_fraction == 0.5
    assert item.evidence_status == "drivable_adjacency_detected"
    stricter = ContactPolicy(foreground_overlap_threshold=0.5001)
    assert associate_road(frame, det, road, policy=stricter).objects[0].evidence_status == "drivable_adjacency_not_detected"


def test_no_road_lane_separate_and_empty_road_mask():
    frame, det, road = fixture(lane_pixels=[(v, u) for v in (4, 5) for u in (4, 5, 6)])
    item = associate_road(frame, det, road).objects[0]
    assert item.drivable_pixels == 0 and item.drivable_overlap_fraction == 0
    assert item.evidence_status == "drivable_adjacency_not_detected"
    assert road.lane_marking_mask.sum() == 6


def test_partial_coverage_and_band_height_change_non_square():
    frame, det, road = fixture(width=20, height=9, road_pixels=[(4, 4), (5, 4), (6, 4)])
    default = associate_road(frame, det, road).objects[0]
    taller = associate_road(frame, det, road, policy=ContactPolicy(min_band_height_px=3)).objects[0]
    assert (default.region_pixels, default.drivable_pixels, default.drivable_overlap_fraction) == (6, 2, 2/6)
    assert (taller.region_pixels, taller.drivable_pixels, taller.drivable_overlap_fraction) == (9, 3, 3/9)


def test_tiny_degenerate_and_zero_pixel_regions():
    frame, det, road = fixture(boxes=(BoxXYXY(2.1, 1, 2.4, 3),))
    item = associate_road(frame, det, road).objects[0]
    assert item.region_pixels == 0 and item.drivable_overlap_fraction is None
    assert item.evidence_status == "unknown_invalid_region"
    with pytest.raises(ValueError):
        BoxXYXY(2, 2, 2, 4)
    frame, det, road = fixture(boxes=(BoxXYXY(2, 1, 4, 3),))
    item = associate_road(frame, det, road).objects[0]
    assert item.region_pixels < 6 and item.evidence_status == "unknown_insufficient_evidence"


def test_boundary_truncation_occlusion_and_missing_road():
    boxes = (BoxXYXY(2, 3, 8, 8), BoxXYXY(0, 1, 6, 4), BoxXYXY(2, 1, 8, 4))
    frame, det, road = fixture(boxes=boxes)
    result = associate_road(frame, det, road, visibility={2: "occluded"})
    assert [item.evidence_status for item in result.objects] == [
        "unknown_image_boundary", "unknown_image_boundary", "unknown_occluded"]
    assert "occluded" in result.objects[2].reason
    assert associate_road(frame, det, road, visibility={1: "truncated"}).objects[1].evidence_status == "unknown_truncated"
    missing = associate_road(frame, det, None)
    assert missing.road_status == "unavailable_missing_road_prediction"
    assert missing.objects[2].drivable_overlap_fraction is None
    assert missing.objects[2].evidence_status == "unknown_insufficient_evidence"


def test_left_right_bottom_edges_are_unknown_not_proven_truncation():
    boxes = (BoxXYXY(0, 1, 6, 4), BoxXYXY(6, 1, 12, 4),
             BoxXYXY(2, 3, 8, 8), BoxXYXY(2, 1, 8, 4))
    frame, det, road = fixture(boxes=boxes)
    statuses = [item.evidence_status for item in associate_road(frame, det, road).objects]
    assert statuses == ["unknown_image_boundary"] * 3 + ["drivable_adjacency_not_detected"]
    assert all("unverified" in item.reason for item in associate_road(frame, det, road).objects[:3])


def test_coverage_is_distinct_from_explicit_truncation_and_occlusion():
    frame, det, road = fixture(boxes=(BoxXYXY(1, 1, 11, 6.5),))
    policy = ContactPolicy(min_band_height_px=3)
    item = associate_road(frame, det, road, policy=policy).objects[0]
    assert item.ideal_region_pixels == 18 and item.region_pixels == 6
    assert item.evidence_status == "unknown_insufficient_coverage"
    assert associate_road(frame, det, road, policy=policy,
                          visibility={0: "truncated"}).objects[0].evidence_status == "unknown_truncated"
    assert associate_road(frame, det, road, policy=policy,
                          visibility={0: "occluded"}).objects[0].evidence_status == "unknown_occluded"


def test_clipping_and_multiple_overlaps_independent():
    boxes = (BoxXYXY(2, 1, 8, 6.5), BoxXYXY(2, 1, 8, 4), BoxXYXY(2, 1, 8, 4))
    frame, det, road = fixture(height=8, boxes=boxes, road_pixels=[(4, 4), (4, 5), (4, 6), (5, 4)])
    items = associate_road(frame, det, road).objects
    assert items[0].region_pixels == 3  # only row 7 survives below 6.5
    assert items[0].evidence_status == "unknown_insufficient_evidence"
    assert items[1].drivable_pixels == items[2].drivable_pixels == 4
    assert items[1].drivable_overlap_fraction == items[2].drivable_overlap_fraction == 4/6


def test_bad_ids_dimensions_and_visibility_hints_rejected():
    frame, det, road = fixture()
    other_frame = Frame(np.zeros((8, 12, 3), np.uint8), "other")
    with pytest.raises(ValueError, match="IDs/dimensions"):
        associate_road(other_frame, det, road)
    wrong_road = FrameRoadSegmentation("f", (12, 7), np.zeros((7, 12), bool),
                                       np.zeros((7, 12), bool), "road", "c", "a"*64, "cpu", {})
    with pytest.raises(ValueError, match="IDs/dimensions"):
        associate_road(frame, det, wrong_road)
    with pytest.raises(ValueError, match="Visibility"):
        associate_road(frame, det, road, visibility={9: "occluded"})


@pytest.mark.parametrize("kwargs", [
    {"width_fraction": 0}, {"height_fraction": float("nan")},
    {"min_band_height_px": 0}, {"min_pixels": 0},
    {"foreground_overlap_threshold": 1.01}, {"min_visible_fraction": -0.1},
])
def test_bad_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        ContactPolicy(**kwargs)
