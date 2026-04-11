"""Synthetic matching, M4 reuse, metrics and frozen-protocol tests."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from perception_stack.frame import Frame
from perception_stack.fusion import fuse_frame
from perception_stack.kitti_eval.dataset import KittiObjectFrame, parse_calibration, parse_labels
from perception_stack.kitti_eval.matching import box_iou, match_frame
from perception_stack.kitti_eval.metrics import (
    ProxyRow, aggregate_frames, evaluate_frame, frame_bootstrap_ci, reference_range_bin,
)
from perception_stack.kitti_eval.protocol import ClassMapping, EvaluationProtocol
from perception_stack.kitti_eval.report import write_json, write_markdown_report, write_plots, write_rows_csv
from perception_stack.kitti_eval.splits import freeze_or_verify_manifest, select_frame_ids
from perception_stack.results import BoxXYXY, Detection, FrameDepth, FrameDetections


ROOT = Path(__file__).resolve().parents[1]
CAR = "Car 0 0 0 0 0 4 4 1 1 2 0 0 10 0"
PERSON = "Pedestrian 0 1 0 0 0 4 4 1 1 2 0 0 12 0"
DONTCARE = "DontCare -1 -1 -10 5 0 9 4 -1 -1 -1 -1000 -1000 -1000 -10"
CALIB = "P2: 2 0 1 4 0 2 1 6 0 0 1 0\n"


def mapping():
    return ClassMapping.load(ROOT / "configs/kitti_class_mapping_v1.json")


def detections(*items):
    return FrameDetections("000001", (10, 10), tuple(
        Detection(2 if name == "car" else 0, name, conf, BoxXYXY(*box), mask)
        for name, conf, box, mask in items), "synthetic")


def test_iou_and_one_to_one_class_consistent_matching():
    assert box_iou(BoxXYXY(0, 0, 4, 4), BoxXYXY(2, 0, 6, 4)) == pytest.approx(1 / 3)
    annotations = parse_labels(CAR + "\n" + PERSON)
    det = detections(("car", .8, (0, 0, 4, 4), None),
                     ("car", .9, (0, 0, 4, 4), None),
                     ("person", .7, (0, 0, 4, 4), None),
                     ("bus", .9, (0, 0, 4, 4), None))
    result = match_frame(det, annotations, mapping())
    assert [(p.detection_id, p.reference_id) for p in result.pairs] == [(1, 0), (2, 1)]
    assert result.unmatched_detection_ids == (0,)
    assert result.unsupported_detection_ids == (3,)
    assert result.unmatched_reference_ids == ()
    assert result.to_dict()["total_detections"] == 4


def test_iou_order_ties_and_reference_conflicts():
    annotations = parse_labels(CAR + "\n" + CAR.replace("0 0 4 4", "1 0 5 4"))
    det = detections(("car", .8, (0, 0, 4, 4), None),
                     ("car", .8, (1, 0, 5, 4), None))
    result = match_frame(det, annotations, mapping())
    assert {(p.detection_id, p.reference_id) for p in result.pairs} == {(0, 0), (1, 1)}
    same_refs = parse_labels(CAR + "\n" + CAR)
    result = match_frame(det, same_refs, mapping())
    assert [(p.detection_id, p.reference_id) for p in result.pairs] == [(0, 0), (1, 1)]


def test_dontcare_ignored_and_empty_cases():
    det = detections(("car", .8, (5, 0, 9, 4), None))
    result = match_frame(det, parse_labels(CAR + "\n" + DONTCARE), mapping())
    assert result.ignored_detection_ids == (0,)
    assert result.unmatched_reference_ids == (0,)
    assert result.to_dict()["ignored_references"] == 1
    assert match_frame(detections(), parse_labels(CAR), mapping()).unmatched_reference_ids == (0,)
    assert match_frame(det, (), mapping()).unmatched_detection_ids == (0,)
    assert match_frame(det, parse_labels(DONTCARE), mapping()).pairs == ()
    with pytest.raises(ValueError, match="Duplicate"):
        same = parse_labels(CAR)
        match_frame(det, (same[0], same[0]), mapping())


def synthetic_evaluation(frame_id="000001", *, mask=True, reference_z=10):
    values = np.full((10, 10), 20, np.float32)
    values[:4, :4] = 10
    instance = np.zeros((10, 10), bool)
    instance[:4, :4] = True
    frame = Frame(np.zeros((10, 10, 3), np.uint8), frame_id)
    det = FrameDetections(frame_id, frame.size,
                          (Detection(2, "car", .8, BoxXYXY(0, 0, 10, 10),
                                     instance if mask else None),), "synthetic")
    depth = FrameDepth(frame_id, frame.size, values, np.ones_like(instance),
                       "synthetic-depth", "a" * 64, "cpu", 392, {})
    fused = fuse_frame(frame, det, depth)
    annotations = parse_labels(CAR.replace("0 0 10 0", f"0 0 {reference_z} 0")
                               .replace("0 0 4 4", "0 0 10 10"))
    sample = KittiObjectFrame(frame, annotations, parse_calibration(CALIB))
    return evaluate_frame(sample, det, fused, mapping())


def test_m4_reuse_paired_errors_and_missing_coverage():
    evaluation = synthetic_evaluation()
    row = evaluation.rows[0]
    assert row.reference_z_m == 10
    assert row.box_median_z_m == 20 and row.mask_median_z_m == 10
    assert row.box_signed_proxy_error_m == 10 and row.mask_signed_proxy_error_m == 0
    assert row.paired_mask_minus_box_absolute_proxy_error_m == -10
    assert row.box_valid_pixels == 100 and row.mask_valid_pixels == 16
    summary = aggregate_frames((evaluation,), min_bootstrap_frames=2)
    assert summary["primary_paired"]["paired_objects"] == 1
    assert summary["primary_paired"]["box"]["rmse_object_reference_z_proxy_m"] == 10
    assert summary["primary_paired"]["mask_lower_error_percent"] == 100
    assert summary["class_coverage"]["Car"]["eligible_references"] == 1
    assert summary["frame_bootstrap_mean_difference_95_ci"]["status"] == "insufficient_distinct_frames"
    no_mask = aggregate_frames((synthetic_evaluation(mask=False),), min_bootstrap_frames=2)
    assert no_mask["counts"]["matched_pairs"] == 1
    assert no_mask["method_coverage_on_all_matched"]["mask_missing"] == 1
    assert no_mask["primary_paired"]["paired_objects"] == 0
    assert no_mask["primary_paired"]["mask"]["mean_absolute_proxy_error_m"] is None
    assert aggregate_frames(())["status"] == "no_real_frames_evaluated"


def test_reference_range_boundaries_and_invalid():
    assert [reference_range_bin(z) for z in (0.1, 15, 30, 50, 80)] == [
        "0-15m", "15-30m", "30-50m", "50-80m", "80m-plus"]
    for z in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            reference_range_bin(z)


def test_frame_bootstrap_is_deterministic_and_frame_clustered():
    base = synthetic_evaluation().rows[0]
    rows = [replace(base, frame_id="a", paired_mask_minus_box_absolute_proxy_error_m=-10),
            replace(base, frame_id="a", paired_mask_minus_box_absolute_proxy_error_m=-10),
            replace(base, frame_id="b", paired_mask_minus_box_absolute_proxy_error_m=10)]
    first = frame_bootstrap_ci(rows, seed=12, replicates=200, min_distinct_frames=2)
    assert first == frame_bootstrap_ci(rows, seed=12, replicates=200, min_distinct_frames=2)
    assert first["status"] == "ok" and first["resampling_unit"] == "frame"
    assert first["mean_difference_95_percentile_ci_m"] == [-10, 10]
    with pytest.raises(ValueError):
        frame_bootstrap_ci(rows, seed=12, replicates=0, min_distinct_frames=2)


def test_frozen_mapping_protocol_and_split(tmp_path):
    protocol = EvaluationProtocol.load(ROOT / "configs/kitti_eval_v1.json")
    assert protocol.mapping.coco_to_kitti == {"car": "Car", "person": "Pedestrian"}
    assert protocol.data["scale_policy"] == "fixed_pretrained_metric_checkpoint_no_label_fit"
    ids = tuple(f"{i:06d}" for i in range(400))
    selected = select_frame_ids(ids, protocol)
    assert len(selected["pilot_frame_ids"]) == 16
    assert len(selected["heldout_frame_ids"]) == 256
    assert set(selected["pilot_frame_ids"]) <= set(selected["development_frame_ids"])
    assert set(selected["development_frame_ids"]).isdisjoint(selected["heldout_frame_ids"])
    assert selected == select_frame_ids(tuple(reversed(ids)), protocol)
    path = tmp_path / "split.json"
    assert freeze_or_verify_manifest(path, ids, protocol) == freeze_or_verify_manifest(path, ids, protocol)
    with pytest.raises(ValueError, match="Frozen manifest differs"):
        freeze_or_verify_manifest(path, ids[:-1], protocol)
    with pytest.raises(ValueError, match="Duplicate"):
        select_frame_ids(ids + (ids[0],), protocol)
    with pytest.raises(ValueError, match="Need at least"):
        select_frame_ids(ids[:10], protocol)


def test_protocol_rejects_heldout_label_scaling_and_mapping_changes(tmp_path):
    source = ROOT / "configs/kitti_eval_v1.json"
    for key, value in (("scale_policy", "fit_per_test_image"),
                       ("matching_iou_threshold", .1), ("detector_confidence", .05)):
        data = json.loads(source.read_text(encoding="utf-8"))
        data[key] = value
        path = tmp_path / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError):
            EvaluationProtocol.load(path)
    changed = json.loads((ROOT / "configs/kitti_class_mapping_v1.json").read_text(encoding="utf-8"))
    changed["coco_to_kitti"]["bus"] = "Car"
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError):
        ClassMapping.load(path)


def test_synthetic_report_artifacts_are_explicitly_proxy_labeled(tmp_path):
    evaluation = synthetic_evaluation()
    summary = aggregate_frames((evaluation,), min_bootstrap_frames=2)
    rows = list(evaluation.rows)
    write_rows_csv(tmp_path / "rows.csv", rows)
    write_json(tmp_path / "summary.json", summary)
    paths = write_plots(tmp_path, rows, summary)
    write_markdown_report(tmp_path / "report.md", summary, subset="synthetic",
                          selected_count=1, processed_count=1)
    assert len(paths) >= 6 and all(Path(path).is_file() for path in paths)
    assert "object-reference Z proxy" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "frame_id" in (tmp_path / "rows.csv").read_text(encoding="utf-8")
    assert write_plots(tmp_path / "no-pairs", [], summary) == []
