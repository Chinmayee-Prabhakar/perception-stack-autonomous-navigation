"""Report frozen held-out paired statistics, coverage, and drive-aware uncertainty."""

from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from perception_stack.kitti_lidar.heldout_statistics import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, TIE_TOLERANCE_M,
    clustered_bootstrap, drive_equal_mean, pair_at_support, paired_metrics,
)


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/kitti_m5_l3/heldout"


def _load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _write(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _group(pairs: list[dict], key):
    groups = defaultdict(list)
    for row in pairs:
        groups[str(key(row))].append(row)
    return {name: paired_metrics(values) for name, values in sorted(groups.items())}


def _depth_bin(z: float) -> str:
    if z < 15:
        return "under_15m"
    if z < 30:
        return "15_to_under_30m"
    if z < 50:
        return "30_to_under_50m"
    return "50m_plus"


def _support_bin(n: int) -> str:
    if n < 3:
        return "under_3"
    if n < 10:
        return "3_to_9"
    if n < 50:
        return "10_to_49"
    if n < 200:
        return "50_to_199"
    return "200_plus"


def _case(row: dict) -> dict:
    identity = {key: row[key] for key in ("frame_id", "reference_id", "detection_id", "drive_id",
                                             "class_name", "status", "occluded", "truncated", "support_bins",
                                             "lidar_z_m", "support_z_spread_m",
                                             "original_m5_bottom_center_z_m", "frozen_box_median_z_m",
                                             "frozen_mask_median_z_m", "annotation_box_xyxy",
                                             "detector_box_xyxy", "rejected_counts")}
    identity["paired_error"] = row["paired_error"]
    return identity


def _top(rows: list[dict], key, *, reverse: bool = True, count: int = 12) -> list[dict]:
    return [_case(row) for row in sorted(rows, key=lambda row: (key(row), row["frame_id"],
                                                         row["reference_id"]), reverse=reverse)[:count]]


def main() -> None:
    rows = _load("object_records.json")
    frames = _load("frame_records.json")
    geometry = _load("geometry_provenance.json")
    extracted = _load("extraction_manifest.json")
    if len(rows) != 785 or len(frames) != 256 or geometry["verified_scan_count"] != 256 or extracted["extracted_count"] != 256:
        raise ValueError("Frozen held-out denominator/source count differs")
    if len({(row["frame_id"], row["reference_id"], row["detection_id"]) for row in rows}) != 785:
        raise ValueError("Repeated frozen match identity")
    if len({row["frame_id"] for row in rows}) != 240:
        raise ValueError("Frozen contributing-frame count changed")
    pairs = [pair_at_support(row, 3) for row in rows]
    pairs = [row for row in pairs if row is not None]
    if len(pairs) != sum(row["status"] == "supported" and row["paired_error"] is not None for row in rows):
        raise ValueError("Primary paired coverage and saved references disagree")
    for pair in pairs:
        source = next(row for row in rows if row["frame_id"] == pair["frame_id"] and
                      row["reference_id"] == pair["reference_id"] and
                      row["detection_id"] == pair["detection_id"])
        if abs(pair["mask_minus_box_absolute_error_m"] - source["paired_error"]["mask_minus_box_absolute_m"]) > 1e-12:
            raise ValueError("Frozen-primary paired error mismatch")
    fields = list(pairs[0]) if pairs else []
    with (OUT / "paired_evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)

    drive = clustered_bootstrap(pairs, "drive_id")
    frame = clustered_bootstrap(pairs, "frame_id")
    _write("bootstrap_draws.json", {"drive_cluster": drive, "frame_cluster_sensitivity": frame})
    thresholds = {}
    for minimum in (1, 3, 5, 10):
        qualifying = [pair_at_support(row, minimum) for row in rows]
        qualifying = [row for row in qualifying if row is not None]
        thresholds[str(minimum)] = {"eligible_pairs": len(qualifying),
                                   "statistics": paired_metrics(qualifying)}
    supported = [row for row in rows if row["status"] == "supported"]
    unsupported = [row for row in rows if row["status"] != "supported"]
    if len(supported) + len(unsupported) != 785:
        raise AssertionError("Coverage partition failed")
    rejected = sum((Counter(row["rejected_counts"]) for row in rows), Counter())
    support_hist = Counter("3+" if row["support_bins"] >= 3 else str(row["support_bins"])
                           for row in rows)
    coverage = {
        "selected_heldout_frames": 256, "verified_matching_scans": 256,
        "frozen_matched_objects": 785, "frozen_contributing_frames": 240,
        "supported_reference_count": len(supported), "unsupported_reference_count": len(unsupported),
        "primary_paired_count": len(pairs),
        "supported_without_both_frozen_medians": len(supported) - len(pairs),
        "support_bins_zero_one_two_three_plus": {key: support_hist.get(key, 0) for key in ("0", "1", "2", "3+")},
        "unsupported_reasons": dict(Counter(row["status"] for row in unsupported)),
        "supported_by_class": dict(Counter(row["class_name"] for row in supported)),
        "unsupported_by_class": dict(Counter(row["class_name"] for row in unsupported)),
        "supported_by_occlusion": dict(Counter(str(row["occluded"]) for row in supported)),
        "unsupported_by_occlusion": dict(Counter(str(row["occluded"]) for row in unsupported)),
        "supported_by_truncation": dict(Counter("positive" if row["truncated"] > 0 else "zero" for row in supported)),
        "unsupported_by_truncation": dict(Counter("positive" if row["truncated"] > 0 else "zero" for row in unsupported)),
        "object_wise_rejections": dict(rejected),
        "object_wise_rejection_semantics": "The same projected return may be checked against multiple matched objects; totals are not distinct points.",
        "per_frame_supported_counts": dict(sorted(Counter(row["frame_id"] for row in pairs).items())),
        "per_drive_supported_counts": dict(sorted(Counter(row["drive_id"] for row in pairs).items())),
        "all_frozen_matches_by_drive": dict(sorted(Counter(row["drive_id"] for row in rows).items())),
        "unsupported_by_drive": dict(sorted(Counter(row["drive_id"] for row in unsupported).items())),
        "supported_by_annotated_bottom_center_depth": dict(Counter(
            _depth_bin(row["original_m5_bottom_center_z_m"]) for row in supported)),
        "unsupported_by_annotated_bottom_center_depth": dict(Counter(
            _depth_bin(row["original_m5_bottom_center_z_m"]) for row in unsupported)),
        "all_object_support_bin_min": min(row["support_bins"] for row in rows),
        "all_object_support_bin_max": max(row["support_bins"] for row in rows),
        "supported_lidar_z_min_m": min((row["lidar_z_m"] for row in supported), default=None),
        "supported_lidar_z_max_m": max((row["lidar_z_m"] for row in supported), default=None),
        "archive_sha256": extracted["archive_sha256"],
        "protocol_sha256": geometry["protocol_sha256"],
    }
    _write("coverage_summary.json", coverage)
    summary = {
        "primary_estimand": "paired frozen M4 box/mask median errors to common sparse LiDAR object-surface camera-2 axial-Z proxy, conditional on three accepted bins",
        "primary": paired_metrics(pairs),
        "drive_cluster_95_interval_mean_delta_m": drive["interval_95_m"],
        "drive_cluster_contributing_drives": drive["group_count"],
        "frame_cluster_sensitivity_95_interval_mean_delta_m": frame["interval_95_m"],
        "drive_equal_sensitivity": drive_equal_mean(pairs),
        "support_threshold_sensitivity": thresholds,
        "class_descriptive": _group(pairs, lambda row: row["class_name"]),
        "depth_descriptive": _group(pairs, lambda row: _depth_bin(row["reference_lidar_z_m"])),
        "occlusion_descriptive": _group(pairs, lambda row: row["occluded"]),
        "truncation_descriptive": _group(pairs, lambda row: "positive" if row["truncated"] > 0 else "zero"),
        "support_count_descriptive": _group(pairs, lambda row: _support_bin(row["support_bins"])),
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "tie_tolerance_m": TIE_TOLERANCE_M,
        "split_limit": "Frame-disjoint, not drive-disjoint; 34 development drives and ten pilot drives overlap held-out. Drive clustering cannot undo development exposure.",
        "reference_limit": "Sparse annotation-gated LiDAR observed surface proxy, not dense RGB-visible-surface ground truth or KITTI 3D-box bottom-center Z.",
    }
    _write("statistical_summary.json", summary)
    supported_with_error = [row for row in rows if row["paired_error"] is not None]
    failure = {
        "largest_absolute_monocular_error": _top(supported_with_error,
            lambda row: max(abs(row["paired_error"]["box_signed_m"]),
                            abs(row["paired_error"]["mask_signed_m"]))),
        "largest_mask_disadvantage": _top(supported_with_error,
            lambda row: row["paired_error"]["mask_minus_box_absolute_m"]),
        "largest_mask_advantage": _top(supported_with_error,
            lambda row: row["paired_error"]["mask_minus_box_absolute_m"], reverse=False),
        "largest_retained_z_spread": _top(rows,
            lambda row: row["support_z_spread_m"] or 0),
        "lowest_support": _top(rows, lambda row: row["support_bins"], reverse=False),
        "largest_bottom_center_vs_sparse_z_disagreement": _top(supported,
            lambda row: abs(row["original_m5_bottom_center_z_m"] - row["lidar_z_m"])),
        "occluded_and_truncated": _top([row for row in rows if row["occluded"] >= 2 or row["truncated"] > 0],
            lambda row: (row["occluded"], row["truncated"])),
        "unsupported_examples": [_case(row) for row in unsupported[:20]],
        "small_detector_boxes": _top(rows,
            lambda row: (row["detector_box_xyxy"][2] - row["detector_box_xyxy"][0]) *
                        (row["detector_box_xyxy"][3] - row["detector_box_xyxy"][1]), reverse=False),
    }
    _write("failure_cases.json", failure)
    print(json.dumps({"coverage": {key: coverage[key] for key in ("supported_reference_count",
                                                                "unsupported_reference_count", "primary_paired_count")},
                      "primary": summary["primary"],
                      "drive_interval": drive["interval_95_m"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
