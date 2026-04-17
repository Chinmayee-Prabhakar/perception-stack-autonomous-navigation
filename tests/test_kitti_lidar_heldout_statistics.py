"""Invented paired errors and clusters; no real held-out scan in tests."""

import pytest

from perception_stack.kitti_lidar.heldout_statistics import (
    clustered_bootstrap, drive_equal_mean, pair_at_support, paired_metrics,
)


def _record(bins, *, box=10.0, mask=12.0, status="supported"):
    measured = [{"camera2_z_m": z} for z in bins]
    return {"frame_id": "000001", "reference_id": 0, "detection_id": 1,
            "drive_id": "drive_a", "class_name": "Car", "occluded": 0,
            "truncated": 0.0, "accepted_bins": measured,
            "status": status, "lidar_z_m": 10.0 if status == "supported" else None,
            "frozen_box_median_z_m": box, "frozen_mask_median_z_m": mask,
            "original_m5_bottom_center_z_m": 11.0}


def test_pair_uses_common_median_and_preserves_frozen_predictions():
    record = _record([9.0, 10.0, 11.0])
    pair = pair_at_support(record, 3)
    assert pair["reference_lidar_z_m"] == 10.0
    assert pair["box_signed_error_m"] == 0.0
    assert pair["mask_signed_error_m"] == 2.0
    assert pair["mask_minus_box_absolute_error_m"] == 2.0
    assert record["accepted_bins"][0]["camera2_z_m"] == 9.0


def test_support_sensitivity_does_not_change_primary_three_bin_rule():
    record = _record([10.0, 12.0], status="unsupported_insufficient_bins")
    assert pair_at_support(record, 3) is None
    assert pair_at_support(record, 1)["reference_lidar_z_m"] == 11.0
    assert pair_at_support(record, 5) is None
    with pytest.raises(ValueError):
        pair_at_support(record, 2)


def test_primary_reference_mismatch_is_rejected():
    record = _record([9.0, 10.0, 11.0])
    record["lidar_z_m"] = 7.0
    with pytest.raises(ValueError):
        pair_at_support(record, 3)


def test_paired_metric_signs_and_ties():
    rows = [
        {"box_signed_error_m": 2.0, "mask_signed_error_m": 1.0,
         "mask_minus_box_absolute_error_m": -1.0},
        {"box_signed_error_m": -1.0, "mask_signed_error_m": -2.0,
         "mask_minus_box_absolute_error_m": 1.0},
        {"box_signed_error_m": 0.5, "mask_signed_error_m": -0.5,
         "mask_minus_box_absolute_error_m": 0.0},
    ]
    result = paired_metrics(rows)
    assert result["paired_count"] == 3
    assert result["mean_mask_minus_box_absolute_error_m"] == 0.0
    assert result["mask_win_fraction"] == pytest.approx(1 / 3)
    assert result["box_win_fraction"] == pytest.approx(1 / 3)
    assert result["tie_fraction"] == pytest.approx(1 / 3)
    assert result["box_mae_m"] == result["mask_mae_m"]


def test_cluster_draw_retains_every_object_in_sampled_drive():
    rows = [
        {"drive_id": "a", "frame_id": "1", "mask_minus_box_absolute_error_m": 0.0},
        {"drive_id": "a", "frame_id": "2", "mask_minus_box_absolute_error_m": 2.0},
        {"drive_id": "b", "frame_id": "3", "mask_minus_box_absolute_error_m": 10.0},
    ]
    first = clustered_bootstrap(rows, "drive_id", draws=30, seed=549, minimum_groups=2)
    second = clustered_bootstrap(rows, "drive_id", draws=30, seed=549, minimum_groups=2)
    assert first["draws"] == second["draws"]
    assert first["object_counts_by_group"] == {"a": 2, "b": 1}
    assert set(first["draws"]) <= {1.0, 4.0, 10.0}
    assert drive_equal_mean(rows)["mean_of_drive_means_m"] == 5.5


def test_too_few_drives_suppresses_interval():
    rows = [{"drive_id": str(i), "mask_minus_box_absolute_error_m": float(i)}
            for i in range(9)]
    result = clustered_bootstrap(rows, "drive_id")
    assert result["interval_95_m"] is None
    assert result["draws"] == []
