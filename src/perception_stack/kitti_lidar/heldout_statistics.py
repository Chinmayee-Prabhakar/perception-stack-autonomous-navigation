"""Descriptive paired LiDAR-proxy statistics and fixed clustered uncertainty."""

from collections import defaultdict
import math

import numpy as np


TIE_TOLERANCE_M = 1e-9
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 549


def pair_at_support(record: dict, minimum_bins: int) -> dict | None:
    """Use the same accepted bins under a prespecified support sensitivity rule."""
    if minimum_bins not in (1, 3, 5, 10):
        raise ValueError("Only prespecified sensitivity thresholds are allowed")
    bins = record["accepted_bins"]
    if len(bins) < minimum_bins:
        return None
    box = record["frozen_box_median_z_m"]
    mask = record["frozen_mask_median_z_m"]
    if box is None or mask is None:
        return None
    reference = float(np.median([point["camera2_z_m"] for point in bins]))
    if minimum_bins == 3 and record["status"] == "supported":
        if not math.isclose(reference, record["lidar_z_m"], abs_tol=1e-12):
            raise ValueError("Primary sparse reference differs from accepted v1 median")
    box_error = box - reference
    mask_error = mask - reference
    return {"frame_id": record["frame_id"], "reference_id": record["reference_id"],
            "detection_id": record["detection_id"], "drive_id": record["drive_id"],
            "class_name": record["class_name"], "occluded": record["occluded"],
            "truncated": record["truncated"], "support_bins": len(bins),
            "reference_lidar_z_m": reference,
            "original_bottom_center_z_m": record["original_m5_bottom_center_z_m"],
            "box_frozen_z_m": box, "mask_frozen_z_m": mask,
            "box_signed_error_m": box_error, "mask_signed_error_m": mask_error,
            "box_absolute_error_m": abs(box_error), "mask_absolute_error_m": abs(mask_error),
            "mask_minus_box_absolute_error_m": abs(mask_error) - abs(box_error)}


def paired_metrics(pairs: list[dict]) -> dict:
    if not pairs:
        return {"paired_count": 0}
    box = np.asarray([row["box_signed_error_m"] for row in pairs], dtype=np.float64)
    mask = np.asarray([row["mask_signed_error_m"] for row in pairs], dtype=np.float64)
    delta = np.abs(mask) - np.abs(box)
    ties = np.abs(delta) <= TIE_TOLERANCE_M
    return {"paired_count": len(pairs),
            "box_signed_mean_m": float(box.mean()),
            "mask_signed_mean_m": float(mask.mean()),
            "box_mae_m": float(np.abs(box).mean()),
            "mask_mae_m": float(np.abs(mask).mean()),
            "box_median_absolute_error_m": float(np.median(np.abs(box))),
            "mask_median_absolute_error_m": float(np.median(np.abs(mask))),
            "box_rmse_m": float(np.sqrt(np.mean(box * box))),
            "mask_rmse_m": float(np.sqrt(np.mean(mask * mask))),
            "mean_mask_minus_box_absolute_error_m": float(delta.mean()),
            "median_mask_minus_box_absolute_error_m": float(np.median(delta)),
            "mask_win_fraction": float(np.mean((delta < -TIE_TOLERANCE_M))),
            "box_win_fraction": float(np.mean((delta > TIE_TOLERANCE_M))),
            "tie_fraction": float(np.mean(ties))}


def clustered_bootstrap(pairs: list[dict], group_key: str, *,
                        draws: int = BOOTSTRAP_DRAWS,
                        seed: int = BOOTSTRAP_SEED,
                        minimum_groups: int = 10) -> dict:
    groups = defaultdict(list)
    for row in pairs:
        groups[row[group_key]].append(row["mask_minus_box_absolute_error_m"])
    keys = sorted(groups)
    counts = {key: len(groups[key]) for key in keys}
    if len(keys) < minimum_groups:
        return {"group_key": group_key, "group_count": len(keys),
                "object_counts_by_group": counts, "draws": [],
                "interval_95_m": None, "reason": "fewer_than_ten_contributing_groups"}
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(groups[key], dtype=np.float64) for key in keys]
    results = []
    for _ in range(draws):
        indices = rng.integers(0, len(keys), size=len(keys))
        values = np.concatenate([arrays[index] for index in indices])
        results.append(float(values.mean()))
    return {"group_key": group_key, "group_count": len(keys),
            "object_counts_by_group": counts, "draws": results,
            "seed": seed, "draw_count": draws,
            "interval_95_m": [float(np.percentile(results, 2.5)),
                               float(np.percentile(results, 97.5))]}


def drive_equal_mean(pairs: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in pairs:
        groups[row["drive_id"]].append(row["mask_minus_box_absolute_error_m"])
    means = {key: float(np.mean(values)) for key, values in sorted(groups.items())}
    return {"contributing_drives": len(groups),
            "mean_of_drive_means_m": float(np.mean(list(means.values()))) if means else None,
            "per_drive_mean_m": means}
