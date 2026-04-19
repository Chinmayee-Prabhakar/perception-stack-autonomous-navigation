"""Recalculate published paired summaries and charts without model inference.

This reads only the compact, trackable results/ snapshot. It never reads or
modifies frozen outputs, checkpoints, or KITTI data.
"""

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path

import numpy as np

from perception_stack.kitti_lidar.heldout_statistics import (
    clustered_bootstrap,
    drive_equal_mean,
    paired_metrics,
)


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=1e-10):
        raise ValueError(f"{name} differs from archived summary: {actual} vs {expected}")


def recalculate() -> dict:
    m5_rows = _read_csv(RESULTS / "m5_annotated_pairs.csv")
    m5_saved = json.loads((RESULTS / "m5_annotated_summary.json").read_text(encoding="utf-8"))
    if len(m5_rows) != 785 or m5_saved["primary_paired"]["paired_objects"] != 785:
        raise ValueError("Original M5 paired denominator changed")
    m5_box = np.array([float(row["box_absolute_proxy_error_m"]) for row in m5_rows])
    m5_mask = np.array([float(row["mask_absolute_proxy_error_m"]) for row in m5_rows])
    m5_delta = m5_mask - m5_box
    m5_groups = defaultdict(list)
    for row in m5_rows:
        m5_groups[row["frame_id"]].append(float(row["paired_mask_minus_box_absolute_proxy_error_m"]))
    frame_ids = sorted(m5_groups)
    rng = np.random.default_rng(m5_saved["frame_bootstrap_mean_difference_95_ci"]["seed"])
    draws = []
    for _ in range(m5_saved["frame_bootstrap_mean_difference_95_ci"]["replicates"]):
        sample = rng.choice(frame_ids, size=len(frame_ids), replace=True)
        draws.append(float(np.mean([value for frame_id in sample for value in m5_groups[frame_id]])))
    m5_ci = [float(x) for x in np.percentile(draws, (2.5, 97.5))]
    _close(float(m5_box.mean()), m5_saved["primary_paired"]["box"]["mean_absolute_proxy_error_m"], "M5 box MAE")
    _close(float(m5_mask.mean()), m5_saved["primary_paired"]["mask"]["mean_absolute_proxy_error_m"], "M5 mask MAE")
    _close(float(m5_delta.mean()), m5_saved["primary_paired"]["mean_mask_minus_box_absolute_proxy_error_m"], "M5 paired mean")
    for actual, expected in zip(m5_ci, m5_saved["frame_bootstrap_mean_difference_95_ci"]["mean_difference_95_percentile_ci_m"]):
        _close(actual, expected, "M5 frame-bootstrap interval")

    lidar_raw = _read_csv(RESULTS / "m5_l3_sparse_pairs.csv")
    lidar_saved = json.loads((RESULTS / "m5_l3_sparse_summary.json").read_text(encoding="utf-8"))
    coverage = json.loads((RESULTS / "m5_l3_coverage.json").read_text(encoding="utf-8"))
    fields = ("box_signed_error_m", "mask_signed_error_m", "mask_minus_box_absolute_error_m",
              "box_absolute_error_m", "mask_absolute_error_m")
    lidar = [{**row, **{field: float(row[field]) for field in fields}} for row in lidar_raw]
    if len(lidar) != 780 or coverage["frozen_matched_objects"] != 785 or coverage["unsupported_reference_count"] != 5:
        raise ValueError("Sparse LiDAR coverage differs from accepted denominator")
    primary = paired_metrics(lidar)
    for name, expected in lidar_saved["primary"].items():
        _close(primary[name], expected, f"LiDAR {name}")
    cluster = clustered_bootstrap(lidar, "drive_id")
    if cluster["group_count"] != lidar_saved["drive_cluster_contributing_drives"]:
        raise ValueError("Drive count changed")
    for actual, expected in zip(cluster["interval_95_m"], lidar_saved["drive_cluster_95_interval_mean_delta_m"]):
        _close(actual, expected, "Drive-cluster interval")
    drive = drive_equal_mean(lidar)
    _close(drive["mean_of_drive_means_m"], lidar_saved["drive_equal_sensitivity"]["mean_of_drive_means_m"],
           "Drive-equal mean")
    return {
        "m5_annotated_reference": {"paired_count": len(m5_rows), "box_mae_m": float(m5_box.mean()),
                                    "mask_mae_m": float(m5_mask.mean()), "mean_delta_m": float(m5_delta.mean()),
                                    "frame_bootstrap_95_m": m5_ci},
        "m5_l3_sparse_reference": {"paired_count": len(lidar), "box_mae_m": primary["box_mae_m"],
                                    "mask_mae_m": primary["mask_mae_m"],
                                    "mean_delta_m": primary["mean_mask_minus_box_absolute_error_m"],
                                    "mask_win_fraction": primary["mask_win_fraction"],
                                    "drive_cluster_95_m": cluster["interval_95_m"],
                                    "contributing_drives": cluster["group_count"],
                                    "drive_equal_mean_delta_m": drive["mean_of_drive_means_m"],
                                    "supported": coverage["supported_reference_count"],
                                    "unsupported": coverage["unsupported_reference_count"]},
    }


def charts(result: dict, output: Path) -> None:
    runtime = REPO / "work/runtime/matplotlib"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(runtime)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ("#156A83", "#E1883A")

    def mae_chart(name: str, title: str, subtitle: str, filename: str) -> None:
        row = result[name]
        fig, ax = plt.subplots(figsize=(6.3, 3.8), layout="constrained")
        bars = ax.bar(("Box", "Mask"), (row["box_mae_m"], row["mask_mae_m"]), color=palette, width=.55)
        ax.bar_label(bars, fmt="%.3f m", padding=4)
        ax.set_ylim(0, max(row["box_mae_m"], row["mask_mae_m"]) * 1.25)
        ax.set_ylabel("Mean absolute error (m)")
        ax.set_title(title + "\n" + subtitle, loc="left", fontsize=11)
        fig.savefig(output / filename, dpi=170)
        plt.close(fig)

    mae_chart("m5_annotated_reference", "Annotated bottom-center Z proxy",
              "785 frozen pairs; frame bootstrap", "m5_annotated_mae.png")
    mae_chart("m5_l3_sparse_reference", "Sparse LiDAR surface Z proxy",
              "780 supported pairs; drive-cluster bootstrap", "m5_l3_sparse_mae.png")

    lidar = _read_csv(RESULTS / "m5_l3_sparse_pairs.csv")
    delta = np.array([float(row["mask_minus_box_absolute_error_m"]) for row in lidar])
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    ax.hist(delta, bins=45, color="#156A83", edgecolor="white")
    ax.axvline(0, color="#333333", linewidth=1)
    ax.axvline(delta.mean(), color="#CB6535", linestyle="--", label=f"Object mean {delta.mean():+.3f} m")
    ax.set(xlabel="Mask minus box absolute error (m)", ylabel="Supported object pairs",
           title="Paired sparse-reference differences")
    ax.legend()
    fig.savefig(output / "paired_difference.png", dpi=170)
    plt.close(fig)

    sparse = result["m5_l3_sparse_reference"]
    fig, ax = plt.subplots(figsize=(6.3, 3.8), layout="constrained")
    values = (sparse["mean_delta_m"], sparse["drive_equal_mean_delta_m"])
    bars = ax.bar(("Object weighted", "Drive equal"), values, color=("#CB6535", "#156A83"), width=.55)
    ax.bar_label(bars, labels=[f"{value:+.3f} m" for value in values], padding=4)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set(ylabel="Mean mask minus box absolute error (m)",
           title="Weighting changes the mean comparison")
    ax.set_ylim(min(values) - .09, max(values) + .09)
    fig.savefig(output / "drive_weighting.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="check numbers without writing charts")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/public_results_reproduced"),
                        help="fresh directory within ignored outputs/")
    args = parser.parse_args()
    result = recalculate()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.check_only:
        return
    target = (REPO / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_root = (REPO / "outputs").resolve()
    if not target.is_relative_to(output_root) or target == output_root:
        raise ValueError("Chart output must be a child of ignored outputs/")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Choose a fresh empty output directory")
    target.mkdir(parents=True, exist_ok=True)
    charts(result, target)
    (target / "recalculated_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                                       encoding="utf-8")
    print(f"Wrote recalculated charts to {target}")


if __name__ == "__main__":
    main()
