"""Write machine-readable evaluation artifacts and explicitly labeled plots."""

import csv
import json
from pathlib import Path

from .metrics import ProxyRow


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_rows_csv(path: Path, rows: list[ProxyRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ProxyRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def write_plots(output_dir: Path, rows: list[ProxyRow], summary: dict) -> list[str]:
    paired = [row for row in rows if row.paired_mask_minus_box_absolute_proxy_error_m is not None]
    if not paired:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    figure, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([row.box_absolute_proxy_error_m for row in paired],
               [row.mask_absolute_proxy_error_m for row in paired], s=14, alpha=0.7)
    limit = max(max(row.box_absolute_proxy_error_m, row.mask_absolute_proxy_error_m) for row in paired)
    ax.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Box absolute object-reference Z proxy error (m)",
           ylabel="Mask absolute object-reference Z proxy error (m)",
           title=f"Paired proxy errors (n={len(paired)})")
    figure.tight_layout()
    path = plots / "paired_box_vs_mask_proxy_error.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    files.append(str(path))

    figure, ax = plt.subplots(figsize=(7, 4))
    differences = [row.paired_mask_minus_box_absolute_proxy_error_m for row in paired]
    ax.scatter(range(len(differences)), differences, s=12, alpha=0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Paired matched-object index", ylabel="Mask minus box absolute proxy error (m)",
           title="Per-object paired proxy differences")
    figure.tight_layout()
    path = plots / "paired_difference_by_object.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    files.append(str(path))

    figure, ax = plt.subplots(figsize=(7, 4))
    ax.scatter([row.reference_z_m for row in paired],
               [row.paired_mask_minus_box_absolute_proxy_error_m for row in paired],
               s=14, alpha=0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="KITTI annotated bottom-center reference Z (m)",
           ylabel="Mask minus box absolute proxy error (m)",
           title="Paired proxy difference by reference depth")
    figure.tight_layout()
    path = plots / "paired_difference_by_reference_z.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    files.append(str(path))

    figure, ax = plt.subplots(figsize=(6, 4))
    names = ["Matched", "Unmatched reference", "Unmatched detection", "Ignored detection"]
    totals = summary["counts"]
    ax.bar(names, [totals["matched_pairs"], totals["unmatched_references"],
                   totals["unmatched_detections"], totals["ignored_detections"]])
    ax.tick_params(axis="x", rotation=20)
    ax.set(ylabel="Count", title="Object matching outcomes")
    figure.tight_layout()
    path = plots / "matching_outcomes.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    files.append(str(path))

    counts = summary["method_coverage_on_all_matched"]
    figure, ax = plt.subplots(figsize=(6, 4))
    names = ["Matched", "Box valid", "Mask valid", "Both valid"]
    values = [counts["denominator_matched"], counts["box_valid"],
              counts["mask_valid"], counts["both_valid_primary_paired"]]
    ax.bar(names, values)
    ax.set(ylabel="Object count", title="Matched-object coverage and failures")
    figure.tight_layout()
    path = plots / "matching_coverage.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    files.append(str(path))
    for field, title in (("class_name", "Class"), ("occluded", "Occlusion state")):
        keys = sorted({getattr(row, field) for row in paired}, key=str)
        figure, ax = plt.subplots(figsize=(max(5, 1.5 * len(keys)), 4))
        ax.boxplot([[row.paired_mask_minus_box_absolute_proxy_error_m for row in paired
                     if getattr(row, field) == key] for key in keys], tick_labels=[str(key) for key in keys])
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel=title, ylabel="Mask minus box absolute proxy error (m)",
               title=f"Exploratory paired difference by {title.lower()}")
        figure.tight_layout()
        path = plots / f"paired_difference_by_{field}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        files.append(str(path))

    figure, ax = plt.subplots(figsize=(6, 4))
    ax.hist([row.mask_area_pixels for row in rows if row.mask_area_pixels is not None], bins=20)
    ax.set(xlabel="Predicted full instance-mask pixel count", ylabel="Matched objects",
           title="Mask area distribution; object-reference Z proxy study")
    figure.tight_layout()
    path = plots / "mask_area_distribution.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    files.append(str(path))
    return files


def write_markdown_report(path: Path, summary: dict, *, subset: str, selected_count: int,
                          processed_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if processed_count == 0:
        status = ("Real KITTI data not yet approved or available"
                  if subset == "none" else "Selected real-data run incomplete before any frame was processed")
        text = (
            "# KITTI object-reference Z proxy evaluation\n\n"
            f"**Status: {status}.** No pilot or held-out images "
            "have been evaluated in this output. No proxy accuracy numbers or surface-depth measurements exist.\n\n"
            "The offline loader, matching, paired metrics and frame bootstrap are covered by synthetic "
            "tests. A frozen real-data split manifest will be created only after approved official "
            "training images, labels and calibration are present.\n\n"
            "The sparse visible-surface depth diagnostic is pending an independently aligned reference.\n"
        )
    else:
        counts = summary["counts"]
        paired = summary["primary_paired"]
        coverage = summary["method_coverage_on_all_matched"]
        ci = summary["frame_bootstrap_mean_difference_95_ci"]
        def format_m(value):
            return "unavailable" if value is None else f"{value:.3f} m"
        interval = ci["mean_difference_95_percentile_ci_m"]
        interval_text = ("unavailable" if interval is None else
                         f"[{interval[0]:.3f}, {interval[1]:.3f}] m")
        groups = summary["exploratory_subgroups"]
        exploratory = ""
        for field, label in (("class_name", "Class"), ("range_bin", "Reference Z range"),
                             ("occluded", "Occlusion")):
            values = groups.get(field, {})
            if values:
                exploratory += f"\n### {label} (exploratory)\n\n| Group | Paired objects | Mean mask − box absolute proxy error |\n| --- | ---: | ---: |\n"
                exploratory += "".join(
                    f"| {name} | {value['paired_objects']} | "
                    f"{format_m(value['mean_mask_minus_box_absolute_proxy_error_m'])} |\n"
                    for name, value in values.items())
        text = (
            "# KITTI object-reference Z proxy evaluation\n\n"
            f"Subset: **{subset}**. Processed **{processed_count}/{selected_count}** selected frames. "
            f"Matched objects: **{counts['matched_pairs']}**; both methods valid: "
            f"**{paired['paired_objects']}**.\n\n"
            f"Execution status: **{summary['status']}**."
            + (f" Failure: **{summary['evaluation_failure']}**." if summary.get("evaluation_failure") else "") + "\n\n"
            "The reference is KITTI's annotated bottom-center 3D-box camera-axis Z. "
            "This is a proxy comparison, not direct visible-surface depth accuracy or "
            "object-center 3D localization. Negative paired mask-minus-box absolute error "
            "favors the mask method for this proxy only.\n\n"
            f"Eligible references: **{counts['eligible_references']}**; unmatched references: "
            f"**{counts['unmatched_references']}**. Eligible detections: "
            f"**{counts['eligible_detections']}**; unmatched detections: "
            f"**{counts['unmatched_detections']}**; ignored detections: "
            f"**{counts['ignored_detections']}**. Among matched objects, box valid: "
            f"**{coverage['box_valid']}** and mask valid: **{coverage['mask_valid']}**.\n\n"
            "| Primary paired proxy metric | Box | Mask |\n| --- | ---: | ---: |\n"
            f"| Mean absolute error | {format_m(paired['box']['mean_absolute_proxy_error_m'])} | {format_m(paired['mask']['mean_absolute_proxy_error_m'])} |\n"
            f"| Median absolute error | {format_m(paired['box']['median_absolute_proxy_error_m'])} | {format_m(paired['mask']['median_absolute_proxy_error_m'])} |\n"
            f"| Signed mean bias | {format_m(paired['box']['signed_mean_proxy_bias_m'])} | {format_m(paired['mask']['signed_mean_proxy_bias_m'])} |\n"
            f"| RMSE | {format_m(paired['box']['rmse_object_reference_z_proxy_m'])} | {format_m(paired['mask']['rmse_object_reference_z_proxy_m'])} |\n\n"
            f"Mean paired mask − box absolute proxy error: **{format_m(paired['mean_mask_minus_box_absolute_proxy_error_m'])}**; "
            f"median: **{format_m(paired['median_mask_minus_box_absolute_proxy_error_m'])}**. "
            f"Mask lower / box lower / tied: **{paired['mask_lower_error_percent']:.1f}% / "
            f"{paired['box_lower_error_percent']:.1f}% / {paired['tie_percent']:.1f}%**. "
            f"Frame-bootstrap 95% interval for the mean difference: **{interval_text}** "
            f"({ci['status']}; {ci['distinct_frames']} contributing frames; "
            f"seed {ci['seed']}, {ci['replicates']} replicates).\n\n"
            "Method failures remain in matched-object coverage denominators. "
            "The primary paired metrics use only matches where both methods returned valid Z. "
            "Frame-bootstrap confidence intervals are omitted when too few distinct frames contribute. "
            "Drive identity is unavailable, so the frame bootstrap cannot account for related frames "
            "from one drive; subgroup comparisons are exploratory.\n"
            f"{exploratory}\n"
            "Surface-depth diagnostic: **pending** independent aligned reference and association policy. "
            "VKITTI-to-real domain shift, occlusion, mask contamination, saturation, and drive "
            "correlation limit interpretation.\n"
        )
    path.write_text(text, encoding="utf-8")
