"""Object-reference Z proxy errors, paired denominators and frame bootstrap."""

from dataclasses import asdict, dataclass
import math

import numpy as np

from perception_stack.fusion.results import FrameFusionResults
from perception_stack.results import FrameDetections
from .dataset import KittiAnnotation, KittiObjectFrame
from .matching import FrameMatches, match_frame
from .protocol import ClassMapping


def reference_range_bin(z_m: float) -> str:
    if not math.isfinite(z_m) or z_m <= 0:
        raise ValueError("Reference camera-axis Z must be finite and positive")
    if z_m < 15:
        return "0-15m"
    if z_m < 30:
        return "15-30m"
    if z_m < 50:
        return "30-50m"
    if z_m < 80:
        return "50-80m"
    return "80m-plus"


@dataclass(frozen=True)
class ProxyRow:
    frame_id: str
    reference_id: int
    detection_id: int
    class_name: str
    confidence: float
    match_iou: float
    reference_z_m: float
    box_median_z_m: float | None
    mask_median_z_m: float | None
    box_signed_proxy_error_m: float | None
    mask_signed_proxy_error_m: float | None
    box_absolute_proxy_error_m: float | None
    mask_absolute_proxy_error_m: float | None
    paired_mask_minus_box_absolute_proxy_error_m: float | None
    box_valid_pixels: int
    mask_valid_pixels: int
    mask_area_pixels: int | None
    occluded: int
    truncated: float
    box_status: str
    mask_status: str
    range_bin: str
    image_horizontal_half: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FrameEvaluation:
    frame_id: str
    matches: FrameMatches
    rows: tuple[ProxyRow, ...]

    def to_dict(self) -> dict:
        return {"frame_id": self.frame_id, "matching": self.matches.to_dict(),
                "paired_object_reference_z_proxy_rows": [row.to_dict() for row in self.rows]}


def evaluate_frame(
    sample: KittiObjectFrame, detections: FrameDetections,
    fusion: FrameFusionResults, mapping: ClassMapping, *,
    iou_threshold: float = 0.5, ignored_ioa_threshold: float = 0.5,
    min_reference_height_px: float = 0, max_truncation: float = 1,
    allowed_occlusion: tuple[int, ...] = (0, 1, 2, 3),
) -> FrameEvaluation:
    frame = sample.frame
    if frame.frame_id != detections.frame_id or frame.frame_id != fusion.frame_id:
        raise ValueError("Evaluation frame IDs differ")
    if frame.size != detections.image_size or frame.size != fusion.image_size:
        raise ValueError("Evaluation image dimensions differ")
    if len(detections.detections) != len(fusion.objects):
        raise ValueError("Fusion and detection counts differ")
    for index, (detection, fused) in enumerate(zip(detections.detections, fusion.objects)):
        if fused.detection_id != index or fused.class_name != detection.class_name or fused.box != detection.box:
            raise ValueError("Fusion objects differ from matched detections")
    matches = match_frame(
        detections, sample.annotations, mapping, iou_threshold=iou_threshold,
        ignored_ioa_threshold=ignored_ioa_threshold,
        min_reference_height_px=min_reference_height_px,
        max_truncation=max_truncation, allowed_occlusion=allowed_occlusion,
    )
    by_id: dict[int, KittiAnnotation] = {a.object_id: a for a in sample.annotations}
    rows = []
    for pair in matches.pairs:
        annotation = by_id[pair.reference_id]
        fused = fusion.objects[pair.detection_id]
        detection = detections.detections[pair.detection_id]
        assert annotation.reference_z_m is not None
        reference_z = annotation.reference_z_m
        box_stats = fused.box_method.statistics
        mask_stats = fused.mask_method.statistics
        box_z = box_stats.median_z_m if box_stats.status == "ok" else None
        mask_z = mask_stats.median_z_m if mask_stats.status == "ok" else None
        box_signed = box_z - reference_z if box_z is not None else None
        mask_signed = mask_z - reference_z if mask_z is not None else None
        box_abs = abs(box_signed) if box_signed is not None else None
        mask_abs = abs(mask_signed) if mask_signed is not None else None
        paired_difference = mask_abs - box_abs if box_abs is not None and mask_abs is not None else None
        rows.append(ProxyRow(
            frame.frame_id, pair.reference_id, pair.detection_id, annotation.class_name,
            detection.confidence, pair.iou, reference_z, box_z, mask_z,
            box_signed, mask_signed, box_abs, mask_abs, paired_difference,
            box_stats.valid_pixels, mask_stats.valid_pixels,
            int(detection.mask.sum()) if detection.mask is not None else None,
            annotation.occluded, annotation.truncated, box_stats.status, mask_stats.status,
            reference_range_bin(reference_z),
            "left" if (detection.box.x1 + detection.box.x2) / 2 < frame.size[0] / 2 else "right",
        ))
    return FrameEvaluation(frame.frame_id, matches, tuple(rows))


def _method_metrics(rows: list[ProxyRow], method: str) -> dict:
    errors = np.asarray([getattr(row, f"{method}_signed_proxy_error_m") for row in rows], dtype=np.float64)
    return {
        "mean_absolute_proxy_error_m": float(np.abs(errors).mean()) if errors.size else None,
        "median_absolute_proxy_error_m": float(np.median(np.abs(errors))) if errors.size else None,
        "signed_mean_proxy_bias_m": float(errors.mean()) if errors.size else None,
        "rmse_object_reference_z_proxy_m": float(np.sqrt(np.mean(errors * errors))) if errors.size else None,
    }


def _paired_metrics(rows: list[ProxyRow]) -> dict:
    if not rows:
        return {"paired_objects": 0, "box": _method_metrics([], "box"),
                "mask": _method_metrics([], "mask"),
                "mean_mask_minus_box_absolute_proxy_error_m": None,
                "median_mask_minus_box_absolute_proxy_error_m": None,
                "mask_lower_error_percent": None, "box_lower_error_percent": None,
                "tie_percent": None}
    differences = np.asarray([row.paired_mask_minus_box_absolute_proxy_error_m for row in rows], dtype=np.float64)
    tie = np.abs(differences) <= 1e-9
    return {
        "paired_objects": len(rows), "box": _method_metrics(rows, "box"),
        "mask": _method_metrics(rows, "mask"),
        "mean_mask_minus_box_absolute_proxy_error_m": float(differences.mean()),
        "median_mask_minus_box_absolute_proxy_error_m": float(np.median(differences)),
        "mask_lower_error_percent": float(100 * np.count_nonzero(differences < -1e-9) / len(rows)),
        "box_lower_error_percent": float(100 * np.count_nonzero(differences > 1e-9) / len(rows)),
        "tie_percent": float(100 * np.count_nonzero(tie) / len(rows)),
    }


def frame_bootstrap_ci(
    rows: list[ProxyRow], *, seed: int, replicates: int,
    min_distinct_frames: int,
) -> dict:
    if type(seed) is not int or type(replicates) is not int or replicates < 1 or type(min_distinct_frames) is not int or min_distinct_frames < 2:
        raise ValueError("Bootstrap seed, replicates and minimum frame count must be valid")
    groups: dict[str, list[float]] = {}
    for row in rows:
        if row.paired_mask_minus_box_absolute_proxy_error_m is not None:
            groups.setdefault(row.frame_id, []).append(row.paired_mask_minus_box_absolute_proxy_error_m)
    frame_ids = sorted(groups)
    if len(frame_ids) < min_distinct_frames:
        return {"status": "insufficient_distinct_frames", "distinct_frames": len(frame_ids),
                "required_frames": min_distinct_frames, "seed": seed, "replicates": replicates,
                "mean_difference_95_percentile_ci_m": None}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(replicates):
        sampled = rng.choice(frame_ids, size=len(frame_ids), replace=True)
        values = [value for frame_id in sampled for value in groups[frame_id]]
        means.append(float(np.mean(values)))
    lo, hi = np.percentile(means, (2.5, 97.5))
    return {"status": "ok", "distinct_frames": len(frame_ids),
            "required_frames": min_distinct_frames, "seed": seed, "replicates": replicates,
            "mean_difference_95_percentile_ci_m": [float(lo), float(hi)],
            "resampling_unit": "frame"}


def aggregate_frames(
    frames: tuple[FrameEvaluation, ...], *, seed: int = 549,
    replicates: int = 1000, min_bootstrap_frames: int = 10,
) -> dict:
    ids = [frame.frame_id for frame in frames]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluated frame IDs")
    rows = [row for frame in frames for row in frame.rows]
    paired = [row for row in rows if row.paired_mask_minus_box_absolute_proxy_error_m is not None]
    matched = len(rows)
    box_available = sum(row.box_status == "ok" for row in rows)
    mask_available = sum(row.mask_status == "ok" for row in rows)
    totals = {
        "frames_processed": len(frames), "total_references": 0, "eligible_references": 0,
        "ignored_references": 0, "total_detections": 0, "eligible_detections": 0,
        "unsupported_detections": 0, "ignored_detections": 0,
        "matched_pairs": 0, "unmatched_references": 0, "unmatched_detections": 0,
    }
    class_coverage: dict[str, dict[str, int]] = {}
    ious = []
    for frame in frames:
        match = frame.matches.to_dict()
        for key in totals:
            if key != "frames_processed":
                totals[key] += match[key]
        ious.extend(match["match_ious"])
        for class_name in frame.matches.eligible_reference_classes.values():
            class_coverage.setdefault(class_name, {"eligible_references": 0, "matched": 0,
                                                   "unmatched_references": 0, "box_valid": 0,
                                                   "mask_valid": 0, "paired": 0,
                                                   "eligible_detections": 0,
                                                   "unmatched_detections": 0,
                                                   "ignored_detections": 0})
        for class_name in frame.matches.eligible_detection_classes.values():
            class_coverage.setdefault(class_name, {"eligible_references": 0, "matched": 0,
                                                   "unmatched_references": 0, "box_valid": 0,
                                                   "mask_valid": 0, "paired": 0,
                                                   "eligible_detections": 0,
                                                   "unmatched_detections": 0,
                                                   "ignored_detections": 0})
        for ref_id in frame.matches.eligible_reference_ids:
            # Matched IDs are already represented by rows. For unmatched IDs,
            # the reference class is carried by FrameMatches below.
            class_name = frame.matches.eligible_reference_classes[ref_id]
            class_coverage[class_name]["eligible_references"] += 1
            class_coverage[class_name]["unmatched_references"] += ref_id in frame.matches.unmatched_reference_ids
        for det_id in frame.matches.eligible_detection_ids:
            class_name = frame.matches.eligible_detection_classes[det_id]
            class_coverage[class_name]["eligible_detections"] += 1
            class_coverage[class_name]["unmatched_detections"] += det_id in frame.matches.unmatched_detection_ids
            class_coverage[class_name]["ignored_detections"] += det_id in frame.matches.ignored_detection_ids
        for row in frame.rows:
            item = class_coverage[row.class_name]
            item["matched"] += 1
            item["box_valid"] += row.box_status == "ok"
            item["mask_valid"] += row.mask_status == "ok"
            item["paired"] += row.paired_mask_minus_box_absolute_proxy_error_m is not None
    subgroups: dict[str, dict] = {}
    for field in ("class_name", "range_bin", "occluded", "image_horizontal_half"):
        for value in sorted({getattr(row, field) for row in paired}, key=str):
            group_rows = [row for row in paired if getattr(row, field) == value]
            subgroups.setdefault(field, {})[str(value)] = {
                **_paired_metrics(group_rows), "interpretation": "exploratory; no subgroup significance claim"}
    for name, predicate in (("none", lambda row: row.truncated == 0),
                            ("partial", lambda row: 0 < row.truncated <= 0.5),
                            ("severe", lambda row: row.truncated > 0.5)):
        group_rows = [row for row in paired if predicate(row)]
        if group_rows:
            subgroups.setdefault("truncation", {})[name] = {
                **_paired_metrics(group_rows), "interpretation": "exploratory; no subgroup significance claim"}
    for name, predicate in (("under_100", lambda row: row.mask_area_pixels is not None and row.mask_area_pixels < 100),
                            ("100_to_999", lambda row: row.mask_area_pixels is not None and 100 <= row.mask_area_pixels < 1000),
                            ("1000_plus", lambda row: row.mask_area_pixels is not None and row.mask_area_pixels >= 1000)):
        group_rows = [row for row in paired if predicate(row)]
        if group_rows:
            subgroups.setdefault("mask_area_pixels", {})[name] = {
                **_paired_metrics(group_rows), "interpretation": "exploratory; no subgroup significance claim"}
    return {
        "analysis": "KITTI annotated 3D bottom-center object-reference Z proxy; not visible-surface accuracy",
        "status": "computed" if frames else "no_real_frames_evaluated",
        "counts": totals,
        "method_coverage_on_all_matched": {
            "denominator_matched": matched,
            "box_valid": box_available, "box_missing": matched - box_available,
            "mask_valid": mask_available, "mask_missing": matched - mask_available,
            "both_valid_primary_paired": len(paired),
        },
        "primary_paired": _paired_metrics(paired),
        "frame_bootstrap_mean_difference_95_ci": frame_bootstrap_ci(
            paired, seed=seed, replicates=replicates,
            min_distinct_frames=min_bootstrap_frames),
        "class_coverage": dict(sorted(class_coverage.items())),
        "match_iou": {"count": len(ious), "median": float(np.median(ious)) if ious else None,
                      "mean": float(np.mean(ious)) if ious else None},
        "exploratory_subgroups": subgroups,
        "surface_depth_diagnostic": {"status": "pending_independent_aligned_reference",
                                     "measurements": None},
    }
