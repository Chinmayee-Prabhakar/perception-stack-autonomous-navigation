"""Prediction-independent oriented-cuboid association of sparse scan returns."""

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from perception_stack.kitti_eval.dataset import KittiAnnotation
from perception_stack.results import BoxXYXY
from .geometry import ProjectedReturn, ProjectionResult
from .protocol import LidarProtocol


REJECTION_CODES = ("outside_target_cuboid", "competing_cuboid",
                   "outside_annotation_box_2d", "outside_detection_box_2d",
                   "same_pixel_occluded")
REFERENCE_STATUSES = ("supported", "unsupported_insufficient_bins")


def _digest(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class SourceDigests:
    scan_sha256: str
    calibration_sha256: str
    protocol_sha256: str
    frozen_m5_frame_sha256: str

    def __post_init__(self) -> None:
        if not all(_digest(getattr(self, field)) for field in self.__dataclass_fields__):
            raise ValueError("Every source requires a lowercase SHA-256 digest")


@dataclass(frozen=True)
class FrozenObjectMatch:
    """An existing M5 row; its IDs/predictions must be read, never recomputed here."""

    frame_id: str
    reference_id: int
    detection_id: int
    reference_class_name: str
    mapped_detection_class_name: str
    detector_box: BoxXYXY
    box_median_z_m: float | None
    mask_median_z_m: float | None

    def __post_init__(self) -> None:
        if not self.frame_id or self.reference_id < 0 or self.detection_id < 0:
            raise ValueError("Frozen match requires valid frame/reference/detection IDs")
        if self.reference_class_name != self.mapped_detection_class_name:
            raise ValueError("Frozen match must preserve same-class mapping")
        if not isinstance(self.detector_box, BoxXYXY):
            raise TypeError("Frozen detector box is required")
        for value in (self.box_median_z_m, self.mask_median_z_m):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Frozen M4 medians must be finite positive Z or absent")


@dataclass(frozen=True)
class SparseReference:
    match: FrozenObjectMatch
    status: str
    accepted_bins: tuple[ProjectedReturn, ...]
    lidar_z_m: float | None
    support_bins: int
    candidate_in_target_count: int
    rejected_counts: Mapping[str, int]
    scan_total_rows: int
    scan_rejected_rows: int
    scan_nonfinite_xyz_rows: int
    scan_nonfinite_intensity_rows: int
    projected_valid_rows: int
    projection_rejected_counts: Mapping[str, int]
    sources: SourceDigests

    def __post_init__(self) -> None:
        for field in ("rejected_counts", "projection_rejected_counts"):
            object.__setattr__(self, field, MappingProxyType(dict(getattr(self, field))))

    def to_dict(self) -> dict:
        values = [point.camera2_z_m for point in self.accepted_bins]
        return {
            "frame_id": self.match.frame_id,
            "reference_id": self.match.reference_id,
            "detection_id": self.match.detection_id,
            "status": self.status,
            "reference_kind": "sparse_lidar_object_surface_camera2_axial_z",
            "lidar_z_m": self.lidar_z_m,
            "support_bins": self.support_bins,
            "support_z_min_m": min(values) if values else None,
            "support_z_max_m": max(values) if values else None,
            "candidate_in_target_count": self.candidate_in_target_count,
            "rejected_counts": dict(self.rejected_counts),
            "scan_total_rows": self.scan_total_rows,
            "scan_rejected_rows": self.scan_rejected_rows,
            "scan_nonfinite_xyz_rows": self.scan_nonfinite_xyz_rows,
            "scan_nonfinite_intensity_rows": self.scan_nonfinite_intensity_rows,
            "projected_valid_rows": self.projected_valid_rows,
            "projection_rejected_counts": dict(self.projection_rejected_counts),
            "accepted_bins": [{"pixel": list(point.pixel), "u": point.u, "v": point.v,
                               "camera2_z_m": point.camera2_z_m,
                               "reference_xyz_m": list(point.reference_xyz_m),
                               "original_scan_index": point.original_index}
                              for point in self.accepted_bins],
            "source_sha256": vars(self.sources),
            "frozen_box_median_z_m": self.match.box_median_z_m,
            "frozen_mask_median_z_m": self.match.mask_median_z_m,
        }


@dataclass(frozen=True)
class PairedSurfaceError:
    frame_id: str
    reference_id: int
    detection_id: int
    drive_id: str
    lidar_z_m: float
    box_signed_error_m: float
    mask_signed_error_m: float
    mask_minus_box_absolute_error_m: float


def _inside_half_open(box: BoxXYXY, u: float, v: float) -> bool:
    return box.x1 <= u < box.x2 and box.y1 <= v < box.y2


def inside_oriented_cuboid(reference_xyz_m: tuple[float, float, float],
                           annotation: KittiAnnotation, *, numeric_tolerance_m: float = 1e-12) -> bool:
    """KITTI x along length, y in [-h,0] from bottom, z along width."""
    if annotation.validity_status != "valid":
        return False
    if any(d <= 0 for d in annotation.dimensions_hwl_m):
        return False
    if numeric_tolerance_m != 1e-12:
        raise ValueError("Cuboid numeric tolerance is fixed, never a support-tuning parameter")
    dx, dy, dz = np.asarray(reference_xyz_m, dtype=np.float64) - annotation.location_xyz_m
    if not np.isfinite((dx, dy, dz)).all():
        return False
    c, s = math.cos(annotation.rotation_y), math.sin(annotation.rotation_y)
    local_x = c * dx - s * dz
    local_z = s * dx + c * dz
    h, w, length = annotation.dimensions_hwl_m
    t = numeric_tolerance_m
    return bool(-length / 2 - t <= local_x <= length / 2 + t and
                -h - t <= dy <= t and -w / 2 - t <= local_z <= w / 2 + t)


def associate_object(projection: ProjectionResult, annotations: tuple[KittiAnnotation, ...],
                     match: FrozenObjectMatch, sources: SourceDigests,
                     protocol: LidarProtocol) -> SparseReference:
    if projection.frame_id != match.frame_id or projection.scan_sha256 != sources.scan_sha256:
        raise ValueError("Projected scan does not match frozen frame/scan provenance")
    if projection.calibration_sha256 != sources.calibration_sha256 or protocol.sha256 != sources.protocol_sha256:
        raise ValueError("Calibration or protocol provenance mismatch")
    refs = {annotation.object_id: annotation for annotation in annotations}
    if len(refs) != len(annotations) or match.reference_id not in refs:
        raise ValueError("Missing or duplicate frozen reference annotation")
    target = refs[match.reference_id]
    if target.validity_status != "valid" or target.class_name != match.reference_class_name:
        raise ValueError("Frozen match and annotation class/status disagree")
    # Resolve all pixel winners first. An unannotated nearer point still occludes a target.
    winners: dict[tuple[int, int], ProjectedReturn] = {}
    for point in projection.returns:
        current = winners.get(point.pixel)
        if current is None or (point.camera2_z_m, point.original_index) < (current.camera2_z_m, current.original_index):
            winners[point.pixel] = point
    accepted = []
    rejected = {code: 0 for code in REJECTION_CODES}
    candidate_count = 0
    for point in projection.returns:
        xyz = point.reference_xyz_m
        if not inside_oriented_cuboid(xyz, target, numeric_tolerance_m=protocol.boundary_tolerance_m):
            rejected["outside_target_cuboid"] += 1
            continue
        candidate_count += 1
        if any(other.object_id != target.object_id and
               inside_oriented_cuboid(xyz, other, numeric_tolerance_m=protocol.boundary_tolerance_m)
               for other in annotations):
            rejected["competing_cuboid"] += 1
            continue
        if not _inside_half_open(target.box, point.u, point.v):
            rejected["outside_annotation_box_2d"] += 1
            continue
        if not _inside_half_open(match.detector_box, point.u, point.v):
            rejected["outside_detection_box_2d"] += 1
            continue
        if winners[point.pixel] is not point:
            rejected["same_pixel_occluded"] += 1
            continue
        accepted.append(point)
    accepted.sort(key=lambda point: (point.pixel[1], point.pixel[0], point.original_index))
    assert len({point.pixel for point in accepted}) == len(accepted)
    supported = len(accepted) >= protocol.min_bins
    return SparseReference(match, "supported" if supported else "unsupported_insufficient_bins",
                           tuple(accepted), float(np.median([p.camera2_z_m for p in accepted])) if supported else None,
                           len(accepted), candidate_count, rejected, projection.scan_total_rows,
                           projection.scan_rejected_rows, projection.scan_nonfinite_xyz_rows,
                           projection.scan_nonfinite_intensity_rows, len(projection.returns),
                           dict(projection.rejected_counts), sources)


def paired_surface_error(reference: SparseReference, *, drive_id: str) -> PairedSurfaceError:
    """Pure metric interface; future real evaluation must report unsupported denominator."""
    box = reference.match.box_median_z_m
    mask = reference.match.mask_median_z_m
    if reference.status != "supported" or reference.lidar_z_m is None or box is None or mask is None:
        raise ValueError("Paired errors require one supported reference and both frozen medians")
    if not drive_id:
        raise ValueError("Drive grouping is required for paired uncertainty")
    e_box = box - reference.lidar_z_m
    e_mask = mask - reference.lidar_z_m
    return PairedSurfaceError(reference.match.frame_id, reference.match.reference_id,
                              reference.match.detection_id, drive_id, reference.lidar_z_m,
                              e_box, e_mask, abs(e_mask) - abs(e_box))
