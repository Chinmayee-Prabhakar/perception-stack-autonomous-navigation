"""Deterministic paired box/mask sampling on one aligned frame and depth map."""

from copy import deepcopy

import numpy as np

from perception_stack.frame import Frame
from perception_stack.results import BoxXYXY, FrameDepth, FrameDetections
from .geometry import backproject_pixel
from .results import (
    DepthSamplingStatistics, FrameFusionResults, FusionConfig, MethodFusion,
    ObjectDepthFusion, SurfaceSample,
)


def box_region(box: BoxXYXY, image_size: tuple[int, int]) -> np.ndarray:
    """Integer pixel center (u,v) is in clipped [x1,x2) x [y1,y2)."""
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    x1, x2 = np.clip((box.x1, box.x2), 0, width)
    y1, y2 = np.clip((box.y1, box.y2), 0, height)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((height, width), dtype=np.bool_)
    columns = (np.arange(width) >= x1) & (np.arange(width) < x2)
    rows = (np.arange(height) >= y1) & (np.arange(height) < y2)
    return rows[:, None] & columns[None, :]


def _unavailable(status: str, reason: str) -> DepthSamplingStatistics:
    return DepthSamplingStatistics(status, reason, 0, 0, 0.0,
                                   None, None, None, None, None, None, None, 0, 0.0, None)


def sample_region(
    region: np.ndarray, depth: FrameDepth, config: FusionConfig,
) -> DepthSamplingStatistics:
    """Same validity and median policy for either sampling region."""
    if region.shape != depth.depth_m.shape or region.dtype != np.bool_:
        raise ValueError("Sampling region must be an aligned boolean image")
    region_pixels = int(region.sum())
    if not region_pixels:
        return _unavailable("empty_region", "No pixel centers belong to this sampling region")
    valid = region & depth.valid_mask
    valid_pixels = int(valid.sum())
    fraction = valid_pixels / region_pixels
    if not valid_pixels:
        return DepthSamplingStatistics("no_valid_depth", "Region has no finite positive depth",
                                       region_pixels, 0, 0.0, None, None, None, None,
                                       None, None, None, 0, 0.0, None)
    values = depth.depth_m[valid].astype(np.float64)
    median = float(np.median(values))
    q25, q75 = np.percentile(values, (25, 75))
    near_cap = int(np.count_nonzero(values >= config.near_cap_ratio * depth.nominal_max_depth_m))

    # Anchor is the centroid of *all* region pixel centers, independent of validity.
    region_v, region_u = np.nonzero(region)
    anchor_u, anchor_v = float(region_u.mean()), float(region_v.mean())
    candidate_v, candidate_u = np.nonzero(valid)
    distance_sq = (candidate_u - anchor_u)**2 + (candidate_v - anchor_v)**2
    # np.lexsort sorts on the final key first; v then u is the last tie break.
    index = int(np.lexsort((candidate_u, candidate_v, distance_sq, np.abs(values - median)))[0])
    sample = SurfaceSample(int(candidate_u[index]), int(candidate_v[index]), float(values[index]))

    if valid_pixels < config.min_valid_pixels:
        status, reason = "insufficient_valid_pixels", f"Need at least {config.min_valid_pixels} valid pixels"
    elif fraction < config.min_valid_fraction:
        status, reason = "low_valid_fraction", f"Need valid fraction >= {config.min_valid_fraction}"
    else:
        status, reason = "ok", None
    return DepthSamplingStatistics(
        status, reason, region_pixels, valid_pixels, fraction,
        median, float(values.mean()), float(values.min()), float(values.max()),
        float(values.std()), float(q75 - q25), float(np.median(np.abs(values - median))),
        near_cap, near_cap / valid_pixels, sample,
    )


def fuse_frame(
    frame: Frame, detections: FrameDetections, depth: FrameDepth, *,
    config: FusionConfig = FusionConfig(),
    calibration_provenance: str | None = None,
    rectified: bool = False,
    detector_checkpoint_sha256: str | None = None,
) -> FrameFusionResults:
    """Compare box and mask depth; backproject only with trusted aligned calibration."""
    if frame.frame_id != detections.frame_id or frame.frame_id != depth.frame_id:
        raise ValueError("Frame, detection, and depth IDs must match")
    if frame.size != detections.image_size or frame.size != depth.image_size:
        raise ValueError("Frame, detection, and depth dimensions must match")
    if depth.depth_convention != "camera_axis_z" or depth.units != "meters":
        raise ValueError("Fusion requires metric camera-axis Z depth")
    if not np.isfinite(depth.nominal_max_depth_m):
        raise ValueError("Nominal maximum depth must be finite")
    if calibration_provenance is not None and not calibration_provenance.strip():
        raise ValueError("Calibration provenance cannot be blank")
    if frame.intrinsics is None:
        calibration_status = "unavailable_no_intrinsics"
        geometry_reason = "No calibrated intrinsics were supplied"
    elif not calibration_provenance:
        calibration_status = "unavailable_unverified_provenance"
        geometry_reason = "Intrinsics lack calibration provenance"
    elif not rectified:
        calibration_status = "unavailable_unrectified"
        geometry_reason = "Rectified pinhole camera origin was not confirmed"
    else:
        calibration_status = "available_rectified_intrinsics"
        geometry_reason = None

    def method(statistics: DepthSamplingStatistics) -> MethodFusion:
        if statistics.status != "ok" or statistics.selected_sample is None:
            return MethodFusion(statistics, None, "unavailable_sampling", statistics.reason)
        if geometry_reason is not None:
            return MethodFusion(statistics, None, calibration_status, geometry_reason)
        sample = statistics.selected_sample
        point = backproject_pixel(sample.u, sample.v, sample.depth_z_m,
                                  frame.intrinsics, image_size=frame.size, rectified=True)
        return MethodFusion(statistics, point, "ok", None)

    objects = []
    for index, detection in enumerate(detections.detections):
        box = box_region(detection.box, frame.size)
        if box.any():
            box_stats = sample_region(box, depth, config)
        else:
            box_stats = _unavailable("empty_box_region", "Clipped box contains no pixel centers")
        if detection.mask is None:
            mask_stats = _unavailable("missing_mask", "Detection has no instance mask")
        elif not detection.mask.any():
            mask_stats = _unavailable("empty_mask", "Instance mask contains no positive pixels")
        else:
            mask_region = box & detection.mask
            if mask_region.any():
                mask_stats = sample_region(mask_region, depth, config)
            else:
                mask_stats = _unavailable("empty_mask_box_intersection", "Mask and box do not overlap at pixel centers")
        objects.append(ObjectDepthFusion(
            index, detection.class_id, detection.class_name, detection.confidence,
            detection.box, detection.mask is not None, method(box_stats), method(mask_stats),
        ))
    return FrameFusionResults(
        frame.frame_id, frame.size, detections.model_id, detector_checkpoint_sha256,
        depth.model_id, depth.checkpoint_sha256, depth.device, depth.input_size,
        deepcopy(dict(depth.preprocessing)), calibration_provenance, calibration_status,
        config, tuple(objects),
    )
