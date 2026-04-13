"""Immutable, JSON-ready results for paired object-depth sampling."""

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from types import MappingProxyType
from typing import Mapping

from perception_stack.results import BoxXYXY
from .source import DETECTOR_INPUT_SCHEMA


@dataclass(frozen=True)
class FusionConfig:
    """Fixed policies shared by box and mask methods; no label-based fitting."""

    min_valid_pixels: int = 3
    min_valid_fraction: float = 0.10
    near_cap_ratio: float = 0.99

    def __post_init__(self) -> None:
        if type(self.min_valid_pixels) is not int or self.min_valid_pixels < 1:
            raise ValueError("min_valid_pixels must be at least one")
        if not math.isfinite(self.min_valid_fraction) or not 0 <= self.min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be finite and in [0, 1]")
        if not math.isfinite(self.near_cap_ratio) or not 0 < self.near_cap_ratio <= 1:
            raise ValueError("near_cap_ratio must be finite and in (0, 1]")

    def to_dict(self) -> dict:
        return {
            "estimator": "median_valid_camera_axis_z_m",
            "min_valid_pixels": self.min_valid_pixels,
            "min_valid_fraction": self.min_valid_fraction,
            "near_cap_ratio": self.near_cap_ratio,
            "box_pixel_rule": "integer pixel center (u,v) belongs iff x1<=u<x2 and y1<=v<y2 after clipping",
            "mask_rule": "boolean instance mask AND box region; no missing-mask fallback",
            "validity_rule": "FrameDepth.valid_mask: finite and positive meters",
            "selection_rule": "closest actual Z to region median, then squared distance to region-pixel centroid, then v, then u",
            "pixel_center_rule": "integer u,v are centers of image pixels; top-left center is (0,0)",
        }


@dataclass(frozen=True)
class SurfaceSample:
    u: int
    v: int
    depth_z_m: float

    def to_dict(self) -> dict:
        return {"u": self.u, "v": self.v, "depth_z_m": self.depth_z_m}


@dataclass(frozen=True)
class SurfacePoint:
    x_m: float
    y_m: float
    z_m: float
    radial_range_m: float

    def to_dict(self) -> dict:
        return {"x_m": self.x_m, "y_m": self.y_m, "z_m": self.z_m,
                "radial_range_m": self.radial_range_m,
                "reference": "predicted_visible_scene_surface_at_selected_pixel"}


@dataclass(frozen=True)
class DepthSamplingStatistics:
    status: str
    reason: str | None
    region_pixels: int
    valid_pixels: int
    valid_fraction: float
    median_z_m: float | None
    mean_z_m: float | None
    min_z_m: float | None
    max_z_m: float | None
    std_z_m: float | None
    iqr_z_m: float | None
    mad_z_m: float | None
    near_cap_pixels: int
    near_cap_fraction_of_valid: float
    selected_sample: SurfaceSample | None

    def to_dict(self) -> dict:
        return {
            "status": self.status, "reason": self.reason,
            "region_pixels": self.region_pixels, "valid_pixels": self.valid_pixels,
            "valid_fraction": self.valid_fraction,
            "median_z_m": self.median_z_m, "mean_z_m": self.mean_z_m,
            "min_z_m": self.min_z_m, "max_z_m": self.max_z_m,
            "std_z_m": self.std_z_m, "iqr_z_m": self.iqr_z_m, "mad_z_m": self.mad_z_m,
            "near_cap_pixels": self.near_cap_pixels,
            "near_cap_fraction_of_valid": self.near_cap_fraction_of_valid,
            "selected_sample": self.selected_sample.to_dict() if self.selected_sample else None,
        }


@dataclass(frozen=True)
class MethodFusion:
    statistics: DepthSamplingStatistics
    surface_point: SurfacePoint | None
    geometry_status: str
    geometry_reason: str | None

    def to_dict(self) -> dict:
        return {
            "statistics": self.statistics.to_dict(),
            "surface_point": self.surface_point.to_dict() if self.surface_point else None,
            "geometry_status": self.geometry_status, "geometry_reason": self.geometry_reason,
        }


@dataclass(frozen=True)
class ObjectDepthFusion:
    detection_id: int
    class_id: int
    class_name: str
    confidence: float
    box: BoxXYXY
    instance_mask_available: bool
    box_method: MethodFusion
    mask_method: MethodFusion

    def to_dict(self) -> dict:
        box_z = self.box_method.statistics.median_z_m if self.box_method.statistics.status == "ok" else None
        mask_z = self.mask_method.statistics.median_z_m if self.mask_method.statistics.status == "ok" else None
        return {
            "detection_id": self.detection_id, "class_id": self.class_id,
            "class_name": self.class_name, "confidence": self.confidence,
            "box_xyxy": list(self.box.xyxy),
            "instance_mask_available": self.instance_mask_available,
            "reference_semantics": "region median is aggregate Z; selected ray is an estimated visible surface, not object center",
            "box": self.box_method.to_dict(), "mask": self.mask_method.to_dict(),
            "mask_minus_box_median_z_m": mask_z - box_z if box_z is not None and mask_z is not None else None,
        }


@dataclass(frozen=True)
class FrameFusionResults:
    frame_id: str
    image_size: tuple[int, int]
    detector_model_id: str
    detector_checkpoint_sha256: str | None
    depth_model_id: str
    depth_checkpoint_sha256: str
    depth_device: str
    depth_input_size: int
    depth_preprocessing: Mapping
    calibration_provenance: str | None
    calibration_status: str
    config: FusionConfig
    objects: tuple[ObjectDepthFusion, ...]
    detector_input_fingerprint_sha256: str | None = None

    def __post_init__(self) -> None:
        # Freeze nested model metadata so callers cannot alter a completed M4
        # result through a shared list/dictionary after inference.
        def freeze(value):
            if isinstance(value, Mapping):
                return MappingProxyType({key: freeze(item) for key, item in value.items()})
            if isinstance(value, (list, tuple)):
                return tuple(freeze(item) for item in value)
            return deepcopy(value)

        object.__setattr__(self, "depth_preprocessing", freeze(self.depth_preprocessing))
        object.__setattr__(self, "objects", tuple(self.objects))

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id, "image_size": list(self.image_size),
            "detector_model_id": self.detector_model_id,
            "detector_checkpoint_sha256": self.detector_checkpoint_sha256,
            "detector_input_fingerprint_schema": DETECTOR_INPUT_SCHEMA if self.detector_input_fingerprint_sha256 else None,
            "detector_input_fingerprint_sha256": self.detector_input_fingerprint_sha256,
            "depth_model_id": self.depth_model_id,
            "depth_checkpoint_sha256": self.depth_checkpoint_sha256,
            "depth_device": self.depth_device, "depth_input_size": self.depth_input_size,
            "depth_preprocessing": json.loads(json.dumps(self.depth_preprocessing, default=_json_metadata)),
            "calibration_provenance": self.calibration_provenance,
            "calibration_status": self.calibration_status,
            "config": self.config.to_dict(),
            "objects": [obj.to_dict() for obj in self.objects],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _json_metadata(value):
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Unsupported metadata value: {type(value).__name__}")
