"""Compact typed scene assembly from unchanged M4 and road predictions."""

from dataclasses import dataclass
import json
import math

from perception_stack.frame import Frame
from perception_stack.fusion.geometry import backproject_pixel
from perception_stack.fusion.results import FrameFusionResults, ObjectDepthFusion, SurfacePoint
from perception_stack.fusion.source import DETECTOR_INPUT_SCHEMA, detector_input_fingerprint
from perception_stack.results import FrameDetections, FrameRoadSegmentation
from .association import FrameRoadAssociations, ObjectRoadAssociation, validate_association_source


@dataclass(frozen=True)
class SceneObject:
    fusion: ObjectDepthFusion
    road: ObjectRoadAssociation

    @property
    def plotted_surface(self) -> tuple[str, SurfacePoint] | None:
        # A selected actual-depth pixel, never a median assigned to an arbitrary ray.
        for name, method in (("mask", self.fusion.mask_method), ("box", self.fusion.box_method)):
            if method.statistics.status == "ok" and method.surface_point is not None:
                return name, method.surface_point
        return None

    def to_dict(self, models: dict) -> dict:
        value = self.fusion.to_dict()
        value["road_association"] = self.road.to_dict()
        value["model_provenance"] = models.copy()
        value["plotted_surface_method"] = self.plotted_surface[0] if self.plotted_surface else None
        warnings = []
        for name, method in (("box", self.fusion.box_method), ("mask", self.fusion.mask_method)):
            if method.statistics.status != "ok":
                warnings.append(f"{name}_depth:{method.statistics.status}")
            if method.geometry_status != "ok":
                warnings.append(f"{name}_geometry:{method.geometry_status}")
        if self.road.evidence_status.startswith("unknown_"):
            warnings.append(f"road:{self.road.evidence_status}")
        value["warnings"] = warnings
        return value


@dataclass(frozen=True)
class FrameScene:
    frame_id: str
    image_size: tuple[int, int]
    fusion: FrameFusionResults
    associations: FrameRoadAssociations
    road_summary_json: str | None
    objects: tuple[SceneObject, ...]

    @property
    def metric_xz_status(self) -> str:
        if self.fusion.calibration_status != "available_rectified_intrinsics":
            return self.fusion.calibration_status
        if not any(obj.plotted_surface is not None for obj in self.objects):
            return "unavailable_no_calibrated_surface_points"
        return "available_approximate_surface_points"

    def to_dict(self) -> dict:
        models = {
            "detector_model_id": self.fusion.detector_model_id,
            "detector_checkpoint_sha256": self.fusion.detector_checkpoint_sha256,
            "depth_model_id": self.fusion.depth_model_id,
            "depth_checkpoint_sha256": self.fusion.depth_checkpoint_sha256,
            "road_model_id": self.associations.road_model_id,
            "road_checkpoint_sha256": self.associations.road_checkpoint_sha256,
        }
        return {
            "schema": "perception_scene_v1",
            "frame_id": self.frame_id,
            "image_size": list(self.image_size),
            "geometry_semantics": "M4 selected-pixel predicted visible-surface points; not object centers, footprints, or ground contact",
            "models": models,
            "fusion_source": {
                "schema": DETECTOR_INPUT_SCHEMA,
                "detector_input_sha256": self.fusion.detector_input_fingerprint_sha256,
                "fingerprint_semantics": "local detector-input consistency only; not model authenticity or prediction accuracy",
            },
            "depth_device": self.fusion.depth_device,
            "depth_input_size": self.fusion.depth_input_size,
            "depth_preprocessing": self.fusion.to_dict()["depth_preprocessing"],
            "fusion_policy": self.fusion.config.to_dict(),
            "road_association_policy": self.associations.policy.to_dict(),
            "association_source": {
                "road_source_sha256": self.associations.road_source_sha256,
                "detection_source_sha256": self.associations.detection_source_sha256,
                "policy_sha256": self.associations.policy_sha256,
                "visibility_hints": [{"detection_id": index, "state": state}
                                     for index, state in self.associations.visibility_hints],
                "fingerprint_semantics": "content consistency only; not model authenticity or prediction accuracy",
            },
            "road_segmentation": json.loads(self.road_summary_json) if self.road_summary_json else None,
            "road_status": self.associations.road_status,
            "calibration": {
                "status": self.fusion.calibration_status,
                "provenance": self.fusion.calibration_provenance,
                "geometry_note": "simple rectified pinhole camera only; raw KITTI P2 is not accepted",
            },
            "visualization": {
                "camera_overlay": "available_predicted_perception",
                "metric_xz": self.metric_xz_status,
                "metric_xz_axes": "X camera right (m), Z forward (m)" if self.metric_xz_status.startswith("available_") else None,
            },
            "objects": [obj.to_dict(models) for obj in self.objects],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def build_scene(
    frame: Frame, detections: FrameDetections,
    road: FrameRoadSegmentation | None,
    fusion: FrameFusionResults,
    associations: FrameRoadAssociations,
) -> FrameScene:
    """Validate all identities and retain M4 values without re-estimating depth."""
    expected = (frame.frame_id, frame.size)
    for result in (detections, fusion, associations):
        if (result.frame_id, result.image_size) != expected:
            raise ValueError("Scene input IDs/dimensions must match")
    if road is not None and (road.frame_id, road.image_size) != expected:
        raise ValueError("Scene road IDs/dimensions must match")
    validate_association_source(frame, detections, road, associations)
    if fusion.detector_input_fingerprint_sha256 is None:
        raise ValueError("Scene fusion source fingerprint is missing; rerun M4 fusion with these detections")
    if fusion.detector_input_fingerprint_sha256 != detector_input_fingerprint(
        detections, fusion.detector_checkpoint_sha256
    ):
        raise ValueError("Scene fusion source fingerprint does not match detector inputs or instance masks")
    if len(fusion.objects) != len(associations.objects) or len(fusion.objects) != len(detections.detections):
        raise ValueError("Scene object counts must match")
    if fusion.detector_model_id != detections.model_id:
        raise ValueError("Scene detector provenance must match detections")
    if (road is None) != (associations.road_status == "unavailable_missing_road_prediction"):
        raise ValueError("Scene road availability must match associations")
    if fusion.calibration_status == "available_rectified_intrinsics" and frame.intrinsics is None:
        raise ValueError("Available scene calibration requires frame intrinsics")
    for index, (det, fused, assoc) in enumerate(zip(detections.detections, fusion.objects, associations.objects)):
        if (fused.detection_id != index or assoc.detection_id != index or fused.box != det.box
                or fused.class_id != det.class_id or fused.class_name != det.class_name
                or fused.confidence != det.confidence
                or fused.instance_mask_available != (det.mask is not None)):
            raise ValueError("Scene detection IDs and predictions must align")
        for method in (fused.box_method, fused.mask_method):
            point = method.surface_point
            if point is None:
                continue
            sample = method.statistics.selected_sample
            if fusion.calibration_status != "available_rectified_intrinsics" or sample is None or method.statistics.status != "ok":
                raise ValueError("Scene surface point requires valid selected ray and calibration")
            expected_point = backproject_pixel(sample.u, sample.v, sample.depth_z_m,
                                               frame.intrinsics, image_size=frame.size, rectified=True)
            if not all(math.isclose(getattr(point, key), getattr(expected_point, key),
                                    rel_tol=1e-6, abs_tol=1e-6)
                       for key in ("x_m", "y_m", "z_m", "radial_range_m")):
                raise ValueError("Scene surface point must match M4 selected-ray backprojection")
    summary_json = json.dumps(road.to_summary(), sort_keys=True, allow_nan=False) if road else None
    scene = FrameScene(frame.frame_id, frame.size, fusion, associations, summary_json,
                       tuple(SceneObject(fused, assoc) for fused, assoc in zip(fusion.objects, associations.objects)))
    scene.to_json()  # Reject non-finite upstream values before artifact creation.
    return scene
