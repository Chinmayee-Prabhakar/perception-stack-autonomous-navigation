"""Deterministic image-space pavement evidence adjacent to detection boxes.

The fixed default samples the central 60% of the box width in a band starting
at its exclusive bottom edge. No pixel is interpreted as a physical contact
point, and lane markings never contribute to drivable-area counts.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Mapping

import numpy as np

from perception_stack.frame import Frame
from perception_stack.fusion.results import FrameFusionResults
from perception_stack.results import BoxXYXY, FrameDetections, FrameRoadSegmentation


@dataclass(frozen=True)
class ContactPolicy:
    width_fraction: float = 0.60
    height_fraction: float = 0.15
    min_band_height_px: int = 2
    min_pixels: int = 6
    foreground_overlap_threshold: float = 0.50
    min_visible_fraction: float = 0.50

    def __post_init__(self) -> None:
        if not math.isfinite(self.width_fraction) or not 0 < self.width_fraction <= 1:
            raise ValueError("width_fraction must be in (0,1]")
        if not math.isfinite(self.height_fraction) or self.height_fraction <= 0:
            raise ValueError("height_fraction must be positive")
        if type(self.min_band_height_px) is not int or self.min_band_height_px < 1:
            raise ValueError("min_band_height_px must be positive")
        if type(self.min_pixels) is not int or self.min_pixels < 1:
            raise ValueError("min_pixels must be positive")
        for name in ("foreground_overlap_threshold", "min_visible_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")

    def to_dict(self) -> dict:
        return {
            "placement": "below_box_adjacent",
            "width_fraction": self.width_fraction,
            "height_fraction": self.height_fraction,
            "min_band_height_px": self.min_band_height_px,
            "min_pixels": self.min_pixels,
            "foreground_overlap_threshold": self.foreground_overlap_threshold,
            "min_visible_fraction": self.min_visible_fraction,
            "pixel_rule": "integer pixel center (u,v) in clipped [left,right) x [top,bottom); y starts at exclusive box bottom",
            "boundary_rule": "box touching any image edge is unknown_image_boundary, not proven truncation; explicit hints override",
            "semantics": "predicted image-space drivable adjacency, not occupancy probability or ground contact",
        }


@dataclass(frozen=True)
class ObjectRoadAssociation:
    detection_id: int
    contact_region_xyxy: tuple[int, int, int, int]
    ideal_region_pixels: int
    region_pixels: int
    drivable_pixels: int
    drivable_overlap_fraction: float | None
    evidence_status: str
    reason: str | None

    def to_dict(self) -> dict:
        return {
            "detection_id": self.detection_id,
            "contact_region_xyxy": list(self.contact_region_xyxy),
            "ideal_region_pixels": self.ideal_region_pixels,
            "region_pixels": self.region_pixels,
            "drivable_pixels": self.drivable_pixels,
            "drivable_overlap_fraction": self.drivable_overlap_fraction,
            "evidence_status": self.evidence_status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FrameRoadAssociations:
    frame_id: str
    image_size: tuple[int, int]
    policy: ContactPolicy
    road_model_id: str | None
    road_checkpoint_sha256: str | None
    road_checkpoint_id: str | None
    road_status: str
    objects: tuple[ObjectRoadAssociation, ...]
    road_source_sha256: str | None
    detection_source_sha256: str
    policy_sha256: str
    visibility_hints: tuple[tuple[int, str], ...]

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "image_size": list(self.image_size),
            "policy": self.policy.to_dict(),
            "road_model_id": self.road_model_id,
            "road_checkpoint_sha256": self.road_checkpoint_sha256,
            "road_checkpoint_id": self.road_checkpoint_id,
            "road_status": self.road_status,
            "road_source_sha256": self.road_source_sha256,
            "detection_source_sha256": self.detection_source_sha256,
            "policy_sha256": self.policy_sha256,
            "visibility_hints": [{"detection_id": index, "state": state}
                                 for index, state in self.visibility_hints],
            "objects": [item.to_dict() for item in self.objects],
        }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest_json(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def policy_fingerprint(policy: ContactPolicy) -> str:
    """Stable SHA-256 of the complete JSON policy, including pixel semantics."""
    return _digest_json(policy.to_dict())


def detection_source_fingerprint(detections: FrameDetections) -> str:
    """Bind detector order, identities and boxes without hashing full instance masks."""
    return _digest_json({
        "schema": "detection-association-source-v1",
        "frame_id": detections.frame_id,
        "image_size": detections.image_size,
        "model_id": detections.model_id,
        "detections": [{"detection_id": index, "class_id": item.class_id,
                        "class_name": item.class_name, "confidence": item.confidence,
                        "box_xyxy": item.box.xyxy}
                       for index, item in enumerate(detections.detections)],
    })


def road_source_fingerprint(road: FrameRoadSegmentation) -> str:
    """SHA-256 of source identity and both full-resolution boolean prediction masks.

    Header is canonical UTF-8 JSON with sorted keys. Masks are row-major C-order,
    one byte per bool (0 or 1), in drivable-then-lane order. Each block has an
    eight-byte big-endian length. Boolean storage is endian-independent. This
    binds content; it does not authenticate a model or prove prediction accuracy.
    """
    header = _canonical_bytes({
        "schema": "road-prediction-source-v1",
        "frame_id": road.frame_id, "image_size": road.image_size,
        "model_id": road.model_id, "checkpoint_id": road.checkpoint_id,
        "checkpoint_sha256": road.checkpoint_sha256, "device": road.device,
        "preprocessing": road.preprocessing,
        "mask_encoding": "C-order bool, one byte per pixel, false=0 true=1",
    })
    digest = sha256(b"perception-road-source-v1\0")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    for name, mask in ((b"drivable\0", road.drivable_area_mask),
                       (b"lane\0", road.lane_marking_mask)):
        if mask.dtype != np.bool_ or mask.shape != (road.image_size[1], road.image_size[0]):
            raise ValueError("Road source mask must be aligned boolean pixels")
        digest.update(name)
        contiguous = np.ascontiguousarray(mask)
        raw = memoryview(contiguous).cast("B")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _contact_box(box: BoxXYXY, image_size: tuple[int, int], policy: ContactPolicy) -> tuple[tuple[int, int, int, int], int]:
    width, height = image_size
    half_width = (box.x2 - box.x1) * policy.width_fraction / 2
    center = (box.x1 + box.x2) / 2
    band_height = max(policy.min_band_height_px, math.ceil((box.y2 - box.y1) * policy.height_fraction))
    # Integer u,v denote pixel centers. ceil implements [left,right) membership.
    left, right = math.ceil(center - half_width), math.ceil(center + half_width)
    top = math.ceil(box.y2)
    bottom = math.ceil(box.y2 + band_height)
    ideal = max(0, right - left) * max(0, bottom - top)
    return (max(0, min(width, left)), max(0, min(height, top)),
            max(0, min(width, right)), max(0, min(height, bottom))), ideal


def associate_road(
    frame: Frame, detections: FrameDetections,
    road: FrameRoadSegmentation | None,
    fusion: FrameFusionResults | None = None, *,
    policy: ContactPolicy = ContactPolicy(),
    visibility: Mapping[int, str] | None = None,
) -> FrameRoadAssociations:
    """Associate each detection independently; visibility hints are optional annotations.

    Hints may be ``occluded`` or ``truncated``; the detector itself does not
    infer these conditions. Missing road predictions never imply background.
    """
    if (detections.frame_id, detections.image_size) != (frame.frame_id, frame.size):
        raise ValueError("Frame and detection IDs/dimensions must match")
    if road is not None and (road.frame_id, road.image_size) != (frame.frame_id, frame.size):
        raise ValueError("Frame and road IDs/dimensions must match")
    if fusion is not None:
        if (fusion.frame_id, fusion.image_size) != (frame.frame_id, frame.size):
            raise ValueError("Frame and fusion IDs/dimensions must match")
        if len(fusion.objects) != len(detections.detections) or any(
            (item.detection_id, item.box) != (index, det.box)
            for index, (item, det) in enumerate(zip(fusion.objects, detections.detections))
        ):
            raise ValueError("Fusion objects must align with detection IDs and boxes")
    visibility = {} if visibility is None else dict(visibility)
    if any(type(index) is not int or index < 0 or index >= len(detections.detections)
           or type(state) is not str or state not in {"occluded", "truncated"}
           for index, state in visibility.items()):
        raise ValueError("Visibility hints require valid detection IDs and occluded/truncated states")
    objects = []
    for index, detection in enumerate(detections.detections):
        (left, top, right, bottom), ideal = _contact_box(detection.box, frame.size, policy)
        count = max(0, right - left) * max(0, bottom - top)
        drivable = int(road.drivable_area_mask[top:bottom, left:right].sum()) if road is not None and count else 0
        overlap = drivable / count if road is not None and count else None
        hint = visibility.get(index)
        if hint == "truncated":
            status, reason = "unknown_truncated", "object is explicitly annotated truncated; box-bottom adjacency is unreliable"
        elif hint == "occluded":
            status, reason = "unknown_occluded", "object is explicitly annotated occluded; box-bottom adjacency is unreliable"
        elif (detection.box.x1 <= 0 or detection.box.x2 >= frame.size[0]
              or detection.box.y1 <= 0 or detection.box.y2 >= frame.size[1]):
            status, reason = "unknown_image_boundary", "box touches an image edge; object extent may be clipped but truncation is unverified"
        elif ideal == 0 or count == 0:
            status, reason = "unknown_invalid_region", "contact band contains no in-image pixel centers"
        elif count / ideal < policy.min_visible_fraction:
            status, reason = "unknown_insufficient_coverage", "too much of the contact band lies outside the image"
        elif count < policy.min_pixels:
            status, reason = "unknown_insufficient_evidence", "contact band has too few sampled pixels"
        elif road is None:
            status, reason = "unknown_insufficient_evidence", "road segmentation is unavailable"
        elif overlap >= policy.foreground_overlap_threshold:
            status, reason = "drivable_adjacency_detected", None
        else:
            status, reason = "drivable_adjacency_not_detected", None
        objects.append(ObjectRoadAssociation(index, (left, top, right, bottom), ideal,
                                             count, drivable, overlap, status, reason))
    return FrameRoadAssociations(
        frame.frame_id, frame.size, policy,
        road.model_id if road else None,
        road.checkpoint_sha256 if road else None,
        road.checkpoint_id if road else None,
        "available" if road else "unavailable_missing_road_prediction",
        tuple(objects),
        road_source_fingerprint(road) if road else None,
        detection_source_fingerprint(detections),
        policy_fingerprint(policy),
        tuple(sorted(visibility.items())),
    )


def validate_association_source(
    frame: Frame, detections: FrameDetections,
    road: FrameRoadSegmentation | None, associations: FrameRoadAssociations,
) -> None:
    """Reject stale source data and contradictory derived diagnostics.

    Recomputes only cheap contact-band counts from supplied immutable masks;
    no detector, depth estimator or road model is rerun.
    """
    expected_identity = (frame.frame_id, frame.size)
    if (detections.frame_id, detections.image_size) != expected_identity:
        raise ValueError("Association frame and detection IDs/dimensions mismatch")
    if (associations.frame_id, associations.image_size) != expected_identity:
        raise ValueError("Association frame ID or dimensions mismatch")
    if road is not None and (road.frame_id, road.image_size) != expected_identity:
        raise ValueError("Association road frame ID or dimensions mismatch")
    if not isinstance(associations.policy, ContactPolicy):
        raise ValueError("Association policy has invalid type")
    if associations.policy_sha256 != policy_fingerprint(associations.policy):
        raise ValueError("Association policy fingerprint mismatch")
    if associations.detection_source_sha256 != detection_source_fingerprint(detections):
        raise ValueError("Association detection source fingerprint mismatch")
    status = "available" if road is not None else "unavailable_missing_road_prediction"
    if associations.road_status != status:
        raise ValueError("Association road availability mismatch")
    if road is None:
        if any(value is not None for value in (associations.road_model_id,
                associations.road_checkpoint_id, associations.road_checkpoint_sha256,
                associations.road_source_sha256)):
            raise ValueError("Association missing-road source must have null provenance")
    else:
        if associations.road_checkpoint_id != road.checkpoint_id:
            raise ValueError("Association road checkpoint ID mismatch")
        if (associations.road_model_id, associations.road_checkpoint_sha256) != (
                road.model_id, road.checkpoint_sha256):
            raise ValueError("Association road model/checkpoint provenance mismatch")
        if associations.road_source_sha256 != road_source_fingerprint(road):
            raise ValueError("Association road source mask fingerprint mismatch")
    hints = associations.visibility_hints
    if not isinstance(hints, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 for item in hints):
        raise ValueError("Association visibility hints are invalid or duplicated")
    if (any(type(index) is not int or index < 0 or index >= len(detections.detections)
            or type(state) is not str or state not in {"occluded", "truncated"}
            for index, state in hints)
            or tuple(sorted(hints)) != hints or len(dict(hints)) != len(hints)):
        raise ValueError("Association visibility hints are invalid or duplicated")
    expected = associate_road(frame, detections, road, policy=associations.policy,
                              visibility=dict(hints))
    if len(associations.objects) != len(expected.objects):
        raise ValueError("Association object count does not match detections")
    fields = ("detection_id", "contact_region_xyxy", "ideal_region_pixels", "region_pixels",
              "drivable_pixels", "drivable_overlap_fraction", "evidence_status", "reason")
    for index, (actual, correct) in enumerate(zip(associations.objects, expected.objects)):
        for field in fields:
            if getattr(actual, field) != getattr(correct, field):
                raise ValueError(f"Association object {index} {field} inconsistent with source/policy")
