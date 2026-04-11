"""Deterministic class-consistent one-to-one 2D box matching."""

from dataclasses import dataclass

from perception_stack.results import BoxXYXY, FrameDetections
from .dataset import KittiAnnotation
from .protocol import ClassMapping


def _intersection(a: BoxXYXY, b: BoxXYXY) -> float:
    return max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))


def _area(box: BoxXYXY) -> float:
    return (box.x2 - box.x1) * (box.y2 - box.y1)


def box_iou(a: BoxXYXY, b: BoxXYXY) -> float:
    intersection = _intersection(a, b)
    union = _area(a) + _area(b) - intersection
    return intersection / union if union > 0 else 0.0


def intersection_over_detection_area(detection: BoxXYXY, ignored: BoxXYXY) -> float:
    return _intersection(detection, ignored) / _area(detection)


@dataclass(frozen=True)
class MatchPair:
    detection_id: int
    reference_id: int
    iou: float

    def to_dict(self) -> dict:
        return {"detection_id": self.detection_id, "reference_id": self.reference_id,
                "iou": self.iou}


@dataclass(frozen=True)
class FrameMatches:
    frame_id: str
    pairs: tuple[MatchPair, ...]
    eligible_reference_ids: tuple[int, ...]
    ignored_reference_ids: tuple[int, ...]
    eligible_detection_ids: tuple[int, ...]
    unsupported_detection_ids: tuple[int, ...]
    ignored_detection_ids: tuple[int, ...]
    unmatched_reference_ids: tuple[int, ...]
    unmatched_detection_ids: tuple[int, ...]
    reference_class_counts: dict[str, int]
    detection_class_counts: dict[str, int]
    eligible_reference_classes: dict[int, str]
    eligible_detection_classes: dict[int, str]

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id, "pairs": [pair.to_dict() for pair in self.pairs],
            "matched_pairs": len(self.pairs),
            "total_references": len(self.eligible_reference_ids) + len(self.ignored_reference_ids),
            "eligible_references": len(self.eligible_reference_ids),
            "ignored_references": len(self.ignored_reference_ids),
            "total_detections": len(self.eligible_detection_ids) + len(self.unsupported_detection_ids),
            "eligible_detections": len(self.eligible_detection_ids),
            "unsupported_detections": len(self.unsupported_detection_ids),
            "ignored_detections": len(self.ignored_detection_ids),
            "unmatched_references": len(self.unmatched_reference_ids),
            "unmatched_detections": len(self.unmatched_detection_ids),
            "eligible_reference_ids": list(self.eligible_reference_ids),
            "ignored_reference_ids": list(self.ignored_reference_ids),
            "eligible_detection_ids": list(self.eligible_detection_ids),
            "unsupported_detection_ids": list(self.unsupported_detection_ids),
            "ignored_detection_ids": list(self.ignored_detection_ids),
            "unmatched_reference_ids": list(self.unmatched_reference_ids),
            "unmatched_detection_ids": list(self.unmatched_detection_ids),
            "reference_class_counts": dict(sorted(self.reference_class_counts.items())),
            "detection_class_counts": dict(sorted(self.detection_class_counts.items())),
            "eligible_reference_classes": dict(sorted(self.eligible_reference_classes.items())),
            "eligible_detection_classes": dict(sorted(self.eligible_detection_classes.items())),
            "match_ious": [pair.iou for pair in self.pairs],
        }


def match_frame(
    detections: FrameDetections, annotations: tuple[KittiAnnotation, ...],
    mapping: ClassMapping, *, iou_threshold: float = 0.5,
    ignored_ioa_threshold: float = 0.5, min_reference_height_px: float = 0,
    max_truncation: float = 1.0, allowed_occlusion: tuple[int, ...] = (0, 1, 2, 3),
) -> FrameMatches:
    if not 0 < iou_threshold <= 1 or not 0 < ignored_ioa_threshold <= 1:
        raise ValueError("Matching thresholds must be in (0, 1]")
    if min_reference_height_px < 0 or not 0 <= max_truncation <= 1:
        raise ValueError("Invalid annotation inclusion policy")
    if len({annotation.object_id for annotation in annotations}) != len(annotations):
        raise ValueError("Duplicate KITTI object IDs")
    eligible_refs = [a for a in annotations if a.class_name in mapping.coco_to_kitti.values()
                     and a.validity_status == "valid"
                     and a.box.y2 - a.box.y1 >= min_reference_height_px
                     and a.truncated <= max_truncation and a.occluded in allowed_occlusion]
    eligible_ids = {a.object_id for a in eligible_refs}
    ignored_refs = [a for a in annotations if a.object_id not in eligible_ids]
    eligible_det = [(i, d) for i, d in enumerate(detections.detections)
                    if d.class_name in mapping.coco_to_kitti]
    supported_ids = {i for i, _ in eligible_det}
    unsupported = tuple(i for i in range(len(detections.detections)) if i not in supported_ids)

    # Greedy maximum IoU first; confidence, reference ID and detection ID break ties.
    candidates = []
    for i, detection in eligible_det:
        target_class = mapping.coco_to_kitti[detection.class_name]
        for annotation in eligible_refs:
            if annotation.class_name != target_class:
                continue
            iou = box_iou(detection.box, annotation.box)
            if iou >= iou_threshold:
                candidates.append((-iou, -detection.confidence, annotation.object_id, i))
    candidates.sort()
    used_det: set[int] = set()
    used_ref: set[int] = set()
    pairs = []
    for neg_iou, _, reference_id, detection_id in candidates:
        if detection_id in used_det or reference_id in used_ref:
            continue
        pairs.append(MatchPair(detection_id, reference_id, -neg_iou))
        used_det.add(detection_id)
        used_ref.add(reference_id)

    ignored_detections = []
    unmatched_detections = []
    for i, detection in eligible_det:
        if i in used_det:
            continue
        if any(intersection_over_detection_area(detection.box, ref.box) >= ignored_ioa_threshold
               for ref in ignored_refs):
            ignored_detections.append(i)
        else:
            unmatched_detections.append(i)
    reference_class_counts = {}
    detection_class_counts = {}
    for annotation in annotations:
        reference_class_counts[annotation.class_name] = reference_class_counts.get(annotation.class_name, 0) + 1
    for detection in detections.detections:
        detection_class_counts[detection.class_name] = detection_class_counts.get(detection.class_name, 0) + 1
    return FrameMatches(
        detections.frame_id, tuple(pairs), tuple(sorted(eligible_ids)),
        tuple(sorted(ref.object_id for ref in ignored_refs)),
        tuple(i for i, _ in eligible_det), unsupported,
        tuple(ignored_detections),
        tuple(sorted(eligible_ids - used_ref)), tuple(unmatched_detections),
        reference_class_counts, detection_class_counts,
        {annotation.object_id: annotation.class_name for annotation in eligible_refs},
        {i: mapping.coco_to_kitti[detection.class_name] for i, detection in eligible_det},
    )
