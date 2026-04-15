"""Evaluate only frozen M5 pilot pairs against verified sparse KITTI Object scans."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from perception_stack.detectors import YOLOInstanceSegmenter
from perception_stack.kitti_eval.dataset import KittiObjectDataset
from perception_stack.kitti_eval.matching import box_iou
from perception_stack.kitti_eval.protocol import EvaluationProtocol
from perception_stack.kitti_lidar import (
    FrozenObjectMatch, LidarProtocol, SourceDigests, associate_object,
    load_calibrated_projector, load_verified_object_scan,
)


REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "datasets/kitti_object"
OUTPUT = REPO / "outputs/kitti_m5_l2/pilot"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counter(rows, key: str, status: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows if row["status"] == status).items()))


def main() -> None:
    split = _json(REPO / "configs/kitti_split_manifest_v1.json")
    ids = split["pilot_frame_ids"]
    assert len(ids) == len(set(ids)) == 16
    geometry = _json(OUTPUT / "geometry_provenance.json")
    if geometry["selected_count"] != 16 or geometry["verified_scan_count"] != 16:
        raise ValueError("Sixteen verified geometry frames are required")
    geometry_rows = {row["frame_id"]: row for row in geometry["frames"]}
    assert set(geometry_rows) == set(ids)
    extraction = _json(REPO / "outputs/kitti_m5_l2_preflight/extraction_manifest.json")
    scan_rows = {row["frame_id"]: row for row in extraction["scans"]}
    assert set(scan_rows) == set(ids)

    protocol_path = REPO / "configs/kitti_lidar_protocol_v1.json"
    protocol = LidarProtocol.load(protocol_path)
    assert protocol.sha256 == "02451176def64af5ea83adc3c63df0ee37351176b008ec947f7da20c3f1f75fc"
    m5_protocol = EvaluationProtocol.load(REPO / "configs/kitti_eval_v1.json")
    checkpoint = REPO / m5_protocol.data["detector_weights"]
    if _sha(checkpoint) != m5_protocol.data["detector_checkpoint_sha256"]:
        raise ValueError("Frozen detector checkpoint changed")
    depth_checkpoint = REPO / m5_protocol.data["depth_weights"]
    if _sha(depth_checkpoint) != m5_protocol.data["depth_checkpoint_sha256"]:
        raise ValueError("Frozen depth checkpoint changed")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO / "work/runtime"))
    detector = YOLOInstanceSegmenter(checkpoint,
                                     confidence=m5_protocol.data["detector_confidence"],
                                     image_size=m5_protocol.data["detector_image_size"],
                                     device="cpu")
    dataset = KittiObjectDataset(DATASET)
    objects = []
    detector_geometry = []
    frame_records = []
    for frame_id in ids:
        sample = dataset.load(frame_id)
        source = geometry_rows[frame_id]["source_sha256"]
        scan = load_verified_object_scan(DATASET / f"training/velodyne/{frame_id}.bin",
                                         frame_id=frame_id,
                                         expected_sha256=scan_rows[frame_id]["sha256"])
        projector = load_calibrated_projector(DATASET / f"training/calib/{frame_id}.txt",
                                              frame_id=frame_id, image_size=sample.frame.size,
                                              expected_sha256=source["calibration"])
        projection = projector.project(scan)
        assert projection.scan_sha256 == source["scan"]
        assert len(projection.returns) == geometry_rows[frame_id]["projection_in_image_rows"]
        m5_path = REPO / f"outputs/kitti_m5/pilot/frames/{frame_id}.json"
        frozen = _json(m5_path)
        assert frozen["frame_id"] == frame_id == frozen["matching"]["frame_id"]
        pairs = frozen["matching"]["pairs"]
        frozen_rows = {(row["reference_id"], row["detection_id"]): row
                       for row in frozen["paired_object_reference_z_proxy_rows"]}
        assert len(pairs) == len(frozen_rows) == frozen["matching"]["matched_pairs"]
        detections = detector.predict(sample.frame)
        if len(detections.detections) != frozen["matching"]["total_detections"]:
            raise ValueError(f"Detector count differs from frozen M5: {frame_id}")
        if dict(sorted(Counter(d.class_name for d in detections.detections).items())) != frozen["matching"]["detection_class_counts"]:
            raise ValueError(f"Detector classes differ from frozen M5: {frame_id}")
        detector_geometry.append({"frame_id": frame_id, "detections": detections.to_summary()["detections"]})
        sources = SourceDigests(scan.source_sha256, projector.calibration_sha256,
                                protocol.sha256, _sha(m5_path))
        for pair in pairs:
            reference_id = pair["reference_id"]
            detection_id = pair["detection_id"]
            row = frozen_rows[(reference_id, detection_id)]
            annotation = sample.annotations[reference_id]
            detection = detections.detections[detection_id]
            mapped = m5_protocol.mapping.coco_to_kitti.get(detection.class_name)
            if mapped != row["class_name"] or annotation.class_name != row["class_name"]:
                raise ValueError(f"Frozen class/ID differs from reconstructed detector: {frame_id}")
            if abs(detection.confidence - row["confidence"]) > 1e-5:
                raise ValueError(f"Frozen detector confidence differs: {frame_id} det {detection_id}")
            if abs(box_iou(detection.box, annotation.box) - pair["iou"]) > 1e-5:
                raise ValueError(f"Frozen detection box/IoU differs: {frame_id} det {detection_id}")
            match = FrozenObjectMatch(frame_id, reference_id, detection_id,
                                      row["class_name"], mapped, detection.box,
                                      row["box_median_z_m"], row["mask_median_z_m"])
            reference = associate_object(projection, sample.annotations, match, sources, protocol)
            record = reference.to_dict()
            record.update({
                "class_name": row["class_name"], "occluded": annotation.occluded,
                "truncated": annotation.truncated,
                "annotation_box_xyxy": list(annotation.box.xyxy),
                "detector_box_xyxy": list(detection.box.xyxy),
                "frozen_match_iou": pair["iou"],
                "frozen_detector_confidence": row["confidence"],
                "original_m5_bottom_center_z_m": row["reference_z_m"],
                "support_z_spread_m": (record["support_z_max_m"] - record["support_z_min_m"]
                                       if record["support_z_min_m"] is not None else None),
            })
            if reference.status == "supported" and match.box_median_z_m is not None and match.mask_median_z_m is not None:
                box_error = match.box_median_z_m - reference.lidar_z_m
                mask_error = match.mask_median_z_m - reference.lidar_z_m
                record["pilot_depth_error_diagnostic"] = {
                    "box_signed_m": box_error, "mask_signed_m": mask_error,
                    "mask_minus_box_absolute_m": abs(mask_error) - abs(box_error),
                }
            else:
                record["pilot_depth_error_diagnostic"] = None
            objects.append(record)
        frame_records.append({"frame_id": frame_id, "frozen_m5_match_count": len(pairs),
                              "reconstructed_detector_count": len(detections.detections),
                              "in_image_returns": len(projection.returns),
                              "sources": vars(sources)})
        print(f"{frame_id}: {len(pairs)} frozen matches, "
              f"{sum(row['frame_id'] == frame_id and row['status'] == 'supported' for row in objects)} supported", flush=True)

    if len(objects) != protocol.data["pilot_matched_denominator"]:
        raise AssertionError("Pilot object denominator differs from frozen v1 protocol")
    assert {row["frame_id"] for row in objects} <= set(ids)
    statuses = Counter(row["status"] for row in objects)
    bins = Counter("3+" if row["support_bins"] >= 3 else str(row["support_bins"]) for row in objects)
    exclusions = sum((Counter(row["rejected_counts"]) for row in objects), Counter())
    supported = [row for row in objects if row["status"] == "supported"]
    errors = [row["pilot_depth_error_diagnostic"] for row in supported
              if row["pilot_depth_error_diagnostic"] is not None]
    summary = {
        "scope": "descriptive 16-frame frozen pilot; no held-out LiDAR evaluation",
        "selected_frames": len(ids), "verified_matching_scans": len(frame_records),
        "geometry_plausible_frames_pending_user_visual_sign_off": len(frame_records),
        "frozen_matched_objects": len(objects),
        "support_bin_categories": {key: bins.get(key, 0) for key in ("0", "1", "2", "3+")},
        "status_counts": dict(statuses),
        "supported_by_class": _counter(objects, "class_name", "supported"),
        "unsupported_by_class": _counter(objects, "class_name", "unsupported_insufficient_bins"),
        "supported_by_occlusion": _counter(objects, "occluded", "supported"),
        "unsupported_by_occlusion": _counter(objects, "occluded", "unsupported_insufficient_bins"),
        "supported_by_truncation": dict(sorted(Counter("positive" if row["truncated"] > 0 else "zero"
                                                       for row in supported).items())),
        "unsupported_by_truncation": dict(sorted(Counter("positive" if row["truncated"] > 0 else "zero"
                                                         for row in objects if row["status"] != "supported").items())),
        "association_rejection_counts_across_object_checks": dict(exclusions),
        "association_rejection_count_semantics": "Object-wise counts repeat global projected points for each of the 38 frozen matches.",
        "supported_lidar_z_min_m": min((row["lidar_z_m"] for row in supported), default=None),
        "supported_lidar_z_median_m": float(np.median([row["lidar_z_m"] for row in supported])) if supported else None,
        "supported_lidar_z_max_m": max((row["lidar_z_m"] for row in supported), default=None),
        "supported_bins_median": float(np.median([row["support_bins"] for row in supported])) if supported else None,
        "supported_z_spread_median": float(np.median([row["support_z_spread_m"] for row in supported])) if supported else None,
        "pilot_descriptive_paired_objects": len(errors),
        "pilot_descriptive_box_mae_m": float(np.mean([abs(row["box_signed_m"]) for row in errors])) if errors else None,
        "pilot_descriptive_mask_mae_m": float(np.mean([abs(row["mask_signed_m"]) for row in errors])) if errors else None,
        "pilot_descriptive_mask_minus_box_absolute_mean_m": float(np.mean([row["mask_minus_box_absolute_m"] for row in errors])) if errors else None,
        "pilot_error_interpretation": "Pilot diagnostic only; not confirmatory and not comparable as one accuracy metric with original M5 bottom-center Z proxy.",
        "protocol_sha256": protocol.sha256,
        "frozen_split_manifest_sha256": _sha(REPO / "configs/kitti_split_manifest_v1.json"),
        "frozen_m5_config_sha256": _sha(REPO / "configs/kitti_eval_v1.json"),
        "detector_checkpoint_sha256": m5_protocol.data["detector_checkpoint_sha256"],
        "depth_checkpoint_sha256": m5_protocol.data["depth_checkpoint_sha256"],
        "visual_sign_off": "pending_user_review",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, payload in (("object_records.json", objects), ("detector_geometry.json", detector_geometry),
                          ("frame_records.json", frame_records), ("pilot_summary.json", summary)):
        (OUTPUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"matched": len(objects), "support": summary["support_bin_categories"],
                      "paired": len(errors)}, sort_keys=True))


if __name__ == "__main__":
    main()
