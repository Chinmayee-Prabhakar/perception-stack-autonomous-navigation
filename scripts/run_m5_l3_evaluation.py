"""Apply unchanged sparse-reference policy to all frozen M5 held-out pairs."""

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import zipfile

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
OUT = REPO / "outputs/kitti_m5_l3/heldout"
PROTOCOL_SHA256 = "02451176def64af5ea83adc3c63df0ee37351176b008ec947f7da20c3f1f75fc"
DEVKIT_SHA256 = "ce0b76b69c0c5f89690a0d65b7302bbbdb962a0c7e8aba6efc7050d1b04b4cf1"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _drive_map(ids: list[str]) -> dict[str, str]:
    archive = Path.home() / "Downloads/devkit_object.zip"
    if _sha(archive) != DEVKIT_SHA256:
        raise ValueError("Official Object devkit mapping archive changed")
    with zipfile.ZipFile(archive) as source:
        permutation = [int(v) for v in source.read("mapping/train_rand.txt").decode().strip().split(",")]
        mapping = [line.split() for line in source.read("mapping/train_mapping.txt").decode().splitlines()]
    if len(permutation) != 7481 or len(mapping) != 7481 or set(permutation) != set(range(1, 7482)):
        raise ValueError("Object-to-Raw drive mapping invalid")
    result = {}
    for frame_id in ids:
        row = mapping[permutation[int(frame_id)] - 1]
        if len(row) != 3 or not row[1].endswith("_sync"):
            raise ValueError(f"Object drive identity absent: {frame_id}")
        result[frame_id] = row[1]
    return result


def main() -> None:
    split = _read(REPO / "configs/kitti_split_manifest_v1.json")
    ids = split["heldout_frame_ids"]
    assert len(ids) == len(set(ids)) == 256
    drives = _drive_map(ids)
    geometry = _read(OUT / "geometry_provenance.json")
    extraction = _read(OUT / "extraction_manifest.json")
    if geometry["verified_scan_count"] != 256 or extraction["extracted_count"] != 256:
        raise ValueError("Held-out source verification incomplete")
    geometry_rows = {row["frame_id"]: row for row in geometry["frames"]}
    scan_rows = {row["frame_id"]: row for row in extraction["scans"]}
    if set(geometry_rows) != set(ids) or set(scan_rows) != set(ids):
        raise ValueError("Frozen 256-frame source identity mismatch")
    protocol = LidarProtocol.load(REPO / "configs/kitti_lidar_protocol_v1.json")
    if protocol.sha256 != PROTOCOL_SHA256:
        raise ValueError("Approved LiDAR protocol changed")
    m5 = EvaluationProtocol.load(REPO / "configs/kitti_eval_v1.json")
    checkpoint = REPO / m5.data["detector_weights"]
    depth_checkpoint = REPO / m5.data["depth_weights"]
    if (_sha(checkpoint) != m5.data["detector_checkpoint_sha256"] or
            _sha(depth_checkpoint) != m5.data["depth_checkpoint_sha256"]):
        raise ValueError("Frozen model checkpoint changed")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO / "work/runtime"))
    detector = YOLOInstanceSegmenter(checkpoint, confidence=m5.data["detector_confidence"],
                                     image_size=m5.data["detector_image_size"], device="cpu")
    dataset = KittiObjectDataset(DATASET)
    objects = []
    frames = []
    detector_geometry = []
    for number, frame_id in enumerate(ids, 1):
        sample = dataset.load(frame_id)
        verified = geometry_rows[frame_id]
        scan = load_verified_object_scan(DATASET / f"training/velodyne/{frame_id}.bin",
                                         frame_id=frame_id,
                                         expected_sha256=scan_rows[frame_id]["sha256"])
        projector = load_calibrated_projector(DATASET / f"training/calib/{frame_id}.txt",
                                              frame_id=frame_id, image_size=sample.frame.size,
                                              expected_sha256=verified["source_sha256"]["calibration"])
        projection = projector.project(scan)
        if (projection.scan_sha256 != verified["source_sha256"]["scan"] or
                len(projection.returns) != verified["projection_in_image_rows"]):
            raise ValueError(f"Scan/projection provenance changed: {frame_id}")
        m5_path = REPO / f"outputs/kitti_m5/heldout/frames/{frame_id}.json"
        frozen = _read(m5_path)
        pairs = frozen["matching"]["pairs"]
        frozen_rows = {(row["reference_id"], row["detection_id"]): row
                       for row in frozen["paired_object_reference_z_proxy_rows"]}
        if (frozen["frame_id"] != frame_id or
                len(pairs) != len(frozen_rows) or
                len(pairs) != frozen["matching"]["matched_pairs"] or
                _sha(m5_path) != verified["source_sha256"]["frozen_m5_frame"]):
            raise ValueError(f"Frozen M5 match record changed: {frame_id}")
        detections = detector.predict(sample.frame)
        if len(detections.detections) != frozen["matching"]["total_detections"]:
            raise ValueError(f"Detector count differs from frozen M5: {frame_id}")
        if dict(sorted(Counter(d.class_name for d in detections.detections).items())) != frozen["matching"]["detection_class_counts"]:
            raise ValueError(f"Detector class sequence/count differs from frozen M5: {frame_id}")
        det_rows = detections.to_summary()["detections"]
        identity = {"frame_id": frame_id,
                    "image_sha256": verified["source_sha256"]["image"],
                    "detector_checkpoint_sha256": m5.data["detector_checkpoint_sha256"],
                    "frozen_m5_config_sha256": _sha(REPO / "configs/kitti_eval_v1.json"),
                    "ordered_detections": det_rows}
        fingerprint = sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False).encode("utf-8")).hexdigest()
        detector_geometry.append({**identity, "geometry_fingerprint_sha256": fingerprint,
                                  "archived_independent_box_coordinates": False,
                                  "coordinate_comparison": "unavailable; frozen M5 stored only confidence/IoU, not box coordinates"})
        sources = SourceDigests(scan.source_sha256, projector.calibration_sha256,
                                protocol.sha256, _sha(m5_path))
        frame_supported = 0
        for pair in pairs:
            reference_id, detection_id = pair["reference_id"], pair["detection_id"]
            row = frozen_rows[(reference_id, detection_id)]
            annotation = sample.annotations[reference_id]
            detection = detections.detections[detection_id]
            mapped = m5.mapping.coco_to_kitti.get(detection.class_name)
            if mapped != row["class_name"] or annotation.class_name != row["class_name"]:
                raise ValueError(f"Frozen class or detection order mismatch: {frame_id} det {detection_id}")
            if abs(detection.confidence - row["confidence"]) > 1e-5:
                raise ValueError(f"Frozen detector confidence mismatch: {frame_id} det {detection_id}")
            actual_iou = box_iou(detection.box, annotation.box)
            if abs(actual_iou - pair["iou"]) > 1e-5:
                raise ValueError(f"Frozen detection IoU mismatch: {frame_id} det {detection_id}")
            match = FrozenObjectMatch(frame_id, reference_id, detection_id,
                                      row["class_name"], mapped, detection.box,
                                      row["box_median_z_m"], row["mask_median_z_m"])
            reference = associate_object(projection, sample.annotations, match, sources, protocol)
            record = reference.to_dict()
            record.update({"drive_id": drives[frame_id], "class_name": row["class_name"],
                           "occluded": annotation.occluded, "truncated": annotation.truncated,
                           "annotation_box_xyxy": list(annotation.box.xyxy),
                           "detector_box_xyxy": list(detection.box.xyxy),
                           "detector_geometry_fingerprint_sha256": fingerprint,
                           "frozen_match_iou": pair["iou"],
                           "reconstructed_match_iou": actual_iou,
                           "frozen_detector_confidence": row["confidence"],
                           "original_m5_bottom_center_z_m": row["reference_z_m"],
                           "support_z_spread_m": (record["support_z_max_m"] - record["support_z_min_m"]
                                                  if record["support_z_min_m"] is not None else None)})
            if record["status"] == "supported":
                frame_supported += 1
                if match.box_median_z_m is not None and match.mask_median_z_m is not None:
                    e_box = match.box_median_z_m - reference.lidar_z_m
                    e_mask = match.mask_median_z_m - reference.lidar_z_m
                    record["paired_error"] = {"box_signed_m": e_box, "mask_signed_m": e_mask,
                                              "mask_minus_box_absolute_m": abs(e_mask) - abs(e_box)}
                else:
                    record["paired_error"] = None
            else:
                record["paired_error"] = None
            objects.append(record)
        frames.append({"frame_id": frame_id, "drive_id": drives[frame_id],
                       "frozen_m5_match_count": len(pairs), "supported_count": frame_supported,
                       "reconstructed_detector_count": len(det_rows),
                       "detector_geometry_fingerprint_sha256": fingerprint,
                       "in_image_returns": len(projection.returns),
                       "sources": vars(sources)})
        if number % 16 == 0 or number == len(ids):
            print(f"Evaluated {number}/{len(ids)} frozen held-out frames; {len(objects)} matches", flush=True)
    if len(objects) != protocol.data["heldout_matched_denominator"]:
        raise ValueError("Original 785 frozen matched-pair denominator changed")
    _write(OUT / "object_records.json", objects)
    _write(OUT / "frame_records.json", frames)
    _write(OUT / "detector_geometry.json", detector_geometry)
    _write(OUT / "drive_mapping.json", {"mapping_archive_sha256": DEVKIT_SHA256,
                                       "frame_to_raw_drive": drives,
                                       "development_heldout_drives_overlap_count": 34,
                                       "pilot_heldout_drives_overlap_count": 10})
    print(json.dumps({"frozen_matched": len(objects),
                      "supported": sum(row["status"] == "supported" for row in objects)}))


if __name__ == "__main__":
    main()
