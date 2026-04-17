"""Validate held-out Object sources and projection with the accepted M5-L1 geometry."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import numpy as np

from perception_stack.kitti_eval.dataset import KittiObjectDataset
from perception_stack.kitti_lidar import load_calibrated_projector, load_verified_object_scan


REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "datasets/kitti_object"
OUT = REPO / "outputs/kitti_m5_l3/heldout"


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    split = json.loads((REPO / "configs/kitti_split_manifest_v1.json").read_text(encoding="utf-8"))
    ids = split["heldout_frame_ids"]
    assert len(ids) == len(set(ids)) == 256
    provenance = json.loads((REPO / "outputs/kitti_m5/dataset_provenance.json").read_text(encoding="utf-8"))
    extracted = json.loads((OUT / "extraction_manifest.json").read_text(encoding="utf-8"))
    if extracted["extracted_count"] != 256:
        raise ValueError("Expected exactly 256 selected held-out scans")
    scan_rows = {row["frame_id"]: row for row in extracted["scans"]}
    if set(scan_rows) != set(ids):
        raise ValueError("Extracted scan IDs differ from frozen held-out split")
    calib_zip = Path.home() / "Downloads/data_object_calib.zip"
    label_zip = Path.home() / "Downloads/data_object_label_2.zip"
    if _sha(calib_zip) != provenance["archive_metadata"][calib_zip.name]["sha256"]:
        raise ValueError("Approved calibration ZIP changed")
    if _sha(label_zip) != provenance["archive_metadata"][label_zip.name]["sha256"]:
        raise ValueError("Approved label ZIP changed")
    dataset = KittiObjectDataset(DATASET)
    frames = []
    with zipfile.ZipFile(calib_zip) as calibrations, zipfile.ZipFile(label_zip) as labels:
        for index, frame_id in enumerate(ids, 1):
            sample = dataset.load(frame_id)
            image = DATASET / f"training/image_2/{frame_id}.png"
            calibration = DATASET / f"training/calib/{frame_id}.txt"
            label = DATASET / f"training/label_2/{frame_id}.txt"
            scan_path = DATASET / f"training/velodyne/{frame_id}.bin"
            image_sha = _sha(image)
            if image_sha != provenance["selected_image_files"][frame_id]["sha256"]:
                raise ValueError(f"Frozen M5 image source changed: {frame_id}")
            calib_bytes = calibration.read_bytes()
            label_bytes = label.read_bytes()
            if calib_bytes != calibrations.read(f"training/calib/{frame_id}.txt"):
                raise ValueError(f"Object calibration archive mismatch: {frame_id}")
            if label_bytes != labels.read(f"training/label_2/{frame_id}.txt"):
                raise ValueError(f"Object annotation archive mismatch: {frame_id}")
            scan = load_verified_object_scan(scan_path, frame_id=frame_id,
                                             expected_sha256=scan_rows[frame_id]["sha256"])
            if scan_path.stat().st_size != scan_rows[frame_id]["bytes"]:
                raise ValueError(f"Held-out scan size mismatch: {frame_id}")
            if (scan.total_rows != scan_rows[frame_id]["total_xyzi_rows"] or
                    scan.retained_rows != scan_rows[frame_id]["finite_xyzi_rows"] or
                    scan.nonfinite_xyz_rows != scan_rows[frame_id]["nonfinite_xyz_rows"] or
                    scan.nonfinite_intensity_rows != scan_rows[frame_id]["nonfinite_intensity_rows"]):
                raise ValueError(f"Held-out scan finite-row accounting mismatch: {frame_id}")
            projector = load_calibrated_projector(calibration, frame_id=frame_id,
                                                  image_size=sample.frame.size,
                                                  expected_sha256=sha256(calib_bytes).hexdigest())
            projection = projector.project(scan)
            if len(projection.returns) + sum(projection.rejected_counts.values()) != scan.retained_rows:
                raise ValueError(f"Projection count partition mismatch: {frame_id}")
            if any(point.camera2_z_m <= 0 or not (0 <= point.u < sample.frame.size[0] and
                                                   0 <= point.v < sample.frame.size[1])
                   for point in projection.returns):
                raise ValueError(f"Retained projection invalid: {frame_id}")
            sampled = projection.returns[::max(1, len(projection.returns) // 500)]
            residual = max((projector.factorization_residual(point.reference_xyz_m)
                            for point in sampled), default=0.0)
            if residual > 1e-7:
                raise ValueError(f"Translated P2 factorization mismatch: {frame_id}")
            z = np.asarray([point.camera2_z_m for point in projection.returns])
            if not len(z):
                raise ValueError(f"No in-image Object returns: {frame_id}")
            m5_path = REPO / f"outputs/kitti_m5/heldout/frames/{frame_id}.json"
            frozen = json.loads(m5_path.read_text(encoding="utf-8"))
            if frozen["frame_id"] != frame_id or frozen["matching"]["frame_id"] != frame_id:
                raise ValueError(f"Frozen frame identity mismatch: {frame_id}")
            frames.append({
                "frame_id": frame_id, "image_size": list(sample.frame.size),
                "annotation_count": len(sample.annotations),
                "frozen_m5_match_count": frozen["matching"]["matched_pairs"],
                "scan_bytes": scan_path.stat().st_size,
                "scan_total_rows": scan.total_rows, "scan_finite_rows": scan.retained_rows,
                "scan_nonfinite_xyz_rows": scan.nonfinite_xyz_rows,
                "scan_nonfinite_intensity_rows": scan.nonfinite_intensity_rows,
                "projection_in_image_rows": len(projection.returns),
                "projection_rejected_counts": dict(projection.rejected_counts),
                "camera2_z_min_m": float(z.min()), "camera2_z_median_m": float(np.median(z)),
                "camera2_z_max_m": float(z.max()),
                "camera2_center_ref_m": list(projector.camera2_center_ref_m),
                "direct_vs_factorized_p2_max_residual": residual,
                "source_sha256": {"image": image_sha, "label": sha256(label_bytes).hexdigest(),
                                  "calibration": projector.calibration_sha256,
                                  "scan": scan.source_sha256, "frozen_m5_frame": _sha(m5_path)},
                "visual_alignment_review": "pending_user_review",
            })
            if index % 16 == 0 or index == len(ids):
                print(f"Validated geometry {index}/{len(ids)} held-out frames", flush=True)
    result = {"scope": "original frozen 256 held-out Object frames",
              "archive_sha256": extracted["archive_sha256"],
              "protocol_sha256": _sha(REPO / "configs/kitti_lidar_protocol_v1.json"),
              "selected_count": len(ids), "verified_scan_count": len(frames),
              "total_scan_rows": sum(row["scan_total_rows"] for row in frames),
              "total_in_image_returns": sum(row["projection_in_image_rows"] for row in frames),
              "total_projection_rejected_counts": dict(sum((Counter(row["projection_rejected_counts"])
                                                        for row in frames), Counter())),
              "frames": frames}
    (OUT / "geometry_provenance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
