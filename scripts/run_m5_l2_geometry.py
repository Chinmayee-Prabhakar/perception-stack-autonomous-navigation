"""Verify selected KITTI Object pilot sources and render discrete real LiDAR overlays."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from perception_stack.kitti_eval.dataset import KittiObjectDataset
from perception_stack.kitti_lidar import load_calibrated_projector, load_verified_object_scan


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "outputs/kitti_m5_l2/pilot"
DATASET = REPO / "datasets/kitti_object"
DOWNLOADS = Path.home() / "Downloads"


def digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _archive(path: Path, expected: str) -> zipfile.ZipFile:
    if digest(path.read_bytes()) != expected:
        raise ValueError(f"Approved KITTI archive changed: {path.name}")
    return zipfile.ZipFile(path)


def _overlay(image_rgb: np.ndarray, returns, frame_id: str) -> Image.Image:
    image = Image.fromarray(image_rgb).convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 52), "#15212c")
    canvas.paste(image)
    draw = ImageDraw.Draw(canvas)
    palette = ((15, "#fae36b"), (30, "#ff9c53"), (50, "#f2485c"),
               (80, "#b65ee8"), (float("inf"), "#69d9ff"))
    for point in returns:
        color = next(color for limit, color in palette if point.camera2_z_m < limit)
        u, v = point.pixel
        draw.ellipse((u - 1, v - 1, u + 1, v + 1), fill=color)
    draw.text((10, image.height + 5),
              f"KITTI Object image_2 {frame_id} | dots: measured Velodyne returns projected by P2 | no interpolation",
              fill="white")
    draw.text((10, image.height + 24),
              "Measured camera-2 Z: yellow <15m | orange 15-30m | red 30-50m | purple 50-80m | cyan >=80m",
              fill="white")
    return canvas


def main() -> None:
    split = json.loads((REPO / "configs/kitti_split_manifest_v1.json").read_text(encoding="utf-8"))
    ids = split["pilot_frame_ids"]
    assert len(ids) == len(set(ids)) == 16
    inventory = json.loads((REPO / "outputs/kitti_m5_l2_preflight/archive_inventory.json").read_text(encoding="utf-8"))
    extraction = json.loads((REPO / "outputs/kitti_m5_l2_preflight/extraction_manifest.json").read_text(encoding="utf-8"))
    assert inventory["archive_sha256"] == extraction["archive_sha256"]
    assert extraction["extracted_count"] == 16
    scan_rows = {row["frame_id"]: row for row in extraction["scans"]}
    assert set(scan_rows) == set(ids)
    assert {path.stem for path in (DATASET / "training/velodyne").glob("*.bin")} == set(ids)

    provenance = json.loads((REPO / "outputs/kitti_m5/dataset_provenance.json").read_text(encoding="utf-8"))
    with _archive(DOWNLOADS / "data_object_calib.zip",
                  provenance["archive_metadata"]["data_object_calib.zip"]["sha256"]) as calib_archive, \
         _archive(DOWNLOADS / "data_object_label_2.zip",
                  provenance["archive_metadata"]["data_object_label_2.zip"]["sha256"]) as label_archive:
        dataset = KittiObjectDataset(DATASET)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        visual = OUTPUT / "visuals"
        visual.mkdir(exist_ok=True)
        rows = []
        thumbnails = []
        for frame_id in ids:
            sample = dataset.load(frame_id)
            image_path = DATASET / f"training/image_2/{frame_id}.png"
            image_bytes = image_path.read_bytes()
            if digest(image_bytes) != provenance["selected_image_files"][frame_id]["sha256"]:
                raise ValueError(f"Frozen image bytes changed: {frame_id}")
            calibration_path = DATASET / f"training/calib/{frame_id}.txt"
            label_path = DATASET / f"training/label_2/{frame_id}.txt"
            calibration_bytes = calibration_path.read_bytes()
            label_bytes = label_path.read_bytes()
            if calibration_bytes != calib_archive.read(f"training/calib/{frame_id}.txt"):
                raise ValueError(f"Calibration differs from approved Object archive: {frame_id}")
            if label_bytes != label_archive.read(f"training/label_2/{frame_id}.txt"):
                raise ValueError(f"Label differs from approved Object archive: {frame_id}")
            scan_path = DATASET / f"training/velodyne/{frame_id}.bin"
            scan = load_verified_object_scan(scan_path, frame_id=frame_id,
                                             expected_sha256=scan_rows[frame_id]["sha256"])
            if scan_path.stat().st_size != scan_rows[frame_id]["bytes"]:
                raise ValueError(f"Extracted scan size changed: {frame_id}")
            projector = load_calibrated_projector(calibration_path, frame_id=frame_id,
                                                  image_size=sample.frame.size,
                                                  expected_sha256=digest(calibration_bytes))
            projection = projector.project(scan)
            if len(projection.returns) + sum(projection.rejected_counts.values()) != scan.retained_rows:
                raise AssertionError(f"Projection counts fail to partition scan: {frame_id}")
            if any(point.camera2_z_m <= 0 for point in projection.returns):
                raise AssertionError(f"Nonpositive retained camera-2 Z: {frame_id}")
            sampled = projection.returns[::max(1, len(projection.returns) // 500)]
            residual = max((projector.factorization_residual(point.reference_xyz_m)
                            for point in sampled), default=0.0)
            if residual > 1e-7:
                raise ValueError(f"Direct/factorized P2 mismatch: {frame_id}, {residual}")
            z = np.asarray([point.camera2_z_m for point in projection.returns])
            if not len(z):
                raise ValueError(f"No in-image LiDAR returns: {frame_id}")
            picture = _overlay(sample.frame.image_rgb, projection.returns, frame_id)
            picture_path = visual / f"projection_{frame_id}.png"
            picture.save(picture_path)
            thumb = picture.copy()
            thumb.thumbnail((620, 225))
            thumbnails.append(thumb)
            rows.append({
                "frame_id": frame_id,
                "image_size": list(sample.frame.size),
                "scan_bytes": scan_path.stat().st_size,
                "scan_total_rows": scan.total_rows,
                "scan_finite_rows": scan.retained_rows,
                "scan_nonfinite_xyz_rows": scan.nonfinite_xyz_rows,
                "scan_nonfinite_intensity_rows": scan.nonfinite_intensity_rows,
                "projection_in_image_rows": len(projection.returns),
                "projection_rejected_counts": dict(projection.rejected_counts),
                "camera2_z_min_m": float(z.min()),
                "camera2_z_median_m": float(np.median(z)),
                "camera2_z_p99_m": float(np.percentile(z, 99)),
                "camera2_z_max_m": float(z.max()),
                "camera2_center_ref_m": list(projector.camera2_center_ref_m),
                "direct_vs_factorized_p2_max_residual": residual,
                "annotation_count": len(sample.annotations),
                "source_sha256": {"image": digest(image_bytes), "label": digest(label_bytes),
                                  "calibration": projector.calibration_sha256,
                                  "scan": scan.source_sha256},
                "overlay": str(picture_path.relative_to(REPO)).replace("\\", "/"),
                "visual_alignment_review": "pending_human_review",
            })
            print(f"{frame_id}: {scan.total_rows} scan rows, {len(projection.returns)} in-image returns, residual {residual:.2e}", flush=True)
    # A contact sheet is an index; full-resolution overlays remain the review evidence.
    tile_w, tile_h = 620, 225
    sheet = Image.new("RGB", (tile_w * 2, tile_h * 8), "#15212c")
    for i, thumb in enumerate(thumbnails):
        sheet.paste(thumb, ((i % 2) * tile_w, (i // 2) * tile_h))
    sheet_path = visual / "projection_contact_sheet.png"
    sheet.save(sheet_path)
    report = {
        "scope": "16 frozen M5 pilot frames only; no held-out LiDAR evaluation",
        "archive_sha256": inventory["archive_sha256"],
        "protocol_sha256": digest((REPO / "configs/kitti_lidar_protocol_v1.json").read_bytes()),
        "frames": rows,
        "selected_count": len(ids),
        "verified_scan_count": len(rows),
        "total_scan_rows": sum(row["scan_total_rows"] for row in rows),
        "total_in_image_returns": sum(row["projection_in_image_rows"] for row in rows),
        "total_projection_rejected_counts": dict(sum((Counter(row["projection_rejected_counts"])
                                                    for row in rows), Counter())),
        "contact_sheet": str(sheet_path.relative_to(REPO)).replace("\\", "/"),
        "visual_sign_off": "pending_user_review",
    }
    (OUTPUT / "geometry_provenance.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                                      encoding="utf-8")


if __name__ == "__main__":
    main()
