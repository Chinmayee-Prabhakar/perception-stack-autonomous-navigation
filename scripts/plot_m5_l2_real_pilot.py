"""Source-faithful crops of measured pilot returns and frozen object associations."""

from hashlib import sha256
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from perception_stack.kitti_eval.dataset import KittiObjectDataset
from perception_stack.kitti_lidar import load_calibrated_projector, load_verified_object_scan
from perception_stack.kitti_lidar.association import inside_oriented_cuboid
from perception_stack.results import BoxXYXY


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "outputs/kitti_m5_l2/pilot"
DATASET = REPO / "datasets/kitti_object"
CASES = (
    ("object_association_006612_ref01", "006612", 1,
     "Frozen pair: annotated box, detector box, measured retained bins and detector-gate exclusions"),
    ("valid_sparse_003601_ref04", "003601", 4,
     "Valid sparse reference: nine distinct measured bins; median camera-2 Z"),
    ("occlusion_uncertainty_005465_ref04", "005465", 4,
     "Potential uncertainty: annotated occlusion 3 and two same-pixel visibility conflicts"),
)


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _inside(box: BoxXYXY, u: float, v: float) -> bool:
    return box.x1 <= u < box.x2 and box.y1 <= v < box.y2


def _classify(projection, annotations, reference, detector_box):
    target = annotations[reference["reference_id"]]
    winners = {}
    for point in projection.returns:
        current = winners.get(point.pixel)
        if current is None or (point.camera2_z_m, point.original_index) < (current.camera2_z_m, current.original_index):
            winners[point.pixel] = point
    kept, competing, two_d_gate, foreground = [], [], [], []
    for point in projection.returns:
        if not inside_oriented_cuboid(point.reference_xyz_m, target):
            continue
        if any(other.object_id != target.object_id and inside_oriented_cuboid(point.reference_xyz_m, other)
               for other in annotations):
            competing.append(point)
        elif not _inside(target.box, point.u, point.v) or not _inside(detector_box, point.u, point.v):
            two_d_gate.append(point)
        elif winners[point.pixel] is not point:
            foreground.append(point)
        else:
            kept.append(point)
    expected = {point["original_scan_index"] for point in reference["accepted_bins"]}
    assert {point.original_index for point in kept} == expected
    assert len(foreground) == reference["rejected_counts"]["same_pixel_occluded"]
    assert len(competing) == reference["rejected_counts"]["competing_cuboid"]
    assert len(two_d_gate) == (reference["rejected_counts"]["outside_annotation_box_2d"]
                               + reference["rejected_counts"]["outside_detection_box_2d"])
    return kept, competing, two_d_gate, foreground


def _make_case(name, frame_id, reference_id, description, records, source):
    record = next(row for row in records if row["frame_id"] == frame_id and row["reference_id"] == reference_id)
    sample = KittiObjectDataset(DATASET).load(frame_id)
    scan = load_verified_object_scan(DATASET / f"training/velodyne/{frame_id}.bin", frame_id=frame_id,
                                     expected_sha256=source[frame_id]["scan"])
    projector = load_calibrated_projector(DATASET / f"training/calib/{frame_id}.txt", frame_id=frame_id,
                                          image_size=sample.frame.size,
                                          expected_sha256=source[frame_id]["calibration"])
    projection = projector.project(scan)
    detector_box = BoxXYXY(*record["detector_box_xyxy"])
    kept, competing, gate, foreground = _classify(projection, sample.annotations, record, detector_box)
    annotation = sample.annotations[reference_id]
    width, height = sample.frame.size
    union = (min(annotation.box.x1, detector_box.x1), min(annotation.box.y1, detector_box.y1),
             max(annotation.box.x2, detector_box.x2), max(annotation.box.y2, detector_box.y2))
    margin = max(25, int(max(union[2] - union[0], union[3] - union[1]) * .3))
    left, top = max(0, int(union[0]) - margin), max(0, int(union[1]) - margin)
    right, bottom = min(width, int(union[2]) + margin), min(height, int(union[3]) + margin)
    original = Image.fromarray(sample.frame.image_rgb).crop((left, top, right, bottom))
    scale = max(2, min(5, int(900 / max(original.size))))
    crop = original.resize((original.width * scale, original.height * scale), Image.Resampling.NEAREST)
    panel_width = 520
    canvas = Image.new("RGB", (crop.width + panel_width, max(crop.height, 650)), "#14212b")
    canvas.paste(crop)
    draw = ImageDraw.Draw(canvas)

    def coords(u, v):
        return ((u - left) * scale, (v - top) * scale)

    for point in gate:
        x, y = coords(point.u, point.v)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#ff9c53")
    for point in competing:
        x, y = coords(point.u, point.v)
        draw.line((x - 4, y - 4, x + 4, y + 4), fill="#dd80ff", width=2)
        draw.line((x - 4, y + 4, x + 4, y - 4), fill="#dd80ff", width=2)
    for point in foreground:
        x, y = coords(point.u, point.v)
        draw.line((x - 5, y - 5, x + 5, y + 5), fill="#ff4055", width=3)
        draw.line((x - 5, y + 5, x + 5, y - 5), fill="#ff4055", width=3)
    for point in kept:
        x, y = coords(point.u, point.v)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#f8ed6b", outline="#111111")

    def box_rect(box, color):
        a = coords(box.x1, box.y1)
        b = coords(box.x2, box.y2)
        draw.rectangle((*a, *b), outline=color, width=3)

    box_rect(annotation.box, "#51ed94")
    box_rect(detector_box, "#5cdbfa")
    x = crop.width + 16
    f16 = _font(16)
    f14 = _font(14)
    lines = [
        (name.replace("_", " "), "#ffffff", f16),
        (f"KITTI Object image_2 {frame_id}  ref #{reference_id} / det #{record['detection_id']}", "#ffffff", f14),
        ("Original camera image crop; nearest-neighbor display enlargement", "#bac9d6", f14),
        ("", "#ffffff", f14),
        ("ANNOTATED: green 2D box; KITTI 3D cuboid gate", "#51ed94", f14),
        (f"Class {record['class_name']} | occlusion {record['occluded']} | truncation {record['truncated']:.2f}", "#bac9d6", f14),
        ("PREDICTED: cyan frozen-match detector box", "#5cdbfa", f14),
        (f"Frozen M4 box median Z {record['frozen_box_median_z_m']:.2f} m; mask median Z {record['frozen_mask_median_z_m']:.2f} m", "#bac9d6", f14),
        ("", "#ffffff", f14),
        ("MEASURED: yellow retained Velodyne pixel bins", "#f8ed6b", f14),
        (f"{len(kept)} bins | median camera-2 Z {record['lidar_z_m']:.3f} m", "#ffffff", f14),
        (f"Z range {record['support_z_min_m']:.3f}-{record['support_z_max_m']:.3f} m", "#bac9d6", f14),
        (f"Orange: {len(gate)} 2D-gate rejects; purple: {len(competing)} competing cuboid", "#ff9c53", f14),
        (f"Red X: {len(foreground)} target returns losing global same-pixel visibility", "#ff4055", f14),
        ("No interpolation; points are discrete measured returns.", "#bac9d6", f14),
        ("Sparse support cannot establish exact RGB-visible surface.", "#bac9d6", f14),
    ]
    if len(kept) <= 12:
        lines.append(("", "#ffffff", f14))
        lines.append(("All retained measured bins (pixel x,y : Z m):", "#f8ed6b", f14))
        for point in kept:
            lines.append((f"({point.pixel[0]},{point.pixel[1]}) : {point.camera2_z_m:.3f}", "#bac9d6", f14))
    else:
        lines.append(("Example retained measured bins (pixel x,y : Z m):", "#f8ed6b", f14))
        for point in kept[::max(1, len(kept) // 5)][:5]:
            lines.append((f"({point.pixel[0]},{point.pixel[1]}) : {point.camera2_z_m:.3f}", "#bac9d6", f14))
    y = 16
    for message, color, font in lines:
        draw.text((x, y), message, fill=color, font=font)
        y += 23 if message else 15
    if y > canvas.height - 10:
        raise ValueError(f"Legend exceeds output canvas for {name}")
    path = OUTPUT / "visuals" / f"{name}.png"
    canvas.save(path)
    return {"name": name, "frame_id": frame_id, "reference_id": reference_id,
            "description": description, "path": str(path.relative_to(REPO)).replace("\\", "/"),
            "sha256": sha256(path.read_bytes()).hexdigest(), "retained_bins": len(kept),
            "two_d_gate_rejects": len(gate), "same_pixel_occluded": len(foreground),
            "annotation_occlusion": record["occluded"], "user_visual_sign_off": "pending"}


def main() -> None:
    records = json.loads((OUTPUT / "object_records.json").read_text(encoding="utf-8"))
    geometry = json.loads((OUTPUT / "geometry_provenance.json").read_text(encoding="utf-8"))
    source = {row["frame_id"]: row["source_sha256"] for row in geometry["frames"]}
    artifacts = [_make_case(*case, records, source) for case in CASES]
    assert all(row["status"] == "supported" for row in records)
    manifest = {"artifacts": artifacts,
                "insufficient_support_artifact": None,
                "insufficient_support_reason": "No frozen pilot match had fewer than three accepted bins.",
                "all_visual_sign_off": "pending_user_review"}
    (OUTPUT / "visual_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                  encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
