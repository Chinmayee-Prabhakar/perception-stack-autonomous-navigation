"""Source-faithful held-out LiDAR overlays and prespecified failure-case crops."""

from hashlib import sha256
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw

from perception_stack.kitti_eval.dataset import KittiObjectDataset
from perception_stack.kitti_lidar import load_calibrated_projector, load_verified_object_scan
from perception_stack.results import BoxXYXY

from plot_m5_l2_real_pilot import _classify, _font
from run_m5_l2_geometry import _overlay


REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "datasets/kitti_object"
OUT = REPO / "outputs/kitti_m5_l3/heldout"
VISUAL = OUT / "visuals"


def _load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _source(frame_id, source):
    sample = KittiObjectDataset(DATASET).load(frame_id)
    scan = load_verified_object_scan(DATASET / f"training/velodyne/{frame_id}.bin",
                                     frame_id=frame_id,
                                     expected_sha256=source[frame_id]["scan"])
    projector = load_calibrated_projector(DATASET / f"training/calib/{frame_id}.txt",
                                          frame_id=frame_id, image_size=sample.frame.size,
                                          expected_sha256=source[frame_id]["calibration"])
    return sample, projector.project(scan)


def _case_key(row):
    return (row["frame_id"], row["reference_id"], row["detection_id"])


def _select(rows):
    supported = [row for row in rows if row["status"] == "supported" and row["paired_error"]]
    cases = []
    seen = set()

    def add(name, candidates, key, description, *, reverse=True):
        ranked = sorted(candidates, key=lambda row: (key(row), _case_key(row)), reverse=reverse)
        chosen = next((row for row in ranked if _case_key(row) not in seen), None)
        if chosen is not None:
            cases.append((name, chosen, description))
            seen.add(_case_key(chosen))

    add("well_supported", supported, lambda row: row["support_bins"],
        "High retained-bin support; support is not proof of RGB-visible-face identity")
    add("minimal_valid", supported, lambda row: row["support_bins"],
        "Fewest valid bins under the unchanged three-bin primary rule", reverse=False)
    add("prediction_disagreement", supported,
        lambda row: abs(row["paired_error"]["mask_minus_box_absolute_m"]),
        "Large absolute difference between frozen box and mask errors")
    add("large_measured_error", supported,
        lambda row: max(abs(row["paired_error"]["box_signed_m"]),
                        abs(row["paired_error"]["mask_signed_m"])),
        "Large error to sparse measured-surface proxy; cause is not established")
    uncertainty = [row for row in rows if row["occluded"] >= 2 or
                   row["rejected_counts"]["same_pixel_occluded"] > 0 or
                   row["rejected_counts"]["competing_cuboid"] > 0]
    if uncertainty:
        add("visibility_uncertainty", uncertainty,
            lambda row: (row["rejected_counts"]["same_pixel_occluded"],
                         row["occluded"], row["truncated"]),
            "Occlusion or observed same-pixel conflict; exact RGB surface remains uncertain")
    unsupported = [row for row in rows if row["status"] != "supported"]
    if unsupported:
        add("unsupported_reference", unsupported, lambda row: row["support_bins"],
            "Fewer than three accepted bins: no primary LiDAR reference", reverse=False)
    return cases


def _draw_case(name, row, description, source):
    frame_id = row["frame_id"]
    sample, projection = _source(frame_id, source)
    detector_box = BoxXYXY(*row["detector_box_xyxy"])
    kept, competing, gate, foreground = _classify(projection, sample.annotations, row, detector_box)
    annotation = sample.annotations[row["reference_id"]]
    width, height = sample.frame.size
    union = (min(annotation.box.x1, detector_box.x1), min(annotation.box.y1, detector_box.y1),
             max(annotation.box.x2, detector_box.x2), max(annotation.box.y2, detector_box.y2))
    margin = max(25, int(max(union[2] - union[0], union[3] - union[1]) * .3))
    left, top = max(0, int(union[0]) - margin), max(0, int(union[1]) - margin)
    right, bottom = min(width, int(union[2]) + margin), min(height, int(union[3]) + margin)
    original = Image.fromarray(sample.frame.image_rgb).crop((left, top, right, bottom))
    scale = max(2, min(5, int(900 / max(original.size))))
    crop = original.resize((original.width * scale, original.height * scale), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (crop.width + 590, max(crop.height, 700)), "#14212b")
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
        draw.line((x - 5, y - 5, x + 5, y + 5), fill="#ff4055", width=2)
        draw.line((x - 5, y + 5, x + 5, y - 5), fill="#ff4055", width=2)
    for point in kept:
        x, y = coords(point.u, point.v)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#f8ed6b", outline="#111111")
    for box, color in ((annotation.box, "#51ed94"), (detector_box, "#5cdbfa")):
        draw.rectangle((*coords(box.x1, box.y1), *coords(box.x2, box.y2)), outline=color, width=3)
    x, y = crop.width + 16, 16
    f = _font(16)
    lidar = "unsupported (<3 bins)" if row["lidar_z_m"] is None else f"{row['lidar_z_m']:.3f} m"
    box = row["frozen_box_median_z_m"]
    mask = row["frozen_mask_median_z_m"]
    lines = [
        f"{name.replace('_', ' ').upper()} | KITTI Object {frame_id}",
        f"Frozen ref #{row['reference_id']} / detector #{row['detection_id']} | {row['class_name']}",
        description,
        "Original pixels; nearest-neighbor enlargement for display",
        "Green: annotated 2D box; cyan: reconstructed frozen detector box",
        f"KITTI occlusion {row['occluded']} | truncation {row['truncated']:.2f}",
        f"Predicted M4 box/mask median Z: {box:.3f} / {mask:.3f} m",
        f"Measured sparse LiDAR camera-2 median Z: {lidar}",
        f"Yellow: {len(kept)} accepted measured bins; no interpolation",
        f"Orange: {len(gate)} 2D-gate rejects; purple: {len(competing)} competing cuboid",
        f"Red X: {len(foreground)} same-pixel visibility losses",
        f"LiDAR Z spread: {row['support_z_spread_m']:.3f} m" if row["support_z_spread_m"] is not None else "LiDAR Z spread: unavailable",
    ]
    if row["paired_error"] is not None:
        error = row["paired_error"]
        lines.extend((f"Box/mask error to sparse Z: {error['box_signed_m']:+.3f} / {error['mask_signed_m']:+.3f} m",
                      f"Mask-minus-box absolute error: {error['mask_minus_box_absolute_m']:+.3f} m"))
    lines.extend(("", "KITTI Object, CC BY-NC-SA 3.0",
                  "Sparse return is not dense RGB-visible-surface truth."))
    if len(kept) <= 10:
        lines.extend(("", "All accepted measured bins (pixel x,y : Z m):"))
        lines.extend(f"({point.pixel[0]},{point.pixel[1]}) : {point.camera2_z_m:.3f}"
                     for point in kept)
    else:
        lines.extend(("", "Example accepted bins (pixel x,y : Z m):"))
        lines.extend(f"({point.pixel[0]},{point.pixel[1]}) : {point.camera2_z_m:.3f}"
                     for point in kept[::max(1, len(kept) // 5)][:5])
    for line in lines:
        for segment in textwrap.wrap(line, width=62) if line else [""]:
            draw.text((x, y), segment, fill="white", font=f)
            y += 24 if segment else 13
    if y > canvas.height - 5:
        raise ValueError(f"Figure legend overflow: {name}")
    path = VISUAL / f"{name}_{frame_id}_ref{row['reference_id']:02d}.png"
    canvas.save(path)
    return {"name": name, "frame_id": frame_id, "reference_id": row["reference_id"],
            "detection_id": row["detection_id"], "description": description,
            "path": str(path.relative_to(REPO)).replace("\\", "/"),
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "support_bins": len(kept), "user_visual_sign_off": "pending"}


def main():
    rows = _load("object_records.json")
    geometry = _load("geometry_provenance.json")
    source = {row["frame_id"]: row["source_sha256"] for row in geometry["frames"]}
    VISUAL.mkdir(parents=True, exist_ok=True)
    cases = _select(rows)
    frame_ids = list(dict.fromkeys(row["frame_id"] for _, row, _ in cases))
    for row in geometry["frames"]:
        if len(frame_ids) == 8:
            break
        if row["frame_id"] not in frame_ids:
            frame_ids.append(row["frame_id"])
    projection_artifacts = []
    thumbnails = []
    for frame_id in frame_ids:
        sample, projection = _source(frame_id, source)
        picture = _overlay(sample.frame.image_rgb, projection.returns, frame_id)
        path = VISUAL / f"projection_{frame_id}.png"
        picture.save(path)
        projection_artifacts.append({"frame_id": frame_id,
                                     "path": str(path.relative_to(REPO)).replace("\\", "/"),
                                     "sha256": sha256(path.read_bytes()).hexdigest(),
                                     "in_image_measured_returns": len(projection.returns)})
        thumbnail = picture.copy()
        thumbnail.thumbnail((620, 225))
        thumbnails.append(thumbnail)
    sheet = Image.new("RGB", (1240, 900), "#14212b")
    for index, thumbnail in enumerate(thumbnails):
        sheet.paste(thumbnail, ((index % 2) * 620, (index // 2) * 225))
    sheet_path = VISUAL / "projection_contact_sheet.png"
    sheet.save(sheet_path)
    artifacts = [_draw_case(name, row, description, source) for name, row, description in cases]
    manifest = {"contact_sheet": {"path": str(sheet_path.relative_to(REPO)).replace("\\", "/"),
                                  "sha256": sha256(sheet_path.read_bytes()).hexdigest(),
                                  "frame_ids": frame_ids},
                "full_frame_projection_artifacts": projection_artifacts,
                "object_artifacts": artifacts,
                "unsupported_case_exists": any(row["status"] != "supported" for row in rows),
                "all_visual_sign_off": "pending_user_review"}
    (OUT / "visual_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"contact_sheet": manifest["contact_sheet"],
                      "object_artifacts": artifacts}, indent=2), flush=True)


if __name__ == "__main__":
    main()
