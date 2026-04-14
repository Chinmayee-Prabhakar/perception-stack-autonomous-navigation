"""Original-frame overlay and calibrated camera X-Z surface-point plot."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from perception_stack.frame import Frame
from perception_stack.results import FrameDetections, FrameRoadSegmentation
from .results import FrameScene, build_scene


def _font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_scene_overlay(
    frame: Frame, detections: FrameDetections,
    road: FrameRoadSegmentation | None, scene: FrameScene,
) -> Image.Image:
    """Render predictions at original resolution, keeping semantic masks distinct."""
    expected = (frame.frame_id, frame.size)
    if (detections.frame_id, detections.image_size) != expected or (scene.frame_id, scene.image_size) != expected:
        raise ValueError("Overlay input IDs/dimensions must match")
    if road is not None and (road.frame_id, road.image_size) != expected:
        raise ValueError("Overlay road IDs/dimensions must match")
    if len(detections.detections) != len(scene.objects):
        raise ValueError("Overlay object counts must match")
    verified = build_scene(frame, detections, road, scene.fusion, scene.associations)
    if verified.to_json() != scene.to_json():
        raise ValueError("Overlay scene content inconsistent with supplied prediction sources")
    image = Image.fromarray(frame.image_rgb).convert("RGBA")
    if road is not None:
        for mask, color, opacity in ((road.drivable_area_mask, (0, 205, 90), 70),
                                     (road.lane_marking_mask, (255, 210, 0), 170)):
            tint = Image.new("RGBA", frame.size, color + (255,))
            image.paste(tint, (0, 0), Image.fromarray(np.uint8(mask) * opacity, "L"))
    for det in detections.detections:
        if det.mask is not None:
            tint = Image.new("RGBA", frame.size, (70, 170, 250, 255))
            image.paste(tint, (0, 0), Image.fromarray(np.uint8(det.mask) * 35, "L"))
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font()
    used_id_regions = []
    for index, (det, obj) in enumerate(zip(detections.detections, scene.objects)):
        if obj.fusion.detection_id != index or obj.fusion.box != det.box:
            raise ValueError("Overlay detections must match the scene")
        assoc = obj.road
        color = ((0, 127, 95) if assoc.evidence_status == "drivable_adjacency_detected"
                 else (164, 74, 0) if assoc.evidence_status == "drivable_adjacency_not_detected"
                 else (119, 58, 155))
        draw.rectangle(det.box.xyxy, outline=color, width=2)
        l, t, r, b = assoc.contact_region_xyxy
        if r > l and b > t:
            draw.rectangle((l, t, r - 1, b - 1), outline=(250, 250, 250), width=1)
        id_label = f"#{index}"
        ib = draw.textbbox((0, 0), id_label, font=font)
        tag_w, tag_h = ib[2] - ib[0] + 4, ib[3] - ib[1] + 4
        x = min(max(0, int(det.box.x1)), max(0, frame.size[0] - tag_w))
        y = min(max(0, int(det.box.y1)), max(0, frame.size[1] - tag_h))
        for dx, dy in ((0, 0), (0, -tag_h - 2), (0, tag_h + 2),
                       (-tag_w - 2, 0), (tag_w + 2, 0)):
            cx = min(max(0, x + dx), frame.size[0] - tag_w)
            cy = min(max(0, y + dy), frame.size[1] - tag_h)
            region = (cx, cy, cx + tag_w, cy + tag_h)
            if all(region[2] <= other[0] or region[0] >= other[2]
                   or region[3] <= other[1] or region[1] >= other[3]
                   for other in used_id_regions):
                x, y = cx, cy
                break
        used_id_regions.append((x, y, x + tag_w, y + tag_h))
        draw.rectangle((x, y, x + tag_w, y + tag_h), fill=(18, 18, 18))
        draw.text((x + 2, y + 1), id_label, fill=color, font=font)
        for method, marker_color in ((obj.fusion.box_method, (255, 255, 255)),
                                     (obj.fusion.mask_method, (255, 225, 30))):
            sample = method.statistics.selected_sample
            if method.statistics.status == "ok" and sample is not None:
                draw.ellipse((sample.u - 2, sample.v - 2, sample.u + 2, sample.v + 2),
                             outline=marker_color, width=2)
    return image


def scene_presentation_data(scene: FrameScene) -> dict:
    """Text for the separate legend; no inference values are changed."""
    statuses = {
        "drivable_adjacency_detected": ("Road-adjacent (+)", "#007f5f"),
        "drivable_adjacency_not_detected": ("Not road-adjacent (−)", "#a44a00"),
    }
    rows = []
    for obj in scene.objects:
        box = obj.fusion.box_method.statistics
        mask = obj.fusion.mask_method.statistics
        state, color = statuses.get(obj.road.evidence_status, ("Insufficient evidence (?)", "#773a9b"))
        rows.append({
            "id": obj.fusion.detection_id,
            "class": obj.fusion.class_name,
            "confidence": obj.fusion.confidence,
            "box_z_m": box.median_z_m if box.status == "ok" else None,
            "mask_z_m": mask.median_z_m if mask.status == "ok" else None,
            "adjacency": state,
            "reason": obj.road.reason,
            "color": color,
        })
    return {"frame_id": scene.frame_id, "rows": rows,
            "mask_legend": ("Green: predicted drivable area", "Yellow: predicted lane-marking pixels",
                            "Blue: predicted object instance masks"),
            "adjacency_legend": "Predicted image-space road adjacency: Road-adjacent (+) | Not road-adjacent (−) | Insufficient evidence (?)",
            "adjacency_method": "Uses predicted drivable-area pixels near each detected object's bounding box.",
            "limitation": "Does not establish physical road contact, whether an object is safely on/off the road, occupancy or collision safety."}


def scene_presentation_layout(image_size: tuple[int, int], object_count: int) -> dict:
    """Keep the camera at source scale and bound dense legend height."""
    if min(image_size) <= 0 or object_count < 0:
        raise ValueError("Presentation requires positive image dimensions and nonnegative object count")
    width = max(image_size[0], 960)
    columns = 2
    rows_per_column = max(1, (object_count + columns - 1) // columns)
    panel_height = 78 + rows_per_column * 27 + 76
    return {"canvas_size": (width, image_size[1] + panel_height),
            "columns": columns, "rows_per_column": rows_per_column,
            "panel_height": panel_height}


def render_scene_presentation(
    frame: Frame, detections: FrameDetections,
    road: FrameRoadSegmentation | None, scene: FrameScene,
) -> Image.Image:
    """Original-size camera overlay with a separate, readable information panel."""
    camera = render_scene_overlay(frame, detections, road, scene)
    data = scene_presentation_data(scene)
    font = _font(15)
    title_font = _font(17)
    layout = scene_presentation_layout(frame.size, len(data["rows"]))
    width = layout["canvas_size"][0]
    columns = layout["columns"]
    rows_per_column = layout["rows_per_column"]
    header_height = 78
    row_height = 27
    footer_height = 76
    panel_height = layout["panel_height"]
    canvas = Image.new("RGB", layout["canvas_size"], "#f5f7f9")
    canvas.paste(camera, (0, 0))
    draw = ImageDraw.Draw(canvas)
    top = frame.size[1]
    draw.text((12, top + 7), f"Frame {scene.frame_id} | predicted scene", fill="#17202a", font=title_font)
    for i, line in enumerate(data["mask_legend"]):
        x = 12 + i * (width // 3)
        draw.text((x, top + 34), line, fill="#17202a", font=font)
    draw.text((12, top + 57), "ID on image = ID below. B/M: box/mask median camera-axis Z (m); ? = unavailable.",
              fill="#17202a", font=font)
    col_width = width // columns
    for index, row in enumerate(data["rows"]):
        col, line = divmod(index, rows_per_column)
        x = 12 + col * col_width
        y = top + header_height + line * row_height
        draw.rectangle((x, y + 3, x + 9, y + 17), fill=row["color"])
        bz = f'{row["box_z_m"]:.1f}' if row["box_z_m"] is not None else "?"
        mz = f'{row["mask_z_m"]:.1f}' if row["mask_z_m"] is not None else "?"
        label = f'#{row["id"]} {row["class"]} {row["confidence"]:.2f}  B:{bz} M:{mz}  {row["adjacency"]}'
        draw.text((x + 14, y), label, fill="#17202a", font=font)
    foot = top + panel_height - footer_height
    draw.line((0, foot, width, foot), fill="#b5bdc5", width=1)
    draw.text((12, foot + 3), data["adjacency_legend"], fill="#17202a", font=font)
    draw.text((12, foot + 26), data["adjacency_method"], fill="#17202a", font=font)
    draw.text((12, foot + 49), data["limitation"], fill="#17202a", font=font)
    return canvas


def metric_xz_data(scene: FrameScene) -> dict:
    """Plot data with camera +X right, +Z up-page, and equal meter scales."""
    if scene.metric_xz_status != "available_approximate_surface_points":
        raise ValueError(f"Metric X-Z unavailable: {scene.metric_xz_status}")
    provenance = scene.fusion.calibration_provenance or ""
    is_camera2 = "camera-2" in provenance.lower() and "image_2" in provenance.lower()
    camera = "Camera-2" if is_camera2 else "Camera"
    coordinate_frame = "Rectified camera-2-centered" if is_camera2 else "Rectified camera-centered"
    points = []
    for obj in scene.objects:
        selected = obj.plotted_surface
        if selected is None:
            continue
        method, point = selected
        if not np.isfinite([point.x_m, point.z_m]).all() or point.z_m <= 0:
            raise ValueError("Plot requires finite positive-Z surface points")
        points.append({"detection_id": obj.fusion.detection_id,
                       "class_name": obj.fusion.class_name,
                       "method": method, "x_m": point.x_m, "z_m": point.z_m,
                       "reference": "predicted_visible_surface"})
    return {"x_axis": f"{camera} X right (m)", "z_axis": f"{camera} Z forward (m)",
            "origin_xz_m": [0.0, 0.0], "aspect": "equal_metric",
            "coordinate_frame": coordinate_frame,
            "title": f"{coordinate_frame} predicted visible-surface points",
            "interpretation": "Selected M4 image rays; not object centers or KITTI box bottom centers. Z is camera-axis depth, not radial range. Not a validated metric reconstruction or collision-avoidance representation.",
            "points": points}


def render_metric_xz(scene: FrameScene) -> Image.Image:
    """Render only M4 calibrated selected-ray surface points; no road plane."""
    data = metric_xz_data(scene)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
    ax.scatter([0], [0], marker="^", c="black", s=110, label="Camera origin")
    for point in data["points"]:
        ax.scatter([point["x_m"]], [point["z_m"]], c="#2366c2", s=60)
        rightmost = point["x_m"] == max(item["x_m"] for item in data["points"])
        ax.annotate(f'#{point["detection_id"]}',
                    (point["x_m"], point["z_m"]), xytext=(-5 if rightmost else 5, 5),
                    ha="right" if rightmost else "left", textcoords="offset points")
    xs = [0] + [point["x_m"] for point in data["points"]]
    zs = [0] + [point["z_m"] for point in data["points"]]
    margin = max(1.0, max(zs) * 0.08)
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(0, max(zs) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(data["x_axis"])
    ax.set_ylabel(data["z_axis"])
    ax.set_title(data["title"])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.text(0.73, 0.80, "Selected M4 rays", fontsize=10, weight="bold")
    for index, point in enumerate(data["points"]):
        fig.text(0.73, 0.77 - index * 0.035,
                 f'#{point["detection_id"]} {point["class_name"]} ({point["method"]})', fontsize=9)
    fig.text(0.05, 0.055, "Selected M4 predicted visible-scene-surface points; not object centers or KITTI box bottom centers.", fontsize=9)
    fig.text(0.05, 0.033, "Camera-axis Z differs from radial range; no validated metric reconstruction or collision-avoidance claim.", fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 0.72, 1))
    stream = BytesIO()
    fig.savefig(stream, format="png")
    plt.close(fig)
    stream.seek(0)
    with Image.open(stream) as image:
        return image.convert("RGB").copy()
