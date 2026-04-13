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
    panel_rows = []
    for index, (det, obj) in enumerate(zip(detections.detections, scene.objects)):
        if obj.fusion.detection_id != index or obj.fusion.box != det.box:
            raise ValueError("Overlay detections must match the scene")
        assoc = obj.road
        color = ((50, 230, 130) if assoc.evidence_status == "drivable_adjacency_detected"
                 else (245, 175, 50) if assoc.evidence_status == "drivable_adjacency_not_detected"
                 else (245, 95, 120))
        draw.rectangle(det.box.xyxy, outline=color, width=2)
        l, t, r, b = assoc.contact_region_xyxy
        if r > l and b > t:
            draw.rectangle((l, t, r - 1, b - 1), outline=(250, 250, 250), width=1)
        box = obj.fusion.box_method.statistics
        mask = obj.fusion.mask_method.statistics
        bz = f"{box.median_z_m:.1f}" if box.status == "ok" else "?"
        mz = f"{mask.median_z_m:.1f}" if mask.status == "ok" else "?"
        road_label = {"drivable_adjacency_detected": "adj+",
                      "drivable_adjacency_not_detected": "adj-"}.get(assoc.evidence_status, "adj?")
        label = f"#{index} {det.class_name} {det.confidence:.2f} B:{bz}m M:{mz}m {road_label}"
        panel_rows.append((label, color))
        id_label = f"#{index}"
        ib = draw.textbbox((0, 0), id_label, font=font)
        x = min(max(0, int(det.box.x1)), max(0, frame.size[0] - (ib[2] - ib[0]) - 4))
        y = min(max(0, int(det.box.y1)), max(0, frame.size[1] - (ib[3] - ib[1]) - 4))
        draw.rectangle((x, y, x + ib[2] - ib[0] + 4, y + ib[3] - ib[1] + 4), fill=(18, 18, 18))
        draw.text((x + 2, y + 1), id_label, fill=color, font=font)
        for method, marker_color in ((obj.fusion.box_method, (255, 255, 255)),
                                     (obj.fusion.mask_method, (255, 225, 30))):
            sample = method.statistics.selected_sample
            if method.statistics.status == "ok" and sample is not None:
                draw.ellipse((sample.u - 2, sample.v - 2, sample.u + 2, sample.v + 2),
                             outline=marker_color, width=2)
    # Compact legend; detailed reasons and counts live in scene JSON.
    legend_lines = (
        "PREDICTED | green: drivable area | yellow: lane | blue: instance",
        "adj+ / adj- / adj? = predicted drivable adjacency detected / not detected / unknown",
    )
    legend_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in legend_lines) + 8
    legend_height = 36
    if frame.size[0] >= legend_width and frame.size[1] >= legend_height:
        draw.rectangle((0, 0, legend_width, legend_height), fill=(18, 18, 18))
        for line_index, line in enumerate(legend_lines):
            draw.text((4, 2 + line_index * 17), line, fill="white", font=font)
    if panel_rows:
        widest = max(draw.textbbox((0, 0), label, font=font)[2] for label, _ in panel_rows)
        panel_width = widest + 12
        panel_height = len(panel_rows) * 20 + 8
        if panel_width <= frame.size[0] and panel_height + 44 <= frame.size[1]:
            px, py = frame.size[0] - panel_width, 42
            draw.rectangle((px, py, frame.size[0] - 1, py + panel_height), fill=(18, 18, 18))
            for row, (label, color) in enumerate(panel_rows):
                draw.text((px + 5, py + 4 + row * 20), label, fill=color, font=font)
    return image


def metric_xz_data(scene: FrameScene) -> dict:
    """Plot data with camera +X right, +Z up-page, and equal meter scales."""
    if scene.metric_xz_status != "available_approximate_surface_points":
        raise ValueError(f"Metric X-Z unavailable: {scene.metric_xz_status}")
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
    return {"x_axis": "Camera X right (m)", "z_axis": "Camera Z forward (m)",
            "origin_xz_m": [0.0, 0.0], "aspect": "equal_metric",
            "title": "Approximate camera-coordinate visible-surface geometry (predicted)",
            "points": points}


def render_metric_xz(scene: FrameScene) -> Image.Image:
    """Render only M4 calibrated selected-ray surface points; no road plane."""
    data = metric_xz_data(scene)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    ax.scatter([0], [0], marker="^", c="black", s=110, label="Camera origin")
    for point in data["points"]:
        ax.scatter([point["x_m"]], [point["z_m"]], c="#2366c2", s=60)
        rightmost = point["x_m"] == max(item["x_m"] for item in data["points"])
        ax.annotate(f'#{point["detection_id"]} {point["class_name"]} ({point["method"]})',
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
    fig.tight_layout()
    stream = BytesIO()
    fig.savefig(stream, format="png")
    plt.close(fig)
    stream.seek(0)
    with Image.open(stream) as image:
        return image.convert("RGB").copy()
