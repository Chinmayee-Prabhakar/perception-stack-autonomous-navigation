"""Compact original-frame box/mask depth comparison overlay."""

from PIL import Image, ImageDraw

from perception_stack.frame import Frame
from perception_stack.fusion.results import FrameFusionResults


def render_fusion(frame: Frame, result: FrameFusionResults) -> Image.Image:
    if frame.frame_id != result.frame_id or frame.size != result.image_size:
        raise ValueError("Fusion overlay requires the matching original frame")
    image = Image.fromarray(frame.image_rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    palette = ((0, 215, 255), (255, 170, 35), (115, 235, 85),
               (255, 90, 150), (200, 145, 255), (255, 230, 60))
    for obj in result.objects:
        color = palette[obj.detection_id % len(palette)]
        draw.rectangle(obj.box.xyxy, outline=color, width=3)
        box_stat = obj.box_method.statistics
        mask_stat = obj.mask_method.statistics
        box_text = f"{box_stat.median_z_m:.1f}m" if box_stat.status == "ok" else f"n/a({box_stat.status})"
        mask_text = f"{mask_stat.median_z_m:.1f}m" if mask_stat.status == "ok" else f"n/a({mask_stat.status})"
        label = f"{obj.class_name} {obj.confidence:.2f}  B:{box_text} M:{mask_text}"
        bounds = draw.textbbox((0, 0), label)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        x = max(0, min(int(obj.box.x1), frame.size[0] - text_width - 6))
        y = max(0, int(obj.box.y1) - text_height - 5)
        draw.rectangle((x, y, x + text_width + 6, y + text_height + 4), fill=(0, 0, 0))
        draw.text((x + 3, y + 2), label, fill=color)
        for method, marker_color in ((obj.box_method, (255, 255, 255)),
                                     (obj.mask_method, (255, 230, 30))):
            sample = method.statistics.selected_sample
            if sample is not None and method.statistics.status == "ok":
                u, v = sample.u, sample.v
                draw.ellipse((u - 3, v - 3, u + 3, v + 3), outline=marker_color, width=2)
    return image
