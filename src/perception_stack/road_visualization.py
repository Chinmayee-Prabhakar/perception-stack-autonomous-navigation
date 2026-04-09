"""Original-resolution semantic road/lane overlays and binary mask PNGs."""

import numpy as np
from PIL import Image, ImageDraw

from perception_stack.frame import Frame
from perception_stack.results import FrameRoadSegmentation


DRIVABLE_RGB = (0, 210, 95)
LANE_RGB = (255, 215, 0)


def mask_png(mask: np.ndarray) -> Image.Image:
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Mask PNG requires a HxW boolean array")
    return Image.fromarray(np.uint8(mask) * 255, mode="L")


def render_road_overlay(frame: Frame, result: FrameRoadSegmentation) -> Image.Image:
    if (frame.frame_id, frame.size) != (result.frame_id, result.image_size):
        raise ValueError("Road/lane result does not belong to this frame")
    base = Image.fromarray(frame.image_rgb).convert("RGBA")
    for mask, color, opacity in (
        (result.drivable_area_mask, DRIVABLE_RGB, 80),
        (result.lane_marking_mask, LANE_RGB, 170),
    ):
        tint = Image.new("RGBA", frame.size, color + (255,))
        alpha = Image.fromarray(np.uint8(mask) * opacity, mode="L")
        base.paste(tint, (0, 0), alpha)
    image = base.convert("RGB")
    draw = ImageDraw.Draw(image)
    legend_width = min(frame.size[0], 196)
    legend_height = min(frame.size[1], 48)
    draw.rectangle((0, 0, legend_width, legend_height), fill=(24, 24, 24))
    if legend_width >= 150 and legend_height >= 42:
        draw.rectangle((6, 7, 18, 18), fill=DRIVABLE_RGB)
        draw.text((24, 6), "Drivable area", fill="white")
        draw.rectangle((6, 27, 18, 38), fill=LANE_RGB)
        draw.text((24, 26), "Lane markings", fill="white")
    return image
