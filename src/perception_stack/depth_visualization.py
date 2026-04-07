"""Fixed-range display of metric depth; numerical arrays are never normalized."""

import cv2
import numpy as np
from PIL import Image, ImageDraw

from perception_stack.results import FrameDepth


def render_depth(result: FrameDepth) -> Image.Image:
    """Turbo color map over 0-80 m, with an explicit meter scale and black invalid pixels."""
    upper = result.nominal_max_depth_m
    display = np.zeros(result.depth_m.shape, dtype=np.uint8)
    display[result.valid_mask] = np.rint(
        np.clip(result.depth_m[result.valid_mask] / upper, 0, 1) * 255
    ).astype(np.uint8)
    colored = cv2.cvtColor(cv2.applyColorMap(display, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    colored[~result.valid_mask] = 0
    height, width = result.depth_m.shape
    canvas = Image.new("RGB", (width + 90, height), "white")
    canvas.paste(Image.fromarray(colored), (0, 0))
    draw = ImageDraw.Draw(canvas)
    bar_top, bar_bottom = 26, max(26, height - 26)
    if bar_bottom > bar_top:
        shades = np.rint(np.linspace(255, 0, bar_bottom - bar_top + 1)).astype(np.uint8)
        bar = cv2.cvtColor(
            cv2.applyColorMap(np.repeat(shades[:, None], 16, axis=1), cv2.COLORMAP_TURBO),
            cv2.COLOR_BGR2RGB,
        )
        canvas.paste(Image.fromarray(bar), (width + 6, bar_top))
        for value in (0, 20, 40, 60, 80):
            y = int(round(bar_bottom - value / 80 * (bar_bottom - bar_top)))
            draw.text((width + 26, max(0, min(height - 12, y - 6))), f"{value} m", fill="black")
    draw.text((width + 5, 4), "Z depth", fill="black")
    return canvas
