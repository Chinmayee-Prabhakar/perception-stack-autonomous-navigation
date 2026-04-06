"""Simple instance mask and box overlay in original RGB frame coordinates."""

import numpy as np
from PIL import Image, ImageDraw

from perception_stack.frame import Frame
from perception_stack.results import FrameDetections


_PALETTE = ((0, 200, 255), (255, 160, 20), (130, 235, 60), (245, 80, 145),
            (180, 120, 255), (255, 220, 60))


def draw_detections(frame: Frame, results: FrameDetections) -> Image.Image:
    if (frame.frame_id, frame.size) != (results.frame_id, results.image_size):
        raise ValueError("Results do not belong to this frame")
    image = Image.fromarray(frame.image_rgb).convert("RGBA")
    for detection in results.detections:
        color = _PALETTE[detection.class_id % len(_PALETTE)]
        if detection.mask is not None:
            alpha = Image.fromarray(np.uint8(detection.mask) * 92, mode="L")
            tint = Image.new("RGBA", frame.size, color + (255,))
            image.paste(tint, (0, 0), alpha)
    draw = ImageDraw.Draw(image)
    for detection in results.detections:
        color = _PALETTE[detection.class_id % len(_PALETTE)]
        x1, y1, x2, y2 = detection.box.xyxy
        draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=3)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        bounds = draw.textbbox((0, 0), label)
        tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
        label_y = max(0, int(y1) - th - 5)
        draw.rectangle((int(x1), label_y, int(x1) + tw + 6, label_y + th + 4), fill=color + (255,))
        draw.text((int(x1) + 3, label_y + 2), label, fill=(0, 0, 0, 255))
    return image.convert("RGB")
