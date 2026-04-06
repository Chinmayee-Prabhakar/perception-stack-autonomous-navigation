"""Run the first milestone on one local image."""

import argparse
import json
import logging
import os
from pathlib import Path

from perception_stack.detectors import YOLOInstanceSegmenter
from perception_stack.frame import Frame
from perception_stack.visualization import draw_detections


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-image instance segmentation")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.json"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/overlay.png"))
    parser.add_argument("--json", type=Path, default=Path("outputs/detections.json"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", type=str)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--imgsz", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = json.loads(args.config.read_text(encoding="utf-8"))
    weights = args.weights or Path(settings["weights"])
    device = args.device or settings["device"]
    confidence = args.conf if args.conf is not None else settings["confidence"]
    image_size = args.imgsz if args.imgsz is not None else settings["image_size"]

    runtime_dir = Path("work/runtime").resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(runtime_dir / "matplotlib"))

    frame = Frame.from_image(args.input)
    LOGGER.info("Loaded %s (%s x %s)", args.input, *frame.size)
    segmenter = YOLOInstanceSegmenter(weights, confidence=confidence,
                                      image_size=image_size, device=device)
    results = segmenter.predict(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    draw_detections(frame, results).save(args.output)
    summary = results.to_summary()
    summary["source"] = str(args.input)
    summary["device"] = device
    summary["image_size_setting"] = image_size
    summary["confidence_setting"] = confidence
    args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("%d detections; overlay: %s; JSON: %s",
                len(results.detections), args.output, args.json)


if __name__ == "__main__":
    main()
