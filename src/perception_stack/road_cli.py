"""One-image TwinLiteNetPlus Small road/lane demonstration."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from perception_stack.frame import Frame
from perception_stack.road import TwinLiteRoadSegmenter
from perception_stack.road_assets import CHECKPOINT_PATH, CHECKPOINT_URL, SOURCE_REVISION
from perception_stack.road_visualization import mask_png, render_road_overlay


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-image drivable-area and lane-marking segmentation")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/road_overlay.png"))
    parser.add_argument("--drivable-mask", type=Path, default=Path("outputs/drivable_area.png"))
    parser.add_argument("--lane-mask", type=Path, default=Path("outputs/lane_markings.png"))
    parser.add_argument("--npz", type=Path, default=Path("outputs/road_masks.npz"))
    parser.add_argument("--meta", type=Path, default=Path("outputs/road_metadata.json"))
    parser.add_argument("--weights", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    frame = Frame.from_image(args.input)
    result = TwinLiteRoadSegmenter(args.weights, device=args.device).predict(frame)
    for path in (args.output, args.drivable_mask, args.lane_mask, args.npz, args.meta):
        path.parent.mkdir(parents=True, exist_ok=True)
    render_road_overlay(frame, result).save(args.output)
    mask_png(result.drivable_area_mask).save(args.drivable_mask)
    mask_png(result.lane_marking_mask).save(args.lane_mask)
    np.savez_compressed(args.npz, drivable_area_mask=result.drivable_area_mask,
                        lane_marking_mask=result.lane_marking_mask)
    metadata = result.to_summary()
    metadata.update({
        "source_image": str(args.input),
        "checkpoint_file": str(args.weights),
        "checkpoint_source_url": CHECKPOINT_URL,
        "source_code_revision": SOURCE_REVISION,
        "source_license": "MIT",
        "checkpoint_sha256_provenance": "computed from authors' Google Drive file; no publisher SHA-256 located",
        "artifacts": {
            "overlay_png": str(args.output),
            "drivable_png": str(args.drivable_mask),
            "lane_png": str(args.lane_mask),
            "binary_masks_npz": str(args.npz),
        },
    })
    args.meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Road/lane on %s x %s: drivable %.2f%%; lane %.2f%%; device %s",
                *frame.size, 100 * metadata["drivable_area_fraction"],
                100 * metadata["lane_marking_fraction"], result.device)
    LOGGER.info("Saved %s, %s, %s, %s, %s", args.output, args.drivable_mask,
                args.lane_mask, args.npz, args.meta)


if __name__ == "__main__":
    main()
