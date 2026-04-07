"""One-image metric-depth demonstration, independent of the segmentation CLI."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from perception_stack.depth import MetricDepthEstimator
from perception_stack.depth_assets import CHECKPOINT_PATH, CHECKPOINT_URL, SOURCE_REVISION
from perception_stack.depth.metric import DEFAULT_INPUT_SIZE
from perception_stack.depth_visualization import render_depth
from perception_stack.frame import Frame


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-image metric depth estimation")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/depth.png"))
    parser.add_argument("--npy", type=Path, default=Path("outputs/depth.npy"))
    parser.add_argument("--mask", type=Path, default=Path("outputs/depth_valid.npy"))
    parser.add_argument("--meta", type=Path, default=Path("outputs/depth_metadata.json"))
    parser.add_argument("--weights", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    frame = Frame.from_image(args.input)
    estimator = MetricDepthEstimator(args.weights, device=args.device, input_size=args.input_size)
    result = estimator.predict(frame)
    for path in (args.output, args.npy, args.mask, args.meta):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.npy, result.depth_m)
    np.save(args.mask, result.valid_mask)
    render_depth(result).save(args.output)
    metadata = result.to_summary()
    metadata.update({
        "source_image": str(args.input),
        "checkpoint_file": str(args.weights),
        "checkpoint_source_url": CHECKPOINT_URL,
        "source_code_revision": SOURCE_REVISION,
        "model_card_url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small",
        "model_license": "Apache-2.0",
        "encoder": "vits",
        "architecture": "DepthAnythingV2 metric DPT, features=64, out_channels=[48,96,192,384], max_depth=80",
        "training_domain": "Virtual KITTI 2 synthetic outdoor scenes",
        "visualization": "OpenCV Turbo, fixed 0-80 m camera-axis Z scale; black=invalid; values clipped only for display",
        "artifacts": {"png": str(args.output), "depth_npy": str(args.npy), "valid_mask_npy": str(args.mask)},
    })
    args.meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Depth %s x %s; valid %.2f%%; near cap %.2f%% of valid; device %s",
                *frame.size, 100 * metadata["valid_fraction"],
                metadata["near_cap_percent_of_valid"], result.device)
    LOGGER.info("Saved %s, %s, %s, %s", args.output, args.npy, args.mask, args.meta)


if __name__ == "__main__":
    main()
