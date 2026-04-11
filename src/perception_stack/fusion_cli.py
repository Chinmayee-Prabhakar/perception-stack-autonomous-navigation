"""One-image paired YOLO mask/box depth fusion; no default camera calibration."""

import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path

from perception_stack.depth import MetricDepthEstimator
from perception_stack.depth.metric import DEFAULT_INPUT_SIZE
from perception_stack.depth_assets import CHECKPOINT_PATH as DEPTH_CHECKPOINT_PATH
from perception_stack.detectors import YOLOInstanceSegmenter
from perception_stack.frame import CameraIntrinsics, Frame
from perception_stack.fusion import FusionConfig, fuse_frame
from perception_stack.fusion_visualization import render_fusion


LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rectified_intrinsics(path: Path) -> tuple[CameraIntrinsics, str]:
    """User-supplied rectified camera intrinsics; deliberately rejects raw P2."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"width", "height", "fx", "fy", "cx", "cy", "rectified", "provenance"}:
        raise ValueError("Calibration JSON requires width,height,fx,fy,cx,cy,rectified,provenance; raw P2 is unsupported")
    if data["rectified"] is not True or not isinstance(data["provenance"], str) or not data["provenance"].strip():
        raise ValueError("Calibration requires rectified=true and nonempty provenance")
    if not all(type(data[key]) is int for key in ("width", "height")):
        raise ValueError("Calibration width and height must be integers")
    if not all(type(data[key]) in (int, float) for key in ("fx", "fy", "cx", "cy")):
        raise ValueError("Calibration pinhole parameters must be numeric")
    return CameraIntrinsics(*(data[key] for key in ("width", "height", "fx", "fy", "cx", "cy"))), data["provenance"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare box and instance-mask metric-depth sampling")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=Path("outputs/fusion.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/fusion_overlay.png"))
    parser.add_argument("--config", type=Path, default=Path("configs/demo.json"))
    parser.add_argument("--detector-weights", type=Path)
    parser.add_argument("--depth-weights", type=Path, default=DEPTH_CHECKPOINT_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--depth-input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--min-valid-pixels", type=int, default=3)
    parser.add_argument("--min-valid-fraction", type=float, default=0.10)
    parser.add_argument("--intrinsics-json", type=Path, help="verified rectified pinhole calibration; omit for bus demo")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = json.loads(args.config.read_text(encoding="utf-8"))
    detector_weights = args.detector_weights or Path(settings["weights"])
    confidence = args.conf if args.conf is not None else settings["confidence"]
    image_size = args.imgsz if args.imgsz is not None else settings["image_size"]
    runtime_dir = Path("work/runtime").resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(runtime_dir / "matplotlib"))

    intrinsics, provenance = load_rectified_intrinsics(args.intrinsics_json) if args.intrinsics_json else (None, None)
    frame = Frame.from_image(args.input, intrinsics=intrinsics)
    config = FusionConfig(args.min_valid_pixels, args.min_valid_fraction)

    # Keep only one pretrained model resident at a time on the CPU memory path.
    import torch
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch installation; use --device cpu")
    detector = YOLOInstanceSegmenter(detector_weights, confidence=confidence,
                                     image_size=image_size, device="0" if device == "cuda" else "cpu")
    detections = detector.predict(frame)
    del detector
    gc.collect()
    estimator = MetricDepthEstimator(args.depth_weights, device=device, input_size=args.depth_input_size)
    depth = estimator.predict(frame)
    del estimator
    gc.collect()

    results = fuse_frame(frame, detections, depth, config=config,
                         calibration_provenance=provenance, rectified=intrinsics is not None,
                         detector_checkpoint_sha256=_sha256(detector_weights))
    summary = results.to_dict()
    summary["source_image"] = str(args.input)
    summary["detector_checkpoint_file"] = str(detector_weights)
    summary["depth_checkpoint_file"] = str(args.depth_weights)
    summary["detector_confidence_setting"] = confidence
    summary["detector_image_size_setting"] = image_size
    summary["geometry_note"] = "Only selected valid image-ray depth can define a visible surface point; bus demo has no camera calibration"
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    render_fusion(frame, results).save(args.output)
    LOGGER.info("%d detections, %d valid box depths, %d valid mask depths; calibration %s; device %s",
                len(results.objects),
                sum(obj.box_method.statistics.status == "ok" for obj in results.objects),
                sum(obj.mask_method.statistics.status == "ok" for obj in results.objects),
                results.calibration_status, depth.device)
    LOGGER.info("Saved JSON %s and overlay %s", args.json, args.output)


if __name__ == "__main__":
    main()
