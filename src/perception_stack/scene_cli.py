"""One-image CPU-friendly integrated scene inference; calibration is opt-in."""

import argparse
import gc
import json
import logging
import os
from pathlib import Path

from perception_stack.depth import MetricDepthEstimator
from perception_stack.depth.metric import DEFAULT_INPUT_SIZE
from perception_stack.depth_assets import CHECKPOINT_PATH as DEPTH_CHECKPOINT_PATH
from perception_stack.detectors import YOLOInstanceSegmenter
from perception_stack.frame import Frame
from perception_stack.fusion import FusionConfig, fuse_frame
from perception_stack.fusion_cli import _sha256, load_rectified_intrinsics
from perception_stack.road import TwinLiteRoadSegmenter
from perception_stack.road_assets import CHECKPOINT_PATH as ROAD_CHECKPOINT_PATH
from perception_stack.scene import ContactPolicy, associate_road, build_scene
from perception_stack.scene.visualization import render_metric_xz, render_scene_presentation


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated predicted perception scene on one RGB image")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=Path("outputs/scene.json"))
    parser.add_argument("--overlay", type=Path, default=Path("outputs/scene_overlay.png"))
    parser.add_argument("--drivable-mask", type=Path, help="optional exact predicted drivable mask PNG")
    parser.add_argument("--lane-mask", type=Path, help="optional exact predicted lane-marking mask PNG")
    parser.add_argument("--association-json", type=Path, default=Path("outputs/scene_road_association.json"))
    parser.add_argument("--xz-plot", type=Path, default=Path("outputs/scene_xz.png"))
    parser.add_argument("--config", type=Path, default=Path("configs/demo.json"))
    parser.add_argument("--detector-weights", type=Path)
    parser.add_argument("--depth-weights", type=Path, default=DEPTH_CHECKPOINT_PATH)
    parser.add_argument("--road-weights", type=Path, default=ROAD_CHECKPOINT_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--depth-input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--intrinsics-json", type=Path,
                        help="verified rectified pinhole intrinsics; never pass raw KITTI P2")
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
    fusion = fuse_frame(frame, detections, depth, config=FusionConfig(),
                        calibration_provenance=provenance, rectified=intrinsics is not None,
                        detector_checkpoint_sha256=_sha256(detector_weights))
    del depth
    gc.collect()
    segmenter = TwinLiteRoadSegmenter(args.road_weights, device=device)
    road = segmenter.predict(frame)
    del segmenter
    gc.collect()
    associations = associate_road(frame, detections, road, fusion, policy=ContactPolicy())
    scene = build_scene(frame, detections, road, fusion, associations)
    for path in (args.json, args.overlay, args.association_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    render_scene_presentation(frame, detections, road, scene).save(args.overlay)
    if args.drivable_mask:
        from PIL import Image
        import numpy as np
        args.drivable_mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.uint8(road.drivable_area_mask) * 255, "L").save(args.drivable_mask)
    if args.lane_mask:
        from PIL import Image
        import numpy as np
        args.lane_mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.uint8(road.lane_marking_mask) * 255, "L").save(args.lane_mask)
    args.association_json.write_text(json.dumps(associations.to_dict(), indent=2,
                                                sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    xz_path = None
    if scene.metric_xz_status == "available_approximate_surface_points":
        args.xz_plot.parent.mkdir(parents=True, exist_ok=True)
        render_metric_xz(scene).save(args.xz_plot)
        xz_path = str(args.xz_plot)
    summary = scene.to_dict()
    summary["source_image"] = str(args.input)
    summary["artifacts"] = {"overlay_png": str(args.overlay),
                            "road_association_json": str(args.association_json),
                            "metric_xz_png": xz_path}
    args.json.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    LOGGER.info("Frame %s %sx%s: %d detections; box/mask valid %d/%d; drivable/lane pixels %d/%d",
                frame.frame_id, *frame.size, len(scene.objects),
                sum(obj.fusion.box_method.statistics.status == "ok" for obj in scene.objects),
                sum(obj.fusion.mask_method.statistics.status == "ok" for obj in scene.objects),
                int(road.drivable_area_mask.sum()), int(road.lane_marking_mask.sum()))
    LOGGER.info("Road evidence statuses: %s", [obj.road.evidence_status for obj in scene.objects])
    LOGGER.info("Calibration %s; metric X-Z %s; device %s", fusion.calibration_status,
                scene.metric_xz_status, device)
    LOGGER.info("Saved %s, %s, %s%s", args.json, args.overlay, args.association_json,
                f", {xz_path}" if xz_path else "")


if __name__ == "__main__":
    main()
