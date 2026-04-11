"""Frozen CPU KITTI object-reference Z proxy evaluation; no dataset download."""

import argparse
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import time

from perception_stack.depth import MetricDepthEstimator
from perception_stack.detectors import YOLOInstanceSegmenter
from perception_stack.fusion import fuse_frame
from perception_stack.kitti_eval.dataset import KittiObjectDataset
from perception_stack.kitti_eval.metrics import aggregate_frames, evaluate_frame
from perception_stack.kitti_eval.protocol import EvaluationProtocol
from perception_stack.kitti_eval.report import write_json, write_markdown_report, write_plots, write_rows_csv
from perception_stack.kitti_eval.splits import freeze_or_verify_manifest


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path_string: str, repository: Path) -> Path:
    path = Path(path_string)
    return path if path.is_absolute() else repository / path


def main() -> None:
    parser = argparse.ArgumentParser(description="KITTI training-only box/mask object-reference Z proxy evaluation")
    parser.add_argument("--config", type=Path, default=Path("configs/kitti_eval_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kitti_m5"))
    parser.add_argument("--dataset-root", type=Path, help="approved official KITTI object training extraction")
    parser.add_argument("--mode", choices=("plan", "freeze", "run"), default="plan")
    parser.add_argument("--subset", choices=("pilot", "heldout"), default="pilot")
    parser.add_argument("--pilot-reviewed", action="store_true", help="required for heldout execution after pilot diagnostics")
    args = parser.parse_args()

    protocol = EvaluationProtocol.load(args.config)
    repository = args.config.resolve().parents[1]
    root = args.dataset_root or _resolve(protocol.data["dataset_root"], repository)
    output_root = args.output_dir
    output = output_root / ("plan" if args.mode == "plan" else args.subset)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "frozen_protocol.json", protocol.data)
    write_json(output / "class_mapping.json", {
        "version": protocol.mapping.version,
        "coco_to_kitti": protocol.mapping.coco_to_kitti,
        "ignored_kitti_categories": sorted(protocol.mapping.ignored_kitti_categories),
        "sha256": protocol.mapping.digest,
    })
    if args.mode == "plan":
        pending = aggregate_frames((), seed=protocol.data["bootstrap_seed"],
                                   replicates=protocol.data["bootstrap_replicates"],
                                   min_bootstrap_frames=protocol.data["bootstrap_min_distinct_frames"])
        pending.update({"dataset_status": "official_KITTI_training_data_not_yet_approved_or_available",
                        "protocol_sha256": protocol.digest, "selected_frame_ids": None,
                        "pilot_evaluation": "pending", "heldout_evaluation": "pending",
                        "actual_images_tested": 0, "actual_objects_tested": 0})
        write_json(output / "infrastructure_summary.json", pending)
        write_json(output / "split_manifest_status.json", {"status": "pending_approved_dataset", "frame_ids": None})
        write_markdown_report(output / "results_report.md", pending, subset="none",
                              selected_count=0, processed_count=0)
        print("Offline infrastructure prepared; 0 real KITTI frames evaluated; no accuracy metrics computed")
        return

    dataset = KittiObjectDataset(root)
    ids = dataset.frame_ids()
    source_validation = dataset.validate_source_records(ids)
    source_digest = dataset.source_label_calibration_sha256(ids)
    manifest = freeze_or_verify_manifest(output_root / "frozen_split_manifest.json", ids, protocol,
                                         source_label_calibration_sha256=source_digest)
    write_json(output_root / "source_validation.json", source_validation)
    if args.mode == "freeze":
        print(f"Frozen {len(manifest['pilot_frame_ids'])} pilot and {len(manifest['heldout_frame_ids'])} held-out frame IDs before inference")
        return
    if args.subset == "heldout" and not args.pilot_reviewed:
        raise ValueError("Held-out execution requires --pilot-reviewed after pilot diagnostic review")
    subset_key = "pilot_frame_ids" if args.subset == "pilot" else "heldout_frame_ids"
    selected = manifest[subset_key]

    runtime_dir = repository / "work/runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(runtime_dir / "matplotlib"))
    detector_weights = _resolve(protocol.data["detector_weights"], repository)
    depth_weights = _resolve(protocol.data["depth_weights"], repository)
    for path, expected in ((detector_weights, protocol.data["detector_checkpoint_sha256"]),
                           (depth_weights, protocol.data["depth_checkpoint_sha256"])):
        if not path.is_file() or _hash_file(path) != expected:
            raise ValueError(f"Missing or changed pinned checkpoint: {path}")
    detector = YOLOInstanceSegmenter(detector_weights,
                                     confidence=protocol.data["detector_confidence"],
                                     image_size=protocol.data["detector_image_size"], device="cpu")
    estimator = MetricDepthEstimator(depth_weights, device="cpu",
                                     input_size=protocol.data["depth_input_size"])
    evaluated = []
    timings = []
    evaluation_failure = None
    for frame_id in selected:
        try:
            sample = dataset.load(frame_id)
            start = time.perf_counter()
            detections = detector.predict(sample.frame)
            detection_s = time.perf_counter() - start
            start = time.perf_counter()
            depth = estimator.predict(sample.frame)
            depth_s = time.perf_counter() - start
            fused = fuse_frame(sample.frame, detections, depth,
                               config=protocol.fusion_config,
                               detector_checkpoint_sha256=protocol.data["detector_checkpoint_sha256"])
            result = evaluate_frame(
                sample, detections, fused, protocol.mapping,
                iou_threshold=protocol.data["matching_iou_threshold"],
                ignored_ioa_threshold=protocol.data["ignored_reference_ioa_threshold"],
                min_reference_height_px=protocol.data["minimum_reference_box_height_px"],
                max_truncation=protocol.data["maximum_truncation"],
                allowed_occlusion=tuple(protocol.data["allowed_occlusion"]),
            )
            evaluated.append(result)
            timings.append({"frame_id": frame_id, "detector_seconds": detection_s,
                            "depth_seconds": depth_s, "device": "cpu"})
            write_json(output / "frames" / f"{frame_id}.json", result.to_dict())
            print(f"{frame_id}: {len(result.matches.pairs)} matched; detector {detection_s:.2f}s; depth {depth_s:.2f}s")
        except Exception as error:
            evaluation_failure = {"frame_id": frame_id, "error_type": type(error).__name__,
                                  "message": str(error)}
            break
    del detector, estimator
    gc.collect()
    summary = aggregate_frames(tuple(evaluated), seed=protocol.data["bootstrap_seed"],
                               replicates=protocol.data["bootstrap_replicates"],
                               min_bootstrap_frames=protocol.data["bootstrap_min_distinct_frames"])
    rows = [row for frame in evaluated for row in frame.rows]
    summary.update({
        "subset": args.subset, "selected_frame_ids": selected,
        "selected_frame_count": len(selected), "actual_images_tested": len(evaluated),
        "actual_objects_tested": len(rows), "protocol_sha256": protocol.digest,
        "class_mapping_sha256": protocol.mapping.digest,
        "detector_model": "YOLO11n-seg", "detector_checkpoint_sha256": protocol.data["detector_checkpoint_sha256"],
        "depth_model": "Depth Anything V2 Metric VKITTI Small",
        "depth_checkpoint_sha256": protocol.data["depth_checkpoint_sha256"],
        "calibration_policy": "P2 parsed and retained; primary comparison uses annotated camera-axis Z only; no P2 inverse or object-center localization",
        "inference_timings": timings,
        "split_limitation": manifest["drive_grouping"],
        "no_test_label_scale_fit": True,
        "evaluation_failure": evaluation_failure,
    })
    if evaluation_failure is not None:
        summary["status"] = "incomplete_selected_subset"
    write_rows_csv(output / "paired_object_results.csv", rows)
    write_json(output / "matching_diagnostics.json", [frame.matches.to_dict() for frame in evaluated])
    summary["plots"] = write_plots(output, rows, summary)
    write_json(output / "summary_metrics.json", summary)
    write_markdown_report(output / "results_report.md", summary, subset=args.subset,
                          selected_count=len(selected), processed_count=len(evaluated))
    print(f"Completed {len(evaluated)}/{len(selected)} {args.subset} frames; matched={summary['counts']['matched_pairs']}; paired={summary['primary_paired']['paired_objects']}")
    if evaluation_failure is not None:
        raise SystemExit(f"Incomplete evaluation; see summary_metrics.json: {evaluation_failure}")


if __name__ == "__main__":
    main()
