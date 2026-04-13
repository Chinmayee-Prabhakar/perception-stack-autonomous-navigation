"""Write a deterministic synthetic calibrated M6 plot and geometry fixture."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("work/runtime/matplotlib").resolve()))

from perception_stack.scene.synthetic import synthetic_scene
from perception_stack.scene.visualization import render_metric_xz, render_scene_overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m6_synthetic"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, detections, road, scene, geometry = synthetic_scene()
    render_metric_xz(scene).save(args.output_dir / "synthetic_camera_xz.png")
    render_scene_overlay(frame, detections, road, scene).save(args.output_dir / "synthetic_overlay.png")
    (args.output_dir / "synthetic_geometry.json").write_text(
        json.dumps(geometry, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "synthetic_scene.json").write_text(scene.to_json(), encoding="utf-8")
    print(f"Synthetic calibrated X-Z points: {geometry['expected_camera_xz_m']}")
    print(f"Saved synthetic artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
