"""Opt-in real YOLO and metric-depth CPU fusion smoke test."""

import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest


@pytest.mark.skipif(os.environ.get("PERCEPTION_RUN_INTEGRATION") != "1", reason="opt-in real model test")
def test_real_cpu_fusion_cli_without_calibration(tmp_path: Path):
    output_json = tmp_path / "fusion.json"
    overlay = tmp_path / "fusion.png"
    subprocess.run([
        sys.executable, "-m", "perception_stack.fusion_cli",
        "--input", "data/examples/bus.jpg", "--json", str(output_json),
        "--output", str(overlay), "--device", "cpu",
    ], check=True, capture_output=True, text=True)
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["image_size"] == [810, 1080]
    assert payload["calibration_status"] == "unavailable_no_intrinsics"
    assert payload["detector_checkpoint_sha256"]
    assert payload["depth_checkpoint_sha256"]
    assert payload["objects"]
    assert any(obj["box"]["statistics"]["status"] == "ok" for obj in payload["objects"])
    assert any(obj["mask"]["statistics"]["status"] == "ok" for obj in payload["objects"])
    assert all(obj["box"]["surface_point"] is None and obj["mask"]["surface_point"] is None
               for obj in payload["objects"])
    with Image.open(overlay) as image:
        assert image.size == (810, 1080)
