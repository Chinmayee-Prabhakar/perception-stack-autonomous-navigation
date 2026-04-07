"""Opt-in checkpoint inference; ordinary pytest remains offline and lightweight."""

import os
from pathlib import Path

import numpy as np
import pytest

from perception_stack.depth import MetricDepthEstimator
from perception_stack.depth_visualization import render_depth
from perception_stack.frame import Frame


@pytest.mark.skipif(os.environ.get("PERCEPTION_RUN_INTEGRATION") != "1", reason="opt-in real model test")
def test_official_metric_checkpoint_on_bus_image(tmp_path: Path) -> None:
    frame = Frame.from_image("data/examples/bus.jpg")
    result = MetricDepthEstimator(device="cpu", input_size=392).predict(frame)
    assert result.depth_m.shape == frame.image_rgb.shape[:2]
    assert result.depth_m.dtype == np.float32
    assert result.valid_mask.any()
    assert np.isfinite(result.depth_m[result.valid_mask]).all()
    depth_file = tmp_path / "depth.npy"
    mask_file = tmp_path / "valid.npy"
    image_file = tmp_path / "depth.png"
    np.save(depth_file, result.depth_m)
    np.save(mask_file, result.valid_mask)
    render_depth(result).save(image_file)
    assert depth_file.is_file() and mask_file.is_file() and image_file.is_file()
    assert np.load(depth_file).shape == result.depth_m.shape
