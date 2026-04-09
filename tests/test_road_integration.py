"""Opt-in test of the authors' real checkpoint on the existing bus image."""

import os

import numpy as np
import pytest

from perception_stack.frame import Frame
from perception_stack.road import TwinLiteRoadSegmenter


@pytest.mark.skipif(os.environ.get("PERCEPTION_RUN_INTEGRATION") != "1", reason="opt-in real model test")
def test_official_small_checkpoint_on_bus() -> None:
    frame = Frame.from_image("data/examples/bus.jpg")
    result = TwinLiteRoadSegmenter(device="cpu").predict(frame)
    assert result.drivable_area_mask.shape == frame.image_rgb.shape[:2]
    assert result.lane_marking_mask.shape == frame.image_rgb.shape[:2]
    assert result.drivable_area_mask.dtype == np.bool_
    assert result.lane_marking_mask.dtype == np.bool_
    assert result.preprocessing["model_input_hw"] == [384, 640]
