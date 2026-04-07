"""Contract for future frame-aligned depth estimators."""

from typing import Protocol

from perception_stack.frame import Frame
from perception_stack.results import FrameDepth


class DepthEstimator(Protocol):
    def predict(self, frame: Frame) -> FrameDepth: ...
