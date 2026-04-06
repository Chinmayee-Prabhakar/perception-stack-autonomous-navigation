"""Contract for future interchangeable object instance adapters."""

from typing import Protocol

from perception_stack.frame import Frame
from perception_stack.results import FrameDetections


class InstanceSegmenter(Protocol):
    def predict(self, frame: Frame) -> FrameDetections: ...
