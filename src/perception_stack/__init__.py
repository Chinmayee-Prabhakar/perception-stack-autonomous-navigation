"""Model-independent perception data contracts and first-milestone adapter."""

from .frame import CameraIntrinsics, Frame
from .results import BoxXYXY, Detection, FrameDetections

__all__ = ["BoxXYXY", "CameraIntrinsics", "Detection", "Frame", "FrameDetections"]
