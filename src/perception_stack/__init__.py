"""Model-independent frame, detection, and depth result contracts."""

from .frame import CameraIntrinsics, Frame
from .results import BoxXYXY, Detection, FrameDepth, FrameDetections

__all__ = ["BoxXYXY", "CameraIntrinsics", "Detection", "Frame", "FrameDepth", "FrameDetections"]
