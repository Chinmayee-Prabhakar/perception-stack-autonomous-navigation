"""Model-independent frame and prediction result contracts."""

from .frame import CameraIntrinsics, Frame
from .results import BoxXYXY, Detection, FrameDepth, FrameDetections, FrameRoadSegmentation

__all__ = ["BoxXYXY", "CameraIntrinsics", "Detection", "Frame", "FrameDepth", "FrameDetections", "FrameRoadSegmentation"]
