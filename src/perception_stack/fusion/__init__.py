"""Paired object-depth sampling and optional calibrated surface geometry."""

from .core import fuse_frame
from .geometry import backproject_pixel
from .results import (
    DepthSamplingStatistics, FrameFusionResults, FusionConfig, MethodFusion,
    ObjectDepthFusion, SurfacePoint, SurfaceSample,
)

__all__ = [
    "DepthSamplingStatistics", "FrameFusionResults", "FusionConfig", "MethodFusion",
    "ObjectDepthFusion", "SurfacePoint", "SurfaceSample", "backproject_pixel", "fuse_frame",
]
