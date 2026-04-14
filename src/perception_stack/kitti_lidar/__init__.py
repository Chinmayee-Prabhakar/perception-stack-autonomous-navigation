"""Synthetic-first KITTI Object sparse LiDAR reference foundation."""

from .scan import VelodyneScan, read_object_scan
from .geometry import CalibratedProjector, ProjectedReturn, ProjectionResult
from .association import (FrozenObjectMatch, SourceDigests, SparseReference,
                          PairedSurfaceError, associate_object, paired_surface_error)
from .protocol import LidarProtocol

__all__ = ["VelodyneScan", "read_object_scan", "CalibratedProjector",
           "ProjectedReturn", "ProjectionResult", "FrozenObjectMatch",
           "SourceDigests", "SparseReference", "PairedSurfaceError", "associate_object",
           "paired_surface_error", "LidarProtocol"]
