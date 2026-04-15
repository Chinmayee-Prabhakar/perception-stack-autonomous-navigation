"""Synthetic-first KITTI Object sparse LiDAR reference foundation."""

from .scan import VelodyneScan, read_object_scan
from .geometry import CalibratedProjector, ProjectedReturn, ProjectionResult
from .association import (FrozenObjectMatch, SourceDigests, SparseReference,
                          PairedSurfaceError, associate_object, paired_surface_error)
from .protocol import LidarProtocol
from .real_data import load_calibrated_projector, load_verified_object_scan

__all__ = ["VelodyneScan", "read_object_scan", "CalibratedProjector",
           "ProjectedReturn", "ProjectionResult", "FrozenObjectMatch",
           "SourceDigests", "SparseReference", "PairedSurfaceError", "associate_object",
           "paired_surface_error", "LidarProtocol", "load_calibrated_projector",
           "load_verified_object_scan"]
