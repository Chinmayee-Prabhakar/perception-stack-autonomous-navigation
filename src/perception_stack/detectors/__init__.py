"""Model adapters. Only the YOLO instance adapter is implemented in milestone 1."""

from .base import InstanceSegmenter
from .ultralytics_yolo import YOLOInstanceSegmenter

__all__ = ["InstanceSegmenter", "YOLOInstanceSegmenter"]
