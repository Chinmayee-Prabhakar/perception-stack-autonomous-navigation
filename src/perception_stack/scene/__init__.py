"""Road evidence and combined scene results for one original-resolution frame."""

from .association import ContactPolicy, FrameRoadAssociations, associate_road
from .results import FrameScene, build_scene

__all__ = ["ContactPolicy", "FrameRoadAssociations", "associate_road", "FrameScene", "build_scene"]
