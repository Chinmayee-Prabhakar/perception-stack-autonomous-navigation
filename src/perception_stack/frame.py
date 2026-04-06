"""Input frame and calibration metadata. All image coordinates are original pixels."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole metadata for an already aligned image, with pixel units for fx/fy/cx/cy."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera size and focal lengths must be positive")
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all():
            raise ValueError("Camera intrinsics must be finite")


@dataclass(frozen=True)
class Frame:
    image_rgb: np.ndarray
    frame_id: str
    source: str | None = None
    intrinsics: CameraIntrinsics | None = None

    def __post_init__(self) -> None:
        image = np.asarray(self.image_rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("image_rgb must be a HxWx3 uint8 RGB array")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Frame dimensions must be positive")
        if not self.frame_id:
            raise ValueError("frame_id must be nonempty")
        if self.intrinsics and (self.intrinsics.width, self.intrinsics.height) != self.size:
            raise ValueError("Intrinsics dimensions must match the frame")
        owned = np.array(image, copy=True, order="C")
        owned.flags.writeable = False
        object.__setattr__(self, "image_rgb", owned)

    @property
    def size(self) -> tuple[int, int]:
        return (self.image_rgb.shape[1], self.image_rgb.shape[0])

    @classmethod
    def from_image(cls, path: str | Path, *, intrinsics: CameraIntrinsics | None = None) -> "Frame":
        path = Path(path)
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        return cls(rgb, frame_id=path.stem, source=str(path), intrinsics=intrinsics)
