"""Pinhole backprojection for a known rectified camera origin."""

import math

from perception_stack.frame import CameraIntrinsics
from .results import SurfacePoint


def backproject_pixel(
    u: int, v: int, depth_z_m: float, intrinsics: CameraIntrinsics, *,
    image_size: tuple[int, int], rectified: bool,
) -> SurfacePoint:
    """Backproject the actual depth at integer pixel-center (u,v), not a region median.

    The camera frame has +X right, +Y down, +Z forward. Intrinsics must already
    correspond to this rectified image and camera origin; a raw KITTI P2 is not
    interchangeable with this contract because its translation may be nonzero.
    """
    if not isinstance(intrinsics, CameraIntrinsics) or not rectified:
        raise ValueError("A rectified CameraIntrinsics calibration is required")
    if image_size != (intrinsics.width, intrinsics.height):
        raise ValueError("Intrinsics resolution does not match the image")
    if not isinstance(u, int) or not isinstance(v, int) or not (0 <= u < image_size[0] and 0 <= v < image_size[1]):
        raise ValueError("Pixel center must be an in-image integer coordinate")
    if not math.isfinite(depth_z_m) or depth_z_m <= 0:
        raise ValueError("Forward depth Z must be finite and positive")
    z = float(depth_z_m)
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return SurfacePoint(x, y, z, math.sqrt(x*x + y*y + z*z))
