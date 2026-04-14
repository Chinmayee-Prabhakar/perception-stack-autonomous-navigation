"""Full KITTI Velodyne-to-image-2 projection and camera-2 axial depth."""

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from perception_stack.kitti_eval.dataset import KittiCalibration
from .scan import VelodyneScan


@dataclass(frozen=True)
class ProjectedReturn:
    original_index: int
    reference_xyz_m: tuple[float, float, float]
    u: float
    v: float
    pixel: tuple[int, int]
    camera2_z_m: float


@dataclass(frozen=True)
class ProjectionResult:
    frame_id: str
    image_size: tuple[int, int]
    returns: tuple[ProjectedReturn, ...]
    rejected_counts: Mapping[str, int]
    scan_sha256: str
    calibration_sha256: str
    scan_total_rows: int
    scan_rejected_rows: int
    scan_nonfinite_xyz_rows: int
    scan_nonfinite_intensity_rows: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "rejected_counts", MappingProxyType(dict(self.rejected_counts)))


@dataclass(frozen=True, eq=False, slots=True, init=False)
class CalibratedProjector:
    """Project with a stable calibration snapshot; the supplied digest claims source bytes only."""

    _p2: np.ndarray
    _r0_rect: np.ndarray
    _tr_velo_to_cam: np.ndarray
    image_size: tuple[int, int]
    calibration_sha256: str
    camera2_center_ref_m: tuple[float, float, float]

    @staticmethod
    def _snapshot(matrix: np.ndarray) -> np.ndarray:
        """Keep the same float64 values in immutable bytes, not reversible array flags."""
        return np.frombuffer(matrix.tobytes(order="C"), dtype=np.float64).reshape(matrix.shape)

    @property
    def calibration(self) -> KittiCalibration:
        """Preserve the public attribute without exposing retained matrix storage."""
        return KittiCalibration(self._p2, self._r0_rect, self._tr_velo_to_cam)

    def __init__(self, calibration: KittiCalibration, image_size: tuple[int, int], *,
                 calibration_sha256: str):
        if not isinstance(calibration, KittiCalibration):
            raise TypeError("Expected parsed KITTI calibration")
        if calibration.r0_rect is None or calibration.tr_velo_to_cam is None:
            raise ValueError("LiDAR projection requires R0_rect and Tr_velo_to_cam")
        if (len(image_size) != 2 or any(type(v) is not int or v <= 0 for v in image_size)):
            raise ValueError("Image width and height must be positive integers")
        if len(calibration_sha256) != 64 or any(c not in "0123456789abcdef" for c in calibration_sha256):
            raise ValueError("Calibration source requires a lowercase SHA-256")
        # Revalidate because the input object's NumPy arrays may have been mutated
        # since its own construction, then sever every external memory alias.
        validated = KittiCalibration(calibration.p2, calibration.r0_rect, calibration.tr_velo_to_cam)
        p2 = self._snapshot(validated.p2)
        r0_rect = self._snapshot(validated.r0_rect)
        tr_velo_to_cam = self._snapshot(validated.tr_velo_to_cam)
        k = p2[:, :3]
        if not np.allclose(k[2], (0, 0, 1), rtol=0, atol=1e-10):
            raise ValueError("Camera-2 axial Z requires canonical rectified P2 denominator")
        if not np.isfinite(np.linalg.cond(k)) or np.linalg.cond(k) >= 1e12:
            raise ValueError("P2 left block is numerically noninvertible")
        object.__setattr__(self, "_p2", p2)
        object.__setattr__(self, "_r0_rect", r0_rect)
        object.__setattr__(self, "_tr_velo_to_cam", tr_velo_to_cam)
        object.__setattr__(self, "image_size", tuple(image_size))
        object.__setattr__(self, "calibration_sha256", calibration_sha256)
        object.__setattr__(self, "camera2_center_ref_m",
                           tuple(float(x) for x in -np.linalg.solve(k, p2[:, 3])))

    def factorization_residual(self, reference_xyz_m: tuple[float, float, float]) -> float:
        xyz = np.asarray(reference_xyz_m, dtype=np.float64)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Expected finite reference-camera XYZ")
        direct = self._p2 @ np.r_[xyz, 1.0]
        factored = self._p2[:, :3] @ (xyz - self.camera2_center_ref_m)
        return float(np.max(np.abs(direct - factored)))

    def project(self, scan: VelodyneScan) -> ProjectionResult:
        if not isinstance(scan, VelodyneScan):
            raise TypeError("Expected validated VelodyneScan")
        tr = self._tr_velo_to_cam
        ref = (scan.xyz_m.astype(np.float64) @ tr[:, :3].T + tr[:, 3]) @ self._r0_rect.T
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            q = np.column_stack((ref, np.ones(len(ref)))) @ self._p2.T
            camera2_z = ref[:, 2] - self.camera2_center_ref_m[2]
            uv = q[:, :2] / q[:, 2:3]
        width, height = self.image_size
        accepted = []
        counts = {key: 0 for key in ("nonfinite_transform", "nonpositive_camera_z",
                                       "nonpositive_projection_denominator", "nonfinite_uv",
                                       "outside_image")}
        for row, index in enumerate(scan.original_indices):
            if not np.isfinite(ref[row]).all() or not np.isfinite(q[row]).all() or not math.isfinite(camera2_z[row]):
                counts["nonfinite_transform"] += 1
                continue
            if camera2_z[row] <= 0:
                counts["nonpositive_camera_z"] += 1
                continue
            if q[row, 2] <= 0:
                counts["nonpositive_projection_denominator"] += 1
                continue
            u, v = uv[row]
            if not math.isfinite(u) or not math.isfinite(v):
                counts["nonfinite_uv"] += 1
                continue
            if not (0 <= u < width and 0 <= v < height):
                counts["outside_image"] += 1
                continue
            accepted.append(ProjectedReturn(int(index), tuple(float(x) for x in ref[row]),
                                            float(u), float(v), (math.floor(u), math.floor(v)),
                                            float(camera2_z[row])))
        return ProjectionResult(scan.frame_id, self.image_size, tuple(accepted), counts,
                                scan.source_sha256, self.calibration_sha256, scan.total_rows,
                                scan.rejected_rows, scan.nonfinite_xyz_rows,
                                scan.nonfinite_intensity_rows)
