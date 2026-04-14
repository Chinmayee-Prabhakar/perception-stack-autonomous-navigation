"""Strict reader for identified KITTI Object labeled-training Velodyne scans."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

import numpy as np


_FRAME_ID = re.compile(r"[0-9]{6}\Z")


@dataclass(frozen=True)
class VelodyneScan:
    frame_id: str
    xyz_m: np.ndarray
    intensity: np.ndarray
    original_indices: np.ndarray
    source_sha256: str
    total_rows: int
    nonfinite_xyz_rows: int
    nonfinite_intensity_rows: int

    def __post_init__(self) -> None:
        count = len(self.original_indices)
        if self.xyz_m.shape != (count, 3) or self.intensity.shape != (count,):
            raise ValueError("Scan arrays have inconsistent shapes")
        if not np.isfinite(self.xyz_m).all() or not np.isfinite(self.intensity).all():
            raise ValueError("Retained scan points must be finite")
        if self.original_indices.dtype.kind not in "iu" or np.any(self.original_indices < 0):
            raise ValueError("Original scan indices must be nonnegative integers")
        for field in ("xyz_m", "intensity", "original_indices"):
            source = np.asarray(getattr(self, field))
            # A read-only ndarray with owned memory can be made writable again.
            # Immutable bytes prevent writes through flags, views, or source aliases.
            protected = np.frombuffer(source.tobytes(order="C"), dtype=source.dtype).reshape(source.shape)
            object.__setattr__(self, field, protected)

    @property
    def retained_rows(self) -> int:
        return len(self.original_indices)

    @property
    def rejected_rows(self) -> int:
        return self.total_rows - self.retained_rows


def read_object_scan(path: str | Path, *, frame_id: str) -> VelodyneScan:
    """Require the Object training layout; numeric Raw/Odometry names do not qualify."""
    source = Path(path)
    if not _FRAME_ID.fullmatch(frame_id):
        raise ValueError("KITTI Object frame ID must have six digits")
    if source.name != f"{frame_id}.bin" or source.parent.name != "velodyne" or source.parent.parent.name != "training":
        raise ValueError("Expected exact KITTI Object training/velodyne/{frame_id}.bin path")
    if not source.is_file():
        raise FileNotFoundError(f"Missing KITTI Object Velodyne scan: {source}")
    payload = source.read_bytes()
    if len(payload) % 16:
        raise ValueError(f"Malformed KITTI Object Velodyne scan byte length: {len(payload)}")
    rows = np.frombuffer(payload, dtype="<f4").reshape(-1, 4)
    good_xyz = np.isfinite(rows[:, :3]).all(axis=1)
    good_intensity = np.isfinite(rows[:, 3])
    good = good_xyz & good_intensity
    return VelodyneScan(frame_id, rows[good, :3], rows[good, 3],
                        np.flatnonzero(good).astype(np.int64), sha256(payload).hexdigest(),
                        len(rows), int((~good_xyz).sum()), int((~good_intensity).sum()))
