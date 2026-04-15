"""Bind identified KITTI Object scan and calibration files to their source bytes."""

from hashlib import sha256
from pathlib import Path
import re

from perception_stack.kitti_eval.dataset import parse_calibration

from .geometry import CalibratedProjector
from .scan import VelodyneScan, read_object_scan


_FRAME_ID = re.compile(r"[0-9]{6}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _require_file(path: str | Path, frame_id: str, leaf: str, extension: str) -> Path:
    source = Path(path)
    if _FRAME_ID.fullmatch(frame_id) is None:
        raise ValueError("KITTI Object frame ID must have six digits")
    if (source.name != f"{frame_id}.{extension}" or source.parent.name != leaf
            or source.parent.parent.name != "training"):
        raise ValueError(f"Expected exact KITTI Object training/{leaf}/{frame_id}.{extension} path")
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _require_digest(value: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError("Expected a lowercase SHA-256 digest")


def load_verified_object_scan(path: str | Path, *, frame_id: str,
                              expected_sha256: str) -> VelodyneScan:
    """Reject a scan whose actual bytes differ from the approved member inventory."""
    _require_digest(expected_sha256)
    source = _require_file(path, frame_id, "velodyne", "bin")
    scan = read_object_scan(source, frame_id=frame_id)
    if scan.source_sha256 != expected_sha256:
        raise ValueError("Scan source SHA-256 differs from approved archive member")
    return scan


def load_calibrated_projector(path: str | Path, *, frame_id: str,
                              image_size: tuple[int, int],
                              expected_sha256: str | None = None) -> CalibratedProjector:
    """Hash and parse one read of the actual calibration file, then snapshot it."""
    source = _require_file(path, frame_id, "calib", "txt")
    payload = source.read_bytes()
    digest = sha256(payload).hexdigest()
    if expected_sha256 is not None:
        _require_digest(expected_sha256)
        if digest != expected_sha256:
            raise ValueError("Calibration source SHA-256 differs from expected file")
    calibration = parse_calibration(payload.decode("utf-8"))
    if calibration.r0_rect is None or calibration.tr_velo_to_cam is None:
        raise ValueError("KITTI LiDAR calibration requires P2, R0_rect and Tr_velo_to_cam")
    return CalibratedProjector(calibration, image_size, calibration_sha256=digest)
