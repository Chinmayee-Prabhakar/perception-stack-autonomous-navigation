"""Synthetic source-file provenance tests; no real KITTI scan is opened."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from perception_stack.kitti_lidar import load_calibrated_projector, load_verified_object_scan


CALIBRATION = ("P2: 10 0 10 0 0 10 10 0 0 0 1 0\n"
               "R0_rect: 1 0 0 0 1 0 0 0 1\n"
               "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n").encode("ascii")


def _calibration_file(root: Path, payload: bytes = CALIBRATION) -> Path:
    source = root / "training/calib/000001.txt"
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    return source


def _scan_file(root: Path, values: tuple[float, float, float, float] = (0, 0, 10, 1)) -> Path:
    source = root / "training/velodyne/000001.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(np.asarray(values, dtype="<f4").tobytes())
    return source


def test_calibration_actual_file_bytes_bind_projection_and_survive_source_mutation(tmp_path):
    source = _calibration_file(tmp_path)
    expected = sha256(CALIBRATION).hexdigest()
    projector = load_calibrated_projector(source, frame_id="000001", image_size=(20, 20),
                                          expected_sha256=expected)
    scan_path = _scan_file(tmp_path)
    scan = load_verified_object_scan(scan_path, frame_id="000001",
                                     expected_sha256=sha256(scan_path.read_bytes()).hexdigest())
    original = projector.project(scan)
    assert original.calibration_sha256 == expected
    assert original.returns[0].pixel == (10, 10)
    source.write_bytes(CALIBRATION.replace(b"10 0 10 0", b"10 0 10 50"))
    assert projector.project(scan).returns == original.returns
    with pytest.raises(ValueError, match="Calibration source SHA-256"):
        load_calibrated_projector(source, frame_id="000001", image_size=(20, 20),
                                  expected_sha256=expected)


def test_calibration_malformed_or_incomplete_bytes_fail(tmp_path):
    source = _calibration_file(tmp_path, CALIBRATION.replace(b"P2:", b"bad:"))
    with pytest.raises(ValueError, match="requires P2"):
        load_calibrated_projector(source, frame_id="000001", image_size=(20, 20))
    source.write_bytes(CALIBRATION.split(b"Tr_velo_to_cam:")[0])
    with pytest.raises(ValueError, match="requires P2, R0_rect and Tr_velo_to_cam"):
        load_calibrated_projector(source, frame_id="000001", image_size=(20, 20))
    source.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        load_calibrated_projector(source, frame_id="000001", image_size=(20, 20))


def test_calibration_identity_and_expected_digest_are_checked(tmp_path):
    source = _calibration_file(tmp_path)
    with pytest.raises(ValueError, match="training/calib"):
        load_calibrated_projector(source, frame_id="000002", image_size=(20, 20))
    with pytest.raises(ValueError, match="Calibration source SHA-256"):
        load_calibrated_projector(source, frame_id="000001", image_size=(20, 20),
                                  expected_sha256="a" * 64)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_calibrated_projector(source, frame_id="000001", image_size=(20, 20),
                                  expected_sha256="invalid")


def test_scan_inventory_hash_and_frame_identity_are_checked(tmp_path):
    source = _scan_file(tmp_path)
    original = sha256(source.read_bytes()).hexdigest()
    scan = load_verified_object_scan(source, frame_id="000001", expected_sha256=original)
    assert scan.source_sha256 == original and scan.retained_rows == 1
    with pytest.raises(ValueError, match="training/velodyne"):
        load_verified_object_scan(source, frame_id="000002", expected_sha256=original)
    source.write_bytes(np.asarray((1, 0, 10, 1), dtype="<f4").tobytes())
    assert scan.xyz_m[0, 0] == 0
    with pytest.raises(ValueError, match="Scan source SHA-256"):
        load_verified_object_scan(source, frame_id="000001", expected_sha256=original)
