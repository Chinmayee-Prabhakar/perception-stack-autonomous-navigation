"""Entirely synthetic tests for M5-L1; no official scan is opened."""

from dataclasses import FrozenInstanceError
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import pytest

from perception_stack.kitti_eval.dataset import KittiAnnotation, KittiCalibration
from perception_stack.kitti_lidar import (
    CalibratedProjector, FrozenObjectMatch, LidarProtocol, SourceDigests,
    VelodyneScan, associate_object, paired_surface_error, read_object_scan,
)
from perception_stack.kitti_lidar.association import inside_oriented_cuboid, REJECTION_CODES
from perception_stack.results import BoxXYXY


PROTOCOL_FILE = Path(__file__).resolve().parents[1] / "configs/kitti_lidar_protocol_v1.json"
SHA = "a" * 64


def protocol():
    return LidarProtocol.load(PROTOCOL_FILE)


def calib(*, p4=(0., 0., 0.), tr=None, rect=None, focal=10., center=10.):
    p2 = np.array([[focal, 0, center, p4[0]], [0, focal, center, p4[1]],
                   [0, 0, 1, p4[2]]], dtype=float)
    if tr is None:
        tr = np.column_stack((np.eye(3), np.zeros(3)))
    if rect is None:
        rect = np.eye(3)
    return KittiCalibration(p2, rect, tr)


def synthetic_scan(points, frame_id="000001"):
    rows = np.asarray(points, dtype=np.float64).reshape(-1, 4)
    return VelodyneScan(frame_id, rows[:, :3], rows[:, 3],
                        np.arange(len(rows), dtype=np.int64), SHA, len(rows), 0, 0)


def annotation(object_id=0, *, location=(0., 0., 10.), hwl=(2., 2., 4.), yaw=0.,
               box=(0., 0., 20., 20.), cls="Car"):
    return KittiAnnotation(object_id, cls, 0., 0, 0., BoxXYXY(*box), hwl,
                           location, yaw, "valid")


def match(*, annotation_id=0, box=(0., 0., 20., 20.), class_name="Car",
          box_z=9., mask_z=11.):
    return FrozenObjectMatch("000001", annotation_id, 7, class_name, class_name,
                             BoxXYXY(*box), box_z, mask_z)


def evaluate(points, *, annotations=None, frozen=None, calibration=None):
    calibration = calibration or calib()
    scan = synthetic_scan(points)
    projection = CalibratedProjector(calibration, (20, 20), calibration_sha256=SHA).project(scan)
    return associate_object(projection, tuple(annotations or (annotation(),)), frozen or match(),
                            SourceDigests(SHA, SHA, protocol().sha256, SHA), protocol())


def test_versioned_policy_is_frozen_and_separates_estimands(tmp_path):
    current = protocol()
    assert current.sha256 == sha256(PROTOCOL_FILE.read_bytes()).hexdigest()
    assert "box/mask" in current.data["primary_estimand"]
    assert "pixel" in current.data["secondary_estimand"]
    assert current.data["pixel_visibility"].startswith("all valid projected returns")
    altered = dict(current.data, minimum_distinct_bins=2)
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="minimum_distinct_bins"):
        LidarProtocol.load(path)


def test_loaded_policy_cannot_mutate_effective_settings_or_reuse_digest():
    current = protocol()
    reviewed_sha = "02451176def64af5ea83adc3c63df0ee37351176b008ec947f7da20c3f1f75fc"
    assert current.sha256 == reviewed_sha
    current.data["minimum_distinct_bins"] = 2
    current.data["boundary_numeric_tolerance_m"] = 1.0
    current.data["support_sensitivity_bins"].append(20)
    current.data["scan_rejection_codes"][0] = "changed"
    assert current.min_bins == 3
    assert current.boundary_tolerance_m == 1e-12
    assert current.data["support_sensitivity_bins"] == [1, 3, 5, 10]
    assert current.data["scan_rejection_codes"][0] == "nonfinite_xyz"
    assert current.sha256 == reviewed_sha == sha256(current._payload).hexdigest()
    with pytest.raises(FrozenInstanceError):
        current._payload = b"changed"
    with pytest.raises(FrozenInstanceError):
        current.sha256 = "b" * 64
    changed = current._payload.replace(b'"minimum_distinct_bins": 3',
                                       b'"minimum_distinct_bins": 2')
    assert changed != current._payload
    with pytest.raises(ValueError, match="bytes and SHA-256"):
        LidarProtocol(changed, reviewed_sha)
    with pytest.raises(ValueError, match="minimum_distinct_bins"):
        LidarProtocol(changed, sha256(changed).hexdigest())


def test_protocol_mutation_attempt_does_not_change_three_bin_support():
    current = protocol()
    current.data["minimum_distinct_bins"] = 1
    scan = synthetic_scan([(0, -1, 10, 1)])
    projection = CalibratedProjector(calib(), (20, 20), calibration_sha256=SHA).project(scan)
    reference = associate_object(projection, (annotation(),), match(),
                                 SourceDigests(SHA, SHA, current.sha256, SHA), current)
    assert reference.status == "unsupported_insufficient_bins"
    assert reference.support_bins == 1 and reference.lidar_z_m is None


@pytest.mark.parametrize("field", ("xyz_m", "intensity", "original_indices"))
def test_validated_scan_arrays_and_views_have_immutable_backing(field):
    scan = synthetic_scan([(0, -1, 10, 1)])
    array = getattr(scan, field)
    before = array.copy()
    with pytest.raises(ValueError):
        array.flat[0] = 99
    with pytest.raises(ValueError):
        array.flags.writeable = True
    with pytest.raises(ValueError):
        array.reshape(-1).flags.writeable = True
    assert np.array_equal(array, before)
    assert scan.source_sha256 == SHA


def test_scan_constructor_copies_source_aliases_and_preserves_projection():
    xyz = np.array([[0., -1., 10.]])
    intensity = np.array([.5])
    indices = np.array([7], dtype=np.int64)
    scan = VelodyneScan("000001", xyz, intensity, indices, SHA, 1, 0, 0)
    expected = CalibratedProjector(calib(), (20, 20), calibration_sha256=SHA).project(scan)
    xyz[:] = 999
    intensity[:] = 0
    indices[:] = 99
    assert scan.xyz_m.tolist() == [[0, -1, 10]]
    assert scan.intensity.tolist() == [.5]
    assert scan.original_indices.tolist() == [7]
    assert CalibratedProjector(calib(), (20, 20), calibration_sha256=SHA).project(scan) == expected
    assert scan.source_sha256 == SHA


def test_parsed_scan_hash_cannot_be_reused_after_attempted_mutation(tmp_path):
    folder = tmp_path / "training/velodyne"
    folder.mkdir(parents=True)
    path = folder / "000001.bin"
    original = np.array([[0, -1, 10, .5]], dtype="<f4").tobytes()
    path.write_bytes(original)
    scan = read_object_scan(path, frame_id="000001")
    original_projection = CalibratedProjector(calib(), (20, 20), calibration_sha256=SHA).project(scan)
    with pytest.raises(ValueError):
        scan.xyz_m.flags.writeable = True
    with pytest.raises(ValueError):
        scan.xyz_m[0, 2] = 20
    assert scan.source_sha256 == sha256(original).hexdigest()
    assert CalibratedProjector(calib(), (20, 20), calibration_sha256=SHA).project(scan) == original_projection
    changed = np.array([[0, -1, 20, .5]], dtype="<f4").tobytes()
    path.write_bytes(changed)
    changed_scan = read_object_scan(path, frame_id="000001")
    assert changed_scan.source_sha256 == sha256(changed).hexdigest() != scan.source_sha256


def test_projection_and_reference_diagnostics_are_immutable_snapshots():
    scan = synthetic_scan([(0, -1, 10, 1)])
    projection = CalibratedProjector(calib(), (20, 20), calibration_sha256=SHA).project(scan)
    reference = associate_object(projection, (annotation(),), match(),
                                 SourceDigests(SHA, SHA, protocol().sha256, SHA), protocol())
    with pytest.raises(TypeError):
        projection.rejected_counts["outside_image"] = 99
    with pytest.raises(TypeError):
        reference.rejected_counts["outside_target_cuboid"] = 99
    with pytest.raises(TypeError):
        reference.projection_rejected_counts["outside_image"] = 99
    with pytest.raises(FrozenInstanceError):
        reference.sources.scan_sha256 = "b" * 64
    exported = reference.to_dict()
    exported["rejected_counts"]["outside_target_cuboid"] = 99
    assert reference.rejected_counts["outside_target_cuboid"] == 0


def test_scan_parser_valid_empty_indices_and_nonfinite(tmp_path):
    folder = tmp_path / "training/velodyne"
    folder.mkdir(parents=True)
    path = folder / "000001.bin"
    rows = np.array([[1, 2, 3, .4], [np.nan, 2, 3, .5], [4, 5, 6, np.inf],
                     [7, 8, 9, .2], [np.inf, 1, 1, np.nan]], dtype="<f4")
    path.write_bytes(rows.tobytes())
    before = path.read_bytes()
    scan = read_object_scan(path, frame_id="000001")
    assert scan.total_rows == 5 and scan.retained_rows == 2 and scan.rejected_rows == 3
    assert scan.nonfinite_xyz_rows == 2 and scan.nonfinite_intensity_rows == 2
    assert scan.original_indices.tolist() == [0, 3]
    assert scan.xyz_m.tolist() == [[1, 2, 3], [7, 8, 9]]
    assert scan.source_sha256 == sha256(before).hexdigest() and path.read_bytes() == before
    assert not scan.xyz_m.flags.writeable
    path.write_bytes(b"")
    assert read_object_scan(path, frame_id="000001").retained_rows == 0


def test_scan_parser_rejects_missing_malformed_and_nonobject_path(tmp_path):
    folder = tmp_path / "training/velodyne"
    folder.mkdir(parents=True)
    path = folder / "000001.bin"
    with pytest.raises(FileNotFoundError, match="Missing"):
        read_object_scan(path, frame_id="000001")
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="byte length"):
        read_object_scan(path, frame_id="000001")
    path.write_bytes(np.zeros((1, 4), dtype="<f4").tobytes())
    with pytest.raises(ValueError, match="exact KITTI Object"):
        read_object_scan(path, frame_id="000002")
    raw = tmp_path / "raw/velodyne/000001.bin"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="exact KITTI Object"):
        read_object_scan(raw, frame_id="000001")


def test_identity_and_nonzero_p2_translation_camera2_depth():
    identity = CalibratedProjector(calib(), (20, 20), calibration_sha256=SHA)
    point = identity.project(synthetic_scan([(0, 0, 10, 1)])).returns[0]
    assert (point.u, point.v, point.pixel, point.camera2_z_m) == (10, 10, (10, 10), 10)
    translated = CalibratedProjector(calib(p4=(2, -3, 1)), (20, 20), calibration_sha256=SHA)
    p = translated.project(synthetic_scan([(0, 0, 10, 1)])).returns[0]
    assert translated.camera2_center_ref_m == pytest.approx((.8, 1.3, -1))
    assert p.u == pytest.approx(102 / 11) and p.v == pytest.approx(97 / 11)
    assert p.camera2_z_m == 11 and p.camera2_z_m != p.reference_xyz_m[2]
    assert translated.factorization_residual((2, 3, 10)) < 1e-12
    assert math.sqrt(10**2) != math.sqrt(1 + 10**2)  # axial Z differs from radial range


@pytest.mark.parametrize("field,index", [
    ("p2", (0, 3)),
    ("r0_rect", (0, 2)),
    ("tr_velo_to_cam", (0, 3)),
])
def test_projector_snapshots_each_supplied_calibration_matrix(field, index):
    calibration = calib()
    projector = CalibratedProjector(calibration, (20, 20), calibration_sha256="c" * 64)
    scan = synthetic_scan([(0, 0, 10, 1)])
    before = projector.project(scan)
    assert (before.returns[0].u, before.returns[0].v,
            before.returns[0].camera2_z_m) == (10, 10, 10)
    assert before.scan_sha256 == SHA and before.calibration_sha256 == "c" * 64
    center = projector.camera2_center_ref_m
    assert projector.factorization_residual((0, 0, 10)) < 1e-12

    source = getattr(calibration, field)
    source.flags.writeable = True
    source[index] = 50 if field == "p2" else .5 if field == "r0_rect" else 5
    source.flags.writeable = False
    assert projector.project(scan) == before
    assert projector.camera2_center_ref_m == center
    assert projector.factorization_residual((0, 0, 10)) < 1e-12


def test_projector_ignores_source_views_and_exposed_calibration_mutation():
    original_p2 = np.array([[10., 0., 10., 0.], [0., 10., 10., 0.], [0., 0., 1., 0.]])
    calibration = KittiCalibration(original_p2, np.eye(3),
                                   np.column_stack((np.eye(3), np.zeros(3))))
    projector = CalibratedProjector(calibration, (20, 20), calibration_sha256="c" * 64)
    scan = synthetic_scan([(0, 0, 10, 1)])
    before = projector.project(scan)
    original_p2[0, 3] = 70
    calibration.p2.flags.writeable = True
    source_view = calibration.p2[:, :]
    source_view[0, 3] = 50
    exposed = projector.calibration
    exposed.p2.flags.writeable = True
    exposed.p2[0, 3] = 30
    assert projector.calibration.p2[0, 3] == 0
    assert projector.project(scan) == before
    assert projector.camera2_center_ref_m == pytest.approx((0, 0, 0))
    with pytest.raises(FrozenInstanceError):
        projector.calibration_sha256 = "d" * 64


@pytest.mark.parametrize("field", ("_p2", "_r0_rect", "_tr_velo_to_cam"))
def test_projector_retained_matrices_have_immutable_bytes_backing(field):
    projector = CalibratedProjector(calib(), (20, 20), calibration_sha256=SHA)
    array = getattr(projector, field)
    with pytest.raises(ValueError):
        array.flat[0] = 99
    with pytest.raises(ValueError):
        array.flags.writeable = True
    with pytest.raises(ValueError):
        array.reshape(-1).flags.writeable = True
    assert projector.project(synthetic_scan([(0, 0, 10, 1)])).returns[0].u == 10


def test_translated_p2_snapshot_keeps_camera_center_and_factorization():
    calibration = calib(p4=(2, -3, 1))
    projector = CalibratedProjector(calibration, (20, 20), calibration_sha256="c" * 64)
    scan = synthetic_scan([(0, 0, 10, 1)])
    before = projector.project(scan)
    assert projector.camera2_center_ref_m == pytest.approx((.8, 1.3, -1))
    assert (before.returns[0].u, before.returns[0].v,
            before.returns[0].camera2_z_m) == pytest.approx((102 / 11, 97 / 11, 11))
    calibration.p2.flags.writeable = True
    calibration.p2[0, 3] = 52
    assert projector.project(scan) == before
    assert projector.camera2_center_ref_m == pytest.approx((.8, 1.3, -1))
    assert projector.factorization_residual((2, 3, 10)) < 1e-12


def test_projector_revalidates_mutated_calibration_and_retains_source_ids():
    invalid = calib()
    invalid.p2.flags.writeable = True
    invalid.p2[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        CalibratedProjector(invalid, (20, 20), calibration_sha256="c" * 64)
    wrong_shape = calib()
    object.__setattr__(wrong_shape, "r0_rect", np.eye(2))
    with pytest.raises(ValueError, match="shape"):
        CalibratedProjector(wrong_shape, (20, 20), calibration_sha256="c" * 64)
    invalid_rotation = calib()
    invalid_rotation.tr_velo_to_cam.flags.writeable = True
    invalid_rotation.tr_velo_to_cam[0, 0] = 2
    with pytest.raises(ValueError, match="orthonormal"):
        CalibratedProjector(invalid_rotation, (20, 20), calibration_sha256="c" * 64)

    policy = protocol()
    projector = CalibratedProjector(calib(), (20, 20), calibration_sha256="c" * 64)
    result = projector.project(synthetic_scan([(0, 0, 10, 1)]))
    sources = SourceDigests(SHA, "c" * 64, policy.sha256, "d" * 64)
    assert result.scan_sha256 == sources.scan_sha256
    assert result.calibration_sha256 == sources.calibration_sha256
    assert sources.protocol_sha256 == policy.sha256
    with pytest.raises(FrozenInstanceError):
        sources.calibration_sha256 = "e" * 64


def test_rotation_translation_order_with_independent_expected_point():
    # First: R_y(90) (2,0,10)+(1,2,3)=(11,2,1); then R_z(90) -> (-2,11,1).
    ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    tr = np.column_stack((ry, np.array([1, 2, 3])))
    projector = CalibratedProjector(calib(tr=tr, rect=rz, focal=1), (30, 30), calibration_sha256=SHA)
    point = projector.project(synthetic_scan([(2, 0, 10, 1)])).returns[0]
    assert point.reference_xyz_m == pytest.approx((-2, 11, 1))
    assert (point.u, point.v) == pytest.approx((8, 21))


def test_independent_camera_rotation_and_translation_cases():
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    tr = np.column_stack((np.eye(3), np.array([2, 0, 0])))
    translated = CalibratedProjector(calib(tr=tr), (30, 30), calibration_sha256=SHA)
    rotated = CalibratedProjector(calib(rect=rz), (30, 30), calibration_sha256=SHA)
    assert translated.project(synthetic_scan([(0, 0, 10, 1)])).returns[0].u == 12
    point = rotated.project(synthetic_scan([(1, 0, 10, 1)])).returns[0]
    assert point.reference_xyz_m == pytest.approx((0, 1, 10)) and point.v == 11


def test_projection_reasons_boundaries_behind_and_invalid_dimensions():
    projector = CalibratedProjector(calib(focal=1, center=0), (10, 10), calibration_sha256=SHA)
    result = projector.project(synthetic_scan([(0, 0, 1, 1), (10, 0, 1, 1),
                                               (-.01, 0, 1, 1), (0, 0, -1, 1),
                                               (0, 0, 0, 1)]))
    assert [p.pixel for p in result.returns] == [(0, 0)]
    assert result.rejected_counts["outside_image"] == 2
    assert result.rejected_counts["nonpositive_camera_z"] == 2
    with pytest.raises(ValueError, match="width and height"):
        CalibratedProjector(calib(), (0, 10), calibration_sha256=SHA)
    with pytest.raises(ValueError, match="requires"):
        CalibratedProjector(KittiCalibration(calib().p2, None, None), (20, 20), calibration_sha256=SHA)


def test_nonfinite_scan_values_and_projection_overflow_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        VelodyneScan("000001", np.array([[np.nan, 0, 1]]), np.array([1.]),
                     np.array([0]), SHA, 1, 1, 0)
    huge = synthetic_scan([(1e308, 1e308, 1e308, 1)])
    result = CalibratedProjector(calib(), (20, 20), calibration_sha256=SHA).project(huge)
    assert result.rejected_counts["nonfinite_transform"] == 1
    with pytest.raises(ValueError):
        KittiCalibration(np.array([[0., 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]]), np.eye(3),
                         np.column_stack((np.eye(3), np.zeros(3))))


def test_cuboid_dimensions_bottom_center_faces_and_independent_corners():
    a = annotation(location=(0, 2, 10), hwl=(2, 2, 4))
    expected_corners = {(x, y, z) for x in (-2, 2) for y in (0, 2) for z in (9, 11)}
    assert len(expected_corners) == 8
    assert all(inside_oriented_cuboid(corner, a) for corner in expected_corners)
    for outside in ((2.001, 1, 10), (0, -0.001, 10), (0, 2.001, 10),
                    (0, 1, 8.999), (0, 1, 11.001)):
        assert not inside_oriented_cuboid(outside, a)
    projector = CalibratedProjector(calib(focal=10, center=10), (30, 30), calibration_sha256=SHA)
    projected = projector.project(synthetic_scan([(*corner, 1) for corner in sorted(expected_corners)])).returns
    assert len(projected) == 8
    assert {(round(p.u, 8), round(p.v, 8)) for p in projected} == {
        (round(10 + 10*x/z, 8), round(10 + 10*y/z, 8)) for x, y, z in expected_corners}


@pytest.mark.parametrize("yaw,point,inside", [
    (math.pi / 2, (0, -1, 8), True),      # local +length/2 rotates toward -Z
    (math.pi / 2, (1.01, -1, 10), False),  # exceeds local width/2
    (-math.pi / 2, (0, -1, 12), True),
    (math.pi / 4, (math.sqrt(2), -1, 10 - math.sqrt(2)), True),
    (math.pi / 4, (1.5, -1, 10 - math.sqrt(2)), False),
])
def test_cuboid_yaw_sign_and_nontrivial_angles(yaw, point, inside):
    assert inside_oriented_cuboid(point, annotation(yaw=yaw)) is inside


def test_three_distinct_bins_common_median_and_paired_error():
    reference = evaluate([(-1, -1, 10, 1), (0, -1, 10, 1), (1, -1, 10, 1)])
    assert reference.status == "supported" and reference.support_bins == 3
    assert reference.lidar_z_m == 10
    assert [p.pixel for p in reference.accepted_bins] == [(9, 9), (10, 9), (11, 9)]
    assert all(k in reference.rejected_counts for k in REJECTION_CODES)
    assert reference.to_dict()["accepted_bins"][1]["original_scan_index"] == 1
    paired = paired_surface_error(reference, drive_id="synthetic_drive")
    assert (paired.box_signed_error_m, paired.mask_signed_error_m,
            paired.mask_minus_box_absolute_error_m) == (-1, 1, 0)


@pytest.mark.parametrize("count", (0, 1, 2))
def test_zero_one_two_bins_are_explicitly_unsupported(count):
    reference = evaluate([(x, -1, 10, 1) for x in range(count)])
    assert reference.status == "unsupported_insufficient_bins"
    assert reference.support_bins == count and reference.lidar_z_m is None
    assert len(reference.to_dict()["accepted_bins"]) == count
    assert reference.to_dict()["lidar_z_m"] is None
    with pytest.raises(ValueError, match="supported"):
        paired_surface_error(reference, drive_id="synthetic_drive")


def test_nearest_same_pixel_wins_before_association_even_if_unassigned():
    # At pixel (10,9), foreground z=5 is not in the target z=[9,11].
    points = [(0, -1, 10, 1), (0, -.5, 5, 1), (-1, -1, 10, 1), (1, -1, 10, 1)]
    reference = evaluate(points)
    assert reference.support_bins == 2 and reference.lidar_z_m is None
    assert reference.rejected_counts["same_pixel_occluded"] == 1
    assert reference.rejected_counts["outside_target_cuboid"] == 1
    assert {p.original_index for p in reference.accepted_bins} == {2, 3}


def test_nearest_target_return_and_exact_depth_index_tie():
    points = [(0, -1, 10, 1), (0, -.95, 9.5, 1), (0, -.95, 9.5, .5),
              (-1, -1, 10, 1), (1, -1, 10, 1)]
    reference = evaluate(points)
    assert reference.status == "supported" and reference.support_bins == 3
    assert {p.original_index for p in reference.accepted_bins} == {1, 3, 4}
    assert reference.rejected_counts["same_pixel_occluded"] == 2
    assert reference.lidar_z_m == 10


def test_other_annotated_object_in_front_blocks_same_pixel():
    foreground = annotation(1, location=(0, 0, 5), hwl=(1, 1, 2))
    points = [(0, -1, 10, 1), (0, -.5, 5, 1),
              (-1, -1, 10, 1), (1, -1, 10, 1)]
    reference = evaluate(points, annotations=(annotation(), foreground))
    assert reference.rejected_counts["same_pixel_occluded"] == 1
    assert reference.support_bins == 2
    assert {p.original_index for p in reference.accepted_bins} == {2, 3}


def test_competing_annotation_and_two_2d_gates():
    target = annotation()
    competitor = annotation(1, location=(0, 0, 10), hwl=(2, 2, 2), box=(0, 0, 20, 20))
    overlapped = evaluate([(0, -1, 10, 1)], annotations=(target, competitor))
    assert overlapped.rejected_counts["competing_cuboid"] == 1
    annotated_gate = evaluate([(1, -1, 10, 1)], annotations=(annotation(box=(0, 0, 10, 20)),))
    assert annotated_gate.rejected_counts["outside_annotation_box_2d"] == 1
    detection_gate = evaluate([(1, -1, 10, 1)], frozen=match(box=(0, 0, 10, 20)))
    assert detection_gate.rejected_counts["outside_detection_box_2d"] == 1


def test_frozen_same_class_and_provenance_are_enforced():
    with pytest.raises(ValueError, match="same-class"):
        FrozenObjectMatch("000001", 0, 7, "Car", "Pedestrian", BoxXYXY(0, 0, 20, 20), 9, 11)
    result = evaluate([(0, -1, 10, 1)] * 3)
    assert result.status == "unsupported_insufficient_bins"
    projection = CalibratedProjector(calib(), (20, 20), calibration_sha256=SHA).project(synthetic_scan([]))
    with pytest.raises(ValueError, match="provenance"):
        associate_object(projection, (annotation(),), match(),
                         SourceDigests("b" * 64, SHA, protocol().sha256, SHA), protocol())
    with pytest.raises(ValueError, match="class/status"):
        associate_object(projection, (annotation(cls="Pedestrian"),), match(),
                         SourceDigests(SHA, SHA, protocol().sha256, SHA), protocol())


def test_association_determinism_and_count_partition():
    points = [(-1, -1, 10, 1), (0, -1, 10, 1), (1, -1, 10, 1), (5, 0, 10, 1)]
    first = evaluate(points).to_dict()
    second = evaluate(points).to_dict()
    assert first == second
    assert sum(first["rejected_counts"].values()) + first["support_bins"] == first["projected_valid_rows"]
    assert first["scan_total_rows"] == 4
