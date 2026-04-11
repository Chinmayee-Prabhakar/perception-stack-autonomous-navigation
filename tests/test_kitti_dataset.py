"""Offline KITTI object-format fixtures; never fetches data or checkpoints."""

import numpy as np
from PIL import Image
import pytest
from zipfile import ZipFile
import importlib.util
from pathlib import Path

from perception_stack.kitti_eval.dataset import (
    KittiObjectDataset, parse_calibration, parse_labels,
)


CAR = "Car 0.25 2 0.1 1 2 5 6 1.5 1.6 4.0 1 2 10 0"
DONTCARE = "DontCare -1 -1 -10 0 0 3 4 -1 -1 -1 -1000 -1000 -1000 -10"
P2 = "P2: 2 0 1 4 0 2 1 6 0 0 1 0\n"


def test_label_fields_status_and_empty():
    assert parse_labels("") == ()
    car, ignored, unknown, bad_z, bad_size = parse_labels(
        "\n".join((CAR, DONTCARE, CAR.replace("Car", "Alien", 1),
                   CAR.replace("1 2 10 0", "1 2 0 0"),
                   CAR.replace("1.5 1.6 4.0", "0 1.6 4.0"))))
    assert car.object_id == 0 and car.class_name == "Car"
    assert car.truncated == 0.25 and car.occluded == 2
    assert car.box.xyxy == (1, 2, 5, 6)
    assert car.dimensions_hwl_m == (1.5, 1.6, 4.0)
    assert car.location_xyz_m == (1, 2, 10)
    assert car.reference_z_m == 10
    assert ignored.validity_status == "dontcare" and ignored.reference_z_m is None
    assert unknown.validity_status == "unsupported_category"
    assert bad_z.validity_status == "invalid_reference_z"
    assert bad_size.validity_status == "invalid_dimensions"


@pytest.mark.parametrize("line", [
    "Car 0 0", CAR + " 0", CAR.replace("10 0", "bad 0"),
    CAR.replace("5 6", "0 6"), CAR.replace("0.25", "nan"),
    CAR.replace("0.25", "1.5"), CAR.replace(" 2 0.1", " 8 0.1"),
    "   ",
])
def test_malformed_annotation_rejected(line):
    with pytest.raises(ValueError):
        parse_labels(line)


def test_full_p2_translation_and_optional_calibration():
    calibration = parse_calibration(P2 +
        "R0_rect: 1 0 0 0 1 0 0 0 1\n" +
        "Tr_velo_to_cam: 1 0 0 1 0 1 0 2 0 0 1 3\n")
    assert calibration.project_rectified_point((1, 2, 2)) == (4, 6)
    assert calibration.p2[0, 3] == 4 and calibration.r0_rect.shape == (3, 3)
    assert calibration.tr_velo_to_cam.shape == (3, 4)
    assert not calibration.p2.flags.writeable
    with pytest.raises(ValueError, match="behind"):
        calibration.project_rectified_point((0, 0, -1))


@pytest.mark.parametrize("text", [
    "", "P2: 1 2 3", P2 + P2,
    P2.replace("2 0 1 4", "nan 0 1 4"),
    "P2: 0 0 0 0 0 0 0 0 0 0 1 0",
    P2 + "R0_rect: 1 2\n",
    P2 + "R0_rect: 2 0 0 0 1 0 0 0 1\n",
    P2 + "Tr_velo_to_cam: 0 0 0 0 0 0 0 0 0 0 0 0\n",
])
def test_bad_calibration_rejected(text):
    with pytest.raises(ValueError):
        parse_calibration(text)


def test_dataset_layout_subset_extraction_and_corruption(tmp_path):
    root = tmp_path / "kitti"
    for leaf in ("image_2", "label_2", "calib"):
        (root / "training" / leaf).mkdir(parents=True)
    labels = root / "training" / "label_2"
    calib = root / "training" / "calib"
    images = root / "training" / "image_2"
    for frame_id in ("000001", "000002"):
        (labels / f"{frame_id}.txt").write_text(CAR + "\n", encoding="utf-8")
        (calib / f"{frame_id}.txt").write_text(P2, encoding="utf-8")
    dataset = KittiObjectDataset(root)
    assert dataset.frame_ids() == ("000001", "000002")
    assert dataset.validate_source_records(dataset.frame_ids()) == {
        "validated_frames": 2, "annotation_status_counts": {"valid": 2}}
    digest = dataset.source_label_calibration_sha256(dataset.frame_ids())
    assert len(digest) == 64
    (labels / "000002.txt").write_text(CAR.replace("10 0", "11 0") + "\n", encoding="utf-8")
    assert dataset.source_label_calibration_sha256(dataset.frame_ids()) != digest
    with pytest.raises(FileNotFoundError):
        dataset.frame_ids(require_images=True)
    with pytest.raises(FileNotFoundError):
        dataset.load("000001")
    Image.fromarray(np.zeros((8, 9, 3), np.uint8)).save(images / "000001.png")
    sample = dataset.load("000001")
    assert sample.frame.frame_id == "000001" and sample.frame.size == (9, 8)
    assert sample.annotations[0].reference_z_m == 10
    (images / "000002.png").write_bytes(b"not a png")
    with pytest.raises(Exception):
        dataset.load("000002")
    with pytest.raises(ValueError):
        dataset.load("../test")
    (calib / "000001.txt").unlink()
    with pytest.raises(FileNotFoundError):
        dataset.frame_ids()


def test_test_partition_not_accepted(tmp_path):
    (tmp_path / "testing" / "image_2").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        KittiObjectDataset(tmp_path)


def test_subset_extractor_uses_only_constructed_training_members(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_kitti_subset.py"
    spec = importlib.util.spec_from_file_location("prepare_kitti_subset", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive_path = tmp_path / "images.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("training/image_2/000001.png", b"selected")
        archive.writestr("training/image_2/000002.png", b"unselected")
        archive.writestr("../outside.png", b"unsafe")
    destination = tmp_path / "images"
    with ZipFile(archive_path) as archive:
        assert module.training_ids(archive, "image_2", "png") == ("000001", "000002")
        assert module.extract_exact_members(archive, "image_2", "png", ("000001",), destination) == 1
        assert module.extract_exact_members(archive, "image_2", "png", ("000001",), destination) == 0
        with pytest.raises(ValueError):
            module.extract_exact_members(archive, "image_2", "png", ("../outside",), destination)
        (destination / "000001.png").write_bytes(b"changed")
        with pytest.raises(ValueError, match="differs"):
            module.extract_exact_members(archive, "image_2", "png", ("000001",), destination)
    assert not (tmp_path / "outside.png").exists()
    assert not (destination / "000002.png").exists()
