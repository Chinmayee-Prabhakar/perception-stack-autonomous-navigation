"""Strict KITTI object *training* image_2/label_2/calib loader."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

import numpy as np

from perception_stack.frame import Frame
from perception_stack.results import BoxXYXY


OFFICIAL_CLASSES = frozenset({"Car", "Van", "Truck", "Pedestrian", "Person_sitting",
                              "Cyclist", "Tram", "Misc", "DontCare"})
_FRAME_ID = re.compile(r"[0-9]{6}\Z")


@dataclass(frozen=True)
class KittiAnnotation:
    object_id: int  # zero-based line index in label_2/<frame>.txt
    class_name: str
    truncated: float
    occluded: int
    alpha: float
    box: BoxXYXY
    dimensions_hwl_m: tuple[float, float, float]
    location_xyz_m: tuple[float, float, float]
    rotation_y: float
    validity_status: str

    @property
    def reference_z_m(self) -> float | None:
        return self.location_xyz_m[2] if self.validity_status == "valid" else None


@dataclass(frozen=True)
class KittiCalibration:
    """P2 projects rectified reference-camera 3D points into image_2."""

    p2: np.ndarray
    r0_rect: np.ndarray | None
    tr_velo_to_cam: np.ndarray | None

    def __post_init__(self) -> None:
        for name, shape in (("p2", (3, 4)), ("r0_rect", (3, 3)),
                            ("tr_velo_to_cam", (3, 4))):
            value = getattr(self, name)
            if value is None:
                continue
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.shape != shape or not np.isfinite(matrix).all():
                raise ValueError(f"{name} must be finite and have shape {shape}")
            matrix = matrix.copy()
            matrix.flags.writeable = False
            object.__setattr__(self, name, matrix)
        if self.p2[0, 0] <= 0 or self.p2[1, 1] <= 0 or abs(np.linalg.det(self.p2[:, :3])) < 1e-12:
            raise ValueError("P2 must have positive focal terms and nonsingular left block")
        if not np.allclose(self.p2[2, :3], (0, 0, 1), atol=1e-6):
            raise ValueError("Unsupported P2 projective denominator convention")
        for name, rotation in (("R0_rect", self.r0_rect),
                               ("Tr_velo_to_cam", self.tr_velo_to_cam[:, :3] if self.tr_velo_to_cam is not None else None)):
            if rotation is not None and (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
                                         or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)):
                raise ValueError(f"{name} rotation must be proper and orthonormal")

    def project_rectified_point(self, xyz_m: tuple[float, float, float]) -> tuple[float, float]:
        """Use the full 3x4 P2, including its translation; no backprojection claim."""
        xyz = np.asarray(xyz_m, dtype=np.float64)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Point must be finite XYZ")
        homogeneous = np.r_[xyz, 1.0]
        uvw = self.p2 @ homogeneous
        if not np.isfinite(uvw).all() or uvw[2] <= 0:
            raise ValueError("Point is behind the image_2 projection plane")
        return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])


def parse_labels(text: str) -> tuple[KittiAnnotation, ...]:
    records = []
    for line_no, raw in enumerate(text.splitlines()):
        fields = raw.split()
        if not fields:
            raise ValueError(f"Blank KITTI annotation line {line_no + 1}")
        if len(fields) != 15:
            raise ValueError(f"KITTI annotation line {line_no + 1} requires exactly 15 fields")
        cls = fields[0]
        try:
            truncated = float(fields[1])
            occluded_raw = int(fields[2])
            values = tuple(float(value) for value in fields[3:])
        except ValueError as error:
            raise ValueError(f"Invalid numeric KITTI annotation line {line_no + 1}") from error
        if not np.isfinite((truncated, *values)).all():
            raise ValueError(f"Nonfinite KITTI annotation line {line_no + 1}")
        if not 0 <= truncated <= 1 or occluded_raw not in (0, 1, 2, 3):
            if cls != "DontCare":
                raise ValueError(f"Invalid truncation/occlusion on line {line_no + 1}")
        try:
            box = BoxXYXY(*values[1:5])
        except ValueError as error:
            raise ValueError(f"Invalid 2D box on KITTI annotation line {line_no + 1}") from error
        dimensions = values[5:8]
        location = values[8:11]
        if cls == "DontCare":
            status = "dontcare"
        elif cls not in OFFICIAL_CLASSES:
            status = "unsupported_category"
        elif any(d <= 0 for d in dimensions):
            status = "invalid_dimensions"
        elif location[2] <= 0:
            status = "invalid_reference_z"
        else:
            status = "valid"
        records.append(KittiAnnotation(line_no, cls, truncated, occluded_raw, values[0],
                                       box, dimensions, location, values[11], status))
    return tuple(records)


def parse_calibration(text: str) -> KittiCalibration:
    matrices: dict[str, np.ndarray] = {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        if ":" not in raw:
            raise ValueError("Malformed KITTI calibration line")
        key, payload = raw.split(":", 1)
        key = key.strip()
        if key in matrices:
            raise ValueError(f"Duplicate calibration field {key}")
        if key not in {"P2", "R0_rect", "Tr_velo_to_cam"}:
            continue
        fields = payload.split()
        shape = (3, 3) if key == "R0_rect" else (3, 4)
        if len(fields) != shape[0] * shape[1]:
            raise ValueError(f"{key} requires {shape[0] * shape[1]} numbers")
        try:
            matrices[key] = np.asarray([float(value) for value in fields], dtype=np.float64).reshape(shape)
        except ValueError as error:
            raise ValueError(f"Invalid numeric {key}") from error
    if "P2" not in matrices:
        raise ValueError("KITTI object calibration requires P2")
    return KittiCalibration(matrices["P2"], matrices.get("R0_rect"), matrices.get("Tr_velo_to_cam"))


@dataclass(frozen=True)
class KittiObjectFrame:
    frame: Frame
    annotations: tuple[KittiAnnotation, ...]
    calibration: KittiCalibration


def _find_training_dir(root: Path, leaf: str, archive: str) -> Path:
    candidates = [root / "training" / leaf, root / leaf,
                  root / archive / "training" / leaf]
    found = [path for path in candidates if path.is_dir()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one KITTI object training/{leaf} directory under {root}; found {found}")
    return found[0]


class KittiObjectDataset:
    """Never searches KITTI's unlabeled test partition or other projects."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.images = _find_training_dir(self.root, "image_2", "data_object_image_2")
        self.labels = _find_training_dir(self.root, "label_2", "data_object_label_2")
        self.calibrations = _find_training_dir(self.root, "calib", "data_object_calib")

    def frame_ids(self, *, require_images: bool = False) -> tuple[str, ...]:
        """Enumerate labeled IDs; allow image extraction after the split is frozen."""
        ids = tuple(sorted(path.stem for path in self.labels.glob("*.txt")))
        if not ids or any(_FRAME_ID.fullmatch(frame_id) is None for frame_id in ids):
            raise ValueError("KITTI training label IDs must be nonempty six-digit stems")
        for frame_id in ids:
            if require_images and not (self.images / f"{frame_id}.png").is_file():
                raise FileNotFoundError(f"Missing image_2/{frame_id}.png")
            if not (self.calibrations / f"{frame_id}.txt").is_file():
                raise FileNotFoundError(f"Missing calib/{frame_id}.txt")
        return ids

    def source_label_calibration_sha256(self, frame_ids: tuple[str, ...]) -> str:
        """Fingerprint frozen source annotations/calibration without needing all PNGs."""
        digest = sha256()
        for frame_id in frame_ids:
            if _FRAME_ID.fullmatch(frame_id) is None:
                raise ValueError("Invalid source frame ID")
            for kind, folder in (("label_2", self.labels), ("calib", self.calibrations)):
                path = folder / f"{frame_id}.txt"
                payload = path.read_bytes()
                digest.update(f"{frame_id}/{kind}:{len(payload)}:".encode("ascii"))
                digest.update(payload)
        return digest.hexdigest()

    def validate_source_records(self, frame_ids: tuple[str, ...]) -> dict:
        """Validate every frozen label/calibration file before any inference."""
        statuses: dict[str, int] = {}
        for frame_id in frame_ids:
            if _FRAME_ID.fullmatch(frame_id) is None:
                raise ValueError("Invalid source frame ID")
            annotations = parse_labels((self.labels / f"{frame_id}.txt").read_text(encoding="utf-8"))
            parse_calibration((self.calibrations / f"{frame_id}.txt").read_text(encoding="utf-8"))
            for annotation in annotations:
                statuses[annotation.validity_status] = statuses.get(annotation.validity_status, 0) + 1
        return {"validated_frames": len(frame_ids), "annotation_status_counts": dict(sorted(statuses.items()))}

    def load(self, frame_id: str) -> KittiObjectFrame:
        if _FRAME_ID.fullmatch(frame_id) is None:
            raise ValueError("KITTI object training frame ID must contain six digits")
        image_path = self.images / f"{frame_id}.png"
        label_path = self.labels / f"{frame_id}.txt"
        calibration_path = self.calibrations / f"{frame_id}.txt"
        for path in (image_path, label_path, calibration_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing KITTI frame component: {path}")
        frame = Frame.from_image(image_path)
        if frame.frame_id != frame_id:
            raise ValueError("KITTI image identifier mismatch")
        annotations = parse_labels(label_path.read_text(encoding="utf-8"))
        calibration = parse_calibration(calibration_path.read_text(encoding="utf-8"))
        return KittiObjectFrame(frame, annotations, calibration)
