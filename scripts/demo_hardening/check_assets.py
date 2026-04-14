"""Verify pinned model/configuration bytes and camera-2 metadata without downloads."""

import argparse
import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "demo_hardening"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_calibration(intrinsics: dict, derivation: dict, selection: dict) -> None:
    if not intrinsics.get("rectified") or "image_2" not in intrinsics["provenance"] or "camera-2" not in intrinsics["provenance"]:
        raise ValueError("Calibration is not identified as rectified KITTI image-2 camera-2")
    if derivation["frame_id"] != "006781" or [intrinsics["width"], intrinsics["height"]] != selection["image_size"]:
        raise ValueError("Calibration frame or image size mismatch")
    p = derivation["p2_full"]
    fx, fy, cx, cy = (intrinsics[name] for name in ("fx", "fy", "cx", "cy"))
    expected_k = ((fx, 0, cx), (0, fy, cy), (0, 0, 1))
    for row in range(3):
        for column in range(3):
            if not math.isclose(p[row][column], expected_k[row][column], abs_tol=1e-9):
                raise ValueError("Intrinsics differ from the full P2 left block")
    z = -p[2][3]
    center = (-(p[0][3] + cx * z) / fx, -(p[1][3] + cy * z) / fy, z)
    if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
           for a, b in zip(center, derivation["camera2_center_in_rectified_reference_m"])):
        raise ValueError("Full-P2 camera-2 center derivation mismatch")


def check_assets(calibration_source: Path | None = None) -> dict:
    assets = json.loads((CONFIG / "assets.json").read_text(encoding="utf-8"))
    for relative, expected in assets["files"].items():
        path = REPO / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Pinned asset missing or SHA-256 mismatch: {relative}")
    selection = json.loads((CONFIG / "selection.json").read_text(encoding="utf-8"))
    frame = next(record for record in selection["frames"] if record["frame_id"] == "006781")
    intrinsics = json.loads((CONFIG / "006781_intrinsics.json").read_text(encoding="utf-8"))
    derivation = json.loads((CONFIG / "006781_derivation.json").read_text(encoding="utf-8"))
    check_calibration(intrinsics, derivation, frame)
    source_status = "not_provided; original calibration bytes not independently rechecked"
    if calibration_source is not None:
        if file_sha256(calibration_source) != derivation["source_calibration_sha256"]:
            raise ValueError("Original KITTI calibration source SHA-256 mismatch")
        source_status = "sha256_verified"
    return {"pinned_files_sha256_verified": len(assets["files"]),
            "full_p2_camera2_derivation_verified": True,
            "original_calibration_source": source_status}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-source", type=Path,
                        help="optional approved original KITTI calib/006781.txt for SHA-256 verification")
    args = parser.parse_args()
    print(json.dumps(check_assets(args.calibration_source), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
