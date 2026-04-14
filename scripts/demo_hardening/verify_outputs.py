"""Verify D1-R2 science and presentation, or the historical D1 presentation."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "demo_hardening"
IDS = ("006263", "000201", "006781", "003783", "005481", "000185", "007200")
SCENES = (*IDS, "006781_calibrated")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required evidence missing: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalized_scene(path: Path) -> dict:
    value = load_json(path)
    # These are the only source-location and generated-artifact fields.
    value.pop("source_image", None)
    value.pop("artifacts", None)
    return value


def json_digest(value: dict) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8"))


def image_record(path: Path, mode: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required image missing: {path}")
    with Image.open(path) as source:
        image = source.convert(mode)
        pixels = image.tobytes()
        if mode == "L" and set(pixels) - {0, 255}:
            raise ValueError(f"Mask is not binary 0/255: {path}")
        return {"size": list(image.size), "pixel_sha256": digest(pixels),
                "foreground_pixels": pixels.count(255) if mode == "L" else None}


def snapshot(output: Path) -> dict:
    selection = load_json(CONFIG / "selection.json")["frames"]
    inputs = {}
    for record in selection:
        fid = record["frame_id"]
        path = output / "inputs" / f"{fid}.png"
        if not path.is_file() or file_sha256(path) != record["image_sha256"]:
            raise ValueError(f"Approved input image missing or changed: {fid}")
        with Image.open(path) as image:
            if list(image.size) != record["image_size"]:
                raise ValueError(f"Approved input dimensions changed: {fid}")
        inputs[fid] = {"sha256": record["image_sha256"], "size": record["image_size"]}
    scenes = {}
    for fid in SCENES:
        folder = output / "scenes" / fid
        scene = normalized_scene(folder / "scene.json")
        association = load_json(folder / "road_association.json")
        overlay = image_record(folder / "overlay.png", "RGB")
        road = scene["road_segmentation"]
        row = {"scientific_scene_sha256": json_digest(scene),
               "road_association_sha256": json_digest(association),
               "overlay": {key: overlay[key] for key in ("size", "pixel_sha256")},
               "detections": len(scene["objects"]),
               "drivable_pixels": road["drivable_area_pixels"],
               "lane_pixels": road["lane_marking_pixels"],
               "adjacency_states": [obj["road_association"]["evidence_status"]
                                    for obj in scene["objects"]]}
        if fid in IDS:
            row["drivable_mask"] = image_record(folder / "drivable_mask.png", "L")
            row["lane_mask"] = image_record(folder / "lane_mask.png", "L")
            if row["drivable_mask"]["foreground_pixels"] != row["drivable_pixels"]:
                raise ValueError(f"Drivable mask count differs from scene JSON: {fid}")
            if row["lane_mask"]["foreground_pixels"] != row["lane_pixels"]:
                raise ValueError(f"Lane mask count differs from scene JSON: {fid}")
        else:
            plot = image_record(folder / "camera_xz.png", "RGB")
            row["camera_xz"] = {key: plot[key] for key in ("size", "pixel_sha256")}
            if scene["calibration"]["status"] != "available_rectified_intrinsics":
                raise ValueError("Calibrated scene lacks verified intrinsics status")
        scenes[fid] = row
    return {"schema": "demo_hardening_expected_v1", "inputs": inputs, "scenes": scenes}


def compare_scientific(output: Path, baseline: Path, label: str) -> dict:
    if not baseline.is_dir():
        raise FileNotFoundError(f"{label} directory missing: {baseline}")
    for fid in SCENES:
        current = output / "scenes" / fid
        prior = baseline / "scenes" / fid
        if normalized_scene(current / "scene.json") != normalized_scene(prior / "scene.json"):
            raise ValueError(f"{label} scientific scene differs: {fid}")
        if load_json(current / "road_association.json") != load_json(prior / "road_association.json"):
            raise ValueError(f"{label} road association differs: {fid}")
    return {"scientific_scenes_equal": len(SCENES), "road_associations_equal": len(SCENES)}


def compare_d0(output: Path, baseline: Path) -> dict:
    hashes = load_json(CONFIG / "d0_file_hashes.json")
    missing = [name for name in hashes if not (baseline / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Historical D0 integrity check cannot run; missing {len(missing)} files, first: {missing[0]}")
    changed = [name for name, expected in hashes.items() if file_sha256(baseline / name) != expected]
    if changed:
        raise ValueError(f"Historical D0 file hash mismatch: {changed[:3]}")
    return {"historical_d0_files_sha256_verified": len(hashes),
            **compare_scientific(output, baseline, "D0")}


def compare_scientific_snapshot(actual: dict, expected: dict) -> None:
    """Only artifact paths and presentation pixels may differ from accepted D1."""
    if actual["schema"] != expected["schema"] or actual["inputs"] != expected["inputs"]:
        raise ValueError("Accepted D1 input or snapshot schema invariant differs")
    if actual["scenes"].keys() != expected["scenes"].keys():
        raise ValueError("Accepted D1 scene selection differs")
    for fid in SCENES:
        current = {key: value for key, value in actual["scenes"][fid].items()
                   if key not in ("overlay", "camera_xz")}
        prior = {key: value for key, value in expected["scenes"][fid].items()
                 if key not in ("overlay", "camera_xz")}
        if current != prior:
            raise ValueError(f"Accepted D1 scientific scene, association or mask invariant differs: {fid}")
    if actual["scenes"]["006781_calibrated"]["camera_xz"] != expected["scenes"]["006781_calibrated"]["camera_xz"]:
        raise ValueError("Accepted D1 calibrated camera X-Z plot differs")


def verify(output: Path, accepted_d1: Path | None = None,
           historical_d0: Path | None = None,
           presentation_version: str = "r2") -> dict:
    if presentation_version not in ("d1", "r2"):
        raise ValueError(f"Unsupported presentation version: {presentation_version}")
    actual = snapshot(output)
    historical_path = CONFIG / "expected_results.json"
    expected = load_json(historical_path)
    compare_scientific_snapshot(actual, expected)
    if presentation_version == "d1":
        if actual != expected:
            raise ValueError("Historical accepted D1 presentation pixel hashes differ")
        presentation_result = "historical accepted D1 overlay pixels match"
    else:
        manifest = load_json(CONFIG / "presentation_r2.json")
        if manifest.get("schema") != "demo_hardening_presentation_r2_v1":
            raise ValueError("D1-R2 presentation manifest schema differs")
        if manifest.get("accepted_d1_manifest_sha256") != file_sha256(historical_path):
            raise ValueError("D1-R2 presentation manifest is not bound to the accepted D1 scientific baseline")
        if manifest.get("overlays", {}).keys() != actual["scenes"].keys():
            raise ValueError("D1-R2 presentation scene selection differs")
        for fid in SCENES:
            if actual["scenes"][fid]["overlay"] != manifest["overlays"][fid]:
                raise ValueError(f"D1-R2 presentation pixel hash or size differs: {fid}")
        if manifest.get("camera_xz") != actual["scenes"]["006781_calibrated"]["camera_xz"]:
            raise ValueError("D1-R2 calibrated plot pixel hash differs")
        presentation_result = "all eight D1-R2 overlay pixel hashes match; calibrated plot matches D1"
    report = {"presentation_version": presentation_version,
              "accepted_d1_scientific_manifest": "all input, scene, association and mask hashes match; calibrated plot unchanged",
              "presentation_fingerprints": presentation_result,
              "ordinary_scenes_verified": len(IDS), "calibrated_scenes_verified": 1,
              "accepted_d1_directory_comparison": "not_requested; original D1 outputs are optional",
              "historical_d0_comparison": "not_requested; original D0 outputs are optional"}
    if accepted_d1 is not None:
        prior = snapshot(accepted_d1)
        if prior != expected:
            raise ValueError("Original D1 output snapshot differs from historical accepted D1 manifest")
        compare_scientific_snapshot(actual, prior)
        report["accepted_d1_directory_comparison"] = (
            "original D1 inputs, scientific scenes, masks, overlays and plot verified; "
            "current scientific outputs equal")
    if historical_d0 is not None:
        report["historical_d0_comparison"] = compare_d0(output, historical_d0)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accepted-d1-dir", type=Path,
                        help="optional original D1 outputs for direct comparison")
    parser.add_argument("--historical-d0-dir", type=Path,
                        help="optional original D0 outputs; tracked 76-file hash manifest is applied")
    parser.add_argument("--presentation-version", choices=("d1", "r2"), default="r2",
                        help="r2 revised labels (default), or historical accepted D1 pixels")
    parser.add_argument("--report", type=Path, help="optional JSON verification report destination")
    args = parser.parse_args()
    result = verify(args.output_dir, args.accepted_d1_dir, args.historical_d0_dir,
                    args.presentation_version)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
