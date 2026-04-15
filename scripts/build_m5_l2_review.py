"""Build and verify a small M5-L2 pilot review archive without raw scans or models."""

from hashlib import sha256
import json
from pathlib import Path
import zipfile


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/kitti_m5_l2"
FILES = (
    "STATUS.md", "NOTES.md",
    "configs/kitti_lidar_protocol_v1.json",
    "configs/kitti_split_manifest_v1.json",
    "configs/kitti_eval_v1.json",
    "src/perception_stack/kitti_lidar/__init__.py",
    "src/perception_stack/kitti_lidar/real_data.py",
    "tests/test_kitti_lidar_real_data.py",
    "scripts/run_m5_l2_geometry.py",
    "scripts/run_m5_l2_pilot.py",
    "scripts/plot_m5_l2_real_pilot.py",
    "scripts/verify_m5_l2_preservation.py",
    "scripts/build_m5_l2_review.py",
    "outputs/kitti_m5_l2_preflight/archive_inventory.json",
    "outputs/kitti_m5_l2_preflight/extraction_manifest.json",
    "outputs/kitti_m5_l2/pilot/report.md",
    "outputs/kitti_m5_l2/pilot/geometry_provenance.json",
    "outputs/kitti_m5_l2/pilot/pilot_summary.json",
    "outputs/kitti_m5_l2/pilot/object_records.json",
    "outputs/kitti_m5_l2/pilot/frame_records.json",
    "outputs/kitti_m5_l2/pilot/detector_geometry.json",
    "outputs/kitti_m5_l2/pilot/visual_manifest.json",
    "outputs/kitti_m5_l2/pilot/preservation.json",
    "outputs/kitti_m5_l2/pilot/verification.json",
    "outputs/kitti_m5_l2/pilot/visuals/projection_contact_sheet.png",
    "outputs/kitti_m5_l2/pilot/visuals/projection_001558.png",
    "outputs/kitti_m5_l2/pilot/visuals/projection_006498.png",
    "outputs/kitti_m5_l2/pilot/visuals/object_association_006612_ref01.png",
    "outputs/kitti_m5_l2/pilot/visuals/valid_sparse_003601_ref04.png",
    "outputs/kitti_m5_l2/pilot/visuals/occlusion_uncertainty_005465_ref04.png",
)


def main() -> None:
    if len(FILES) != len(set(FILES)):
        raise ValueError("Duplicate review member")
    if any(name.startswith(("datasets/", "models/", ".venv/", "work/")) for name in FILES):
        raise ValueError("Forbidden large or private review member")
    entries = []
    for name in FILES:
        path = REPO / name
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        entries.append({"path": name, "bytes": len(payload), "sha256": sha256(payload).hexdigest()})
    manifest = {"scope": "M5-L2 16-frame pilot review; no raw scans, checkpoint or environment",
                "files": entries}
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    review = OUT / "m5_l2_pilot_review.zip"
    with zipfile.ZipFile(review, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for entry in entries:
            archive.write(REPO / entry["path"], arcname=entry["path"])
        archive.writestr("review_manifest.json", manifest_payload)
    with zipfile.ZipFile(review) as archive:
        if archive.testzip() is not None:
            raise ValueError("Review ZIP failed CRC validation")
        if set(archive.namelist()) != set(FILES) | {"review_manifest.json"}:
            raise ValueError("Review ZIP member list changed")
        for entry in entries:
            if sha256(archive.read(entry["path"])).hexdigest() != entry["sha256"]:
                raise ValueError(f"Review member SHA-256 differs: {entry['path']}")
        if archive.read("review_manifest.json") != manifest_payload:
            raise ValueError("Review manifest bytes changed")
    digest = sha256(review.read_bytes()).hexdigest()
    (OUT / "m5_l2_pilot_review.zip.sha256").write_text(f"{digest}  {review.name}\n", encoding="ascii")
    (OUT / "review_manifest.json").write_bytes(manifest_payload)
    print(json.dumps({"review_zip": str(review), "bytes": review.stat().st_size,
                      "sha256": digest, "member_count": len(entries) + 1}, indent=2))


if __name__ == "__main__":
    main()
