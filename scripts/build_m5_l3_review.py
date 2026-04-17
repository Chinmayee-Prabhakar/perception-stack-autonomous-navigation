"""Package and hash bounded M5-L3 review evidence without raw scan/model assets."""

from hashlib import sha256
import json
from pathlib import Path
import zipfile


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/kitti_m5_l3/heldout"
SOURCE = (
    "STATUS.md", "NOTES.md", "configs/kitti_lidar_protocol_v1.json",
    "configs/kitti_split_manifest_v1.json", "configs/kitti_eval_v1.json",
    "src/perception_stack/kitti_lidar/__init__.py",
    "src/perception_stack/kitti_lidar/association.py",
    "src/perception_stack/kitti_lidar/geometry.py",
    "src/perception_stack/kitti_lidar/protocol.py",
    "src/perception_stack/kitti_lidar/scan.py",
    "src/perception_stack/kitti_lidar/real_data.py",
    "src/perception_stack/kitti_lidar/heldout_statistics.py",
    "scripts/run_m5_l2_geometry.py", "scripts/plot_m5_l2_real_pilot.py",
    "scripts/extract_m5_l3_heldout_scans.py", "scripts/run_m5_l3_geometry.py",
    "scripts/run_m5_l3_evaluation.py", "scripts/run_m5_l3_statistics.py",
    "scripts/build_m5_l3_source_manifest.py",
    "scripts/plot_m5_l3_heldout.py", "scripts/verify_m5_l3_preservation.py",
    "scripts/record_m5_l3_verification.py",
    "scripts/build_m5_l3_review.py", "tests/test_kitti_lidar.py",
    "tests/test_kitti_lidar_real_data.py", "tests/test_kitti_lidar_heldout_statistics.py",
)
EVIDENCE = (
    "extraction_preflight.json", "extraction_manifest.json", "geometry_provenance.json",
    "source_identity_manifest.json",
    "detector_geometry.json", "frame_records.json", "drive_mapping.json",
    "drive_overlap_validation.json",
    "object_records.json", "paired_evaluation.csv", "bootstrap_draws.json",
    "coverage_summary.json", "statistical_summary.json", "failure_cases.json",
    "visual_manifest.json", "preservation.json", "verification.json", "report.md",
    "focused_tests.xml", "offline_tests.xml", "integration_tests.xml",
)


def main() -> None:
    visual = json.loads((OUT / "visual_manifest.json").read_text(encoding="utf-8"))
    visual_paths = [visual["contact_sheet"]["path"]]
    visual_paths += [row["path"] for row in visual["full_frame_projection_artifacts"]]
    visual_paths += [row["path"] for row in visual["object_artifacts"]]
    names = list(SOURCE) + [f"outputs/kitti_m5_l3/heldout/{name}" for name in EVIDENCE] + visual_paths
    if len(names) != len(set(names)):
        raise ValueError("Duplicate review member")
    if any(name.startswith(("datasets/", "models/", ".venv/", "work/")) for name in names):
        raise ValueError("Forbidden raw or environment member")
    entries = []
    for name in names:
        path = REPO / name
        if not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        entries.append({"path": name, "bytes": len(data), "sha256": sha256(data).hexdigest()})
    manifest = {"scope": "M5-L3 frozen held-out KITTI Object sparse LiDAR review; no raw scans or model checkpoints",
                "files": entries}
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    review = OUT / "m5_l3_heldout_review.zip"
    with zipfile.ZipFile(review, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for row in entries:
            archive.write(REPO / row["path"], arcname=row["path"])
        archive.writestr("review_manifest.json", payload)
    with zipfile.ZipFile(review) as archive:
        if archive.testzip() is not None:
            raise ValueError("Review ZIP CRC failure")
        if set(archive.namelist()) != set(names) | {"review_manifest.json"}:
            raise ValueError("Review ZIP file list differs")
        for row in entries:
            if sha256(archive.read(row["path"])).hexdigest() != row["sha256"]:
                raise ValueError(f"Review ZIP member digest mismatch: {row['path']}")
    digest = sha256(review.read_bytes()).hexdigest()
    (OUT / "m5_l3_heldout_review.zip.sha256").write_text(
        f"{digest}  {review.name}\n", encoding="ascii")
    (OUT / "review_manifest.json").write_bytes(payload)
    print(json.dumps({"review_zip": str(review), "bytes": review.stat().st_size,
                      "sha256": digest, "member_count": len(entries) + 1}, indent=2))


if __name__ == "__main__":
    main()
