"""Verify frozen M5 outputs while allowing only the scoped M5-L2 worktree edits."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import zipfile


REPO = Path(__file__).resolve().parents[1]
REVIEW = REPO / "outputs/perception_stack_m5_kitti_review.zip"
OUTPUT = REPO / "outputs/kitti_m5_l2/pilot/preservation.json"
ALLOWED_TRACKED = {"STATUS.md", "NOTES.md", "src/perception_stack/kitti_lidar/__init__.py"}


def main() -> None:
    with zipfile.ZipFile(REVIEW) as archive:
        if archive.testzip() is not None:
            raise ValueError("Accepted M5 review ZIP has a bad member")
        frozen = sorted(name for name in archive.namelist()
                        if name.startswith("outputs/kitti_m5/") and not name.endswith("/"))
        mismatches = []
        for name in frozen:
            local = REPO / name
            if not local.is_file() or sha256(local.read_bytes()).digest() != sha256(archive.read(name)).digest():
                mismatches.append(name)
    changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=REPO,
                                      text=True, encoding="utf-8").splitlines()
    unexpected = sorted(set(changed) - ALLOWED_TRACKED)
    protocol_sha = sha256((REPO / "configs/kitti_lidar_protocol_v1.json").read_bytes()).hexdigest()
    split_sha = sha256((REPO / "configs/kitti_split_manifest_v1.json").read_bytes()).hexdigest()
    m5_config_sha = sha256((REPO / "configs/kitti_eval_v1.json").read_bytes()).hexdigest()
    result = {
        "frozen_m5_files_compared": len(frozen), "frozen_m5_mismatches": mismatches,
        "tracked_changes": changed, "unexpected_tracked_changes": unexpected,
        "lidar_protocol_v1_sha256": protocol_sha,
        "frozen_split_manifest_sha256": split_sha,
        "frozen_m5_config_sha256": m5_config_sha,
        "frozen_m5_m6_d1_tracked_source_config_unchanged": not unexpected,
        "legacy_m5_l1_checker_limitation": "Its isolation assertion allows only STATUS.md and NOTES.md; this milestone intentionally exports new LiDAR loader functions.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert len(frozen) == 309 and not mismatches and not unexpected
    assert protocol_sha == "02451176def64af5ea83adc3c63df0ee37351176b008ec947f7da20c3f1f75fc"
    assert split_sha == "91d81d8600c8c66994e64a6aeb32756d7bb86762138d2959633b665c2e8b464b"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
