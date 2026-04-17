"""Verify frozen M5 and accepted M5-L2 outputs remain byte-identical."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import zipfile


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/kitti_m5_l3/heldout"
M5 = REPO / "outputs/perception_stack_m5_kitti_review.zip"
M5_L2 = REPO / "outputs/kitti_m5_l2/m5_l2_pilot_review.zip"
PROTOCOL_SHA256 = "02451176def64af5ea83adc3c63df0ee37351176b008ec947f7da20c3f1f75fc"


def _compare(archive_path: Path, prefix: str) -> dict:
    mismatches = []
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Accepted review ZIP CRC failed: {archive_path}")
        names = sorted(name for name in archive.namelist() if name.startswith(prefix) and
                       not name.endswith("/"))
        for name in names:
            local = REPO / name
            if not local.is_file() or sha256(local.read_bytes()).digest() != sha256(archive.read(name)).digest():
                mismatches.append(name)
    return {"compared": len(names), "mismatches": mismatches}


def main() -> None:
    m5 = _compare(M5, "outputs/kitti_m5/")
    m5_l2 = _compare(M5_L2, "outputs/kitti_m5_l2/pilot/")
    preflight = _compare(M5_L2, "outputs/kitti_m5_l2_preflight/")
    changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=REPO,
                                      text=True, encoding="utf-8").splitlines()
    unexpected = sorted(set(changed) - {"STATUS.md", "NOTES.md"})
    protocol = sha256((REPO / "configs/kitti_lidar_protocol_v1.json").read_bytes()).hexdigest()
    result = {"frozen_m5": m5, "accepted_m5_l2_pilot": m5_l2,
              "accepted_m5_l2_preflight": preflight, "tracked_changes": changed,
              "unexpected_tracked_changes": unexpected, "protocol_v1_sha256": protocol}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preservation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8")
    if m5["compared"] != 309 or m5["mismatches"] or m5_l2["mismatches"] or preflight["mismatches"]:
        raise ValueError("Accepted M5 or M5-L2 outputs differ")
    if unexpected or protocol != PROTOCOL_SHA256:
        raise ValueError("Tracked accepted scientific source or LiDAR protocol differs")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
