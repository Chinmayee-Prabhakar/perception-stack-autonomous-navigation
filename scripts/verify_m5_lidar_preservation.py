"""Read-only comparison of frozen M5 outputs with the accepted local review ZIP."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
review = repo / "outputs/perception_stack_m5_kitti_review.zip"
out = repo / "outputs/kitti_m5_l1"
out.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(review) as archive:
    assert archive.testzip() is None
    frozen_names = [name for name in archive.namelist()
                    if name.startswith("outputs/kitti_m5/") and not name.endswith("/")]
    mismatches = []
    for name in frozen_names:
        local = repo / name
        if not local.is_file() or hashlib.sha256(local.read_bytes()).digest() != hashlib.sha256(archive.read(name)).digest():
            mismatches.append(name)

changed_tracked = subprocess.check_output(
    ["git", "diff", "HEAD", "--name-only"], cwd=repo, text=True, encoding="utf-8").splitlines()
unexpected = sorted(set(changed_tracked) - {"STATUS.md", "NOTES.md"})
result = {
    "frozen_m5_review_zip_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
    "frozen_m5_output_files_compared": len(frozen_names),
    "frozen_m5_output_mismatches": mismatches,
    "tracked_files_modified_against_head": changed_tracked,
    "unexpected_tracked_changes": unexpected,
    "frozen_m5_m6_d1_tracked_files_unchanged": not unexpected,
}
assert not mismatches and not unexpected
(out / "preservation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
