"""Create and verify the small M6 review bundle without staging Git files."""

import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/perception_stack_m6_review_final.zip"
FILES = (
    "README.md", "PROJECT_PLAN.md", "REFERENCES.md", "STATUS.md", "NOTES.md",
    "pyproject.toml", "requirements-win-py311.lock",
    "src/perception_stack/fusion/results.py",
    "src/perception_stack/fusion/core.py",
    "src/perception_stack/fusion/source.py",
    "src/perception_stack/scene/__init__.py",
    "src/perception_stack/scene/association.py",
    "src/perception_stack/scene/results.py",
    "src/perception_stack/scene/visualization.py",
    "src/perception_stack/scene/synthetic.py",
    "src/perception_stack/scene_cli.py",
    "scripts/create_synthetic_scene.py", "scripts/package_m6_review.py",
    "tests/test_scene_association.py", "tests/test_scene.py",
    "tests/test_scene_consistency.py",
    "tests/test_fusion_source_consistency.py",
    "outputs/bus_scene.json", "outputs/bus_scene_overlay.png",
    "outputs/bus_scene_road_association.json",
    "outputs/m6_synthetic/synthetic_camera_xz.png",
    "outputs/m6_synthetic/synthetic_overlay.png",
    "outputs/m6_synthetic/synthetic_geometry.json",
    "outputs/m6_synthetic/synthetic_scene.json",
    "outputs/m6_correction_report.md",
)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def main() -> None:
    members = {name: (ROOT / name).read_bytes() for name in FILES}
    members["review/git_diff.patch"] = _git("diff", "--binary").encode("utf-8")
    members["review/git_status.txt"] = (
        "HEAD\n" + _git("log", "-1", "--format=%H %s")
        + "STATUS\n" + _git("status", "--short")
        + "REMOTES\n" + _git("remote", "-v")
        + "STAGED FILES\n" + _git("diff", "--cached", "--name-only")
    ).encode("utf-8")
    manifest = {name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
                for name, content in sorted(members.items())}
    members["review/SHA256SUMS.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in sorted(members.items()):
            archive.writestr(name, content)
    with ZipFile(OUTPUT, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Review ZIP failed its CRC integrity check")
        actual = set(archive.namelist())
        if actual != set(members):
            raise RuntimeError("Review ZIP member list changed")
        stored_manifest = json.loads(archive.read("review/SHA256SUMS.json"))
        for name, expected in stored_manifest.items():
            content = archive.read(name)
            if len(content) != expected["bytes"] or hashlib.sha256(content).hexdigest() != expected["sha256"]:
                raise RuntimeError(f"Review ZIP SHA-256 mismatch: {name}")
    print(f"Verified {len(members)} ZIP members and {len(manifest)} SHA-256 entries: {OUTPUT}")


if __name__ == "__main__":
    main()
