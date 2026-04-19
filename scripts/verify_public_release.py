"""Check proposed public snapshot hashes, links, and accepted table statistics.

This verifier needs no KITTI data, checkpoints, review ZIP, or network access.
"""

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

from reproduce_public_results import recalculate


REPO = Path(__file__).resolve().parents[1]
DOCS = (
    REPO / "README.md",
    REPO / "PROJECT_PLAN.md",
    REPO / "REFERENCES.md",
    REPO / "docs/PROJECT_REPORT.md",
    REPO / "docs/REPRODUCING_RESULTS.md",
    REPO / "docs/PUBLIC_ARTIFACTS.md",
)
LINK = re.compile(r"!?(?:\[[^]]*\])\(([^)]+)\)")
PRIVATE = re.compile(r"(?i)(?:[a-z]:[/\\]users[/\\][^/\\\s]+|/(?:users|home)/[^/\s]+|documents[/\\]codex[/\\]202[0-9])")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_links() -> int:
    checked = 0
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        if PRIVATE.search(text):
            raise ValueError(f"Private absolute path in proposed public document: {path}")
        for target in LINK.findall(text):
            if target.startswith(("https://", "http://", "mailto:")):
                continue
            local = (path.parent / unquote(target.split("#", 1)[0])).resolve()
            if not local.is_relative_to(REPO) or not local.exists():
                raise FileNotFoundError(f"Broken or external local link: {path}: {target}")
            if local.is_relative_to(REPO / "outputs"):
                raise ValueError(f"Public document links to ignored output: {path}: {target}")
            checked += 1
    return checked


def main() -> None:
    manifest = json.loads((REPO / "results/manifest.json").read_text(encoding="utf-8"))
    for relative, record in manifest["files"].items():
        path = (REPO / relative).resolve()
        if not path.is_relative_to(REPO) or not path.is_file():
            raise FileNotFoundError(f"Missing public artifact: {relative}")
        if path.stat().st_size != record["bytes"] or sha(path) != record["public_sha256"]:
            raise ValueError(f"Public artifact changed: {relative}")
    pdf = REPO / "docs/PROJECT_REPORT.pdf"
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
        raise ValueError("Project report PDF missing or malformed")
    payload = pdf.read_bytes()
    if any(value in payload for value in (b"/CreationDate", b"/ModDate", b"C:\\Users\\", b"C:/Users/")):
        raise ValueError("Report PDF contains activity metadata or private path")
    links = check_links()
    metrics = recalculate()
    print(json.dumps({"public_artifacts_verified": len(manifest["files"]),
                      "local_links_verified": links,
                      "project_report_pdf_bytes": pdf.stat().st_size,
                      "paired_counts": [metrics["m5_annotated_reference"]["paired_count"],
                                        metrics["m5_l3_sparse_reference"]["paired_count"]],
                      "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()
