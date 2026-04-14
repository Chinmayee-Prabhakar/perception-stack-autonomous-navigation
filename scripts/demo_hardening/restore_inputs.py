"""Restore only the seven approved KITTI Object images from a local ZIP."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

from PIL import Image


def project_root(script_path: Path = Path(__file__)) -> Path:
    """Resolve assets from this script, independent of the caller's directory."""
    return script_path.resolve().parents[2]


def selection_path(repo: Path) -> Path:
    return repo / "configs" / "demo_hardening" / "selection.json"


def default_output(repo: Path) -> Path:
    return repo / "outputs" / "demo_hardening_reproduced" / "inputs"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_selection(repo: Path) -> list[dict]:
    config = repo / "configs" / "demo_hardening"
    assets = json.loads((config / "assets.json").read_text(encoding="utf-8"))
    selection = selection_path(repo)
    expected = assets["files"]["configs/demo_hardening/selection.json"]
    if sha256(selection.read_bytes()) != expected:
        raise ValueError("Approved selection manifest SHA-256 mismatch")
    records = json.loads(selection.read_text(encoding="utf-8"))["frames"]
    ids = [record["frame_id"] for record in records]
    if ids != ["006263", "000201", "006781", "003783", "005481", "000185", "007200"]:
        raise ValueError("Approved frame order or IDs changed")
    for record in records:
        if record["source_member"] != f'training/image_2/{record["frame_id"]}.png':
            raise ValueError("Unexpected archive member")
    return records


def restore(archive_path: Path, output: Path, records: list[dict]) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    verified = []
    with zipfile.ZipFile(archive_path) as archive:
        for record in records:
            member = record["source_member"]
            with archive.open(member) as source:
                payload = source.read()  # zipfile checks the member CRC.
            digest = sha256(payload)
            if digest != record["image_sha256"] or len(payload) != record["image_bytes"]:
                raise ValueError(f"Image bytes or SHA-256 mismatch: {member}")
            with Image.open(io.BytesIO(payload)) as image:
                if list(image.size) != record["image_size"]:
                    raise ValueError(f"Image dimensions mismatch: {member}")
                image.verify()
            target = output / f'{record["frame_id"]}.png'
            if target.exists() and sha256(target.read_bytes()) != digest:
                raise FileExistsError(f"Existing image differs: {target}")
            if not target.exists():
                target.write_bytes(payload)
            verified.append({"frame_id": record["frame_id"], "source_member": member,
                             "image_size": record["image_size"], "sha256": digest})
    return verified


def main() -> None:
    repo = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True,
                        help="explicit path to the approved local KITTI Object image ZIP")
    parser.add_argument("--output", type=Path, default=default_output(repo))
    args = parser.parse_args()
    for record in restore(args.archive, args.output, load_selection(repo)):
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
