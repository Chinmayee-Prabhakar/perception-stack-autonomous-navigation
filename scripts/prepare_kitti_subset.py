"""Extract only approved KITTI object training records from local official archives.

No network access. Image extraction requires a previously frozen split manifest.
"""

import argparse
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile
import zlib


FRAME_ID = re.compile(r"[0-9]{6}\Z")


def training_ids(archive: ZipFile, leaf: str, extension: str) -> tuple[str, ...]:
    pattern = re.compile(rf"training/{re.escape(leaf)}/([0-9]{{6}})\.{re.escape(extension)}\Z")
    ids = [match.group(1) for item in archive.infolist()
           if (match := pattern.fullmatch(item.filename)) is not None]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate training/{leaf} member ID")
    return tuple(sorted(ids))


def _crc32(path: Path) -> int:
    value = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value = zlib.crc32(chunk, value)
    return value & 0xffffffff


def extract_exact_members(archive: ZipFile, leaf: str, extension: str,
                          frame_ids: tuple[str, ...], destination: Path) -> int:
    """Construct safe member paths from validated IDs; verify each member CRC."""
    if len(frame_ids) != len(set(frame_ids)) or any(FRAME_ID.fullmatch(i) is None for i in frame_ids):
        raise ValueError("Expected distinct six-digit KITTI frame IDs")
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    for frame_id in frame_ids:
        member = archive.getinfo(f"training/{leaf}/{frame_id}.{extension}")
        target = destination / f"{frame_id}.{extension}"
        if target.exists():
            if target.stat().st_size != member.file_size or _crc32(target) != member.CRC:
                raise ValueError(f"Existing file differs from archive: {target}")
            continue
        try:
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            if target.stat().st_size != member.file_size or _crc32(target) != member.CRC:
                raise ValueError(f"Extracted file failed ZIP size/CRC verification: {member.filename}")
            extracted += 1
        except Exception:
            target.unlink(missing_ok=True)
            raise
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare approved KITTI object training subset from local ZIPs")
    parser.add_argument("--stage", choices=("metadata", "pilot", "heldout"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/kitti_object"))
    parser.add_argument("--labels-zip", type=Path)
    parser.add_argument("--calib-zip", type=Path)
    parser.add_argument("--images-zip", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/kitti_m5/frozen_split_manifest.json"))
    args = parser.parse_args()
    if args.stage == "metadata":
        if args.labels_zip is None or args.calib_zip is None:
            parser.error("metadata stage requires --labels-zip and --calib-zip")
        with ZipFile(args.labels_zip) as labels, ZipFile(args.calib_zip) as calib:
            ids = training_ids(labels, "label_2", "txt")
            if len(ids) != 7481 or training_ids(calib, "calib", "txt") != ids:
                raise ValueError("Expected matching 7,481 official KITTI object training label/calibration IDs")
            count_labels = extract_exact_members(labels, "label_2", "txt", ids,
                                                 args.dataset_root / "training/label_2")
            count_calib = extract_exact_members(calib, "calib", "txt", ids,
                                                args.dataset_root / "training/calib")
        (args.dataset_root / "training/image_2").mkdir(parents=True, exist_ok=True)
        print(f"Verified {len(ids)} training IDs; extracted {count_labels} labels and {count_calib} calibrations")
        return
    if args.images_zip is None:
        parser.error("pilot/heldout stage requires --images-zip")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    key = "pilot_frame_ids" if args.stage == "pilot" else "heldout_frame_ids"
    ids = tuple(manifest[key])
    if set(manifest["pilot_frame_ids"]) & set(manifest["heldout_frame_ids"]):
        raise ValueError("Pilot and held-out frame IDs overlap")
    with ZipFile(args.images_zip) as images:
        if len(training_ids(images, "image_2", "png")) != 7481:
            raise ValueError("Expected 7,481 official KITTI object training images")
        count = extract_exact_members(images, "image_2", "png", ids,
                                      args.dataset_root / "training/image_2")
    print(f"Verified {len(ids)} frozen {args.stage} IDs; extracted {count} new images")


if __name__ == "__main__":
    main()
