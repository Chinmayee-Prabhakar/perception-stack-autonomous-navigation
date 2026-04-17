"""Verify and selectively extract only the frozen M5 held-out Object scans."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path.home() / "Downloads/data_object_velodyne.zip"
ARCHIVE_SHA256 = "2a005138e7a7c01eda7abe0ba48139b27d864984f1c1cab38f3df1c54e87eb28"
DEST = REPO / "datasets/kitti_object/training/velodyne"
OUTPUT = REPO / "outputs/kitti_m5_l3/heldout"


def _sha_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true", help="extract after complete preflight")
    args = parser.parse_args()
    split = json.loads((REPO / "configs/kitti_split_manifest_v1.json").read_text(encoding="utf-8"))
    ids = split["heldout_frame_ids"]
    assert len(ids) == len(set(ids)) == 256
    assert not set(ids) & set(split["pilot_frame_ids"])
    if not ARCHIVE.is_file() or ARCHIVE.stat().st_size != 28_750_710_812:
        raise ValueError("Approved original Object archive missing or size changed")
    measured_sha = _sha_file(ARCHIVE)
    if measured_sha != ARCHIVE_SHA256:
        raise ValueError("Approved original Object archive SHA-256 changed")

    with zipfile.ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member path")
        info_by_id = {}
        for frame_id in ids:
            member = f"training/velodyne/{frame_id}.bin"
            info = archive.getinfo(member)
            if info.flag_bits & 1 or info.compress_type != zipfile.ZIP_STORED or info.file_size % 16:
                raise ValueError(f"Invalid Object XYZI member metadata: {member}")
            info_by_id[frame_id] = info
        total_bytes = sum(info.file_size for info in info_by_id.values())
        free_bytes = shutil.disk_usage(REPO).free
        existing = {p.stem for p in DEST.glob("*.bin")}
        if existing - set(split["pilot_frame_ids"]) - set(ids):
            raise ValueError("Unexpected extracted Object scan in destination")
        existing_heldout = sum((DEST / f"{frame_id}.bin").is_file() for frame_id in ids)
        preflight = {"archive_path": str(ARCHIVE), "archive_bytes": ARCHIVE.stat().st_size,
                     "archive_sha256": measured_sha, "selected_count": len(ids),
                     "selected_member_bytes": total_bytes, "free_bytes_before_extraction": free_bytes,
                     "existing_pilot_scan_count": len(existing & set(split["pilot_frame_ids"])),
                     "existing_heldout_scan_count": existing_heldout,
                     "members": [{"frame_id": frame_id, "member": info_by_id[frame_id].filename,
                                  "bytes": info_by_id[frame_id].file_size,
                                  "crc32": f"{info_by_id[frame_id].CRC:08x}"} for frame_id in ids]}
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "extraction_preflight.json").write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({key: preflight[key] for key in (
            "archive_sha256", "selected_count", "selected_member_bytes",
            "free_bytes_before_extraction", "existing_pilot_scan_count")}, indent=2), flush=True)
        if not args.extract:
            return
        if free_bytes < total_bytes - sum(info_by_id[frame_id].file_size for frame_id in ids
                                          if (DEST / f"{frame_id}.bin").is_file()) + 1024 ** 3:
            raise OSError("Less than selected scan bytes plus 1 GiB free space")
        DEST.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, frame_id in enumerate(ids, 1):
            info = info_by_id[frame_id]
            target = DEST / f"{frame_id}.bin"
            temporary = DEST / f".{frame_id}.m5l3-partial"
            digest = sha256()
            copied = 0
            if target.exists():
                with archive.open(info) as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        copied += len(chunk)
                if copied != info.file_size or target.stat().st_size != copied or _sha_file(target) != digest.hexdigest():
                    raise ValueError(f"Existing extracted scan differs from archive member: {frame_id}")
            else:
                if temporary.exists():
                    raise FileExistsError(temporary)
                try:
                    with archive.open(info) as source, temporary.open("xb") as sink:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            sink.write(chunk)
                            digest.update(chunk)
                            copied += len(chunk)
                    if copied != info.file_size or copied % 16:
                        raise ValueError(f"Extracted size differs: {frame_id}")
                    temporary.replace(target)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            values = np.fromfile(target, dtype="<f4").reshape(-1, 4)
            finite_xyz = np.isfinite(values[:, :3]).all(axis=1)
            finite_i = np.isfinite(values[:, 3])
            row = {"frame_id": frame_id, "member": info.filename,
                   "original_archive_path": str(ARCHIVE),
                   "extracted_path": str(target), "bytes": copied,
                   "crc32": f"{info.CRC:08x}", "sha256": digest.hexdigest(),
                   "total_xyzi_rows": len(values),
                   "finite_xyzi_rows": int((finite_xyz & finite_i).sum()),
                   "nonfinite_xyz_rows": int((~finite_xyz).sum()),
                   "nonfinite_intensity_rows": int((finite_xyz & ~finite_i).sum())}
            rows.append(row)
            if index % 32 == 0 or index == len(ids):
                print(f"Verified {index}/{len(ids)} held-out Object scan members", flush=True)
        manifest = {"scope": "original frozen 256 held-out training scans only",
                    "archive_sha256": measured_sha, "extracted_count": len(rows),
                    "extracted_bytes": sum(row["bytes"] for row in rows), "scans": rows}
        (OUTPUT / "extraction_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
