"""Join and verify per-frame archive, image, calibration, M5 and detector identity."""

import json
from pathlib import Path

from run_m5_l3_evaluation import _drive_map, DEVKIT_SHA256


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/kitti_m5_l3/heldout"


def _load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _index(rows):
    result = {row["frame_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate frame identity in M5-L3 source record")
    return result


def main() -> None:
    split = json.loads((REPO / "configs/kitti_split_manifest_v1.json").read_text(encoding="utf-8"))
    ids = split["heldout_frame_ids"]
    extraction = _load("extraction_manifest.json")
    geometry = _load("geometry_provenance.json")
    frame_records = _load("frame_records.json")
    detections = _load("detector_geometry.json")
    scans = _index(extraction["scans"])
    projected = _index(geometry["frames"])
    frames = _index(frame_records)
    boxes = _index(detections)
    if not all(set(value) == set(ids) for value in (scans, projected, frames, boxes)):
        raise ValueError("Source-chain frame IDs differ from original frozen 256")
    combined = []
    for frame_id in ids:
        scan, projection, frame, detector = (table[frame_id] for table in
                                             (scans, projected, frames, boxes))
        sha = projection["source_sha256"]
        if (scan["sha256"] != sha["scan"] or
                frame["sources"]["scan_sha256"] != sha["scan"] or
                frame["sources"]["calibration_sha256"] != sha["calibration"] or
                frame["sources"]["frozen_m5_frame_sha256"] != sha["frozen_m5_frame"] or
                frame["detector_geometry_fingerprint_sha256"] != detector["geometry_fingerprint_sha256"] or
                detector["image_sha256"] != sha["image"]):
            raise ValueError(f"Source-chain hash mismatch: {frame_id}")
        combined.append({"frame_id": frame_id,
                         "archive_path": scan["original_archive_path"],
                         "archive_sha256": extraction["archive_sha256"],
                         "scan_member_path": scan["member"],
                         "scan_crc32": scan["crc32"],
                         "scan_bytes": scan["bytes"],
                         "scan_extracted_path": scan["extracted_path"],
                         "scan_total_xyzi_rows": scan["total_xyzi_rows"],
                         "scan_finite_xyzi_rows": scan["finite_xyzi_rows"],
                         "scan_nonfinite_xyz_rows": scan["nonfinite_xyz_rows"],
                         "scan_nonfinite_intensity_rows": scan["nonfinite_intensity_rows"],
                         "image_sha256": sha["image"],
                         "label_sha256": sha["label"],
                         "calibration_sha256": sha["calibration"],
                         "scan_sha256": sha["scan"],
                         "frozen_m5_frame_sha256": sha["frozen_m5_frame"],
                         "detector_geometry_fingerprint_sha256": detector["geometry_fingerprint_sha256"],
                         "frozen_m5_match_count": frame["frozen_m5_match_count"],
                         "supported_count": frame["supported_count"],
                         "raw_drive_id": frame["drive_id"]})
    output = {"selected_frame_count": len(combined),
              "source_chain": "archive member CRC/SHA and extracted bytes; frozen image and approved label/calibration; frozen M5 JSON; pinned detector geometry",
              "frames": combined}
    (OUT / "source_identity_manifest.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected_drives = _drive_map(list(dict.fromkeys(
        split["development_frame_ids"] + split["pilot_frame_ids"] + ids)))
    heldout_drives = {selected_drives[frame_id] for frame_id in ids}
    development_overlap = {selected_drives[frame_id] for frame_id in split["development_frame_ids"]} & heldout_drives
    pilot_overlap = {selected_drives[frame_id] for frame_id in split["pilot_frame_ids"]} & heldout_drives
    if len(development_overlap) != 34 or len(pilot_overlap) != 10:
        raise ValueError("Official devkit drive overlap differs from accepted M5-L0 audit")
    overlap = {"mapping_archive_sha256": DEVKIT_SHA256,
               "development_heldout_drive_overlap_count": len(development_overlap),
               "pilot_heldout_drive_overlap_count": len(pilot_overlap),
               "heldout_drive_count": len(heldout_drives),
               "frozen_matched_drives": len({frame["drive_id"] for frame in frame_records
                                             if frame["frozen_m5_match_count"] > 0})}
    (OUT / "drive_overlap_validation.json").write_text(
        json.dumps(overlap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"source_chain_verified_frames": len(combined)}))


if __name__ == "__main__":
    main()
