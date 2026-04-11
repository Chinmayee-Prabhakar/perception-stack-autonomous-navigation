"""Freeze frame IDs before any model inference or error inspection."""

from hashlib import sha256
import json
from pathlib import Path

from .protocol import EvaluationProtocol, canonical_hash


def select_frame_ids(frame_ids: tuple[str, ...], protocol: EvaluationProtocol, *,
                     source_label_calibration_sha256: str | None = None) -> dict:
    if source_label_calibration_sha256 is not None and (
        len(source_label_calibration_sha256) != 64 or
        any(char not in "0123456789abcdef" for char in source_label_calibration_sha256)
    ):
        raise ValueError("Source label/calibration fingerprint must be SHA-256")
    ids = tuple(frame_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source frame IDs")
    needed = protocol.data["development_count"] + protocol.data["heldout_count"]
    if len(ids) < needed:
        raise ValueError(f"Need at least {needed} labeled training frames for the frozen split")
    seed = protocol.data["split_seed"]
    ordered = sorted(ids, key=lambda frame_id: (sha256(f"{seed}:{frame_id}".encode()).hexdigest(), frame_id))
    development = ordered[:protocol.data["development_count"]]
    heldout = ordered[protocol.data["development_count"]:needed]
    pilot = development[:protocol.data["pilot_count"]]
    return {
        "protocol_sha256": protocol.digest,
        "class_mapping_sha256": protocol.mapping.digest,
        "source_ids_sha256": canonical_hash(sorted(ids)),
        "source_frame_count": len(ids),
        "source_label_calibration_sha256": source_label_calibration_sha256,
        "selection_rule": protocol.data["split_rule"],
        "seed": seed,
        "pilot_frame_ids": pilot,
        "development_frame_ids": development,
        "heldout_frame_ids": heldout,
        "drive_grouping": "unavailable; hashed frame IDs may contain related drive frames in both subsets",
    }


def freeze_or_verify_manifest(path: Path, frame_ids: tuple[str, ...], protocol: EvaluationProtocol,
                              *, source_label_calibration_sha256: str | None = None) -> dict:
    proposed = select_frame_ids(frame_ids, protocol,
                                source_label_calibration_sha256=source_label_calibration_sha256)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != proposed:
            raise ValueError("Frozen manifest differs from current dataset/protocol; refusing to overwrite")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proposed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return proposed
