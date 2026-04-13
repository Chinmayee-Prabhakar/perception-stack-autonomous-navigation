"""Deterministic provenance for the detector inputs consumed by M4 fusion."""

from hashlib import sha256
import json

import numpy as np

from perception_stack.results import FrameDetections


DETECTOR_INPUT_SCHEMA = "fusion-detector-input-v1"


def detector_input_fingerprint(
    detections: FrameDetections, detector_checkpoint_sha256: str | None = None,
) -> str:
    """Hash detector identity, ordered metadata, and exact full-resolution masks.

    The versioned header is sorted-key compact UTF-8 JSON. Every present mask
    follows in detector order as C-row-major, one-byte boolean pixels, with an
    eight-byte big-endian byte-length prefix. Absent masks have a distinct
    header state and no pixel block. Dimensions and presence are in the header.
    This binds local source data; it does not authenticate model predictions.
    """
    width, height = detections.image_size
    header = {
        "schema": DETECTOR_INPUT_SCHEMA,
        "frame_id": detections.frame_id,
        "image_size": [width, height],
        "model_id": detections.model_id,
        "detector_checkpoint_sha256": detector_checkpoint_sha256,
        "mask_encoding": "C-order bool, one byte per pixel, false=0 true=1",
        "detections": [
            {
                "detection_id": index,
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": float(det.confidence),
                "box_xyxy": [float(value) for value in det.box.xyxy],
                "mask_present": det.mask is not None,
                "mask_shape": list(det.mask.shape) if det.mask is not None else None,
            }
            for index, det in enumerate(detections.detections)
        ],
    }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    digest = sha256((DETECTOR_INPUT_SCHEMA + "\0").encode("ascii"))
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    for det in detections.detections:
        if det.mask is None:
            continue
        if det.mask.dtype != np.bool_ or det.mask.shape != (height, width):
            raise ValueError("Fusion detector input mask must be aligned boolean pixels")
        contiguous = np.ascontiguousarray(det.mask)
        raw = memoryview(contiguous).cast("B")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()
