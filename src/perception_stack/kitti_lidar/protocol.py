"""Versioned M5-L sparse-reference policy, separate from frozen M5 v1."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path


@dataclass(frozen=True)
class LidarProtocol:
    _payload: bytes
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self._payload, bytes) or sha256(self._payload).hexdigest() != self.sha256:
            raise ValueError("Protocol bytes and SHA-256 must agree")
        data = json.loads(self._payload)
        required = {
            "protocol_version": "kitti-object-lidar-surface-v1",
            "boundary_numeric_tolerance_m": 1e-12,
            "minimum_distinct_bins": 3,
            "instance_mask_required_for_primary": False,
            "support_sensitivity_bins": [1, 3, 5, 10],
            "bootstrap_replicates": 1000,
            "bootstrap_seed": 549,
            "bootstrap_min_drives": 10,
            "heldout_matched_denominator": 785,
            "pilot_matched_denominator": 38,
            "scan_rejection_codes": ["nonfinite_xyz", "nonfinite_intensity"],
            "projection_rejection_codes": ["nonfinite_transform", "nonpositive_camera_z",
                                           "nonpositive_projection_denominator", "nonfinite_uv", "outside_image"],
            "association_rejection_codes": ["outside_target_cuboid", "competing_cuboid",
                                            "outside_annotation_box_2d", "outside_detection_box_2d",
                                            "same_pixel_occluded"],
            "reference_status_codes": ["supported", "unsupported_insufficient_bins"],
        }
        for key, value in required.items():
            if data.get(key) != value or type(data.get(key)) is not type(value):
                raise ValueError(f"M5-L protocol field {key} differs from reviewed v1")
        if "before object association" not in data.get("pixel_visibility", ""):
            raise ValueError("Visibility must be resolved before object association")

    @property
    def data(self) -> dict:
        """Return a fresh policy tree; callers cannot alter the hashed policy."""
        return json.loads(self._payload)

    @classmethod
    def load(cls, path: str | Path) -> "LidarProtocol":
        payload = Path(path).read_bytes()
        return cls(payload, sha256(payload).hexdigest())

    @property
    def min_bins(self) -> int:
        return self.data["minimum_distinct_bins"]

    @property
    def boundary_tolerance_m(self) -> float:
        return self.data["boundary_numeric_tolerance_m"]
