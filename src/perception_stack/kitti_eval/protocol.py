"""Versioned, frozen evaluation protocol and conservative class mapping."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from perception_stack.fusion import FusionConfig


def canonical_hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ClassMapping:
    version: str
    coco_to_kitti: dict[str, str]
    ignored_kitti_categories: frozenset[str]
    digest: str

    @classmethod
    def load(cls, path: Path) -> "ClassMapping":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != "kitti-coco-conservative-v1" or data.get("coco_to_kitti") != {"car": "Car", "person": "Pedestrian"}:
            raise ValueError("Primary class mapping must be the frozen conservative v1 policy")
        ignored = frozenset(data.get("ignored_kitti_categories", ()))
        expected = {"Van", "Truck", "Tram", "Misc", "Person_sitting", "Cyclist", "DontCare"}
        if ignored != expected:
            raise ValueError("Ignored KITTI categories differ from frozen primary mapping")
        return cls(data["version"], dict(data["coco_to_kitti"]), ignored, canonical_hash(data))


@dataclass(frozen=True)
class EvaluationProtocol:
    data: dict
    digest: str
    mapping: ClassMapping

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationProtocol":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("protocol_version") != "kitti-object-z-proxy-v1":
            raise ValueError("Unsupported evaluation protocol")
        if data.get("device") != "cpu" or data.get("scale_policy") != "fixed_pretrained_metric_checkpoint_no_label_fit":
            raise ValueError("Primary evaluation requires CPU and fixed, uncalibrated metric checkpoint")
        if data.get("matching_iou_threshold") != 0.5 or data.get("ignored_reference_ioa_threshold") != 0.5:
            raise ValueError("Primary matching thresholds must remain at frozen v1 values")
        if data.get("detector_confidence") != 0.25 or data.get("detector_image_size") != 640 or data.get("depth_input_size") != 392:
            raise ValueError("Pretrained model settings differ from frozen v1 protocol")
        if data.get("fusion_min_valid_pixels") != 3 or data.get("fusion_min_valid_fraction") != 0.10 or data.get("fusion_near_cap_ratio") != 0.99:
            raise ValueError("M4 sampling settings differ from frozen v1 protocol")
        if data.get("minimum_reference_box_height_px") != 0 or data.get("maximum_truncation") != 1.0 or data.get("allowed_occlusion") != [0, 1, 2, 3]:
            raise ValueError("Primary annotation inclusion policy differs from v1")
        if data.get("reference_depth_bins_m") != [0, 15, 30, 50, 80]:
            raise ValueError("Range bins differ from frozen v1 protocol")
        if data.get("surface_depth_diagnostic") != "pending_independent_aligned_reference":
            raise ValueError("Surface-depth diagnostic requires separate protocol approval")
        for key in ("detector_checkpoint_sha256", "depth_checkpoint_sha256"):
            value = data.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Invalid {key}")
        for key in ("split_seed", "development_count", "pilot_count", "heldout_count",
                    "bootstrap_seed", "bootstrap_replicates", "bootstrap_min_distinct_frames"):
            if type(data.get(key)) is not int or data[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if data["pilot_count"] > data["development_count"]:
            raise ValueError("Pilot must be a subset of development")
        if data.get("drive_mapping_file") is not None:
            raise ValueError("Drive-group split needs a separate reviewed protocol; v1 records frame-independence limitation")
        mapping_path = Path(data["class_mapping_file"])
        if not mapping_path.is_absolute():
            mapping_path = path.resolve().parents[1] / mapping_path
        mapping = ClassMapping.load(mapping_path)
        return cls(data, canonical_hash(data), mapping)

    @property
    def fusion_config(self) -> FusionConfig:
        return FusionConfig(self.data["fusion_min_valid_pixels"],
                            self.data["fusion_min_valid_fraction"],
                            self.data["fusion_near_cap_ratio"])
