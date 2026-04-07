"""Official Depth Anything V2 metric VKITTI Small with explicit device control."""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from perception_stack.depth_assets import (
    CHECKPOINT_PATH, MODEL_ID, SOURCE_REVISION, verify_checkpoint,
)
from perception_stack.frame import Frame
from perception_stack.results import FrameDepth


DEFAULT_INPUT_SIZE = 392  # 518 upstream; lower CPU cost and GPU memory on this laptop.
PREPROCESSING = {
    "input_color": "RGB from Frame (equivalent to upstream BGR-to-RGB conversion)",
    "image_scale": "uint8 / 255",
    "normalization_mean_rgb": [0.485, 0.456, 0.406],
    "normalization_std_rgb": [0.229, 0.224, 0.225],
    "resize": "aspect-preserving lower_bound, dimensions multiples of 14, OpenCV cubic",
    "depth_restore": "PyTorch bilinear to original HxW, align_corners=True",
    "upstream_source_revision": SOURCE_REVISION,
}


def resolve_device(requested: str) -> torch.device:
    if requested not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be cpu, cuda, or auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch installation; use --device cpu")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")


def preprocess_frame(frame: Frame, input_size: int) -> torch.Tensor:
    """Match upstream metric image2tensor, starting from Frame's existing RGB."""
    if not isinstance(input_size, int) or input_size < 14:
        raise ValueError("input_size must be at least one 14-pixel ViT patch")
    from perception_stack._vendor.depth_anything_v2_metric.util.transform import (
        NormalizeImage, PrepareForNet, Resize,
    )

    sample = {"image": frame.image_rgb.astype(np.float32) / 255.0}
    sample = Resize(
        width=input_size, height=input_size, resize_target=False,
        keep_aspect_ratio=True, ensure_multiple_of=14, resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_CUBIC,
    )(sample)
    sample = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(sample)
    sample = PrepareForNet()(sample)
    return torch.from_numpy(sample["image"]).unsqueeze(0)


def predict_metric_depth(
    frame: Frame, model: torch.nn.Module, *, device: torch.device,
    input_size: int, checkpoint_sha256: str,
) -> FrameDepth:
    """Run a metric model on one frame; also usable with deterministic fake models."""
    tensor = preprocess_frame(frame, input_size).to(device)
    with torch.inference_mode():
        prediction = model(tensor)
        if not isinstance(prediction, torch.Tensor) or prediction.ndim != 3 or prediction.shape[0] != 1:
            raise ValueError("Metric model must return a 1xHxW depth tensor")
        if not prediction.is_floating_point():
            raise ValueError("Metric model must return floating-point depth")
        if prediction.shape[1] < 1 or prediction.shape[2] < 1:
            raise ValueError("Metric model returned an empty depth map")
        depth = F.interpolate(
            prediction[:, None], (frame.size[1], frame.size[0]),
            mode="bilinear", align_corners=True,
        )[0, 0].to(torch.float32).cpu().numpy().copy()
    valid = np.isfinite(depth) & (depth > 0)
    return FrameDepth(
        frame_id=frame.frame_id, image_size=frame.size, depth_m=depth,
        valid_mask=valid, model_id=MODEL_ID, checkpoint_sha256=checkpoint_sha256,
        device=device.type, input_size=input_size,
        preprocessing={**PREPROCESSING, "model_input_hw": [int(tensor.shape[2]), int(tensor.shape[3])]},
    )


class MetricDepthEstimator:
    """Verified metric state dict, strict architecture load, inference-only execution."""

    def __init__(
        self, weights: str | Path = CHECKPOINT_PATH, *,
        device: str = "cpu", input_size: int = DEFAULT_INPUT_SIZE,
    ) -> None:
        self.device = resolve_device(device)
        self.input_size = input_size
        if not isinstance(input_size, int) or input_size < 14:
            raise ValueError("input_size must be at least 14")
        self.checkpoint_sha256 = verify_checkpoint(weights)
        from perception_stack._vendor.depth_anything_v2_metric.dpt import DepthAnythingV2

        model = DepthAnythingV2(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384],
            max_depth=80.0,
        )
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("Metric checkpoint must contain a state dictionary")
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError("Checkpoint is incompatible with metric VKITTI vits architecture") from error
        self._model = model.to(self.device).eval()

    def predict(self, frame: Frame) -> FrameDepth:
        return predict_metric_depth(
            frame, self._model, device=self.device, input_size=self.input_size,
            checkpoint_sha256=self.checkpoint_sha256,
        )
