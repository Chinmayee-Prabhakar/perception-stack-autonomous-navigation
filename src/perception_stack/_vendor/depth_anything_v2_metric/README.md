# Vendored metric architecture

These files are copied, with whitespace-only cleanup, from `metric_depth/depth_anything_v2`
in [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2),
revision `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`. Only the model,
DINOv2 layers, and image transforms needed for inference are included. The
upstream project is Apache-2.0 licensed; see `LICENSE`. DINOv2-derived files
retain their Meta copyright and license notices.

The project adapter imports this package by its full, metric-specific name.
It calls `DepthAnythingV2.forward` with explicitly preprocessed tensors on the
selected device; it does not call the upstream `infer_image` helper because
that helper independently chooses a device and could override explicit CPU
selection. The selected configuration is `encoder=vits`, `features=64`,
`out_channels=[48,96,192,384]`, `max_depth=80`.
