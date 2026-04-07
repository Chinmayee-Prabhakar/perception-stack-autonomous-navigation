# Status

**P0/P1 remains complete.** In a verification run, the unchanged CPU segmentation CLI was rerun on `data/examples/bus.jpg` and produced six detections.

**Milestone 2 was implemented, verified, reviewed and accepted.** The official Depth Anything V2 Metric VKITTI Small model produces an original-resolution `float32` depth map and boolean validity mask through the shared `Frame` contract. Its 99,221,808-byte checkpoint matched the independently published SHA-256 `9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb`; a strict metric `vits` state-dictionary load succeeded. The CPU demo on `bus.jpg` wrote an 810×1080 depth array, validity mask, fixed 0–80 m colorbar PNG, and JSON metadata under `outputs/`. All 874,800 pixels were valid (finite and positive); min/median/mean/max predictions were 5.553/12.477/14.219/33.389 m. Zero valid pixels were within 1% of the 80 m modeled cap. These are model predictions on an unlabelled example, not accuracy measurements.

The ordinary test run reported **17 passed, 1 skipped** (the intentionally opt-in real-model test). The opt-in integration run reported **1 passed**. No test downloaded weights or required CUDA by default. Current PyTorch is CPU-only despite the verified RTX 3050 Ti Laptop GPU with 4 GB VRAM. The existing Windows lockfile was not upgraded, no CUDA or Blender installation was made, and no remote or push was used. 

The modeled quantity is approximate visible-surface camera-axis **Z** in meters, supported by Virtual KITTI 2's camera-plane depth convention and the official metric training loader. This is distinct from Euclidean range and KITTI's annotated 3D object reference. Synthetic-to-real accuracy has not been evaluated. Object-depth fusion, road/lane segmentation, backprojection, KITTI evaluation, bird's-eye view, and Blender rendering remain unimplemented. Blender remains optional after core quantitative evaluation.

**Next development step: Milestone 3**, the drivable-area/lane adapter and aligned overlays. Do not infer 3D object location or quantitative accuracy from the current depth demo.
