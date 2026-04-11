# Status

## P0/P1 — Shared frame and instance segmentation

Implemented and committed as P0/P1 milestone. The CPU YOLO11n-seg demonstration on `data/examples/bus.jpg` returns six detections. Boxes and optional instance masks use the shared original-frame coordinate contract.

## M2 — Metric monocular depth

Implemented and committed as M2 milestone. The official Depth Anything V2 Metric VKITTI Small checkpoint produces original-resolution `float32` camera-axis Z estimates in meters and a boolean validity mask. Its 99,221,808-byte checkpoint matches the independently published SHA-256 `9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb`; strict metric `vits` loading succeeds.

The CPU bus-image demo produces an 810×1080 array with 874,800 finite positive values. Predicted min/median/mean/max are 5.553/12.477/14.219/33.389 m, with no values within 1% of the model's nominal 80 m cap. These are visible-surface predictions on an unlabeled photo, not KITTI accuracy or 3D object-reference locations. Synthetic-to-real depth accuracy remains unmeasured.

## M3 — Road and lane segmentation

The TwinLiteNetPlus Small BDD100K adapter, original-resolution masks, overlay, checkpoint verification, CPU CLI, and automated tests are implemented and committed as M3 milestone. Its authors' `small.pth` file is 594,777 bytes and has locally computed SHA-256 `0dd3bd7356a5b70e1d21ae60cd09b55641b48bde738393177f94daff17ab6360`. The pinned source revision is `90f1b8695ae311d5123b05f8534b2e11e42499d2`. The inference source carries the upstream MIT license; no separately stated weight license or independently published checkpoint SHA-256 was located. The current PyTorch 2.14.1 installation remains CPU-only, and the dependency lockfile is unchanged.

The bus-image CPU output is 810×1080. A 384×640 model input holds a 384×288 resized image with 176 pixels of padding on each side. Predicted foreground is 147,229 drivable-area pixels (16.83%) and 18,456 lane-marking pixels (2.11%). The green overlay mostly follows foreground pavement, while the yellow lane mask includes conspicuous top and bottom edge false positives. Those image edges have no vertical input padding, so leakage from top or bottom padding cannot explain them. The portrait scene and absence of clear lane markings limit the interpretation; a representative landscape road image with approved provenance is still needed for qualitative validation. Pixel fractions are not accuracy measurements.

The standard fast suite reports **45 passed, 2 skipped**; the skips are the intentionally opt-in real-checkpoint tests. The separate depth and road integration run reports **2 passed** on CPU. The detector, metric-depth, and road CLIs all run on the existing assets, and their pinned asset hashes reverify without downloads. Detailed commands and checks are recorded in [NOTES.md](NOTES.md).

The first/last 12 model-row crop in upstream training loss is a possible boundary reliability concern, not an established explanation of the bus-image false positives. Representative landscape-road validation remains pending by choice.

## M4 — Paired object-depth fusion and visible-surface geometry

M4 implementation is complete and externally reviewed. The box baseline and instance-mask method use the same original-frame detections, depth map, finite-positive validity rule, median statistic, minimum-pixel/fraction policy, and deterministic selected-pixel rule. The mask region is intersected with its box; missing masks are explicitly unavailable. Region-median Z is kept separate from the actual Z at the selected image ray. Synthetic pinhole backprojection tests pass. Rectified backprojection requires aligned intrinsics and nonempty provenance; real calibrated backprojection remains undemonstrated. The bus image has no calibration and receives no 3D coordinates.

The CPU bus demonstration produced six detections, six valid box estimates, and six valid mask estimates. Box/mask medians differ, but the image lacks ground truth and the differences are not accuracy measurements. The fast suite reports **59 passed, 3 skipped**; the depth, road, and fusion real-checkpoint tests report **3 passed**. The detector CLI still reports six detections. Generated JSON and overlay are in `outputs/`.

No remote is configured and nothing was pushed. Domain shift, mask contamination/occlusion, small-mask sensitivity, 80 m depth saturation, and uncalibrated monocular uncertainty remain unresolved. KITTI quantitative accuracy, matching, class mapping, held-out evaluation and object-reference localization are future work. M5 road association and research visualization is the next planned milestone; Blender remains optional after core quantitative evaluation.
