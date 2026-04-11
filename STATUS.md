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

No remote is configured and nothing was pushed. Domain shift, mask contamination/occlusion, small-mask sensitivity, 80 m depth saturation, and uncalibrated monocular uncertainty remain unresolved. M5 KITTI proxy evaluation is recorded below; road association and research visualization move to M6. Blender remains optional after core quantitative evaluation.

## M5 — KITTI object-reference Z proxy evaluation

The evaluation infrastructure and primary real-data study are complete, independently reviewed, and accepted for local version control. The loader validated 7,481 label/calibration pairs from user-supplied local KITTI object archives; no dataset download occurred. The full `P2` matrix, including nonzero translation terms, is retained. A conservative Car/Pedestrian mapping, frozen pilot/development/held-out IDs, one-to-one 2D matching, M4 estimator reuse, paired proxy metrics, CPU CLI, and plots/reports are implemented. The metric-depth checkpoint, scale policy and all thresholds stayed fixed; no KITTI label was used to fit depth scale or select a sampling method. The selected IDs and source fingerprint are preserved in `configs/kitti_split_manifest_v1.json`.

The **pilot** processed 16/16 frames: 52 eligible references, 38 matches, and 38 valid pairs. Its box/mask mean absolute **object-reference Z proxy** errors were 4.538/4.375 m. The **held-out** study processed 256/256 distinct frames: 1,157 eligible references, 785 matches, 372 unmatched references, and 785 valid pairs. Box/mask mean absolute proxy errors were **3.703/3.837 m**; median absolute errors **2.416/2.519 m**; signed mean biases **+3.140/+3.361 m**. Mean paired mask-minus-box absolute proxy error was **+0.134 m** (frame-bootstrap 95% interval **[+0.060,+0.215] m**, 240 contributing frames, seed 549, 1,000 replicates). Mask error was lower for 53.9% of paired objects, but larger mask-worse cases raised its mean. Pilot and held-out findings are reported separately; the pilot is not evidence of a general mask advantage. Frame grouping may not equal drive grouping, so the interval does not capture possible same-drive correlation. These are **annotated bottom-center reference-Z proxy** errors, not visible-surface depth or full 3D localization accuracy.

The independent visible-surface diagnostic remains pending suitable approved aligned depth/LiDAR data and a correspondence policy. M3 representative landscape-road validation remains pending. **30 M5 offline tests passed**; the full fast suite reports **89 passed, 3 skipped**, and the separate opt-in depth/road/fusion integration suite reports **3 passed**. The review package is `outputs/perception_stack_m5_kitti_review.zip`. Earlier M1–M4 commits are preserved. No remote is configured and nothing was pushed. M6 road association and bird's-eye research visualization has not begun.
