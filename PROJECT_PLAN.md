# Project plan

## Research question and scope

Test whether object-instance masks improve monocular object depth estimation and 3D localization versus bounding-box sampling, under identical detector, depth model, camera calibration, object matching, and test split. Networks are pretrained components; fusion and rigorous evaluation are the contribution. The project was inspired by [Project 3, “Einstein Vision,”](https://rbe549.github.io/spring2023/proj/p3/) from WPI's RBE/CS 549 Computer Vision course, taught by Prof. Nitin J. Sanket. The measurable extension is a mask-aware, calibrated fusion study.

## Model decision for this laptop

| Component | Recommended checkpoint | Why / tradeoff |
| --- | --- | --- |
| Instance segmentation | **YOLO11n-seg**, COCO pretrained (`yolo11n-seg.pt`) | Official, compact, maintained Ultralytics API; yields boxes and instance masks and suits 4 GB VRAM. Nano masks and COCO-only classes can miss thin/far road users; compare a larger variant only if resources permit. AGPL-3.0/enterprise licensing applies. **Locked for milestone 1.** |
| Drivable area and lane markings | **TwinLiteNetPlus Small**, BDD100K pretrained | Dedicated, lightweight dual segmentation output fits the laptop better than a large general semantic model; MIT source. Checkpoint distribution is through the authors' Google link and the repository is less actively maintained than Ultralytics, so freeze a retrieved weight digest and validate its input/output contract before integration. **Provisional; no implementation this milestone.** |
| Monocular depth | **Depth Anything V2 Metric VKITTI Small** (`depth_anything_v2_metric_vkitti_vits.pth`, max depth 80 m) | Official outdoor metric model, 24.8M parameters; avoids test-label scale fitting and is feasible on 4 GB with reduced input size. Synthetic VKITTI training creates a KITTI domain gap, so metric accuracy must be measured, not assumed. The relative-depth V2 checkpoint is *not* a metric substitute. **Provisional; no implementation this milestone.** |

Pin package versions and record checkpoint hashes, image preprocessing, resolution, and device with each later experiment. Do not train a new backbone without an observed failure that warrants it.

## Interfaces and geometry contract

- Shared `Frame`: original RGB array, frame ID/source, optional intrinsics. Every model adapter returns outputs aligned to this original image, regardless of internal resizing.
- Shared `FrameDetections`: class ID/name, confidence, float image-space `xyxy` box, and optional full-resolution boolean instance mask. A later road adapter will return separate semantic masks; a depth adapter will return a dense `float32` depth map plus unit/convention metadata. They consume the same frame and do not mutate it.
- KITTI rectified camera coordinates: `X` right, `Y` down, `Z` forward. Pixel coordinates are `u` right, `v` down; camera-frame forward depth is `Z` in meters. Euclidean range is `sqrt(X²+Y²+Z²)` and must never be labeled as `Z`. Verify whether a depth checkpoint predicts axial `Z` or ray distance before assigning units to its output. Use the correct calibrated `P2`/rectification chain and match image resolution before applying intrinsics. Handle the `P2` translation terms and any camera/LiDAR frame transform explicitly, rather than treating `P2[:3,:3]` as the entire geometry.
- KITTI 3D object-label `location` is the 3D box's bottom-center reference in the rectified camera frame. A mask-centroid depth backprojection generally lands on a visible *surface*, not that reference. Preserve distinct estimated-surface-point and estimated-object-reference fields; do not silently equate them.
- Mask and depth validity rules, robust statistic, sampling region, occlusion handling, and uncertainty will be recorded before evaluation. Drivable-area association will use overlap/contact evidence in image space and calibrated ground-plane geometry only where defensible.

## Milestones

1. **Done in this session:** frame abstraction, optional intrinsics metadata, one instance segmentation adapter, standard results, overlay, one-image CLI, asset setup, tests and documentation.
2. Add a dense metric depth adapter with explicit `Z` units, validity mask, resize alignment and model provenance. Add an independently selected, fixed scale method only if using a relative-depth model. No per-test-image scale from the same ground-truth labels used for reporting.
3. Add drivable-area/lane segmentation adapter and overlays. Define how lane markings differ from drivable pixels; inspect output alignment and domain behavior.
4. Implement and ablate fusion: robust depth from instance masks versus box region, possible trimmed/quantile/erosion variants, then camera backprojection of a selected pixel/surface point. Keep camera frame and object-reference claims explicit.
5. Add bird's-eye-view scene visualization from estimated coordinates, with uncertainty/range cues and drivable-area association. This is a 2D research visualization.
6. Evaluate on a fixed KITTI annotated training/validation split. Match predicted detections to annotated 2D boxes by declared class mapping and IoU threshold, with one-to-one assignment and unmatched counts. Compare predicted forward `Z` separately from full 3D object-reference localization error using KITTI **labeled** 3D object locations. Use LiDAR projected via calibration only for a clearly defined sparse surface-depth diagnostic, not as a replacement for 3D object labels. Report by class, range, occlusion/truncation, and matched/unmatched status; prevent train/validation leakage. Freeze metric scaling using an independent calibration split or use a genuinely metric model; never calibrate each test image from its reported ground truth.

## Optional extension after core evaluation: Blender scene rendering

Export predicted object classes, positions, and approximate 3D scene information in a documented scene format. A separate Blender Python script may create basic class-labeled object meshes, a camera, and a road surface where calibration and drivable-area evidence justify their placement. Optionally render with NVIDIA GPU acceleration only after Blender support is verified; CPU rendering remains possible. Label the scene as **predicted/approximate geometry**, not a ground-truth reconstruction. Blender is optional and must not block the core perception pipeline or KITTI evaluation. No Blender integration is part of the current task.

## Acceptance criteria for later quantitative evaluation

Publish split IDs, coordinate transforms, matching rule, model/checkpoint hashes, class map, valid depth definition, scale policy, metrics and confidence intervals. Report absolute `Z` error and signed bias by range/class, plus 3D Euclidean error *only when the estimated point is transformed to the same object-reference definition*. If that reference conversion cannot be justified, report surface-point error separately and do not call it box-center localization accuracy.
