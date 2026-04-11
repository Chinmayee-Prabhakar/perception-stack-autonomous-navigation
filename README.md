# perception-stack

Modular research pipeline for autonomous-navigation perception. **Milestone 1** runs COCO-pretrained instance segmentation on one RGB image. **Milestone 2** independently estimates dense outdoor metric depth with the official Depth Anything V2 Metric VKITTI Small checkpoint. **Milestone 3** independently predicts drivable-area and lane-marking masks with TwinLiteNetPlus Small. **Milestone 4** compares box and instance-mask depth sampling and optionally backprojects an actual sampled surface pixel when verified rectified intrinsics are supplied. **Milestone 5** evaluates the fixed box and mask methods against KITTI annotated object-reference forward Z on a pilot and held-out subset. The revised sequence is in [PROJECT_PLAN.md](PROJECT_PLAN.md).

Repository layout: `src/perception_stack/frame.py` defines shared input; `results.py` defines model-independent detections, depth, and road/lane masks; `detectors/`, `depth/`, and `road/` hold separate adapters; `fusion/` contains paired sampling, typed results, and geometry; CLI modules run demos; `configs/`, `scripts/`, and `tests/` hold settings, asset setup, and contract tests. Generated files go to ignored `outputs/`.

## Windows setup (PowerShell)

From this repository's root, use Python 3.11. The CPU path works everywhere; the local RTX 3050 Ti Laptop GPU can be selected with `--device 0` after installing a CUDA-enabled PyTorch build compatible with the NVIDIA driver. Install commands below use pip's current PyTorch dependency resolution; verify GPU access before selecting it.

```powershell
py -3.11 -m venv .venv
$shortTemp = Join-Path $env:USERPROFILE 'Documents\Codex\work\t'
New-Item -ItemType Directory -Force $shortTemp | Out-Null
$env:TEMP = $shortTemp
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-win-py311.lock
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv\Scripts\python.exe scripts\download_assets.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m perception_stack.cli --input data\examples\bus.jpg --output outputs\bus_overlay.png --json outputs\bus_detections.json --device cpu
```

For CUDA installation details, use the [official PyTorch selector](https://pytorch.org/get-started/locally/). Once `torch.cuda.is_available()` is `True`, change `--device cpu` to `--device 0`. The 4 GB GPU may need `--imgsz 480` if memory is tight.

`scripts/download_assets.py` fetches the small official YOLO11n segmentation checkpoint and Ultralytics demo image into ignored `models/` and `data/examples/` directories. The script prints SHA-256 digests for recording an exact experiment. You may instead provide any local image to `--input`; model weights are never checked into Git. The sample image is for a smoke test, not evaluation data.

`requirements-win-py311.lock` records the tested Windows/Python 3.11 CPU package versions. The dependency ranges in `pyproject.toml` allow updating later; rerun tests and inference before changing the lock. The locked PyTorch wheel is CPU-only. A CUDA build is a separate, deliberate installation step through the PyTorch selector.

## Output and conventions

The CLI writes a mask/box overlay and a compact JSON summary. `Frame` stores `uint8` **RGB** pixels in original image coordinates, with top-left origin, `u` rightward and `v` downward. Boxes use float `xyxy` edges with an exclusive right/bottom edge; masks are full-resolution boolean arrays indexed `[v, u]`. No resize or letterbox coordinates escape the detector adapter. Optional `CameraIntrinsics` metadata stays with the frame for future calibrated modules. It does not imply that depth or 3D localization exists yet.

The adapter converts RGB to the BGR NumPy input expected by Ultralytics, requests original-resolution masks, and rejects misaligned masks. `--conf`, `--imgsz`, `--weights`, and `--device` override `configs/demo.json` values. The JSON summary records mask pixel counts, not the full mask arrays; the in-memory `Detection.mask` holds the full boolean mask for later fusion stages.

The CLI keeps Ultralytics and Matplotlib runtime settings under ignored `work/runtime/`, so it also runs in restricted Windows workspaces. If calling the adapter directly, set `YOLO_CONFIG_DIR` to a writable directory before importing Ultralytics.

## Milestone 2: metric-depth demo

The current Windows Python 3.11 environment has CPU-only PyTorch. From the repository root, after the setup above, run:

```powershell
.\.venv\Scripts\python.exe scripts\download_assets.py --depth
.\.venv\Scripts\python.exe -m perception_stack.depth_cli --input data\examples\bus.jpg --output outputs\bus_depth.png --npy outputs\bus_depth.npy --mask outputs\bus_depth_valid.npy --meta outputs\bus_depth_metadata.json --device cpu
.\.venv\Scripts\python.exe -m pytest -q
$env:PERCEPTION_RUN_INTEGRATION = '1'
.\.venv\Scripts\python.exe -m pytest -q tests\test_depth_integration.py
Remove-Item Env:\PERCEPTION_RUN_INTEGRATION
```

The original asset script command without flags still fetches YOLO11n-seg and `bus.jpg`; `--depth` fetches only the official metric checkpoint, and `--all` fetches all three. The metric checkpoint is `models/depth_anything_v2_metric_vkitti_vits.pth`, about 99 MB, and is ignored by Git. The downloader validates its SHA-256 against the digest **published on the official Hugging Face file page** (`9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb`); an existing invalid file raises an error instead of being silently used or downloaded again. The adapter then verifies the hash, loads a state dictionary with `weights_only=True`, and requires an exact match with the metric `vits` architecture. The metric source is vendored under a separate package at upstream revision `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` with Apache-2.0 license and attribution.

`FrameDepth` contains an original-resolution `float32` array in **meters**, a same-sized boolean validity mask, source/checkpoint identity, execution device, and preprocessing metadata. Valid means finite and greater than zero; values near the model's nominal 80 m cap remain valid and are reported separately as a saturation diagnostic. The saved `.npy` depth array is not normalized or altered to look plausible. The PNG alone uses a fixed 0–80 m OpenCV Turbo display scale and a labeled colorbar; invalid pixels are black. The JSON includes valid-pixel statistics and the checkpoint hash. `bus.jpg` is a demonstration image without depth ground truth, so its numbers are predictions rather than measured distances.

The source dataset, Virtual KITTI 2, defines its 16-bit depth values as centimeters **to the camera plane**. The official metric training loader divides these values by 100, and the official metric model regresses depth with an 80 m bounded head. This supports interpreting output as approximate forward camera-axis **Z**, not Euclidean radial range. A pixel's prediction refers to the visible surface and is not the annotated center or bottom-center of a 3D object. Synthetic training can cause a real-KITTI domain gap; no accuracy claim or test-image scale fitting is made here.

Preprocessing follows the official metric implementation: the RGB `Frame` is scaled to `[0,1]`, resized with lower-bound aspect preservation and OpenCV cubic interpolation to dimensions divisible by the 14-pixel ViT patch size, then ImageNet-normalized. The upstream default input size is 518; this laptop demo uses **392** to reduce CPU work and future 4 GB GPU memory pressure. For the 1080×810 example, the actual model tensor is 518×392 (`H×W`); this is recorded in JSON metadata. The model output is bilinearly restored to the original height and width with `align_corners=True`. The validity mask is formed after restoration, so no boolean interpolation occurs. The adapter supplies the selected device explicitly, avoiding the upstream helper's internal auto-selection. `--device auto` chooses CUDA only if PyTorch reports it available; `--device cuda` errors clearly in the current CPU-only environment. A future CUDA installation should be tested separately; Blender remains optional after quantitative evaluation.

The added direct dependencies (`torch`, `torchvision`, `opencv-python`) were already present at pinned versions in `requirements-win-py311.lock`, so the lockfile did not need package upgrades. The existing editable install remains the setup workflow; it was rechecked with `pip install -e . --no-deps --no-build-isolation` using the documented short temporary directory. The `perception-depth-demo` entry point and `python -m perception_stack.depth_cli` are equivalent. Ordinary unit tests neither download weights nor require a GPU; the real checkpoint smoke test is opt-in.

## Milestone 3: drivable-area and lane-marking demo

Run from the repository root with the existing Python 3.11 CPU environment:

```powershell
.\.venv\Scripts\python.exe scripts\download_assets.py --road
.\.venv\Scripts\python.exe -m perception_stack.road_cli --input data\examples\bus.jpg --output outputs\bus_road_overlay.png --drivable-mask outputs\bus_drivable_area.png --lane-mask outputs\bus_lane_markings.png --npz outputs\bus_road_masks.npz --meta outputs\bus_road_metadata.json --device cpu
.\.venv\Scripts\python.exe -m pytest -q
$env:PERCEPTION_RUN_INTEGRATION = '1'
.\.venv\Scripts\python.exe -m pytest -q tests\test_road_integration.py
Remove-Item Env:\PERCEPTION_RUN_INTEGRATION
```

The `--road` option downloads only the authors' `small.pth` from the [official checkpoint folder](https://drive.google.com/drive/folders/1EqBzUw0b17aEumZmWYrGZmbx_XJqU-vz), storing it as ignored `models/twinlitenetplus_bdd100k_small.pth` (594,777 bytes). Its pinned SHA-256 is `0dd3bd7356a5b70e1d21ae60cd09b55641b48bde738393177f94daff17ab6360`, computed locally from the authors' file. No independently published SHA-256 was found, so this check detects changes from the retrieved file but cannot independently authenticate the publisher. Existing invalid weights are rejected. The adapter uses `torch.load(weights_only=True)` and strict Small architecture matching; normal unit tests never fetch weights. The inference source is pinned to upstream revision `90f1b8695ae311d5123b05f8534b2e11e42499d2` and the necessary model/config files are vendored with the upstream MIT license. A separate checkpoint redistribution license was not located. No package upgrades or lockfile edits were needed.

The two model heads each have two channels: background at 0, foreground at 1. The BDD100K training loader turns its drivable-area and lane-marking annotations into **separate binary targets**. `FrameRoadSegmentation` therefore keeps separate original-frame boolean masks; it does not supply a road-surface class, lane boundaries, lane centerlines, or fitted trajectories. The input `Frame` remains RGB and unchanged. The adapter uses the upstream training **image** resolution 384×640, aspect-preserving OpenCV linear resize, centered 114-value padding, RGB `/255`, and no mean/std normalization. Upstream training resizes labels separately to 640×360 and crops 12 rows from each end of its output for standard BDD landscape images; applying that label rule to arbitrary inference images would misalign predictions. Upstream demo code instead uses a 640-square nominal image with automatic stride padding and an integer inverse-ratio restoration. This adapter records exact padding for each image, removes it, bilinearly restores both class-score channels to original resolution, then uses argmax (channel 1 is foreground); there is no sigmoid or arbitrary threshold. Metadata records the actual resize, padding, device, source revision, and class rule.

The CLI saves a combined RGB overlay with a legend, separate 0/255 PNG masks, a compressed `.npz` containing boolean masks, and JSON metadata. `--device cpu` is the tested path. `--device auto` selects CUDA only when this PyTorch build reports it; `--device cuda` errors clearly in the current CPU-only installation. `bus.jpg` is an unlabeled portrait street photo with no clear lane markings. Its output is useful for checking that masks align with the original image, while visible lane false positives remain and qualitative lane behavior on a representative, user-approved road photo is pending. Pixel fractions are predictions, not accuracy scores.

## Documentation convention

Project-authored Markdown uses milestone labels and technical topics rather than development activity dates or Git date metadata. Git history holds commit timestamps. Preserve scientific publication years, software versions, source revisions, hashes, and legal notices. Apply this convention when documenting future milestones.

The primary research extension compares instance-mask and box-region depth sampling. This project was inspired by [Project 3, “Einstein Vision,”](https://rbe549.github.io/spring2023/proj/p3/) from WPI's RBE/CS 549 Computer Vision course, taught by Prof. Nitin J. Sanket.

Ultralytics code and pretrained models use [AGPL-3.0 or an enterprise license](https://docs.ultralytics.com/help/contributing/). Review the applicable license before distributing or deploying a combined application.

## Milestone 4: paired object-depth fusion

Run from the repository root after the existing detector and metric-depth checkpoints have been downloaded:

```powershell
.\.venv\Scripts\python.exe -m perception_stack.fusion_cli --input data\examples\bus.jpg --json outputs\bus_fusion.json --output outputs\bus_fusion_overlay.png --device cpu
.\.venv\Scripts\python.exe -m pytest -q
$env:PERCEPTION_RUN_INTEGRATION = '1'
.\.venv\Scripts\python.exe -m pytest -q tests\test_depth_integration.py tests\test_road_integration.py tests\test_fusion_integration.py
Remove-Item Env:\PERCEPTION_RUN_INTEGRATION
```

Both methods use the same `FrameDetections`, `FrameDepth`, finite-positive validity mask, estimator, and thresholds. A clipped float box contains pixel center `(u,v)` iff `x1 <= u < x2` and `y1 <= v < y2`, with integer pixel centers and `(0,0)` at the top-left pixel center. The mask method samples `instance_mask AND box`; a missing or empty mask is explicitly unavailable. The primary representative depth is the **median of valid camera-axis Z values in meters**. Default acceptance requires at least **3 valid pixels** and **10% valid fraction** of the sampling region, fixed independently of evaluation labels. One- or two-pixel and sparse samples retain diagnostics and are marked unavailable; `--min-valid-pixels` and `--min-valid-fraction` expose this fixed policy for deliberate experiments. Invalid values are never converted to zeros. The JSON reports median, mean, min/max, population standard deviation, IQR, MAD, region and valid counts, valid fraction, near-cap count/fraction (Z >= 0.99×80 m), status and reason.

A separate selected ray uses an **actual valid pixel**. Among valid region pixels, choose depth closest to the region median; break ties by squared image distance to the centroid of all region pixel centers, then smaller `v`, then smaller `u`. The selected pixel's actual depth can differ from the aggregate median. Box and mask methods use the same rule. The point denotes an estimated visible scene surface, never a KITTI 3D box center or bottom-center. The default bus run supplies no camera intrinsics, so it reports pixels and Z values but sets 3D camera coordinates to unavailable.

For a trusted rectified pinhole camera, `--intrinsics-json` may supply `width,height,fx,fy,cx,cy,rectified:true,provenance`. The resolution must match the image; a raw KITTI `P2` is rejected because its translation and coordinate origin need an explicit calibration chain. Pixel centers use integer `(u,v)`. The camera axes are `X` right, `Y` down, `Z` forward, with `X=(u-cx)Z/fx`, `Y=(v-cy)Z/fy`, and radial range `sqrt(X²+Y²+Z²)` reported separately. The program records the user-supplied provenance; it cannot independently authenticate a calibration file. Do not fabricate intrinsics for `bus.jpg`.

The demo JSON includes detector/depth checkpoint hashes, model IDs, depth preprocessing, configuration, object-level statistics, selected rays, explicit unavailable fields, and optional surface points. The overlay shows boxes, class/confidence, both median Z values, and selected-ray markers. Detections are inferred before the depth model is loaded, reducing simultaneous model residency on the CPU path. The fusion does not use the road/lane predictions yet.

Observed box-minus-mask differences on the unlabeled bus photo are **prediction differences**, not an accuracy comparison. VKITTI-to-real domain shift, occluded/background pixels inside masks, incomplete or noisy masks, small-object sparsity, depth saturation, and lack of calibrated monocular uncertainty can all affect these estimates. M5 defines paired matching, class mapping, a held-out split, and the object-reference convention for quantitative evaluation. The M3 lane-boundary false positives and possible first/last 12 training-loss-row reliability concern remain under review; approved landscape-road validation is pending. Blender remains optional after core evaluation.

## Milestone 5: KITTI object-reference Z proxy evaluation

The offline evaluation is runnable without downloading a dataset:

```powershell
.\.venv\Scripts\python.exe -m perception_stack.kitti_eval_cli --mode plan
.\.venv\Scripts\python.exe -m pytest -q
```

The plan command writes an explicit **pending**, zero-real-frame summary under ignored `outputs/kitti_m5/plan/`; it does not compute accuracy. The user supplied local copies of the [official KITTI object archives](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Their expected `training/` structure and 7,481 label/calibration/image IDs were checked; publisher authenticity cannot be independently established from filenames alone. No data were downloaded. Registration may be required to obtain another copy. The full left-color ZIP is about 12 GB, so `scripts/prepare_kitti_subset.py` extracts only frozen image IDs into ignored `datasets/kitti_object/`.

To recreate the local extraction from approved ZIPs in a PowerShell session:

```powershell
$dl = Join-Path $env:USERPROFILE 'Downloads'
.\.venv\Scripts\python.exe scripts\prepare_kitti_subset.py --stage metadata --labels-zip (Join-Path $dl 'data_object_label_2.zip') --calib-zip (Join-Path $dl 'data_object_calib.zip')
.\.venv\Scripts\python.exe -m perception_stack.kitti_eval_cli --mode freeze --dataset-root datasets\kitti_object
.\.venv\Scripts\python.exe scripts\prepare_kitti_subset.py --stage pilot --images-zip (Join-Path $dl 'data_object_image_2.zip')
.\.venv\Scripts\python.exe -m perception_stack.kitti_eval_cli --mode run --subset pilot --dataset-root datasets\kitti_object
# Inspect pilot diagnostics before evaluating the untouched held-out IDs.
.\.venv\Scripts\python.exe scripts\prepare_kitti_subset.py --stage heldout --images-zip (Join-Path $dl 'data_object_image_2.zip')
.\.venv\Scripts\python.exe -m perception_stack.kitti_eval_cli --mode run --subset heldout --pilot-reviewed --dataset-root datasets\kitti_object
```

The loader accepts the official separate-archive `data_object_image_2/training/image_2`, `data_object_label_2/training/label_2`, and `data_object_calib/training/calib` layout, or a consolidated `training/{image_2,label_2,calib}` layout. It never uses the unlabeled `testing` partition. Labels must have all 15 documented fields. `DontCare`, unsupported categories, nonpositive reference Z, and invalid dimensions are recorded with distinct statuses; malformed fields and missing selected files fail explicitly. The complete `P2` 3×4 matrix, including translation, is retained; optional `R0_rect` and `Tr_velo_to_cam` are parsed for future geometry diagnostics. Primary proxy analysis uses the annotation's forward camera-axis `location_z` directly. It does not invert `P2`, equate surface points to 3D box reference points, or report radial range as Z.

`configs/kitti_class_mapping_v1.json` freezes COCO `car`→KITTI `Car` and `person`→`Pedestrian`. Van, Truck, Tram, Misc, Person_sitting, Cyclist, and DontCare are ignored as references, and other COCO detections remain unsupported without remapping. The fixed matcher greedily takes highest 2D IoU, then highest confidence, then smallest reference and detection IDs, at IoU ≥0.50. Remaining eligible detections overlapping an ignored reference by ≥0.50 of **detection** area are counted as ignored. No depth value enters matching. Counts preserve total/eligible/ignored references and detections, unmatched items, match IoUs, per-class coverage, and missing box or mask estimates.

`configs/kitti_eval_v1.json` freezes CPU model/checkpoint hashes, 640-pixel YOLO input, confidence 0.25, 392-pixel metric-depth input, M4 thresholds, mapping, range bins, and statistics. Among labeled training IDs, SHA-256 ordering with seed 549 selects 64 development frames, its first 16 as the pilot, and a disjoint 256-frame held-out subset. Source label/calibration contents are fingerprinted in the split manifest. The `freeze` command can run after approved labels/calibration arrive, **before extracting all image files**; extract only the frozen pilot and held-out `image_2` IDs. Source-drive grouping cannot be established from the available object archive IDs, so frame-level independence is a limitation. No small-object, truncation, or occlusion exclusion is applied beyond annotation validity. Do not select subsets based on measured errors.

`configs/kitti_split_manifest_v1.json` is the version-controlled copy of the frozen evaluated IDs and source fingerprint; it is byte-identical to `outputs/kitti_m5/frozen_split_manifest.json`. Separate `pilot/` and `heldout/` output directories hold per-frame JSON, per-object CSV, matching diagnostics, summary JSON, labeled plots, and a Markdown report. Each matched object records annotated 3D-box **bottom-center** forward Z, both M4 median Z estimates, signed and absolute proxy errors, paired mask-minus-box absolute proxy error, valid pixels, mask area, occlusion/truncation, and failure statuses. Negative paired differences favor masks **for this proxy**. Primary paired errors use only matched objects with both methods valid; coverage always uses every matched object. A seeded bootstrap resamples frames, with no interval when fewer than ten distinct frames contribute. Range/class/occlusion/truncation/mask-size findings are exploratory. CPU inference times are recorded separately.

| Real-data subset | Selected frames | Eligible references | Matched and paired | Box mean absolute proxy error | Mask mean absolute proxy error | Mean mask − box absolute proxy error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pilot diagnostic | 16 | 52 | 38 | 4.538 m | 4.375 m | −0.163 m |
| Frozen held-out | 256 | 1,157 | 785 | 3.703 m | 3.837 m | +0.134 m |

All selected frames completed. Held-out box/mask median absolute errors were 2.416/2.519 m, signed mean biases +3.140/+3.361 m, and RMSE 5.433/5.630 m. The mean paired difference's deterministic frame-bootstrap 95% interval was **[+0.060,+0.215] m** (240 contributing frames, seed 549, 1,000 replicates). Masks had lower absolute proxy error on 53.9% of pairs, but larger mask-worse cases raised their mean. There were 372 unmatched eligible held-out references, 213 unmatched eligible detections, and 236 detections ignored by the fixed DontCare/ignored-category overlap policy. See `outputs/kitti_m5/heldout/results_report.md` and `summary_metrics.json` for denominators, class/range/occlusion breakdowns, and plots. The pilot is a pipeline diagnostic and is not combined with the held-out result.

These are **object-reference Z proxy** errors: KITTI's annotated 3D-box bottom-center is not the image-visible surface predicted by monocular depth. The frame bootstrap does not account for related frames from one source drive because drive IDs were unavailable. Class/range/occlusion analyses are exploratory, and the 80 m model cap limits far-range interpretation. The optional surface diagnostic needs independently approved and correctly aligned LiDAR/depth data plus a defensible visible-surface association policy. M3 landscape-road validation remains pending. M6 road association and bird's-eye visualization is next; Blender remains optional later.
