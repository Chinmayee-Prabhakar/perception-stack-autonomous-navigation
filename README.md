# perception-stack

Modular research pipeline for autonomous-navigation perception. **Milestone 1** runs COCO-pretrained instance segmentation on one RGB image. **Milestone 2** independently estimates dense outdoor metric depth with the official Depth Anything V2 Metric VKITTI Small checkpoint. **Milestone 3** independently predicts drivable-area and lane-marking masks with TwinLiteNetPlus Small. **Milestone 4** compares box and instance-mask depth sampling and optionally backprojects an actual sampled surface pixel when verified rectified intrinsics are supplied. KITTI evaluation remains planned in [PROJECT_PLAN.md](PROJECT_PLAN.md).

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

Observed box-minus-mask differences on the unlabeled bus photo are **prediction differences**, not an accuracy comparison. VKITTI-to-real domain shift, occluded/background pixels inside masks, incomplete or noisy masks, small-object sparsity, depth saturation, and lack of calibrated monocular uncertainty can all affect these estimates. Future KITTI research needs declared paired matching, class mapping, held-out splits, calibration, and a common object-reference convention. The M3 lane-boundary false positives and possible first/last 12 training-loss-row reliability concern remain under review; approved landscape-road validation is pending. Blender remains optional after core evaluation.
