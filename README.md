# perception-stack

Modular research pipeline for autonomous-navigation perception. **Milestone 1** runs COCO-pretrained instance segmentation on one RGB image. **Milestone 2** independently estimates dense outdoor metric depth with the official Depth Anything V2 Metric VKITTI Small checkpoint. Road/lane segmentation, object-depth fusion, 3D localization, and evaluation remain planned in [PROJECT_PLAN.md](PROJECT_PLAN.md).

Repository layout: `src/perception_stack/frame.py` defines shared input; `results.py` defines model-independent detections and depth; `detectors/` and `depth/` hold separate adapters; `cli.py` and `depth_cli.py` run the independent demos; `configs/`, `scripts/`, and `tests/` hold settings, asset setup, and contract tests. Generated files go to ignored `outputs/`.

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

The primary research extension compares instance-mask and box-region depth sampling. This project was inspired by [Project 3, “Einstein Vision,”](https://rbe549.github.io/spring2023/proj/p3/) from WPI's RBE/CS 549 Computer Vision course, taught by Prof. Nitin J. Sanket.

Ultralytics code and pretrained models use [AGPL-3.0 or an enterprise license](https://docs.ultralytics.com/help/contributing/). Review the applicable license before distributing or deploying a combined application.
