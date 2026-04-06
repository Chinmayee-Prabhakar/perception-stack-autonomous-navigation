# perception-stack

Modular research pipeline for autonomous-navigation perception. **Milestone 1** runs COCO-pretrained instance segmentation on one RGB image, converts predictions to model-independent results, and saves an overlay and JSON summary. Depth, road/lane segmentation, 3D localization, and evaluation are planned in [PROJECT_PLAN.md](PROJECT_PLAN.md); they are not implemented here.

Repository layout: `src/perception_stack/frame.py` defines shared input; `results.py` defines model-independent detections; `detectors/` holds the adapter; `visualization.py` and `cli.py` render/run the demo; `configs/`, `scripts/`, and `tests/` hold settings, asset setup, and contract tests. Generated files go to ignored `outputs/`.

## Windows setup (PowerShell)

From this repository's root, use Python 3.11. The CPU path works everywhere; the local RTX 3050 can be selected with `--device 0` after installing a CUDA-enabled PyTorch build compatible with the NVIDIA driver. Install commands below use pip's current PyTorch dependency resolution; verify GPU access before selecting it.

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

The primary research extension compares instance-mask and box-region depth sampling. This project was inspired by [Project 3, “Einstein Vision,”](https://rbe549.github.io/spring2023/proj/p3/) from WPI's RBE/CS 549 Computer Vision course, taught by Prof. Nitin J. Sanket.

Ultralytics code and pretrained models use [AGPL-3.0 or an enterprise license](https://docs.ultralytics.com/help/contributing/). Review the applicable license before distributing or deploying a combined application.
