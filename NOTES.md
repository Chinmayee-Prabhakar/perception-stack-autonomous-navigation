# Engineering notes

## Local inspection

- Windows PowerShell; Python launcher's default is 3.9.0. Python 3.11.4 is installed at `the Python 3.11 interpreter`; 3.8 and 3.5 are also listed. Use `py -3.11` explicitly.
- The initial `nvidia-smi` display truncated the device name to RTX 3050. A later full query identified the Ti model.
- Windows reports NT build 10.0.26200.0. The installed project environment uses PyTorch 2.14.1+cpu and torchvision 0.29.1+cpu; `torch.cuda.is_available()` is `False` and `torch.version.cuda` is `None`. The demo was run on CPU.
- Before setup, Python 3.11 had NumPy and Pillow but no PyTorch, torchvision, Ultralytics, pytest, or OpenCV. The default Python 3.9 had OpenCV, but no PyTorch or Ultralytics.
- The initial workspace had no project files; the implementation was written in this repository.
- Pip editable builds exceeded Windows' path limit when its temporary directory was inside this deeply nested project folder. Using the shorter writable temporary path outside the project succeeded. Ultralytics settings defaulted to a protected AppData folder, so the CLI sets its runtime config root under ignored `work/runtime/`.

## Questions to test in later milestones

- Which mask sampling rule avoids background, reflective-window, and occluder depth contamination without discarding small/far objects?
- Does the chosen metric-depth model retain useful absolute scale on real KITTI? Measure error by range before changing its scale policy.
- For a 3D object reference, can a ground-contact pixel and an object dimension prior be justified, or should the output remain a surface point? State the distinction in every plot.
- COCO class coverage does not include all KITTI categories and may merge semantically different objects. Freeze a class map and count unmatched labels.

## Hardware and local Git verification

- `pnputil /enum-devices /class Display` lists **NVIDIA GeForce RTX 3050 Ti Laptop GPU** and **AMD Radeon(TM) Graphics**, both started. NVIDIA device ID is `VEN_10DE&DEV_25A0`; AMD device ID is `VEN_1002&DEV_1636`. AMD presence does not imply CUDA support.
- `nvidia-smi -L` confirms the RTX 3050 Ti. `nvidia-smi --query-gpu` reports 4,096 MiB total VRAM (3,962 MiB free at the time of the query), NVIDIA driver 566.07, and compute capability 8.6. NVIDIA reports CUDA driver capability 12.7; this is not a CUDA toolkit or PyTorch CUDA build.
- Registry `ProcessorNameString` identifies an **AMD Ryzen 7 4800H with Radeon Graphics**, correcting the reported Ryzen 7000-series description.
- Project `.venv` remains PyTorch **2.14.1+cpu**: `torch.version.cuda` is `None`, `torch.cuda.is_available()` is `False`, and device count is 0. CPU inference is the tested fallback. Prefer the NVIDIA GPU if a CUDA-enabled PyTorch build is separately installed and verified in an isolated environment for a later milestone; no CUDA installation or lockfile change was made here.
- Git was already initialized with no commits at the start of this task. No remote was configured. The first milestone files were still untracked. Git metadata does not independently establish implementation time.
- The 20 intended source, test, configuration, reproducibility, and documentation files were reviewed by name and size (largest: `PROJECT_PLAN.md`, about 7 KB). `git check-ignore` confirmed exclusion of the virtual environment, model checkpoints, datasets, generated outputs, local secrets, and Blender backup/render artifacts. An earlier sandboxed attempt could not create `.git/index.lock`. A normal Git staging operation confirmed index write access, so the local initial commit could proceed. `git remote -v` was empty, and nothing was pushed. The existing lockfile was not changed.

## Optional Blender visualization plan

After the core perception and KITTI evaluation milestones, an optional exporter can write predicted object classes, approximate 3D positions, and scene metadata. A Blender Python script can render basic object meshes and class labels, plus a calibrated camera and road representation only where geometry supports them. NVIDIA GPU rendering is optional and requires separate Blender compatibility testing. Every render must identify predicted geometry distinctly from ground truth. This extension must not block CPU inference, fusion experiments, or quantitative KITTI evaluation. Blender was not installed or invoked in this task.

## Milestone 2 implementation and checks

- The existing P0/P1 commit P0/P1 milestone was preserved. M2 was reviewed and accepted for one local commit; no remote or push was used. The future M2 commit hash is intentionally omitted from files in that commit.
- Official Depth Anything V2 `metric_depth` source was inspected at Git revision `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`. Only inference architecture, DINOv2 layers, and image transforms were copied to a uniquely named vendored package with Apache-2.0 license and retained source notices. The adapter uses `encoder=vits`, `features=64`, `out_channels=[48,96,192,384]`, `max_depth=80`; it does not call the upstream helper that auto-selects its own device.
- Official Hugging Face `Depth-Anything-V2-Metric-VKITTI-Small` checkpoint `depth_anything_v2_metric_vkitti_vits.pth` downloaded to ignored `models/`: 99,221,808 bytes, local SHA-256 `9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb`, matching the hash published on the official file page. The loader verifies this hash, uses `torch.load(weights_only=True)` and strict state-dictionary matching. No relative-depth checkpoint or test-label scaling is used.
- Virtual KITTI 2 documents depth as centimeters to the camera plane; the official metric training loader divides by 100 to meters. Hence the result is labeled approximate camera-axis `Z`, not radial range. It estimates a visible surface, not a KITTI 3D box center or bottom-center. VKITTI 2 training is synthetic and has non-commercial dataset terms; KITTI accuracy remains unmeasured.
- Preprocessing uses the original RGB `Frame`, `/255`, RGB ImageNet normalization, lower-bound aspect-preserving cubic resize to multiples of the 14-pixel patch size, and bilinear restoration to original height/width with `align_corners=True`. Official input default is 518; this laptop's CPU demo uses 392. Explicit `cpu` always stays CPU; `auto` uses CUDA only if available; unavailable requested `cuda` raises. Current environment remains PyTorch 2.14.1+cpu, and no CUDA installation was attempted. `torch`, `torchvision`, and OpenCV were added as direct project dependencies but already existed in the unchanged lockfile.
- Command `.\.venv\Scripts\python.exe scripts\download_assets.py --depth` succeeded. Command `.\.venv\Scripts\python.exe -m perception_stack.depth_cli --input data\examples\bus.jpg --output outputs\bus_depth.png --npy outputs\bus_depth.npy --mask outputs\bus_depth_valid.npy --meta outputs\bus_depth_metadata.json --device cpu` succeeded. Saved depth shape is `(1080, 810)`, `float32`; mask shape matches and is boolean. There were 874,800 valid pixels (100%). Valid predicted min/median/mean/max were 5.552923/12.477253/14.219467/33.388870 m. No pixels were at or above the 79.2 m near-cap threshold. Visualization is fixed 0–80 m Turbo and affects display only. `bus.jpg` has no depth ground truth.
- `.\.venv\Scripts\python.exe -m pytest -q` reported **17 passed, 1 skipped**; the skip is the opt-in real-model check. With `PERCEPTION_RUN_INTEGRATION=1`, `pytest -q tests\test_depth_integration.py` reported **1 passed**. The unchanged segmentation command was rerun on `bus.jpg` and again produced **six detections**. Blender remains optional after core quantitative evaluation and was not installed or implemented.
- Remaining work begins with Milestone 3 drivable-area/lane segmentation. Fusion, KITTI evaluation, 3D backprojection/localization, BEV, and Blender are outside this milestone.
- The editable package was reinstalled successfully with `pip install -e . --no-deps --no-build-isolation` and the documented short temporary directory. The new `perception-depth-demo --help` entry point worked, and `pip check` found no broken requirements. The final CPU demo metadata recorded actual model input `H×W=(518,392)` after aspect-preserving resize and model license `Apache-2.0`.
