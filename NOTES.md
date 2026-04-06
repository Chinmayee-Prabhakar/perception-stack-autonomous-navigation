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
