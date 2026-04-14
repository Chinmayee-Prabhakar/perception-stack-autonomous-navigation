# Reproduce the fixed seven-scene D1-R2 demonstration

This folder contains the tracked demonstration tools. The approved frame selection, camera-2 intrinsics and full-`P2` derivation, pinned asset hashes, accepted D1 scientific and historical-presentation fingerprints, and D1-R2 presentation fingerprints are in `configs/demo_hardening/`. These small files are source metadata. The image archive, checkpoints, restored inputs, masks, scenes and plots remain external or under ignored `outputs/` directories.

## Required local assets

- The previously approved KITTI Object labeled-training left-color archive, `data_object_image_2.zip`, supplied with an explicit path. `selection.json` names all seven exact `training/image_2/<ID>.png` members, dimensions and SHA-256 values. No other frames are selected.
- The three checkpoint files named in `assets.json` under the repository's `models/` directory. Their pinned SHA-256 values are checked before inference. `configs/demo.json` is tracked and hash-checked.
- A compatible existing Windows Python environment with the dependencies in `requirements-win-py311.lock` and this package importable. The validated path uses Python 3.11 and CPU-only PyTorch. The runner defaults to the repository's `.venv/Scripts/python.exe`; pass `-PythonExecutable` for another existing compatible interpreter. The script neither installs nor downloads anything.
- The tracked `006781_intrinsics.json` and `006781_derivation.json` suffice to reproduce the calibrated camera-2 plot. If the approved original KITTI `calib/006781.txt` is available at `datasets/kitti_object/training/calib/006781.txt`, the runner also verifies its recorded SHA-256. Without it, the full translated `P2` factorization is checked from the tracked derivation, and the original calibration bytes are explicitly reported as unverified.

## Run from a clean checkout

From the repository root in PowerShell, provide the location of the existing approved archive and a fresh ignored output directory:

```powershell
$archive = 'PATH_TO_APPROVED_LOCAL_data_object_image_2.zip'
& .\scripts\demo_hardening\reproduce.ps1 -KittiImageArchive $archive -OutputDirectory 'outputs\demo_hardening_reproduced'
```

The script resolves repository assets relative to its own location, so the repository may live elsewhere and the command may be launched from another working directory. The archive and optional interpreter paths are resolved from the caller. The output directory is resolved relative to the repository if not absolute. It must be a fresh child of ignored `outputs/`; the runner refuses the historical `outputs/demo_validation/` and `outputs/demo_hardening/` directories and will not overwrite a nonempty target.

It checks the pinned assets and camera-2 metadata, restores only seven approved PNGs with ZIP CRC, exact byte hash and dimensions, then invokes the existing `perception_stack.scene_cli` on CPU for each frame. The ordinary scene folders contain `scene.json`, `road_association.json`, `overlay.png`, `drivable_mask.png` and `lane_mask.png`; `scenes/006781_calibrated/` additionally contains `camera_xz.png`. All generated files and logs remain under the chosen ignored output folder. No D0 output directory is used by this command.

The runner ends with `verify_outputs.py --presentation-version r2`. It checks every input, eight normalized scientific scene JSON payloads, eight road-association payloads, seven exact road and lane masks, and the unchanged calibrated plot against the historical accepted D1 `expected_results.json`. It separately checks eight revised overlay pixel hashes against `presentation_r2.json`. Only `scene.json`'s `source_image` and `artifacts` paths are excluded from scientific comparison. The accepted ordinary-scene counts are 50 detections total, with per-frame detections `5, 12, 8, 2, 8, 7, 8` in manifest order. The calibrated frame retains the same eight detections and associations as its ordinary run. Verified render-pixel hashes make any visual difference explicit; compatible font and plotting dependencies are needed to match those hashes.

## Optional independent historical comparisons

Ordinary reproduction needs no historical D0 or D1 output folder. When those archived outputs are locally available, run:

```powershell
.\.venv\Scripts\python.exe scripts\demo_hardening\verify_outputs.py `
  --output-dir outputs\demo_hardening_reproduced `
  --accepted-d1-dir outputs\demo_hardening `
  --historical-d0-dir outputs\demo_validation `
  --presentation-version r2
```

The optional D1 comparison verifies the original D1 directory against the unchanged historical manifest, then compares all current scientific payloads, source images and masks. D1-R2 overlay pixels are intentionally different. To verify the original D1 presentation itself, use `verify_outputs.py --output-dir outputs\demo_hardening --presentation-version d1`. The optional D0 audit first verifies 76 historical D0 files against the tracked `d0_file_hashes.json`, then compares all eight scientific scene and association payloads. D0 overlays are deliberately different because D1 changed the layout. If a requested historical directory or required file is missing, verification fails with a named missing prerequisite. If no historical directory is requested, the JSON report explicitly records that the comparison was not performed. The earlier ignored `outputs/demo_hardening/build_review.py` remains part of historical D1 evidence; this tracked verifier supersedes its unconditional dependency on the ignored `d0_initial_hashes.json` file.

## Interpretation and permissions

These seven frames support qualitative presentation of predicted drivable area and lane-marking pixels; KITTI Object does not provide independent road/lane masks for an accuracy score here. The revised labels describe image-space predicted evidence from drivable-area pixels near each box. A negative label does not mean physically off-road; none of the labels establish road contact, safe on/off-road status, occupancy or collision safety. The calibrated X–Z points are selected predicted visible surfaces in the rectified image-2 camera frame, not KITTI annotated 3D-box bottom centers or independently validated metric distances. Frozen M5 experiments are not rerun or changed. M5-L requires separate authorization and aligned official Velodyne data.

KITTI data are subject to Creative Commons Attribution-NonCommercial-ShareAlike 3.0. Ultralytics software and weights carry AGPL-3.0 or enterprise terms. The vendored TwinLiteNetPlus source retains its MIT notice; separate pretrained-weight redistribution terms were not established. Consult `REFERENCES.md` before sharing derived visuals outside the user's research review.
