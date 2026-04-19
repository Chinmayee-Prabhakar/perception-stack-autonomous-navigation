# Reproducing the published evidence

Run commands from the repository root in a compatible Python 3.11 environment. The Windows CPU lockfile is `requirements-win-py311.lock`. The public `results/` directory is a compact snapshot, not the complete local M5 or M5-L3 audit archive. No raw KITTI image, scan, dense prediction, or model checkpoint is included.

## A. View saved results without a dataset

Open the root README, [short report](PROJECT_REPORT.md), [figure directory](figures/), and [result directory](../results/). The figures are source-faithful copies of accepted local evidence; [PUBLIC_ARTIFACTS.md](PUBLIC_ARTIFACTS.md) records source hashes. The paired tables retain the original matched identities and numerical values. The sparse table contains the 780 supported pairs; the separate coverage record accounts for the five unsupported objects. Viewing these files does not run a model or require KITTI archives.

## B. Recalculate from compact tables

After installing the package and its pinned dependencies, run:

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_public_results.py --check-only
.\.venv\Scripts\python.exe scripts\reproduce_public_results.py --output-dir outputs\public_results_reproduced
```

The first command reads only `results/m5_annotated_pairs.csv`, `results/m5_annotated_summary.json`, `results/m5_l3_sparse_pairs.csv`, `results/m5_l3_sparse_summary.json`, and `results/m5_l3_coverage.json`. It checks the original M5 means and 1,000-draw frame interval against the archived snapshot. It calls the accepted `perception_stack.kitti_lidar.heldout_statistics` implementation for LiDAR paired metrics, the 1,000-draw seed-549 Raw-drive-cluster interval, and drive-equal mean. The second command additionally writes four regenerated charts and a summary to a **fresh** child of ignored `outputs/`; it refuses a nonempty target. It does not recreate measured point associations, unsupported-object details, or the full calibration audit.

## C. Rerun model inference and KITTI evaluation

Full regeneration requires separately obtained official KITTI Object labeled-training `image_2`, `label_2`, `calib`, and matching `velodyne` data, plus the three pinned checkpoints named in the README. The frozen split is `configs/kitti_split_manifest_v1.json`; do not create a new split. The Object benchmark is distinct from KITTI Raw and Odometry. Model checkpoints and KITTI archives have their own access and license terms; see [REFERENCES.md](../REFERENCES.md).

For a single user-supplied image, use `python -m perception_stack.scene_cli` as shown in the README. The CLI writes scene JSON, overlay, and road-association JSON; the source image and model weights are required. It does not use LiDAR.

For the original annotated-reference KITTI evaluation in a **separate workspace**, use a fresh output root and the already tracked split:

1. Use `scripts/prepare_kitti_subset.py --stage metadata` with the official local label and calibration ZIPs. It verifies 7,481 matching training IDs and extracts their small text records under the ignored dataset root.
2. Use the same script with `--stage pilot` and `--stage heldout`, the official image ZIP, and `--manifest configs/kitti_split_manifest_v1.json`. It restores only the already selected images. Do not call the evaluator's `freeze` mode.
3. Copy the tracked `configs/kitti_split_manifest_v1.json` to a **new** output root as `frozen_split_manifest.json` before calling `python -m perception_stack.kitti_eval_cli --mode run`. Run pilot first; the held-out command also requires `--pilot-reviewed`. Supply `--dataset-root` and `--output-dir` explicitly. The CLI validates labels/calibration, checkpoint hashes, matching, and the frozen manifest before computing results. Never point it at preserved local `outputs/kitti_m5/`.

```powershell
$labelsZip = 'PATH_TO_OFFICIAL_data_object_label_2.zip'
$calibZip = 'PATH_TO_OFFICIAL_data_object_calib.zip'
$imagesZip = 'PATH_TO_OFFICIAL_data_object_image_2.zip'
$run = 'outputs\m5_reproduction'
if (Test-Path -LiteralPath $run) { throw 'Choose an unused output directory.' }
New-Item -ItemType Directory -Path $run | Out-Null
Copy-Item -LiteralPath 'configs\kitti_split_manifest_v1.json' -Destination (Join-Path $run 'frozen_split_manifest.json')
.\.venv\Scripts\python.exe scripts\prepare_kitti_subset.py --stage metadata --labels-zip $labelsZip --calib-zip $calibZip --dataset-root datasets\kitti_object
.\.venv\Scripts\python.exe scripts\prepare_kitti_subset.py --stage pilot --images-zip $imagesZip --manifest configs\kitti_split_manifest_v1.json --dataset-root datasets\kitti_object
.\.venv\Scripts\python.exe scripts\prepare_kitti_subset.py --stage heldout --images-zip $imagesZip --manifest configs\kitti_split_manifest_v1.json --dataset-root datasets\kitti_object
.\.venv\Scripts\python.exe -m perception_stack.kitti_eval_cli --mode run --subset pilot --dataset-root datasets\kitti_object --output-dir $run
# Review pilot diagnostics before the held-out run.
.\.venv\Scripts\python.exe -m perception_stack.kitti_eval_cli --mode run --subset heldout --pilot-reviewed --dataset-root datasets\kitti_object --output-dir $run
```

The commands require valid separately obtained archives and pinned weights and perform expensive CPU inference. The preceding input preparation writes only to the ignored dataset root; use a separate workspace if accepted local assets are already present. The `--pilot-reviewed` flag is a gate indicating an actual review, not an automatic verification step.

### Full M5-L3 LiDAR dependency chain

This is a **research-script dependency map**, not a turnkey fresh-checkout command sequence. The scripts use fixed paths below and must run only in an isolated reproduction checkout containing verified source assets. Do not run them over the accepted local output tree.

Before the first LiDAR stage, prepare official KITTI Object `training/image_2`, `training/label_2`, and `training/calib` for the frozen frames under `datasets/kitti_object/` using the preceding subset commands. The original M5 evaluation must have completed with the tracked `configs/kitti_split_manifest_v1.json` and `configs/kitti_eval_v1.json`, producing `frozen_split_manifest.json`, `heldout/frames/{frame_id}.json` for all 256 selected held-out IDs, and its protocol/class-mapping records. The original-M5 example above writes these to **`outputs/m5_reproduction/`**, but the M5-L3 scripts read **`outputs/kitti_m5/`**. In the isolated checkout, make the verified *complete* M5 output tree available at `outputs/kitti_m5/` before LiDAR processing; keep the original run intact and verify frame IDs, source hashes, frozen split, and checkpoint/protocol identities. Merely changing the documented M5 output path or copying the public `results/` snapshot does not supply the required frame records.

There is a further prerequisite: `run_m5_l3_geometry.py` reads `outputs/kitti_m5/dataset_provenance.json` for selected-image SHA-256 values and official label/calibration archive hashes. Neither `prepare_kitti_subset.py` nor `perception_stack.kitti_eval_cli` creates that file. The included scripts therefore **cannot fully regenerate M5-L3 from a clean checkout by themselves**. Supply an independently verified provenance record from the authorized original-M5 source audit, or implement and review a separate provenance-generation step before attempting a full rerun. Do not fabricate it from filenames or copy an unverifiable historical record. The public snapshot intentionally omits that local audit file.

The research scripts also fix `datasets/kitti_object/` and `outputs/kitti_m5_l3/heldout/` relative to their repository, and expect the approved Object Velodyne, calibration, label, and Object devkit ZIPs in the executing user's `Downloads` directory. The scan extractor checks the approved Velodyne archive's byte size and SHA-256; evaluation checks the approved devkit digest, protocol digest, and detector/depth checkpoint hashes. Geometry checks label/calibration ZIP hashes against `dataset_provenance.json`, selected image hashes, and each original M5 frame record. These are actual path and identity requirements, not evidence that the public checkout includes the assets or that a different download is interchangeable.

| Stage and script | Required inputs before this stage | Main files produced under `outputs/kitti_m5_l3/heldout/` |
| --- | --- | --- |
| 1. Prepare original M5 records and official assets (`prepare_kitti_subset.py`, `perception_stack.kitti_eval_cli`) | Official Object image/label/calibration ZIPs, tracked frozen split/config, pinned checkpoints; separately verified `dataset_provenance.json` still required | `outputs/kitti_m5/` frozen split, per-frame held-out JSON, protocol/class records; extracted Object images/labels/calibrations under `datasets/` |
| 2. `extract_m5_l3_heldout_scans.py --extract` | Approved Object Velodyne ZIP at its fixed `Downloads` path, frozen split, adequate disk space | `extraction_preflight.json`, `extraction_manifest.json`; selected scans under `datasets/kitti_object/training/velodyne/` |
| 3. `run_m5_l3_geometry.py` | Extracted scans and manifest; `outputs/kitti_m5/dataset_provenance.json` and `heldout/frames/{frame_id}.json`; matching official images/labels/calibrations and label/calibration ZIPs | `geometry_provenance.json` |
| 4. `run_m5_l3_evaluation.py` | Geometry and extraction records, original M5 per-frame records, official Object devkit, pinned detector/depth checkpoints, matching dataset, fixed protocol v1 | `object_records.json`, `frame_records.json`, `detector_geometry.json`, `drive_mapping.json` |
| 5. `build_m5_l3_source_manifest.py` | Extraction, geometry, `frame_records.json`, `detector_geometry.json`, frozen split, verified Object devkit | `source_identity_manifest.json`, `drive_overlap_validation.json` |
| 6. `run_m5_l3_statistics.py` | `object_records.json`, `frame_records.json`, geometry and extraction records | `paired_evaluation.csv`, `bootstrap_draws.json`, `coverage_summary.json`, `statistical_summary.json`, `failure_cases.json` |
| 7. `plot_m5_l3_heldout.py` | Object and geometry records, official extracted image/calibration/scan files | `visuals/` overlays and crops, `visual_manifest.json` |

The manifest builder must follow evaluation because it reads `geometry_provenance.json`, `frame_records.json`, and `detector_geometry.json`. The compact published records do not contain the measured point associations, local source provenance, or other inputs needed for this chain.

The accepted comparisons use different targets: annotated 3D-box bottom-center camera Z for M5 and median eligible sparse object-surface camera-2 Z for M5-L3. Preserve their separate denominators, the three-bin LiDAR rule, the global same-pixel visibility step, and the frozen matches and predictions. The frame split overlaps Raw drives; clustered resampling quantifies one uncertainty measure but cannot undo development exposure.
