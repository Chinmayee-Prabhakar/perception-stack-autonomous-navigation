# Compact public evidence and provenance

The proposed public snapshot consists of eight selected figures and five compact result files. [The machine manifest](../results/manifest.json) records each public file's exact byte count, SHA-256, source-relative path, source SHA-256, and any transformation. Except for the selected M5 summary described below, copies are byte-identical to accepted local evidence. The ignored `outputs/` tree remains the complete local audit archive and is not needed to view or recalculate the published snapshot.

| Public figure | Accepted source | What it shows | SHA-256 prefix |
| --- | --- | --- | --- |
| [Integrated scene](figures/integrated_scene_000201.png) | `outputs/demo_hardening_r2/regenerated/scenes/000201/overlay.png` | Camera-only detection, depth, road and lane predictions; inspect labels | `3f5129522135` |
| [LiDAR projection](figures/lidar_projection_007111.png) | `outputs/kitti_m5_l3/heldout/visuals/projection_007111.png` | Discrete measured returns; inspect gross camera alignment | `8a4efc11e18c` |
| [Architecture](figures/architecture.png) | `outputs/kitti_m5_l4/figures/architecture.png` | Three RGB inference branches and separate evaluation path | `c6f5d0d89975` |
| [Annotated-reference MAE](figures/m5_annotated_reference_mae.png) | `outputs/kitti_m5_l4/figures/m5_annotated_reference_mae.png` | 785-pair bottom-center-Z proxy comparison | `87f96b81c231` |
| [Sparse-reference MAE](figures/m5_l3_sparse_reference_mae.png) | `outputs/kitti_m5_l4/figures/m5_l3_sparse_reference_mae.png` | 780-pair LiDAR surface-Z proxy comparison | `f827fbdde312` |
| [Paired differences](figures/paired_difference_distribution.png) | `outputs/kitti_m5_l4/figures/paired_difference_distribution.png` | Error distribution and positive tail | `d251fc9fc392` |
| [Drive weighting](figures/drive_weighting.png) | `outputs/kitti_m5_l4/figures/drive_weighting.png` | Object-weighted versus drive-equal mean | `845c31884191` |
| [LiDAR coverage](figures/lidar_coverage.png) | `outputs/kitti_m5_l4/figures/lidar_coverage.png` | 780 supported and five unsupported objects | `889cad9d587b` |

The integrated scene and LiDAR projection contain original KITTI Object camera imagery. KITTI data are [CC BY-NC-SA 3.0](https://www.cvlibs.net/datasets/kitti/); attribution, noncommercial use, and share-alike conditions apply to derived imagery. These visualizations are credited to the KITTI Vision Benchmark Suite (Geiger, Lenz, and Urtasun, CVPR 2012) and this project's prediction/overlay code. No raw scan, image archive, or dense prediction array is included. Architecture and statistical charts are original project figures derived from the saved records. None of these figures proves physical road contact, exact RGB-visible surface depth, safety, or broad method superiority.

## Numerical snapshot

| Public result file | Accepted source | Content |
| --- | --- | --- |
| [`m5_annotated_pairs.csv`](../results/m5_annotated_pairs.csv) | `outputs/kitti_m5/heldout/paired_object_results.csv` | All 785 frozen matched rows and both annotated-reference errors |
| [`m5_annotated_summary.json`](../results/m5_annotated_summary.json) | `outputs/kitti_m5/heldout/summary_metrics.json` | Selected unchanged counts, primary statistics, frame interval, and protocol identifiers |
| [`m5_l3_sparse_pairs.csv`](../results/m5_l3_sparse_pairs.csv) | `outputs/kitti_m5_l3/heldout/paired_evaluation.csv` | All 780 supported paired rows, measured proxy, drive IDs, and frozen predictions |
| [`m5_l3_sparse_summary.json`](../results/m5_l3_sparse_summary.json) | `outputs/kitti_m5_l3/heldout/statistical_summary.json` | Primary metrics, bootstrap and drive-weighting results, sensitivity summaries |
| [`m5_l3_coverage.json`](../results/m5_l3_coverage.json) | `outputs/kitti_m5_l3/heldout/coverage_summary.json` | Original 785-object denominator and five unsupported cases |

The M5 summary is the sole **selected-field derivative**. It preserves `actual_images_tested`, `actual_objects_tested`, `counts`, `primary_paired`, `frame_bootstrap_mean_difference_95_ci`, `protocol_sha256`, `class_mapping_sha256`, `split_limitation`, and `status` exactly. It omits per-frame inference timings, local ignored plot paths, exploratory subgroups, and other audit-only fields. Its `source_sha256` and the manifest bind it to the immutable original. The other four numerical files and all eight figures are byte-identical copies. The compact snapshot cannot independently reconstruct per-point LiDAR association, image/scan provenance, all unsupported object records, or every local preservation check; those remain in the ignored accepted audit archive.

Run `python scripts/reproduce_public_results.py --check-only` to verify the published numerical comparisons without KITTI data or checkpoints. Run `python scripts/verify_public_release.py` to check public file hashes and document links. The full M5-L3 visual diagnostic set remains local pending user visual sign-off; the presentation exports and review ZIPs are excluded from the proposed Git commit.
