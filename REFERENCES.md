# References and provenance

- [Project 3, “Einstein Vision,”](https://rbe549.github.io/spring2023/proj/p3/) from WPI's RBE/CS 549 Computer Vision course, taught by Prof. Nitin J. Sanket. This assignment inspired the driving-scene integration objective.
- [Ultralytics YOLO11 model documentation](https://docs.ultralytics.com/models/yolo11): official COCO-pretrained `yolo11n-seg.pt` instance segmentation checkpoint and model variants.
- [Ultralytics segmentation task documentation](https://docs.ultralytics.com/tasks/segment): mask output and inference API.
- [Ultralytics licensing guidance](https://docs.ultralytics.com/help/contributing/): AGPL-3.0 / enterprise terms for the framework and weights.
- [TwinLiteNetPlus official repository](https://github.com/cream1nve02/TwinLiteNetPlus): BDD100K pretrained drivable-area and lane segmentation variants; MIT source.
- [Depth Anything V2 official metric-depth instructions](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md): outdoor VKITTI small checkpoint and fixed `max_depth=80 m` setting.
- [KITTI object benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php): annotated 2D/3D object data and evaluation context. Its **labeled training data**, split into train/validation for this research, is required to compare against 3D object locations; hidden test annotations cannot be used locally for this analysis.
- [KITTI development kit / data format](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d): consult the calibration and label format when implementing evaluation; `location` is a camera-frame box reference and differs from a visible-surface point.
