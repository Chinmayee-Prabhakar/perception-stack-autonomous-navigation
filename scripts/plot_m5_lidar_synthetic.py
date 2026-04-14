"""Render four discrete, synthetic M5-L1 geometry review images."""

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from perception_stack.kitti_eval.dataset import KittiAnnotation, KittiCalibration
from perception_stack.kitti_lidar import (CalibratedProjector, FrozenObjectMatch,
                                          LidarProtocol, SourceDigests, VelodyneScan,
                                          associate_object)
from perception_stack.results import BoxXYXY


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/kitti_m5_l1/visuals"
OUT.mkdir(parents=True, exist_ok=True)
POLICY = LidarProtocol.load(ROOT / "configs/kitti_lidar_protocol_v1.json")
STAMP = "Synthetic geometry test — not KITTI data or model performance."
SHA = "a" * 64


def projection(points):
    p2 = np.array([[10., 0, 10, 0], [0, 10, 10, 0], [0, 0, 1, 0]])
    calibration = KittiCalibration(p2, np.eye(3), np.column_stack((np.eye(3), np.zeros(3))))
    rows = np.array(points, dtype=float).reshape(-1, 4)
    scan = VelodyneScan("000001", rows[:, :3], rows[:, 3], np.arange(len(rows)),
                        SHA, len(rows), 0, 0)
    return CalibratedProjector(calibration, (20, 20), calibration_sha256=SHA).project(scan)


def reference(points, *, annotation=None):
    annotation = annotation or KittiAnnotation(0, "Car", 0, 0, 0, BoxXYXY(5, 5, 15, 15),
                                               (2, 2, 4), (0, 0, 10), 0, "valid")
    match = FrozenObjectMatch("000001", 0, 7, "Car", "Car", BoxXYXY(6, 6, 14, 14), 9, 11)
    return associate_object(projection(points), (annotation,), match,
                            SourceDigests(SHA, SHA, POLICY.sha256, SHA), POLICY)


def save(fig, name):
    fig.subplots_adjust(bottom=.22)
    fig.text(.5, .015, STAMP, ha="center", fontsize=9, color="#7c2330")
    fig.savefig(OUT / name, dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Pixel-plane example: only discrete projected scan returns, never an interpolated surface.
points = [(-1, -1, 10, 1), (0, -1, 10, 1), (1, -1, 10, 1),
          (0, -.5, 5, 1), (5, 0, 10, 1)]
result = reference(points)
projected = projection(points).returns
accepted_ids = {point.original_index for point in result.accepted_bins}
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_facecolor("#f1f3f5")
offsets = {0: (-48, 14), 1: (-48, -20), 2: (8, 14), 3: (8, -20), 4: (8, 8)}
for point in projected:
    color = "#007d4f" if point.original_index in accepted_ids else "#b94934"
    ax.scatter(point.u, point.v, c=color, s=70, zorder=3)
    ax.annotate(f"#{point.original_index} Z={point.camera2_z_m:g}",
                (point.u, point.v), xytext=offsets[point.original_index],
                textcoords="offset points", fontsize=8)
for xy, size, color, label in [((5, 5), 10, "#2368b4", "annotated box"),
                               ((6, 6), 8, "#9462ad", "matched detector box")]:
    ax.add_patch(plt.Rectangle(xy, size, size, fill=False, linewidth=1.8,
                               edgecolor=color, label=label))
ax.set(xlim=(0, 20), ylim=(20, 0), xlabel="image-2 u (pixels)", ylabel="image-2 v (pixels)",
       title="Projected discrete returns: green retained, red rejected")
ax.set_xticks(range(0, 21, 2)); ax.set_yticks(range(0, 21, 2)); ax.grid(alpha=.15)
ax.legend(loc="lower right")
save(fig, "01_projected_points.png")


# Independently specified yawed cuboid (not derived from the containment function).
yaw = math.pi / 4
c, s = math.cos(yaw), math.sin(yaw)
local = np.array([(x, y, z) for x in (-2, 2) for y in (-2, 0) for z in (-1, 1)])
world = np.column_stack((c * local[:, 0] + s * local[:, 2], local[:, 1],
                         -s * local[:, 0] + c * local[:, 2] + 10))
inside = np.array([[0., -1., 10.], [math.sqrt(2), -1., 10 - math.sqrt(2)]])
outside = np.array([[3., -1., 10.], [0., .4, 10.]])
fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
for i in range(8):
    for j in range(i + 1, 8):
        if np.count_nonzero(local[i] != local[j]) == 1:
            ax.plot(world[[i, j], 0], world[[i, j], 2], world[[i, j], 1], c="#356ca6")
ax.scatter(inside[:, 0], inside[:, 2], inside[:, 1], c="#007d4f", s=55, label="in cuboid")
ax.scatter(outside[:, 0], outside[:, 2], outside[:, 1], c="#b94934", s=55, label="outside")
ax.set(xlabel="reference X (m)", ylabel="reference Z (m)", zlabel="reference Y (m)",
       title="Known yawed KITTI cuboid; bottom center (0,0,10)")
ax.legend(loc="upper left")
save(fig, "02_rotated_cuboid.png")


# Same-pixel collision: foreground wins even when it has no eligible annotation.
collision = [(0, -1, 10, 1), (0, -.5, 5, 1), (-1, -1, 10, 1), (1, -1, 10, 1)]
collision_result = reference(collision)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.scatter([0, 0], [5, 10], s=115, c=["#007d4f", "#b94934"], zorder=3)
ax.annotate("#1 foreground winner; unassigned", (0, 5), xytext=(.18, 5.1), fontsize=10)
ax.annotate("#0 object return; excluded", (0, 10), xytext=(.18, 10.1), fontsize=10)
ax.plot([0, 0], [5, 10], color="#888", linestyle=":")
ax.set(xlim=(-.4, 3.2), ylim=(12, 3), xticks=[0], xticklabels=["same image pixel (10,9)"],
       ylabel="camera-2 axial Z (m)",
       title=f"Global z-buffer first; target support = {collision_result.support_bins} < 3")
ax.grid(axis="y", alpha=.2)
save(fig, "03_zbuffer_collision.png")


# Absence of a measured reference is explicit, not filled or colorized.
counts = []
for n in (0, 1, 2, 3):
    sample = reference([(x, -1, 10, 1) for x in range(n)])
    counts.append(sample.support_bins)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
bars = ax.bar(range(4), counts, color=["#b94934"] * 3 + ["#007d4f"])
ax.axhline(3, c="#333", linestyle="--", label="primary minimum = 3 bins")
for i, bar in enumerate(bars):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .1,
            "unsupported" if i < 3 else "supported", ha="center", fontsize=9)
ax.set(xlim=(-.5, 3.5), ylim=(0, 4.1), xticks=range(4), xlabel="synthetic projected returns",
       ylabel="distinct accepted pixel bins", title="No LiDAR reference is invented below three bins")
ax.legend(loc="upper left")
save(fig, "04_insufficient_support.png")

print("\n".join(str(path) for path in sorted(OUT.glob("*.png"))))
