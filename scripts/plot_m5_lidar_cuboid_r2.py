"""Render only the versioned synthetic cuboid clarification and point audit."""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from perception_stack.kitti_eval.dataset import KittiAnnotation
from perception_stack.kitti_lidar.association import inside_oriented_cuboid
from perception_stack.results import BoxXYXY


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/kitti_m5_l1_r2"
VISUALS = OUT / "visuals"
VISUALS.mkdir(parents=True, exist_ok=True)
IMAGE = VISUALS / "02_rotated_cuboid_clarified.png"
TABLE = OUT / "cuboid_points.json"

YAW = math.pi / 4
C, S = math.cos(YAW), math.sin(YAW)
ANNOTATION = KittiAnnotation(0, "Car", 0., 0, 0., BoxXYXY(0, 0, 20, 20),
                             (2., 2., 4.), (0., 0., 10.), YAW, "valid")

# Expected classifications are specified from the cuboid bounds, independently
# of inside_oriented_cuboid(): local length |L|<=2, width |W|<=1, Y in [-2,0].
POINTS = (
    {"id": "I1", "xyz_m": (0., -1., 10.), "expected_inside": True,
     "explanation": "Center in horizontal footprint and halfway between top and bottom."},
    {"id": "I2", "xyz_m": (math.sqrt(2), -1., 10. - math.sqrt(2)),
     "expected_inside": True,
     "explanation": "On the local +2 m length face; Y and width remain inside."},
    {"id": "O1", "xyz_m": (3., -1., 10.), "expected_inside": False,
     "explanation": "Outside footprint: local length +2.12 m exceeds +2 m and width +2.12 m exceeds +1 m."},
    {"id": "O2", "xyz_m": (0., .4, 10.), "expected_inside": False,
     "explanation": "Y=+0.4 m is below the cuboid bottom Y=0; annotated top is Y=-2 m (camera Y points down)."},
)


def world(length: float, y: float, width: float) -> tuple[float, float, float]:
    """Independently draw yawed length/width axes at bottom center (0,0,10)."""
    return (C * length + S * width, 10. - S * length + C * width, y)


def local_xz(x: float, z: float) -> tuple[float, float]:
    dx, dz = x, z - 10.
    return C * dx - S * dz, S * dx + C * dz


def main() -> None:
    rows = []
    for point in POINTS:
        x, y, z = point["xyz_m"]
        computed = inside_oriented_cuboid((x, y, z), ANNOTATION)
        assert computed is point["expected_inside"], point["id"]
        length, width = local_xz(x, z)
        rows.append({"id": point["id"], "xyz_m": [x, y, z],
                     "local_length_m": length, "local_width_m": width,
                     "expected_inside": point["expected_inside"],
                     "computed_inside": computed,
                     "explanation": point["explanation"]})
    TABLE.write_text(json.dumps({
        "synthetic_only": True,
        "coordinates": "rectified reference camera: X right, Y down, Z forward",
        "cuboid": {"bottom_center_xyz_m": [0, 0, 10], "height_m": 2,
                   "width_m": 2, "length_m": 4, "yaw_rad": YAW,
                   "local_limits_m": {"length": [-2, 2], "width": [-1, 1], "y": [-2, 0]}},
        "points": rows,
    }, indent=2) + "\n", encoding="utf-8")

    fig = plt.figure(figsize=(15, 9), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.42, 1),
                            left=.045, right=.975, bottom=.19, top=.82,
                            wspace=.25, hspace=.36)
    ax3 = fig.add_subplot(grid[:, 0], projection="3d")
    ax_plan = fig.add_subplot(grid[0, 1])
    ax_side = fig.add_subplot(grid[1, 1])

    corners = {(length, y, width): world(length, y, width)
               for length in (-2, 2) for y in (-2, 0) for width in (-1, 1)}
    faces = []
    for y in (-2, 0):
        faces.append([corners[(length, y, width)] for length, width in
                      ((-2, -1), (2, -1), (2, 1), (-2, 1))])
    for length in (-2, 2):
        faces.append([corners[(length, y, width)] for y, width in
                      ((-2, -1), (0, -1), (0, 1), (-2, 1))])
    for width in (-1, 1):
        faces.append([corners[(length, y, width)] for length, y in
                      ((-2, -2), (2, -2), (2, 0), (-2, 0))])
    ax3.add_collection3d(Poly3DCollection(faces, facecolors="#8eb9d6",
                                          edgecolors="#356ca6", linewidths=.8,
                                          alpha=.12))
    for length in (-2, 2):
        for width in (-1, 1):
            top, bottom = corners[(length, -2, width)], corners[(length, 0, width)]
            ax3.plot([top[0], bottom[0]], [top[1], bottom[1]],
                     [top[2], bottom[2]], color="#356ca6", linewidth=1)
    ax3.scatter([0], [10], [0], c="#2d4f79", marker="*", s=160,
                depthshade=False, zorder=10)
    ax3.text(.07, 10.12, .05, "bottom center Y=0", fontsize=8)
    for row in rows:
        x, y, z = row["xyz_m"]
        color = "#007d4f" if row["expected_inside"] else "#c23b32"
        marker = "o" if row["expected_inside"] else "^"
        ax3.scatter([x], [z], [y], c=color, marker=marker, s=95,
                    edgecolors="white", linewidths=.8, depthshade=False, zorder=20)
        ax3.text(x + .08, z + .08, y - .08, row["id"], fontsize=10, fontweight="bold", color=color)
    ax3.set(xlim=(-2.8, 3.5), ylim=(7.3, 12.7), zlim=(.7, -2.5),
            xlabel="X right (m)", ylabel="Z forward (m)", zlabel="Y down (m)")
    ax3.set_box_aspect((1.15, 1, .75))
    ax3.view_init(elev=22, azim=-60)
    ax3.set_title("Perspective view · overlap can mislead", fontsize=12, pad=14)

    footprint = [world(length, 0, width)[:2] for length, width in
                 ((-2, -1), (2, -1), (2, 1), (-2, 1))]
    ax_plan.add_patch(Polygon(footprint, closed=True, facecolor="#8eb9d6",
                              edgecolor="#356ca6", linewidth=1.5, alpha=.22))
    for row in rows:
        x, y, z = row["xyz_m"]
        color = "#007d4f" if row["expected_inside"] else "#c23b32"
        if row["id"] == "O2":
            ax_plan.scatter(x, z, s=220, facecolors="none", edgecolors=color,
                            linewidths=2, marker="o", zorder=5)
        else:
            ax_plan.scatter(x, z, s=68, c=color,
                            marker="o" if row["expected_inside"] else "^", zorder=6)
        offsets = {"I1": (-30, -22), "I2": (8, -18), "O1": (8, 8), "O2": (13, 17)}
        ax_plan.annotate(row["id"], (x, z), xytext=offsets[row["id"]],
                         textcoords="offset points", color=color, fontsize=9,
                         fontweight="bold", arrowprops={"arrowstyle": "-", "color": color, "lw": .7})
    ax_plan.set(xlim=(-2.4, 3.6), ylim=(7.2, 12.5), xlabel="X right (m)",
                ylabel="Z forward (m)", title="Orthographic top: X–Z footprint")
    ax_plan.set_aspect("equal", adjustable="box")
    ax_plan.grid(alpha=.15)
    ax_plan.text(.02, .02, "O2 shares I1's X–Z; Y decides its status.",
                 transform=ax_plan.transAxes, fontsize=8, color="#4b4b4b")

    ax_side.add_patch(Rectangle((-2, -2), 4, 2, facecolor="#8eb9d6",
                                edgecolor="#356ca6", linewidth=1.5, alpha=.22))
    for row in rows:
        length = row["local_length_m"]
        y = row["xyz_m"][1]
        color = "#007d4f" if row["expected_inside"] else "#c23b32"
        ax_side.scatter(length, y, s=75, c=color,
                        marker="o" if row["expected_inside"] else "^", zorder=5)
        offsets = {"I1": (-24, -17), "I2": (-5, -19), "O1": (8, 8), "O2": (9, 0)}
        ax_side.annotate(row["id"], (length, y), xytext=offsets[row["id"]],
                         textcoords="offset points", color=color, fontsize=9, fontweight="bold")
    ax_side.set(xlim=(-2.55, 2.85), ylim=(.8, -2.5),
                xlabel="local length (m); allowed −2 to +2",
                ylabel="camera Y down (m)", title="Orthographic side: length–Y")
    ax_side.grid(alpha=.15)
    ax_side.text(.02, .03, "O2: Y=+0.4 is below bottom Y=0. Top is Y=−2.",
                 transform=ax_side.transAxes, fontsize=8, color="#9b302b")

    fig.suptitle("Yawed synthetic cuboid: bottom center (X,Y,Z) = (0,0,10) m",
                 fontsize=16, y=.97)
    fig.text(.5, .895, "KITTI camera Y points downward. Blue box: length 4 m, width 2 m, height 2 m, yaw +45°."
             " Orthographic views resolve 3D perspective overlap.", ha="center", fontsize=10)
    handles = [Line2D([], [], color="#007d4f", marker="o", linestyle="None", markersize=8,
                      label="Inside (I1, I2)"),
               Line2D([], [], color="#c23b32", marker="^", linestyle="None", markersize=8,
                      label="Outside (O1, O2)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .135), ncol=2,
               fontsize=10, frameon=True)
    fig.text(.5, .085, "O1: outside X–Z footprint (local length and width exceed limits)."
             "    O2: below cuboid bottom despite X–Z overlap.", ha="center", fontsize=9)
    fig.text(.5, .025, "Synthetic geometry test — not KITTI data or model performance.",
             ha="center", fontsize=10, color="#7c2330")
    fig.savefig(IMAGE, dpi=160, facecolor="white")
    plt.close(fig)
    print(IMAGE)
    print(TABLE)


if __name__ == "__main__":
    main()
