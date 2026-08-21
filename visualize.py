import argparse

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import yaml

from data.dataset import SonarDiverDataset, collate_fn
from models.detector import DiverDetector
from models.decode import decode_detections
from models.sparse_ops import build_index_grid
from models.box_utils import box_to_corners, quat_to_rotmat, axis_aligned_iou_3d

# local x/y/z distinguished by line style since color is already used for GT vs pred
AXIS_STYLES = {"x": "-", "y": "--", "z": ":"}

# Edges of the cuboid, indexed into the fixed corner order from box_utils._SIGNS
# (every +/- combination of (l/2, w/2, h/2), sx-major/sy/sz-minor).
_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
]


def plot_box(ax, corners, color, label=None):
    corners = corners.numpy()
    for i, (a, b) in enumerate(_EDGES):
        xs, ys, zs = zip(corners[a], corners[b])
        ax.plot(xs, ys, zs, color=color, linewidth=1.5, label=label if i == 0 else None)


def plot_axes(ax, center, size, quat, color):
    """Draws the box's own rotated x/y/z axes as arrows from its center out to
    the middle of each face (center + R @ e_i * size_i/2), so the box's
    orientation (from its quaternion) is visible, not just its extent."""
    R = quat_to_rotmat(quat.unsqueeze(0))[0]  # world = R @ local + center, so local e_i -> world dir R[:, i]
    center = center.numpy()
    size = size.numpy()
    for i, name in enumerate(("x", "y", "z")):
        tip = center + R[:, i].numpy() * (size[i] / 2)
        ax.quiver(*center, *(tip - center), color=color, linewidth=2.2, linestyle=AXIS_STYLES[name],
                  arrow_length_ratio=0.25)


def best_ious(center, size, other_center, other_size):
    """For each box in (center,size) (N,3)/(N,3), the max axis-aligned IoU against
    any box in (other_center,other_size) (M,3)/(M,3) -- 0.0 if the other set is empty
    or every pair is disjoint. Not a 1:1 match (see test.py for that); just "how well
    does this box overlap with something on the other side" for the annotation."""
    ious = []
    for i in range(center.shape[0]):
        best = 0.0
        for j in range(other_center.shape[0]):
            best = max(best, axis_aligned_iou_3d(center[i], size[i], other_center[j], other_size[j]))
        ious.append(best)
    return ious


def annotate_iou(ax, center, size, iou, color, lift=0.15):
    """Small IoU label floating just above the box (its own color, best-match IoU
    against the other set). `lift` staggers GT vs pred labels apart when their
    boxes nearly coincide."""
    x, y, z = center.numpy()
    z_top = z + size[2].item() / 2 + lift
    ax.text(x, y, z_top, f"IoU {iou:.2f}", color=color, fontsize=8, ha="center", va="bottom")


def find_frame_with_gt(ds, frame_index):
    if frame_index is not None:
        return frame_index
    for i in range(len(ds)):
        if ds[i]["gt_boxes"].shape[0] > 0:
            return i
    raise RuntimeError(f"no frame in this split has any GT objects")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="defaults to the config stored in the checkpoint")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--frame_index", type=int, default=None, help="index within the split; default picks the first frame with >=1 GT object")
    parser.add_argument("--score_threshold", type=float, default=0.3)
    parser.add_argument("--out", default="../figures/example_detection.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = yaml.safe_load(open(args.config)) if args.config else ckpt["cfg"]

    model = DiverDetector(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = SonarDiverDataset(cfg, args.split)
    idx = find_frame_with_gt(ds, args.frame_index)
    sample = ds[idx]
    batch = collate_fn([sample])

    with torch.no_grad():
        pred, stem_coords, gt_boxes_list = model.forward(batch, device)
        stem_grid_size = tuple(int(stem_coords[:, i].max().item()) + 1 if stem_coords.shape[0] > 0 else 1 for i in (1, 2, 3))
        index_grid = build_index_grid(stem_coords, batch["batch_size"], stem_grid_size, device=device)
        eff_voxel_size = model.voxel_size * model.stem_stride
        dets = decode_detections(pred, stem_coords, index_grid, stem_grid_size, model.pc_range, eff_voxel_size, args.score_threshold)

    det = dets[0]
    gt_boxes = gt_boxes_list[0].cpu()
    points = sample["points"].numpy()

    pred_corners = box_to_corners(det["center"], det["size"], det["quat"]) if det["center"].shape[0] > 0 else torch.zeros(0, 8, 3)
    gt_corners = box_to_corners(gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6:10]) if gt_boxes.shape[0] > 0 else torch.zeros(0, 8, 3)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    if points.shape[0] > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c="gray", alpha=0.3, label="points")

    gt_ious = best_ious(gt_boxes[:, :3], gt_boxes[:, 3:6], det["center"], det["size"])
    pred_ious = best_ious(det["center"], det["size"], gt_boxes[:, :3], gt_boxes[:, 3:6])

    for i in range(gt_corners.shape[0]):
        plot_box(ax, gt_corners[i], "red", label="GT" if i == 0 else None)
        plot_axes(ax, gt_boxes[i, :3], gt_boxes[i, 3:6], gt_boxes[i, 6:10], "red")
        annotate_iou(ax, gt_boxes[i, :3], gt_boxes[i, 3:6], gt_ious[i], "red", lift=0.15)
    for i in range(pred_corners.shape[0]):
        plot_box(ax, pred_corners[i], "blue", label="pred" if i == 0 else None)
        plot_axes(ax, det["center"][i], det["size"][i], det["quat"][i], "blue")
        annotate_iou(ax, det["center"][i], det["size"][i], pred_ious[i], "blue", lift=0.40)

    # Zoom to the objects: at full scene scale (~16m x ~20m) a diver-sized box
    # (and its axis arrows) is nearly invisible against the whole point cloud.
    all_corners = torch.cat([gt_corners.reshape(-1, 3), pred_corners.reshape(-1, 3)], dim=0)
    if all_corners.shape[0] > 0:
        lo, hi = all_corners.min(dim=0).values.numpy(), all_corners.max(dim=0).values.numpy()
        pad = np.clip((hi - lo) * 0.6, 0.5, None)
        lo, hi = lo - pad, hi + pad
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"{args.split} frame {idx} ({sample['frame_id']}) -- GT (red) vs pred (blue)")
    handles, labels = ax.get_legend_handles_labels()
    axis_legend = [Line2D([0], [0], color="black", linestyle=ls, label=f"local {name}") for name, ls in AXIS_STYLES.items()]
    ax.legend(handles=handles + axis_legend)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved to {args.out}  (n_gt={gt_corners.shape[0]}, n_det={pred_corners.shape[0]})")


if __name__ == "__main__":
    main()
