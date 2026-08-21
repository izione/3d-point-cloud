"""Visualizes the detection head's center heatmap (sigmoid(center_logit) at
every active stem voxel) as two 2D max-projections: xy (BEV, max over z) and
yz (max over x). This is the same score tensor decode.py thresholds/peak-
picks into boxes -- here it's rendered pre-thresholding so you can see the
raw heatmap shape (diffuse vs. sharp peaks, false-hot regions, z-offset
between GT and peak, etc.).

Frame selection: either
  --frame Person1,89,208        (person, scene number, frame number -- works
                                  for ANY person/scene on disk, incl. Person3/4
                                  which aren't in any configured split)
or
  --split val --frame_index 3   (index within a configured split, as before)
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data.dataset import SonarDiverDataset, collate_fn, _label_path_for
from models.detector import DiverDetector
from models.decode import decode_detections
from models.sparse_ops import build_index_grid

AXIS_NAMES = {1: "x", 2: "y", 3: "z"}


def load_named_frame(cfg, person, scene_num, frame_num):
    """Loads one frame by (person, scene number, frame number) directly off
    disk, bypassing the split scene lists in cfg -- so this works for scenes
    excluded from every split (e.g. Person3/Person4)."""
    root = Path(cfg["DATA"]["ROOT"])
    pc_range = np.array(cfg["DATA"]["POINT_CLOUD_RANGE"], dtype=np.float32)
    scene_id = f"{person}/scene_{int(scene_num):04d}"
    sonar_path = root / scene_id / "sonar" / f"frame_{int(frame_num):06d}.bin"
    if not sonar_path.is_file():
        raise FileNotFoundError(f"no such frame: {sonar_path}")

    data = np.fromfile(sonar_path, dtype=np.float32)
    points = data.reshape(-1, 4) if data.size > 0 else np.zeros((0, 4), dtype=np.float32)
    mask = (
        (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) &
        (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) &
        (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
    )
    points = points[mask]

    label_path = _label_path_for(sonar_path)
    if label_path.is_file():
        with open(label_path) as f:
            label = json.load(f)
        objs = label.get("objects", [])
        gt_boxes = np.zeros((len(objs), 10), dtype=np.float32)
        for i, obj in enumerate(objs):
            c, d, q = obj["centroid"], obj["dimensions"], obj["quaternion"]
            gt_boxes[i] = [c["x"], c["y"], c["z"], d["length"], d["width"], d["height"],
                           q["w"], q["x"], q["y"], q["z"]]
    else:
        gt_boxes = np.zeros((0, 10), dtype=np.float32)

    return {
        "points": torch.from_numpy(points),
        "gt_boxes": torch.from_numpy(gt_boxes),
        "frame_id": str(sonar_path.relative_to(root)),
    }


def find_frame_with_gt(ds, frame_index):
    if frame_index is not None:
        return frame_index
    for i in range(len(ds)):
        if ds[i]["gt_boxes"].shape[0] > 0:
            return i
    raise RuntimeError("no frame in this split has any GT objects")


def parse_frame_arg(s):
    person, scene, frame = s.split(",")
    return person.strip(), int(scene), int(frame)


def max_project(coords_b, scores, axis_a, axis_b):
    """coords_b: (N,4) [batch,x,y,z] filtered to one batch item. Returns a
    (size_b, size_a) array (NaN where no voxel is active) of the max score
    seen at each (axis_a, axis_b) index pair, projecting out the third axis."""
    a = coords_b[:, axis_a].numpy()
    b = coords_b[:, axis_b].numpy()
    s = scores.numpy()
    size_a = int(a.max()) + 1 if a.size > 0 else 1
    size_b = int(b.max()) + 1 if b.size > 0 else 1
    grid = np.full((size_b, size_a), np.nan, dtype=np.float32)
    for ai, bi, si in zip(a, b, s):
        cur = grid[bi, ai]
        if np.isnan(cur) or si > cur:
            grid[bi, ai] = si
    return grid, size_a, size_b


def plot_panel(ax, fig, grid, extent, cmap, points_a, points_b, gt_a, gt_b, det_a, det_b, det_s,
               xlabel, ylabel, title, score_threshold, show_legend):
    ax.set_facecolor("#1a1a2e")
    ax.scatter(points_a, points_b, s=1, c="lightgray", alpha=0.25, zorder=1, label="points")
    im = ax.imshow(grid, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=1,
                    alpha=0.85, zorder=2, aspect="equal")
    if gt_a.size > 0:
        ax.scatter(gt_a, gt_b, s=140, marker="*", facecolors="none",
                   edgecolors="cyan", linewidths=1.5, zorder=3, label="GT center")
    if det_a.size > 0:
        ax.scatter(det_a, det_b, s=60, marker="x", c="lime", linewidths=2,
                   zorder=4, label=f"pred peak (>{score_threshold})")
        for a, b, s in zip(det_a, det_b, det_s):
            ax.text(a, b + 0.3, f"{s:.2f}", color="lime", fontsize=7, ha="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="sigmoid(center_logit)", fraction=0.03, pad=0.02)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/linear_20.pth")
    parser.add_argument("--config", default=None, help="defaults to the config stored in the checkpoint")
    parser.add_argument("--frame", type=parse_frame_arg, default=None,
                         help="Person,scene,frame e.g. Person1,89,208 -- overrides --split/--frame_index")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--frame_index", type=int, default=None, help="index within the split; default picks the first frame with >=1 GT object")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="only used to mark decoded peaks, not to filter the heatmap itself")
    parser.add_argument("--out", default="figures/center_map.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = yaml.safe_load(open(args.config)) if args.config else ckpt["cfg"]

    model = DiverDetector(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.frame is not None:
        person, scene_num, frame_num = args.frame
        sample = load_named_frame(cfg, person, scene_num, frame_num)
        frame_label = f"{person}/scene_{scene_num:04d} frame {frame_num}"
    else:
        ds = SonarDiverDataset(cfg, args.split)
        idx = find_frame_with_gt(ds, args.frame_index)
        sample = ds[idx]
        frame_label = f"{args.split} frame {idx}"

    batch = collate_fn([sample])

    with torch.no_grad():
        losses, pred, stem_coords, assign_result = model.loss(batch, device)
        scores = torch.sigmoid(pred["center_logit"].squeeze(-1)).detach().cpu()
        stem_coords_cpu = stem_coords.detach().cpu()

        stem_grid_size = tuple(int(stem_coords_cpu[:, i].max().item()) + 1 if stem_coords_cpu.shape[0] > 0 else 1 for i in (1, 2, 3))
        index_grid = build_index_grid(stem_coords, batch["batch_size"], stem_grid_size, device=device)
        eff_voxel_size = model.voxel_size * model.stem_stride
        dets = decode_detections(pred, stem_coords, index_grid, stem_grid_size, model.pc_range, eff_voxel_size, args.score_threshold)

    det = dets[0]
    center_loss = losses["center"].item()
    gt_boxes = sample["gt_boxes"]
    points = sample["points"].numpy()
    pc_range = model.pc_range.cpu().numpy()
    eff_vx = eff_voxel_size.cpu().numpy()

    m0 = stem_coords_cpu[:, 0] == 0
    coords_b = stem_coords_cpu[m0]
    scores_b = scores[m0]

    xy, X, Y = max_project(coords_b, scores_b, axis_a=1, axis_b=2)  # (Y,X): project out z
    yz, Y2, Z = max_project(coords_b, scores_b, axis_a=2, axis_b=3)  # (Z,Y): project out x

    x0, y0, z0 = pc_range[0], pc_range[1], pc_range[2]
    vx, vy, vz = eff_vx
    extent_xy = [x0, x0 + X * vx, y0, y0 + Y * vy]
    extent_yz = [y0, y0 + Y2 * vy, z0, z0 + Z * vz]

    gt_c = gt_boxes[:, :3].numpy() if gt_boxes.shape[0] > 0 else np.zeros((0, 3))
    det_c = det["center"].numpy() if det["center"].shape[0] > 0 else np.zeros((0, 3))
    det_s = det["score"].numpy() if det["score"].shape[0] > 0 else np.zeros((0,))

    fig, (ax_xy, ax_yz) = plt.subplots(1, 2, figsize=(15, 7))
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color="#1a1a2e")

    plot_panel(ax_xy, fig, xy, extent_xy, cmap, points[:, 0], points[:, 1],
               gt_c[:, 0], gt_c[:, 1], det_c[:, 0], det_c[:, 1], det_s,
               "x (m)", "y (m)", "xy plane (max over z)", args.score_threshold, show_legend=True)
    plot_panel(ax_yz, fig, yz, extent_yz, cmap, points[:, 1], points[:, 2],
               gt_c[:, 1], gt_c[:, 2], det_c[:, 1], det_c[:, 2], det_s,
               "y (m)", "z (m)", "yz plane (max over x)", args.score_threshold, show_legend=False)

    fig.suptitle(f"center heatmap -- {frame_label} ({sample['frame_id']}) -- center_loss={center_loss:.4f}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved to {args.out}  (n_active_voxels={coords_b.shape[0]}, n_gt={gt_c.shape[0]}, "
          f"n_pred_peaks={det_c.shape[0]}, max_score={scores_b.max().item() if scores_b.numel() else float('nan'):.3f}, "
          f"center_loss={center_loss:.4f})")


if __name__ == "__main__":
    main()
