"""Visualizes the detection head's offset predictions (the sub-voxel
correction added to a voxel's center to get the final box center) as green
quiver arrows on top of the same xy/yz center-heatmap panels from
visualize_center_map.py.

Offset is only meaningful where the center score is actually high (a
low-score voxel's offset was never trained to point anywhere in particular),
so arrows are only drawn for voxels with score >= --score_threshold -- same
threshold decode.py uses to pick peaks, reused here as the "which voxels
matter" cutoff.

Frame selection: either
  --frame Person1,89,208        (person, scene number, frame number -- works
                                  for ANY person/scene on disk, incl. Person3/4
                                  which aren't in any configured split)
or
  --split val --frame_index 3   (index within a configured split, as before)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data.dataset import SonarDiverDataset, collate_fn
from models.detector import DiverDetector
from models.decode import decode_detections
from models.sparse_ops import build_index_grid
from visualize_center_map import (
    load_named_frame, find_frame_with_gt, parse_frame_arg, max_project, plot_panel,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/linear_20.pth")
    parser.add_argument("--config", default=None, help="defaults to the config stored in the checkpoint")
    parser.add_argument("--frame", type=parse_frame_arg, default=None,
                         help="Person,scene,frame e.g. Person1,89,208 -- overrides --split/--frame_index")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--frame_index", type=int, default=None, help="index within the split; default picks the first frame with >=1 GT object")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="offset arrows are only drawn for voxels with score >= this (also used for decode's peak marking)")
    parser.add_argument("--margin_voxels", type=float, default=4, help="zoom the view to the bounding box of all GT centers / pred peaks / arrow tips, padded by this many stem voxels on each side (0 disables zoom)")
    parser.add_argument("--out", default="figures/offset_map.png")
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
        offset = pred["offset"].detach().cpu()  # (N,3) meters, world-aligned: center = voxel_world + offset
        stem_coords_cpu = stem_coords.detach().cpu()

        stem_grid_size = tuple(int(stem_coords_cpu[:, i].max().item()) + 1 if stem_coords_cpu.shape[0] > 0 else 1 for i in (1, 2, 3))
        index_grid = build_index_grid(stem_coords, batch["batch_size"], stem_grid_size, device=device)
        eff_voxel_size = model.voxel_size * model.stem_stride
        dets = decode_detections(pred, stem_coords, index_grid, stem_grid_size, model.pc_range, eff_voxel_size, args.score_threshold)

    det = dets[0]
    center_loss = losses["center"].item()
    offset_loss = losses["offset"].item()
    gt_boxes = sample["gt_boxes"]
    points = sample["points"].numpy()
    pc_range = model.pc_range.cpu().numpy()
    eff_vx = eff_voxel_size.cpu().numpy()

    m0 = stem_coords_cpu[:, 0] == 0
    coords_b = stem_coords_cpu[m0]
    scores_b = scores[m0]
    offset_b = offset[m0]

    xy, X, Y = max_project(coords_b, scores_b, axis_a=1, axis_b=2)  # project out z
    yz, Y2, Z = max_project(coords_b, scores_b, axis_a=2, axis_b=3)  # project out x

    x0, y0, z0 = pc_range[0], pc_range[1], pc_range[2]
    vx, vy, vz = eff_vx
    extent_xy = [x0, x0 + X * vx, y0, y0 + Y * vy]
    extent_yz = [y0, y0 + Y2 * vy, z0, z0 + Z * vz]

    gt_c = gt_boxes[:, :3].numpy() if gt_boxes.shape[0] > 0 else np.zeros((0, 3))
    det_c = det["center"].numpy() if det["center"].shape[0] > 0 else np.zeros((0, 3))
    det_s = det["score"].numpy() if det["score"].shape[0] > 0 else np.zeros((0,))

    # only voxels whose center score clears the threshold have a meaningful offset
    hi = (scores_b >= args.score_threshold).numpy()
    voxel_world_b = pc_range[:3] + (coords_b[:, 1:4].numpy() + 0.5) * eff_vx
    arrow_origin = voxel_world_b[hi]
    arrow_vec = offset_b.numpy()[hi]

    fig, (ax_xy, ax_yz) = plt.subplots(1, 2, figsize=(15, 7))
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color="#1a1a2e")

    plot_panel(ax_xy, fig, xy, extent_xy, cmap, points[:, 0], points[:, 1],
               gt_c[:, 0], gt_c[:, 1], det_c[:, 0], det_c[:, 1], det_s,
               "x (m)", "y (m)", "xy plane (max over z)", args.score_threshold, show_legend=True)
    plot_panel(ax_yz, fig, yz, extent_yz, cmap, points[:, 1], points[:, 2],
               gt_c[:, 1], gt_c[:, 2], det_c[:, 1], det_c[:, 2], det_s,
               "y (m)", "z (m)", "yz plane (max over x)", args.score_threshold, show_legend=False)

    if arrow_origin.shape[0] > 0:
        ax_xy.quiver(arrow_origin[:, 0], arrow_origin[:, 1], arrow_vec[:, 0], arrow_vec[:, 1],
                     color="limegreen", angles="xy", scale_units="xy", scale=1, width=0.005,
                     zorder=5, label=f"offset (score≥{args.score_threshold})")
        ax_yz.quiver(arrow_origin[:, 1], arrow_origin[:, 2], arrow_vec[:, 1], arrow_vec[:, 2],
                     color="limegreen", angles="xy", scale_units="xy", scale=1, width=0.005, zorder=5)
        ax_xy.legend(loc="upper right", fontsize=8)

    # zoom to a tight box around everything "center"-related (GT centers, pred
    # peaks, the high-score voxels the arrows start from, and their tips) --
    # the raw arrows are ~voxel-sized and invisible against the full ~16x23m scene
    if args.margin_voxels > 0:
        arrow_tip = arrow_origin + arrow_vec if arrow_origin.shape[0] > 0 else arrow_origin
        all_centers = np.concatenate([gt_c, det_c, arrow_origin, arrow_tip], axis=0)
        if all_centers.shape[0] > 0:
            pad = args.margin_voxels * eff_vx
            lo = all_centers.min(axis=0) - pad
            hi = all_centers.max(axis=0) + pad
            ax_xy.set_xlim(lo[0], hi[0])
            ax_xy.set_ylim(lo[1], hi[1])
            ax_yz.set_xlim(lo[1], hi[1])
            ax_yz.set_ylim(lo[2], hi[2])

    fig.suptitle(f"offset vectors (score≥{args.score_threshold}) -- {frame_label} ({sample['frame_id']}) -- "
                 f"center_loss={center_loss:.4f}, offset_loss={offset_loss:.4f}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved to {args.out}  (n_active_voxels={coords_b.shape[0]}, n_high_score={arrow_origin.shape[0]}, "
          f"n_gt={gt_c.shape[0]}, n_pred_peaks={det_c.shape[0]}, center_loss={center_loss:.4f}, offset_loss={offset_loss:.4f})")


if __name__ == "__main__":
    main()
