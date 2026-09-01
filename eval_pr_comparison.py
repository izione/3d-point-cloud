"""Compares the 4 backbone/SlotFormer experiments' trained checkpoints on one
PR-curve figure: one subplot per --iou_thresholds value, one line per
checkpoint per subplot (AP in the legend). Reuses test.py's collect_pr_data/
pr_curve_for_threshold/compute_ap so the matching logic (volume-only 3D IoU,
score-ranked greedy 1:1, VOC-style AP) is identical to a single-model
`test.py --pr_curve_out` run -- this script only adds the multi-model overlay.

Default checkpoints are each experiment's best-val_loss epoch (found by
`awk -F, '$1=="val"{print $2,$6}' checkpoints_X/X_loss_history.csv | sort -k2 -n`
as of 2026-08-31): sparse_no_slotformer epoch 29, sparse_slotformer_3l epoch
26, sparse_slotformer_6l epoch 18. dense has no trained checkpoint yet (see
README/session notes -- OOM'd at BATCH_SIZE 64, and BATCH_SIZE 2 measured
~107h for 40 epochs, so it was skipped) and is left out here; pass
--checkpoints to include it once one exists.

Usage:
    python eval_pr_comparison.py
    python eval_pr_comparison.py --split val --iou_thresholds 0.3 0.5 0.7
    python eval_pr_comparison.py --checkpoints my_label=path/to/ckpt.pth ...
"""
import argparse

import torch
from torch.utils.data import DataLoader

import models.slotformer as slotformer
from config_utils import load_config
from data.dataset import SonarDiverDataset, collate_fn
from models.detector import DiverDetector
from test import collect_pr_data, pr_curve_for_threshold

DEFAULT_CHECKPOINTS = {
    "sparse_no_slotformer": "checkpoints_sparse_no_slotformer/sparse_no_slotformer_epoch_29.pth",
    "sparse_slotformer_3l": "checkpoints_sparse_slotformer_3l/sparse_slotformer_3l_epoch_26.pth",
    "sparse_slotformer_6l": "checkpoints_sparse_slotformer_6l/sparse_slotformer_6l_epoch_18.pth",
}

# one distinct color per model, stable across subplots regardless of dict order
MODEL_COLORS = {
    "sparse_no_slotformer": "#e0574a",
    "sparse_slotformer_3l": "#3987e5",
    "sparse_slotformer_6l": "#2c9e5c",
    "dense": "#a35fce",
}
FALLBACK_COLORS = ["#e0574a", "#3987e5", "#2c9e5c", "#a35fce", "#c98a1f", "#4a4a4a"]


def parse_checkpoints(pairs):
    out = {}
    for p in pairs:
        name, path = p.split("=", 1)
        out[name] = path
    return out


@torch.no_grad()
def collect_for_checkpoint(ckpt_path, split, batch_size, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    if ckpt.get("attention_kind"):
        slotformer.ATTENTION_KIND = ckpt["attention_kind"]

    model = DiverDetector(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    ds = SonarDiverDataset(cfg, split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True)

    print(f"  {len(ds)} {split} frames, epoch {ckpt.get('epoch')}, attention_kind={slotformer.ATTENTION_KIND}")
    return collect_pr_data(model, loader, device)


def plot_comparison(results, iou_thresholds, out_path):
    """results: dict[name] -> (frame_data, total_gt). One subplot per IoU
    threshold, one line per model, laid out in a single row."""
    import matplotlib.pyplot as plt

    n = len(iou_thresholds)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.6), facecolor="#f5f8f9")
    if n == 1:
        axes = [axes]

    names = list(results.keys())
    colors = {name: MODEL_COLORS.get(name, FALLBACK_COLORS[i % len(FALLBACK_COLORS)]) for i, name in enumerate(names)}

    for ax, iou_t in zip(axes, iou_thresholds):
        ax.set_facecolor("#ffffff")
        for name in names:
            frame_data, total_gt = results[name]
            recalls, precisions, ap = pr_curve_for_threshold(frame_data, total_gt, iou_t)
            ax.plot(recalls, precisions, color=colors[name], linewidth=2.2, label=f"{name}  (AP {ap:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"IoU >= {iou_t:.2f}")
        ax.grid(color="#dde5e7", linewidth=1)
        for spine_name, spine in ax.spines.items():
            spine.set_visible(spine_name in ("left", "bottom"))
            spine.set_color("#c4ced0")
        ax.legend(fontsize=9, loc="lower left")

    fig.suptitle("Precision-recall by architecture", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor="#f5f8f9", bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=None,
                         help="override the default 3 checkpoints, as name=path pairs "
                              "(e.g. --checkpoints dense=checkpoints_dense/dense_last.pth)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--iou_thresholds", type=float, nargs="+", default=[0.30, 0.35, 0.40])
    parser.add_argument("--out", default="pr_curve_comparison.png")
    args = parser.parse_args()

    checkpoints = parse_checkpoints(args.checkpoints) if args.checkpoints else DEFAULT_CHECKPOINTS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    results = {}
    for name, ckpt_path in checkpoints.items():
        print(f"[{name}] {ckpt_path}")
        results[name] = collect_for_checkpoint(ckpt_path, args.split, args.batch_size, device)

    print(f"\nIoU thresholds: {args.iou_thresholds}")
    for iou_t in args.iou_thresholds:
        print(f"\n--- IoU >= {iou_t:.2f} ---")
        for name in checkpoints:
            frame_data, total_gt = results[name]
            recalls, precisions, ap = pr_curve_for_threshold(frame_data, total_gt, iou_t)
            final_p = precisions[-1] if precisions else float("nan")
            final_r = recalls[-1] if recalls else float("nan")
            print(f"  {name:24s} AP={ap:.4f}  final_precision={final_p:.4f}  final_recall={final_r:.4f}")

    plot_comparison(results, args.iou_thresholds, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
