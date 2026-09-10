"""train.py - CenterPoint reproduction, training entry point.

Trains model.CenterPointVoxelNet against a yaw-only-flattened GT rotation
target by default (dataset.zyaw_flatten -- matches the CenterPoint paper's
own scope: nuScenes/Waymo boxes are yaw-only). Eval GT is always full 3D
(dataset.gt_boxes_from_objects on the UNFLATTENED objects) regardless of
--full3d, so AP3D numbers stay comparable to the sibling voxelnet_baseline
repo's own ablation program either way.

Validates on ALL val-split frames every epoch (no --val_start_epoch skip,
unlike an anchor-head ablation in this same research program -- CenterPoint's
decode is anchor-free and NMS-free, so an undertrained early model doesn't
produce an unbounded pre-NMS candidate blowup; full-frame val stays cheap
from epoch 0).

Usage:
    python train.py                       # yaw-only (paper-faithful) rotation target
    python train.py --full3d              # full 3D rotation target instead
    python train.py --smoke               # tiny run, CLI/pipeline sanity check only
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import config
from center_loss import center_voxelnet_loss
from dataset import VoxelDataset, collate_fn, zyaw_flatten
from eval import evaluate_full
import heatmap_targets as ht
from model import CenterPointVoxelNet

VAL_LOG_IOUS = config.IOU_THRESHOLDS


def lr_at_epoch(epoch, total_epochs, base_lr, decay_frac, decay_factor):
    """Step decay -- VoxelNet paper's own recipe (last ~15% of epochs at lr/10)."""
    if epoch >= total_epochs * decay_frac:
        return base_lr * decay_factor
    return base_lr


def build_batch_center_targets(gt_objects_list, flatten_yaw: bool, device):
    """Per-frame heatmap_targets.build_heatmap_targets, stacked into a batch."""
    hm, mask, off, z, dim, rot = [], [], [], [], [], []
    for objects in gt_objects_list:
        obj = zyaw_flatten(objects) if flatten_yaw else objects
        t = ht.build_heatmap_targets(obj)
        hm.append(t["heatmap"]); mask.append(t["reg_mask"]); off.append(t["offset"])
        z.append(t["z"]); dim.append(t["dim"]); rot.append(t["rot"])
    return {
        "heatmap": torch.from_numpy(np.stack(hm)).to(device),
        "reg_mask": torch.from_numpy(np.stack(mask)).to(device),
        "offset": torch.from_numpy(np.stack(off)).to(device),
        "z": torch.from_numpy(np.stack(z)).to(device),
        "dim": torch.from_numpy(np.stack(dim)).to(device),
        "rot": torch.from_numpy(np.stack(rot)).to(device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--lr_decay_epoch_frac", type=float, default=config.LR_DECAY_EPOCH_FRAC)
    parser.add_argument("--lr_decay_factor", type=float, default=config.LR_DECAY_FACTOR)
    parser.add_argument("--momentum", type=float, default=config.MOMENTUM)
    parser.add_argument("--num_workers", type=int, default=2,
                         help="Colab's CPU core count is usually small (2) -- raise this if "
                              "you have more cores and GPU util looks CPU-starved (nvidia-smi).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt_dir", default="checkpoints_centerpoint")
    parser.add_argument("--full3d", action="store_true",
                         help="train against the full 3D rotation GT instead of the "
                              "paper-faithful yaw-only-flattened target (default off)")
    parser.add_argument("--smoke", action="store_true",
                         help="truncate train/val to a handful of frames -- pipeline sanity check only")
    parser.add_argument("--resume", default=None,
                         help="path to a checkpoint (e.g. checkpoints_centerpoint/epoch_7.pth) to "
                              "resume from -- restores model weights and continues at epoch+1 for "
                              "the remaining --epochs. Best-AP tracking restarts at -1 on resume "
                              "(only affects which epoch this run prints as \"best\" at the end); "
                              "every epoch's checkpoint is still saved regardless, so a disconnect "
                              "never loses a completed epoch.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cpu":
        print("WARNING: no CUDA GPU found -- on Colab, check Runtime > Change runtime type > GPU (T4+)")

    train_ds = VoxelDataset("train")
    val_ds = VoxelDataset("val")
    if args.smoke:
        train_ds.samples = train_ds.samples[:8]
        val_ds.samples = val_ds.samples[:4]
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                               num_workers=args.num_workers, generator=loader_generator)
    print(f"train frames: {len(train_ds)}  val frames: {len(val_ds)}  batches/epoch: {len(train_loader)}")

    model = CenterPointVoxelNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    rot_mode = "full3d" if args.full3d else "yaw-only (paper-faithful)"
    print(f"CenterPoint  rotation_target={rot_mode}  params={n_params:,}")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    loss_keys = ["hm_loss", "reg_loss"]

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from {args.resume} (epoch {ckpt['epoch']}) -- continuing at epoch {start_epoch}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "loss_history.csv"
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    ap_cols = []
    for thr in VAL_LOG_IOUS:
        tag = f"{int(round(thr * 100)):02d}"
        ap_cols += [f"ap_iou{tag}", f"precision_iou{tag}", f"recall_iou{tag}"]
    if log_path.stat().st_size == 0:
        log_writer.writerow(["phase", "epoch", "lr", "n_pos", "total"] + loss_keys + ap_cols)
        log_file.flush()

    best_ap35, best_epoch = -1.0, None
    for epoch in range(start_epoch, args.epochs):
        lr = lr_at_epoch(epoch, args.epochs, args.lr, args.lr_decay_epoch_frac, args.lr_decay_factor)
        for g in optimizer.param_groups:
            g["lr"] = lr

        model.train()
        t0 = time.time()
        running = {k: 0.0 for k in loss_keys}
        running.update(loss=0.0, n_pos=0, n=0)
        iterable = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs - 1}", unit="batch") \
            if HAS_TQDM else train_loader
        for batch in iterable:
            voxel_features = batch["voxel_features"].to(device)
            num_points = batch["num_points"].to(device)
            coords = batch["coords"].to(device)

            pred = model(voxel_features, num_points, coords)
            target = build_batch_center_targets(batch["gt_objects"], not args.full3d, device)
            loss, stats = center_voxelnet_loss(
                pred["heatmap"], pred["offset"], pred["z"], pred["dim"], pred["rot"],
                target["heatmap"], target["reg_mask"], target["offset"], target["z"],
                target["dim"], target["rot"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()

            running["loss"] += loss.item(); running["n_pos"] += stats["n_pos"]; running["n"] += 1
            for k in loss_keys:
                running[k] += stats[k]
            if HAS_TQDM:
                iterable.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr:.4f}", n_pos=stats["n_pos"])

        n = max(running["n"], 1)
        row = ["train", epoch, lr, running["n_pos"] / n, running["loss"] / n]
        row += [running[k] / n for k in loss_keys]
        row += [""] * len(ap_cols)
        log_writer.writerow(row); log_file.flush()
        print(f"epoch {epoch} done in {time.time() - t0:.1f}s avg_loss={running['loss'] / n:.4f} lr={lr:.4f}")

        ckpt_path = ckpt_dir / f"epoch_{epoch}.pth"
        torch.save({"model": model.state_dict(), "epoch": epoch, "full3d": args.full3d}, ckpt_path)

        vt0 = time.time()
        val = evaluate_full(model, device, val_ds)
        row = ["val", epoch, "", "", ""] + [""] * len(loss_keys)
        for thr in VAL_LOG_IOUS:
            ap_t, p_t, r_t = val[thr]
            row += [ap_t, p_t, r_t]
        log_writer.writerow(row); log_file.flush()

        ap35 = val[config.VAL_TARGET_IOU][0]
        improved = ap35 > best_ap35
        if improved:
            best_ap35, best_epoch = ap35, epoch
        log_zone = " ".join(f"iou{int(round(t * 100))}={val[t][0]:.4f}" for t in (0.3, 0.35, 0.4))
        print(f"epoch {epoch} val AP3D[{log_zone}] ({time.time() - vt0:.1f}s)"
              + ("  <- best@35" if improved else ""))

    log_file.close()
    print(f"done. best epoch={best_epoch} AP3D@0.35={best_ap35:.4f} -> {ckpt_dir}/epoch_{best_epoch}.pth")


if __name__ == "__main__":
    main()
