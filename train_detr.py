"""Sibling to train.py for the DETR-head variant (models/detector_detr.py) --
kept separate rather than branching inside train.py because the loss-dict keys
and "how many positives" concept differ enough (bipartite-matched queries vs.
per-voxel assignment) that a shared loop would need more branches than it's
worth. Reuses train.py's build_dataloader/save_checkpoint/LossLogger as-is.
"""
import argparse
import csv
import time
from pathlib import Path

import torch

import models.slotformer as slotformer
from config_utils import load_config
from models.detector_detr import DiverDetectorDETR
from train import build_dataloader, save_checkpoint
from test import iou_matrix, pr_curve_for_threshold

LOSS_KEYS = ["total", "cls", "center", "size", "rotation", "dn_cls", "dn_center", "dn_size", "dn_rotation"]
AP_IOU_THRESHOLD = 0.5


class LossLoggerDetr:
    """Same append-one-row-per-step/epoch CSV convention as train.py's
    LossLogger, just parameterized on this file's own LOSS_KEYS (train.py's
    version hardcodes the dense head's 6 keys, which don't match this loss)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        self.file = open(self.path, "a", newline="")
        self.writer = csv.writer(self.file)
        if is_new:
            self.writer.writerow(["phase", "epoch", "step", "lr", "n_pos"] + LOSS_KEYS + ["ap50"])
            self.file.flush()

    def log_train_step(self, epoch, step, losses, lr, n_pos):
        self.writer.writerow(["train", epoch, step, lr, n_pos] + [losses[k].item() for k in LOSS_KEYS] + [""])
        self.file.flush()

    def log_val_epoch(self, epoch, step, avg_losses, ap50):
        self.writer.writerow(["val", epoch, step, "", ""] + [avg_losses[k] for k in LOSS_KEYS] + [ap50])
        self.file.flush()

    def close(self):
        self.file.close()


@torch.no_grad()
def run_validation(model, val_loader, device):
    """Reuses the same forward pass model.loss() already ran for the val-loss
    terms to also decode + collect each frame's det x GT IoU matrix (test.py's
    pr_curve_for_threshold expects), so AP@0.5 costs no extra forward passes --
    only decode()'s (cheap, per-query threshold) and axis_aligned_iou_3d's
    (closed-form) overhead on top of validation that already ran every epoch."""
    model.eval()
    sums = {k: 0.0 for k in LOSS_KEYS}
    n = 0
    frame_data, total_gt = [], 0
    for batch in val_loader:
        losses, pred, gt_boxes_list, _ = model.loss(batch, device)
        for k in LOSS_KEYS:
            sums[k] += losses[k].item()
        n += 1

        dets = model.decode(pred, score_threshold=0.0)
        for b, gt_boxes in enumerate(gt_boxes_list):
            gt_boxes = gt_boxes.cpu()
            det = dets[b]
            total_gt += gt_boxes.shape[0]
            frame_data.append({"scores": det["score"], "ious": iou_matrix(det, gt_boxes)})
    model.train()
    n = max(n, 1)
    avg_losses = {k: v / n for k, v in sums.items()}
    _, _, ap50 = pr_curve_for_threshold(frame_data, total_gt, AP_IOU_THRESHOLD)
    return avg_losses, ap50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp_detr_head.yaml")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--attention_kind", choices=["softmax", "linear"], default=None)
    args = parser.parse_args()

    exp_name = args.exp_name or Path(args.config).stem
    print(f"exp_name: {exp_name}")

    if args.attention_kind:
        slotformer.ATTENTION_KIND = args.attention_kind
    print(f"SlotFormer attention kind: {slotformer.ATTENTION_KIND}")

    cfg = load_config(args.config)
    opt_cfg = cfg["OPTIMIZATION"]
    num_epochs = args.epochs or opt_cfg["NUM_EPOCHS"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cpu":
        print("WARNING: no CUDA GPU found -- this will be very slow. Intended to run on Colab.")

    train_loader = build_dataloader(cfg, "train", opt_cfg["BATCH_SIZE"], shuffle=True, num_workers=opt_cfg["NUM_WORKERS"])
    val_loader = build_dataloader(cfg, "val", opt_cfg["BATCH_SIZE"], shuffle=False, num_workers=opt_cfg["NUM_WORKERS"])
    print(f"train batches/epoch: {len(train_loader)}  val batches: {len(val_loader)}")

    model = DiverDetectorDETR(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["LR"], weight_decay=opt_cfg["WEIGHT_DECAY"])
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs

    start_epoch, global_step = 0, 0
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1 if ckpt.get("epoch_complete") else ckpt["epoch"]
        global_step = ckpt["step"]
        if not args.attention_kind and ckpt.get("attention_kind"):
            slotformer.ATTENTION_KIND = ckpt["attention_kind"]
        print(f"resumed from {args.resume} at epoch {start_epoch}, step {global_step}")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=opt_cfg["LR"], total_steps=total_steps,
        pct_start=opt_cfg["PCT_START"], div_factor=opt_cfg["DIV_FACTOR"],
        base_momentum=opt_cfg["MOMS"][1], max_momentum=opt_cfg["MOMS"][0],
        last_epoch=(global_step - 1) if ckpt is not None else -1,
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_every_epochs = opt_cfg["CKPT_EVERY_N_EPOCHS"]
    ckpt_every_steps = opt_cfg["CKPT_EVERY_N_STEPS"]
    log_path = Path(args.log_file) if args.log_file else ckpt_dir / f"{exp_name}_loss_history.csv"
    logger = LossLoggerDetr(log_path)
    print(f"logging per-step/per-epoch losses to {log_path}")

    model.train()
    for epoch in range(start_epoch, num_epochs):
        epoch_t0 = time.time()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            losses, _, _, matches = model.loss(batch, device)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg["GRAD_NORM_CLIP"])
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += losses["total"].item()
            lr = scheduler.get_last_lr()[0]
            n_pos = sum(qi.numel() for qi, _ in matches)
            logger.log_train_step(epoch, global_step, losses, lr, n_pos)

            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{steps_per_epoch}  loss={losses['total'].item():.3f}  "
                      f"n_pos={n_pos}  lr={lr:.2e}")

            if ckpt_every_steps and global_step % ckpt_every_steps == 0:
                save_checkpoint(ckpt_dir / f"{exp_name}_step_{global_step}.pth", model, optimizer, scheduler, epoch, global_step, cfg, epoch_complete=False)

        epoch_time = time.time() - epoch_t0
        avg_train_loss = running_loss / max(len(train_loader), 1)
        val_losses, val_ap50 = run_validation(model, val_loader, device)
        logger.log_val_epoch(epoch, global_step, val_losses, val_ap50)
        print(f"epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={val_losses['total']:.4f} "
              f"val_AP@{AP_IOU_THRESHOLD:.1f}={val_ap50:.4f} time={epoch_time:.1f}s")

        if ckpt_every_epochs and (epoch + 1) % ckpt_every_epochs == 0:
            save_checkpoint(ckpt_dir / f"{exp_name}_epoch_{epoch}.pth", model, optimizer, scheduler, epoch, global_step, cfg, epoch_complete=True)

    save_checkpoint(ckpt_dir / f"{exp_name}_last.pth", model, optimizer, scheduler, num_epochs - 1, global_step, cfg, epoch_complete=True)
    logger.close()
    print("training complete.")


if __name__ == "__main__":
    main()
