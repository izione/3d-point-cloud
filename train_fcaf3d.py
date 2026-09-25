"""Sibling to train_detr.py for the FCAF3D-style dense head
(models/detector_fcaf3d.py) -- same overall shape (build_dataloader/
save_checkpoint reused from train.py, chunked-resume mid-epoch skip reused
from train_detr.py's own fix) but no matcher/denoising/aux-loss bookkeeping
since there's no decoder here.
"""
import argparse
import csv
import time
from pathlib import Path

import torch

import models.slotformer as slotformer
from config_utils import load_config
from models.detector_fcaf3d import DiverDetectorFCAF3D
from models.box_utils import oriented_iou_3d_sampled, quat_to_rotmat
from train import build_dataloader, save_checkpoint
from test import pr_curve_for_threshold

LOSS_KEYS = ["total", "cls", "iou", "center", "size", "rotation", "centerness"]

# AP@0.35 with a rotation-AWARE IoU (not the axis-aligned one test.py's own
# iou_matrix uses) -- ground truth here has full 3D orientation, and this
# project's own axis_aligned_iou_3d would silently score two boxes that are
# actually disjoint-but-rotated as heavily overlapping (see
# models/box_utils.py::oriented_iou_3d_sampled's docstring / the test case
# that motivated it). 0.35 rather than 0.5 since decode()'s per-voxel dense
# candidates are still coarser than a single well-centered query -- 0.5 is
# the DETR head's own threshold (configs/exp_detr_head.yaml), not
# transplanted here without evidence it's still the right cutoff.
AP_IOU_THRESHOLD = 0.35
# decode() with score_threshold=0 keeps every active voxel across both levels
# (thousands per frame) -- computing a rotation-aware IoU (grid-sampled, much
# more expensive than the axis-aligned closed form) against that many
# candidates every epoch would be far too slow. Filtering to a plausible
# working point (same values evaluated by hand against the real trained
# checkpoint: recall=0.98, precision=0.76 at these settings) keeps candidate
# counts per frame small enough to run this every epoch without materially
# slowing training.
AP_SCORE_THRESHOLD = 0.1
AP_NMS_RADIUS = 0.5


def rotation_aware_iou_matrix(det: dict, gt_boxes: torch.Tensor) -> torch.Tensor:
    """det: dict with 'center'/'size'/'rot_matrix' (Nd,3)/(Nd,3)/(Nd,3,3) from
    DiverDetectorFCAF3D.decode(). gt_boxes: (Ng,10) [center(3),size(3),quat(4)].
    Returns (Nd,Ng) rotation-aware IoU (see box_utils.oriented_iou_3d_sampled)."""
    nd, ng = det["center"].shape[0], gt_boxes.shape[0]
    ious = torch.zeros(nd, ng)
    if nd == 0 or ng == 0:
        return ious
    gt_R = quat_to_rotmat(gt_boxes[:, 6:10])
    for di in range(nd):
        for gi in range(ng):
            ious[di, gi] = oriented_iou_3d_sampled(
                det["center"][di], det["size"][di], det["rot_matrix"][di],
                gt_boxes[gi, :3], gt_boxes[gi, 3:6], gt_R[gi],
            )
    return ious


class LossLoggerFCAF3D:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        self.file = open(self.path, "a", newline="")
        self.writer = csv.writer(self.file)
        if is_new:
            self.writer.writerow(["phase", "epoch", "step", "lr", "n_pos"] + LOSS_KEYS + ["ap35"])
            self.file.flush()

    def log_train_step(self, epoch, step, losses, lr):
        self.writer.writerow(["train", epoch, step, lr, losses["n_pos"]] + [losses[k].item() for k in LOSS_KEYS] + [""])
        self.file.flush()

    def log_val_epoch(self, epoch, step, avg_losses, ap35):
        self.writer.writerow(["val", epoch, step, "", avg_losses["n_pos"]] + [avg_losses[k] for k in LOSS_KEYS] + [ap35])
        self.file.flush()

    def close(self):
        self.file.close()


@torch.no_grad()
def run_validation(model, val_loader, device):
    """Reuses the same forward pass model.loss() already ran for the val-loss
    terms to also decode (NMS + score threshold, see AP_SCORE_THRESHOLD/
    AP_NMS_RADIUS) and build each frame's rotation-aware IoU matrix, so
    AP@0.35 costs no extra forward passes -- only decode()'s and
    oriented_iou_3d_sampled's overhead on top of validation that already ran
    every epoch."""
    model.eval()
    sums = {k: 0.0 for k in LOSS_KEYS}
    n_pos_total, n = 0, 0
    frame_data, total_gt = [], 0
    for batch in val_loader:
        losses, level_preds, level_batch_idx, gt_boxes_list = model.loss(batch, device)
        for k in LOSS_KEYS:
            sums[k] += losses[k].item()
        n_pos_total += losses["n_pos"]
        n += 1

        dets = model.decode(level_preds, level_batch_idx, len(gt_boxes_list),
                             score_threshold=AP_SCORE_THRESHOLD, nms_radius=AP_NMS_RADIUS)
        for b, gt_boxes in enumerate(gt_boxes_list):
            gt_boxes = gt_boxes.cpu()
            det = dets[b]
            total_gt += gt_boxes.shape[0]
            frame_data.append({"scores": det["score"], "ious": rotation_aware_iou_matrix(det, gt_boxes)})
    model.train()
    n = max(n, 1)
    avg = {k: v / n for k, v in sums.items()}
    avg["n_pos"] = n_pos_total
    _, _, ap = pr_curve_for_threshold(frame_data, total_gt, AP_IOU_THRESHOLD)
    return avg, ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp_fcaf3d.yaml")
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

    model = DiverDetectorFCAF3D(cfg).to(device)

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
    logger = LossLoggerFCAF3D(log_path)
    print(f"logging per-step/per-epoch losses to {log_path}")

    # see train_detr.py's own copy of this fix for why: resuming from a
    # mid-epoch (not epoch-complete) checkpoint must skip the already-trained
    # batches of that epoch, or every resumed chunk quietly re-trains the
    # start of the same epoch instead of making forward progress.
    resume_skip_steps = 0
    if ckpt is not None and not ckpt.get("epoch_complete"):
        resume_skip_steps = global_step - start_epoch * steps_per_epoch
        if resume_skip_steps > 0:
            print(f"resuming mid-epoch: skipping the first {resume_skip_steps} "
                  f"already-trained batches of epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, num_epochs):
        epoch_t0 = time.time()
        running_loss = 0.0
        n_steps_run = 0
        skip = resume_skip_steps if epoch == start_epoch else 0
        for step, batch in enumerate(train_loader):
            if step < skip:
                continue
            losses, _, _, _ = model.loss(batch, device)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg["GRAD_NORM_CLIP"])
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += losses["total"].item()
            n_steps_run += 1
            lr = scheduler.get_last_lr()[0]
            logger.log_train_step(epoch, global_step, losses, lr)

            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{steps_per_epoch}  loss={losses['total'].item():.3f}  "
                      f"n_pos={losses['n_pos']}  lr={lr:.2e}")

            if ckpt_every_steps and global_step % ckpt_every_steps == 0:
                save_checkpoint(ckpt_dir / f"{exp_name}_step_{global_step}.pth", model, optimizer, scheduler, epoch, global_step, cfg, epoch_complete=False)

        epoch_time = time.time() - epoch_t0
        avg_train_loss = running_loss / max(n_steps_run, 1)
        val_losses, val_ap35 = run_validation(model, val_loader, device)
        logger.log_val_epoch(epoch, global_step, val_losses, val_ap35)
        print(f"epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={val_losses['total']:.4f} "
              f"val_AP@{AP_IOU_THRESHOLD:.2f}={val_ap35:.4f} time={epoch_time:.1f}s")

        if ckpt_every_epochs and (epoch + 1) % ckpt_every_epochs == 0:
            save_checkpoint(ckpt_dir / f"{exp_name}_epoch_{epoch}.pth", model, optimizer, scheduler, epoch, global_step, cfg, epoch_complete=True)

    save_checkpoint(ckpt_dir / f"{exp_name}_last.pth", model, optimizer, scheduler, num_epochs - 1, global_step, cfg, epoch_complete=True)
    logger.close()
    print("training complete.")


if __name__ == "__main__":
    main()
