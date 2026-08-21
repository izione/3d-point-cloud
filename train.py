import argparse
import csv
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import models.slotformer as slotformer
from data.dataset import SonarDiverDataset, collate_fn
from models.detector import DiverDetector

LOSS_KEYS = ["total", "center", "corner_aux", "offset", "size", "rotation"]


class LossLogger:
    """Appends one CSV row per train step and per val epoch to `path` (put it
    under --ckpt_dir so it survives a Colab session the same way checkpoints
    do). Kept as one file with a `phase` column rather than separate
    train/val files, and flushed every row -- this is exactly the kind of
    data a disconnect mid-run shouldn't lose. Column order: phase, epoch,
    step, lr, n_pos, then LOSS_KEYS. Reopening an existing file appends
    instead of overwriting, so it's resume-safe without any extra bookkeeping."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        self.file = open(self.path, "a", newline="")
        self.writer = csv.writer(self.file)
        if is_new:
            self.writer.writerow(["phase", "epoch", "step", "lr", "n_pos"] + LOSS_KEYS)
            self.file.flush()

    def log_train_step(self, epoch, step, losses, lr, n_pos):
        self.writer.writerow(["train", epoch, step, lr, n_pos] + [losses[k].item() for k in LOSS_KEYS])
        self.file.flush()

    def log_val_epoch(self, epoch, step, avg_losses):
        self.writer.writerow(["val", epoch, step, "", ""] + [avg_losses[k] for k in LOSS_KEYS])
        self.file.flush()

    def close(self):
        self.file.close()


def build_dataloader(cfg, split, batch_size, shuffle, num_workers):
    ds = SonarDiverDataset(cfg, split)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate_fn, drop_last=shuffle,  # drop_last only for train (shuffle=True)
        pin_memory=True, persistent_workers=(num_workers > 0),
    )


def save_checkpoint(path, model, optimizer, scheduler, epoch, step, cfg, epoch_complete):
    """epoch_complete=True only when saved *after* finishing all of `epoch`'s
    batches (the epoch-boundary checkpoints). Step-checkpoints save mid-epoch,
    so resuming from one must re-run that whole epoch rather than skip to the
    next -- there's no record of exactly which batches were already seen."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "epoch_complete": epoch_complete,
        "cfg": cfg,
        "attention_kind": slotformer.ATTENTION_KIND,
    }, path)


@torch.no_grad()
def run_validation(model, val_loader, device):
    """Returns per-component averages (not just total) so the loss history
    log can show val trends per term too -- useful for spotting a term that's
    overfitting differently from the others when it's time to retune weights."""
    model.eval()
    sums = {k: 0.0 for k in LOSS_KEYS}
    n = 0
    for batch in val_loader:
        losses, _, _, _ = model.loss(batch, device)
        for k in LOSS_KEYS:
            sums[k] += losses[k].item()
        n += 1
    model.train()
    n = max(n, 1)
    return {k: v / n for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="path to a checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="override OPTIMIZATION.NUM_EPOCHS")
    parser.add_argument("--log_file", default=None, help="defaults to <ckpt_dir>/loss_history.csv")
    parser.add_argument("--attention_kind", choices=["softmax", "linear"], default=None,
                         help="override models.slotformer.ATTENTION_KIND (default: whatever's hardcoded there)")
    parser.add_argument("--center_sigma", type=float, default=None,
                         help="override LOSS.CENTER_SIGMA_M (meters) -- the gaussian sigma for the soft "
                              "center/corner heatmap target; shrink this to sharpen center localization")
    args = parser.parse_args()

    if args.attention_kind:
        slotformer.ATTENTION_KIND = args.attention_kind
    print(f"SlotFormer attention kind: {slotformer.ATTENTION_KIND}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.center_sigma is not None:
        cfg["LOSS"]["CENTER_SIGMA_M"] = args.center_sigma
    print(f"LOSS.CENTER_SIGMA_M: {cfg['LOSS']['CENTER_SIGMA_M']}")

    opt_cfg = cfg["OPTIMIZATION"]
    num_epochs = args.epochs or opt_cfg["NUM_EPOCHS"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cpu":
        print("WARNING: no CUDA GPU found -- this will be very slow. Intended to run on Colab.")

    train_loader = build_dataloader(cfg, "train", opt_cfg["BATCH_SIZE"], shuffle=True, num_workers=opt_cfg["NUM_WORKERS"])
    val_loader = build_dataloader(cfg, "val", opt_cfg["BATCH_SIZE"], shuffle=False, num_workers=opt_cfg["NUM_WORKERS"])
    print(f"train batches/epoch: {len(train_loader)}  val batches: {len(val_loader)}")

    model = DiverDetector(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["LR"], weight_decay=opt_cfg["WEIGHT_DECAY"])
    steps_per_epoch = len(train_loader)
    # +15% slack: a mid-epoch resume re-runs that epoch's steps, so the true
    # step count can exceed steps_per_epoch*num_epochs. OneCycleLR raises a
    # hard error if .step() is called past total_steps, so this buffer is
    # what keeps a resumed run from crashing near the end of training. With
    # zero resumes the schedule just never quite reaches its planned minimum
    # LR -- a minor imperfection, safer than the alternative.
    total_steps = int(steps_per_epoch * num_epochs * 1.15)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=opt_cfg["LR"], total_steps=total_steps,
        pct_start=opt_cfg["PCT_START"], div_factor=opt_cfg["DIV_FACTOR"],
        base_momentum=opt_cfg["MOMS"][1], max_momentum=opt_cfg["MOMS"][0],
    )

    start_epoch, global_step = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1 if ckpt.get("epoch_complete") else ckpt["epoch"]
        global_step = ckpt["step"]
        if not args.attention_kind and ckpt.get("attention_kind"):
            slotformer.ATTENTION_KIND = ckpt["attention_kind"]  # keep resumed run consistent with how it was trained
        print(f"resumed from {args.resume} at epoch {start_epoch}, step {global_step} "
              f"(epoch_complete={ckpt.get('epoch_complete')}, attention_kind={slotformer.ATTENTION_KIND})")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_every_epochs = opt_cfg["CKPT_EVERY_N_EPOCHS"]
    ckpt_every_steps = opt_cfg["CKPT_EVERY_N_STEPS"]
    log_path = Path(args.log_file) if args.log_file else ckpt_dir / "loss_history.csv"
    logger = LossLogger(log_path)
    print(f"logging per-step/per-epoch losses to {log_path}")

    model.train()
    for epoch in range(start_epoch, num_epochs):
        epoch_t0 = time.time()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{num_epochs - 1}")
        for batch in pbar:
            losses, _, _, assign_result = model.loss(batch, device)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg["GRAD_NORM_CLIP"])
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += losses["total"].item()
            lr = scheduler.get_last_lr()[0]
            n_pos = int(assign_result["pos_mask"].sum().item())
            logger.log_train_step(epoch, global_step, losses, lr, n_pos)

            pbar.set_postfix({
                "loss": f"{losses['total'].item():.3f}",
                "center": f"{losses['center'].item():.3f}",
                "n_pos": n_pos,
                "lr": f"{lr:.2e}",
            })

            if ckpt_every_steps and global_step % ckpt_every_steps == 0:
                save_checkpoint(ckpt_dir / f"step_{global_step}.pth", model, optimizer, scheduler, epoch, global_step, cfg, epoch_complete=False)

        epoch_time = time.time() - epoch_t0
        avg_train_loss = running_loss / max(len(train_loader), 1)
        val_losses = run_validation(model, val_loader, device)
        logger.log_val_epoch(epoch, global_step, val_losses)
        print(f"epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={val_losses['total']:.4f} time={epoch_time:.1f}s")

        if ckpt_every_epochs and (epoch + 1) % ckpt_every_epochs == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch}.pth", model, optimizer, scheduler, epoch, global_step, cfg, epoch_complete=True)

    save_checkpoint(ckpt_dir / "last.pth", model, optimizer, scheduler, num_epochs - 1, global_step, cfg, epoch_complete=True)
    logger.close()
    print("training complete.")


if __name__ == "__main__":
    main()
