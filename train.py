import argparse
import functools
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SonarDiverDataset, collate_fn
from models.detector import DiverDetector


def build_dataloader(cfg, split, batch_size, shuffle, num_workers):
    ds = SonarDiverDataset(cfg, split)
    pc_range = cfg["DATA"]["POINT_CLOUD_RANGE"]
    voxel_size = cfg["DATA"]["VOXEL_SIZE"]
    grid_size = [round((pc_range[3 + i] - pc_range[i]) / voxel_size[i]) for i in range(3)]
    cfn = functools.partial(collate_fn, pc_range=pc_range, voxel_size=voxel_size, grid_size=grid_size)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=cfn, drop_last=shuffle,  # drop_last only for train (shuffle=True)
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
    }, path)


@torch.no_grad()
def run_validation(model, val_loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        losses, _, _, _ = model.loss(batch, device)
        total += losses["total"].item()
        n += 1
    model.train()
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="path to a checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="override OPTIMIZATION.NUM_EPOCHS")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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
        print(f"resumed from {args.resume} at epoch {start_epoch}, step {global_step} "
              f"(epoch_complete={ckpt.get('epoch_complete')})")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_every_epochs = opt_cfg["CKPT_EVERY_N_EPOCHS"]
    ckpt_every_steps = opt_cfg["CKPT_EVERY_N_STEPS"]

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

            pbar.set_postfix({
                "loss": f"{losses['total'].item():.3f}",
                "center": f"{losses['center'].item():.3f}",
                "n_pos": int(assign_result["pos_mask"].sum().item()),
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if ckpt_every_steps and global_step % ckpt_every_steps == 0:
                save_checkpoint(ckpt_dir / f"step_{global_step}.pth", model, optimizer, scheduler, epoch, global_step, cfg, epoch_complete=False)

        epoch_time = time.time() - epoch_t0
        avg_train_loss = running_loss / max(len(train_loader), 1)
        val_loss = run_validation(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} time={epoch_time:.1f}s")

        if ckpt_every_epochs and (epoch + 1) % ckpt_every_epochs == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch}.pth", model, optimizer, scheduler, epoch, global_step, cfg, epoch_complete=True)

    save_checkpoint(ckpt_dir / "last.pth", model, optimizer, scheduler, num_epochs - 1, global_step, cfg, epoch_complete=True)
    print("training complete.")


if __name__ == "__main__":
    main()
