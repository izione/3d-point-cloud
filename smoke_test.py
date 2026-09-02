"""Dry-runs all 4 experiment configs (configs/exp_*.yaml) end-to-end -- model
construction, forward, loss, backward -- on synthetic random points/GT boxes
instead of the real sonar dataset. Meant to catch config/shape/wiring bugs
*before* the real dataset (data/dataset.py's SonarDiverDataset) is ready,
since that's currently blocked on data collection.

Not a substitute for training on real data (random points obviously won't
converge to anything meaningful) -- just proves each of the 4 architectures
in the "dense vs sparse vs sparse+SlotFormer(3L/6L)" comparison actually
builds and trains one step without crashing.

Usage:
    python smoke_test.py
    python smoke_test.py --configs configs/exp_dense.yaml  # just one
"""
import argparse
import glob
import time

import torch

from config_utils import load_config
from models.detector import DiverDetector

DEFAULT_CONFIGS = sorted(glob.glob("configs/exp_*.yaml") + glob.glob("experiments/*/*.yaml"))


def make_synthetic_batch(cfg, batch_size=2, points_per_sample=2000, num_gt_per_sample=3, device="cpu"):
    """Random points uniformly inside DATA.POINT_CLOUD_RANGE (so voxelization/
    index-grid clipping behaves like on real data) + random GT boxes, in the
    exact dict shape data/dataset.py's collate_fn produces."""
    pc_range = torch.tensor(cfg["DATA"]["POINT_CLOUD_RANGE"], dtype=torch.float32)
    lo, hi = pc_range[:3], pc_range[3:]

    all_points, all_batch_idx, gt_boxes_list = [], [], []
    for b in range(batch_size):
        xyz = lo + torch.rand(points_per_sample, 3) * (hi - lo)
        intensity = torch.rand(points_per_sample, 1)
        pts = torch.cat([xyz, intensity], dim=1)
        all_points.append(pts)
        all_batch_idx.append(torch.full((points_per_sample,), b, dtype=torch.long))

        centers = lo + torch.rand(num_gt_per_sample, 3) * (hi - lo)
        sizes = 0.3 + torch.rand(num_gt_per_sample, 3) * 1.5  # length,width,height in a plausible diver-sized range
        quat = torch.randn(num_gt_per_sample, 4)
        quat = quat / quat.norm(dim=1, keepdim=True)  # random unit quaternion
        gt_boxes_list.append(torch.cat([centers, sizes, quat], dim=1))

    return {
        "points": torch.cat(all_points, dim=0).to(device),
        "point_batch_idx": torch.cat(all_batch_idx, dim=0).to(device),
        "gt_boxes": [g.to(device) for g in gt_boxes_list],
        "frame_ids": [f"synthetic_{b}" for b in range(batch_size)],
        "batch_size": batch_size,
    }


def run_one(config_path, device):
    print(f"\n{'=' * 70}\n{config_path}\n{'=' * 70}")
    cfg = load_config(config_path)
    print(f"BACKBONE.TYPE={cfg['BACKBONE'].get('TYPE', 'auto')}  "
          f"SLOTFORMER.ENABLED={cfg['SLOTFORMER']['ENABLED']}  "
          f"NUM_CYCLES={cfg['SLOTFORMER'].get('NUM_CYCLES')}")

    model = DiverDetector(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = make_synthetic_batch(cfg, device=device)

    t0 = time.perf_counter()
    losses, pred, stem_coords, assign_result = model.loss(batch, device)
    fwd_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    optimizer.zero_grad()
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["OPTIMIZATION"]["GRAD_NORM_CLIP"])
    optimizer.step()
    bwd_s = time.perf_counter() - t0

    n_pos = int(assign_result["pos_mask"].sum().item())
    loss_str = "  ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
    print(f"stem voxels: {stem_coords.shape[0]}  n_pos assigned: {n_pos}")
    print(f"forward: {fwd_s * 1000:.1f}ms  backward+step: {bwd_s * 1000:.1f}ms")
    print(f"losses: {loss_str}")

    for name, t in losses.items():
        assert torch.isfinite(t).all(), f"{config_path}: non-finite loss in '{name}'"
    print("OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if not args.configs:
        raise SystemExit("no configs found (expected configs/exp_*.yaml)")

    failures = []
    for config_path in args.configs:
        try:
            run_one(config_path, device)
        except Exception as e:
            failures.append((config_path, e))
            print(f"FAILED: {config_path}: {e!r}")

    print(f"\n{'=' * 70}\n{len(args.configs) - len(failures)}/{len(args.configs)} configs passed")
    if failures:
        for config_path, e in failures:
            print(f"  FAIL: {config_path}: {e!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
