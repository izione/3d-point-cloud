"""Compares per-frame compute time of three architecture-matched 3D backbones
(same stage_channels / num_blocks_per_stage / kernel / stride from
configs/*.yaml's BACKBONE section, so the comparison isolates "which conv
implementation", not "different-sized models"):

  - sparse (pytorch): models/backbone3d.Sparse3DBackbone -- this project's
    own sparse conv, built entirely on plain PyTorch ops (no native
    extensions), so it runs anywhere including CPU-only Colab images.
  - sparse (spconv):  models/backbone3d_spconv.Sparse3DBackboneSpconv -- a
    real hash-table-based sparse conv backend (CUDA-only). Skipped
    automatically if spconv isn't installed/usable here (see
    models/backbone3d_auto.py's probe).
  - dense:            models/backbone3d_dense.Dense3DBackbone -- an ordinary
    nn.Conv3d backbone over the full dense voxel grid, as a "what if we
    didn't bother with sparsity at all" baseline.

For each *sparse* backbone (pytorch/spconv), also times models/slotformer.py's
SlotFormerBackbone run on that backbone's output voxels, at each
--slotformer_cycles depth (NUM_CYCLES=1 -> 3 layers (one x/y/z pass),
NUM_CYCLES=2 -> 6 layers -- this is the axial-attention refinement stage that
actually runs after the 3D backbone in models/detector.py, see
configs/default.yaml's SLOTFORMER section). SlotFormer is deliberately NOT
run after the dense backbone: it exists to give sparse voxels a long-range
receptive field despite their local kernels, which a dense conv backbone
already has for free from convolving over the whole grid -- dense + SlotFormer
isn't a combination the actual model ever uses.

Everything else (heads, assigner, loss) is left out -- only VFE (to get real
per-voxel input features from real data) -> backbone [-> SlotFormer, for the
sparse backbones] is timed.

Usage:
    python benchmark_backbone.py --config configs/default.yaml
    python benchmark_backbone.py --num_frames 50   # quick pass instead of the whole dataset
    python benchmark_backbone.py --skip_slotformer  # backbone-only, like before
"""
import argparse
import time

import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm

import models.sparse_ops as sparse_ops
from config_utils import load_config
from data.dataset import SonarDiverDataset, voxelize_batch
from models.vfe import VFE
from models.sparse_ops import build_index_grid
from models.backbone3d import Sparse3DBackbone
from models.backbone3d_dense import Dense3DBackbone
from models.backbone3d_auto import spconv_usable
from models.slotformer import SlotFormerBackbone


def build_full_dataset(cfg):
    """All frames across train/val/test pooled together -- this benchmark
    only cares about realistic point-cloud density, not the train/val/test
    split semantics."""
    splits = []
    for split in ("train", "val", "test"):
        try:
            splits.append(SonarDiverDataset(cfg, split))
        except RuntimeError:
            pass
    return ConcatDataset(splits)


def sparse_to_dense(features, coords, batch_size, grid_size):
    """(N,C) sparse voxel features + (N,4) [batch,x,y,z] coords -> dense
    (B,C,X,Y,Z) grid, zero-filled everywhere there's no active voxel."""
    X, Y, Z = grid_size
    C = features.shape[1]
    dense = torch.zeros(batch_size, C, X, Y, Z, device=features.device, dtype=features.dtype)
    if coords.shape[0] > 0:
        dense[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]] = features
    return dense


def summarize(name, times_sec, n_voxels):
    if not times_sec:
        print(f"{name}: no successful runs")
        return
    t = torch.tensor(times_sec)
    v = torch.tensor(n_voxels, dtype=torch.float32)
    print(f"{name}: mean={t.mean() * 1000:.3f}ms  median={t.median() * 1000:.3f}ms  "
          f"min={t.min() * 1000:.3f}ms  max={t.max() * 1000:.3f}ms  "
          f"n={len(times_sec)}  avg_active_voxels(full-res input)={v.mean():.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_root", default=None, help="override DATA.ROOT from the config")
    parser.add_argument("--num_frames", type=int, default=None,
                         help="cap on number of frames to benchmark (default: the whole dataset)")
    parser.add_argument("--warmup", type=int, default=5,
                         help="frames run first and excluded from the average (JIT/cudnn/allocator warmup)")
    parser.add_argument("--skip_dense", action="store_true",
                         help="skip the dense backbone -- its input tensor is the full-res grid (huge at small "
                              "voxel sizes) and can OOM; use this to still get sparse-only numbers if that happens")
    parser.add_argument("--skip_spconv", action="store_true", help="skip the spconv backbone even if it's usable")
    parser.add_argument("--skip_slotformer", action="store_true",
                         help="don't run SlotFormer after the sparse backbones -- backbone-only timing")
    parser.add_argument("--slotformer_cycles", type=int, nargs="+", default=[1, 2],
                         help="NUM_CYCLES values to benchmark SlotFormer at (each cycle = 3 layers, one x/y/z "
                             "pass) -- default compares 1 cycle/3 layers vs 2 cycles/6 layers")
    parser.add_argument("--sparse_conv_mode", choices=["loop", "vectorized"], default=None,
                         help="override models.sparse_ops.CONV_MODE for the pure-PyTorch sparse backbone "
                              "(default: whatever's hardcoded there -- 'loop' is tuned for CPU, 'vectorized' "
                              "batches all k^3 kernel offsets into fewer/larger GPU launches and is usually the "
                              "one to try on CUDA)")
    args = parser.parse_args()

    if args.sparse_conv_mode:
        sparse_ops.CONV_MODE = args.sparse_conv_mode
    print(f"sparse_ops.CONV_MODE: {sparse_ops.CONV_MODE}")

    cfg = load_config(args.config)
    if args.data_root:
        cfg["DATA"]["ROOT"] = args.data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cpu":
        print("WARNING: no CUDA GPU found -- dense conv3d over the full-res grid can be very slow/memory-heavy on CPU.")

    pc_range = torch.tensor(cfg["DATA"]["POINT_CLOUD_RANGE"], dtype=torch.float32, device=device)
    voxel_size = torch.tensor(cfg["DATA"]["VOXEL_SIZE"], dtype=torch.float32, device=device)
    grid_size = tuple(round((pc_range[3 + i].item() - pc_range[i].item()) / voxel_size[i].item()) for i in range(3))
    print(f"full-res grid size (x,y,z): {grid_size}  (voxel_size={cfg['DATA']['VOXEL_SIZE']})")

    vfe = VFE(num_filters=cfg["VFE"]["NUM_FILTERS"]).to(device).eval()

    bcfg = cfg["BACKBONE"]
    backbone_args = (vfe.out_channels, bcfg["STAGE_CHANNELS"], bcfg["NUM_BLOCKS_PER_STAGE"],
                      bcfg["DOWNSAMPLE_KERNEL"], bcfg["DOWNSAMPLE_STRIDE"])
    print(f"backbone: stage_channels={bcfg['STAGE_CHANNELS']}  "
          f"num_blocks_per_stage={bcfg['NUM_BLOCKS_PER_STAGE']}  "
          f"kernel={bcfg['DOWNSAMPLE_KERNEL']}  stride={bcfg['DOWNSAMPLE_STRIDE']} (same for all backbones)")

    sparse_backbone = Sparse3DBackbone(*backbone_args).to(device).eval()

    spconv_backbone = None
    if not args.skip_spconv and spconv_usable():
        from models.backbone3d_spconv import Sparse3DBackboneSpconv
        spconv_backbone = Sparse3DBackboneSpconv(*backbone_args).to(device).eval()
        print("spconv backbone: enabled")
    else:
        print("spconv backbone: skipped (not usable here or --skip_spconv)")

    dense_backbone = None if args.skip_dense else Dense3DBackbone(*backbone_args).to(device).eval()

    scfg = cfg["SLOTFORMER"]
    slotformers = {}  # num_cycles -> module
    if not args.skip_slotformer:
        for c in args.slotformer_cycles:
            slotformers[c] = SlotFormerBackbone(
                bcfg["STAGE_CHANNELS"][-1], scfg["WIN_SIZE"], c, scfg["NUM_HEADS"],
            ).to(device).eval()
        print(f"SlotFormer: win_size={scfg['WIN_SIZE']}  num_heads={scfg['NUM_HEADS']}  "
              f"cycles tested={args.slotformer_cycles} (layers={[3 * c for c in args.slotformer_cycles]})")
    else:
        print("SlotFormer: skipped (--skip_slotformer)")

    sparse_backends = {"sparse (pytorch)": sparse_backbone}
    if spconv_backbone is not None:
        sparse_backends["sparse (spconv)"] = spconv_backbone

    dataset = build_full_dataset(cfg)
    n_total = len(dataset)
    n_frames = min(args.num_frames, n_total) if args.num_frames else n_total
    print(f"dataset frames available: {n_total}, benchmarking: {n_frames}")

    results = {name: ([], []) for name in sparse_backends}
    for name in sparse_backends:
        for c in slotformers:
            results[f"{name} + slotformer({3 * c}L)"] = ([], [])
    if dense_backbone is not None:
        results["dense"] = ([], [])
    skipped = {name: 0 for name in results}

    grid_size_t = torch.tensor(grid_size, device=device)

    def timed(fn):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    with torch.no_grad():
        for i in tqdm(range(n_frames)):
            sample = dataset[i]
            points = sample["points"].to(device)
            if points.shape[0] == 0:
                continue
            point_batch_idx = torch.zeros(points.shape[0], dtype=torch.long, device=device)
            voxel_coords, point_voxel_idx = voxelize_batch(points, point_batch_idx, pc_range, voxel_size, grid_size_t)
            num_voxels = voxel_coords.shape[0]
            if num_voxels == 0:
                continue

            vfe_out = vfe(points, point_voxel_idx, voxel_coords, num_voxels, pc_range, voxel_size)
            index_grid = build_index_grid(voxel_coords, 1, grid_size, device=device)
            is_warmup = i < args.warmup

            for name, backbone in sparse_backends.items():
                try:
                    out = {}
                    bb_elapsed = timed(lambda: out.update(f=backbone(vfe_out, voxel_coords, index_grid, grid_size, 1)))
                    bb_feat, bb_coords = out["f"][0], out["f"][1]
                    if not is_warmup:
                        results[name][0].append(bb_elapsed)
                        results[name][1].append(num_voxels)

                    for c, slotformer in slotformers.items():
                        sf_out = {}
                        sf_elapsed = timed(lambda: sf_out.update(f=slotformer(bb_feat, bb_coords)))
                        combo_name = f"{name} + slotformer({3 * c}L)"
                        if not is_warmup:
                            results[combo_name][0].append(bb_elapsed + sf_elapsed)
                            results[combo_name][1].append(num_voxels)
                except RuntimeError as e:
                    skipped[name] += 1
                    if skipped[name] <= 3:
                        print(f"\n[frame {i}] {name} failed ({e}) -- skipping this frame")

            if dense_backbone is not None:
                try:
                    dense_in = sparse_to_dense(vfe_out, voxel_coords, 1, grid_size)
                    elapsed = timed(lambda: dense_backbone(dense_in))
                    if not is_warmup:
                        results["dense"][0].append(elapsed)
                        results["dense"][1].append(num_voxels)
                    del dense_in
                except RuntimeError as e:
                    skipped["dense"] += 1
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if skipped["dense"] <= 3:
                        print(f"\n[frame {i}] dense backbone failed ({e}) -- skipping this frame "
                              f"(likely OOM on the full-res dense grid)")

    print()
    for name, (times_sec, n_voxels) in results.items():
        summarize(name, times_sec, n_voxels)
        if skipped[name]:
            print(f"  ({skipped[name]} frame(s) skipped, see errors above)")

    baseline = results.get("dense")
    if baseline and baseline[0]:
        dense_mean = sum(baseline[0]) / len(baseline[0])
        print(f"\n--- vs dense ({dense_mean * 1000:.3f}ms/frame; SlotFormer not applicable to dense, see module docstring) ---")
        for name, (times_sec, _) in results.items():
            if name == "dense" or not times_sec:
                continue
            speedup = dense_mean / (sum(times_sec) / len(times_sec))
            print(f"{name}: {speedup:.2f}x faster than dense")


if __name__ == "__main__":
    main()
