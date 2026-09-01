"""Runs all 5 backbone/SlotFormer experiments back-to-back with train.py,
each into its own --ckpt_dir (same naming as the `python train.py ...` command
documented in each configs/exp_*.yaml's header comment) and its own --exp_name
(the short name below, e.g. "sparse_no_slotformer") so checkpoint/loss-log
filenames are self-descriptive even if dirs ever get consolidated:
<ckpt_dir>/<name>_last.pth, <name>_epoch_N.pth, <name>_step_N.pth,
<name>_loss_history.csv.

    1. dense backbone, no SlotFormer        -> configs/exp_dense.yaml
    2. sparse backbone, no SlotFormer       -> configs/exp_sparse_no_slotformer.yaml
    3. sparse backbone + SlotFormer 3L      -> configs/exp_sparse_slotformer_3l.yaml
    4. sparse backbone + SlotFormer 6L      -> configs/exp_sparse_slotformer_6l.yaml
    5. sparse 3D U-Net (encoder-decoder)    -> configs/exp_sparse_unet.yaml

Meant to be run once the real dataset (data/dataset.py's SonarDiverDataset) is
in place -- until then, use smoke_test.py to sanity-check the 5 configs
end-to-end against synthetic data.

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --epochs 1        # quick real-data smoke run
    python run_all_experiments.py --only dense sparse_no_slotformer
"""
import argparse
import subprocess
import sys

EXPERIMENTS = [
    ("dense", "configs/exp_dense.yaml", "checkpoints_dense"),
    ("sparse_no_slotformer", "configs/exp_sparse_no_slotformer.yaml", "checkpoints_sparse_no_slotformer"),
    ("sparse_slotformer_3l", "configs/exp_sparse_slotformer_3l.yaml", "checkpoints_sparse_slotformer_3l"),
    ("sparse_slotformer_6l", "configs/exp_sparse_slotformer_6l.yaml", "checkpoints_sparse_slotformer_6l"),
    ("sparse_unet", "configs/exp_sparse_unet.yaml", "checkpoints_sparse_unet"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=[name for name, _, _ in EXPERIMENTS], default=None,
                         help="run a subset instead of all 4 (by name, e.g. --only dense sparse_slotformer_3l)")
    parser.add_argument("--epochs", type=int, default=None, help="passed through to train.py's --epochs override")
    parser.add_argument("--dry_run", action="store_true", help="print the commands instead of running them")
    args = parser.parse_args()

    experiments = [e for e in EXPERIMENTS if args.only is None or e[0] in args.only]

    for name, config_path, ckpt_dir in experiments:
        cmd = [sys.executable, "train.py", "--config", config_path, "--ckpt_dir", ckpt_dir, "--exp_name", name]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]

        print(f"\n{'=' * 70}\n[{name}] {' '.join(cmd)}\n{'=' * 70}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[{name}] FAILED (exit code {result.returncode}) -- stopping, remaining experiments not run")
            sys.exit(result.returncode)

    print("\nall experiments finished." if not args.dry_run else "\n(dry run -- nothing was executed)")


if __name__ == "__main__":
    main()
