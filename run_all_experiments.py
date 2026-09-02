"""Runs any subset of this project's backbone/SlotFormer experiments back-to-back
with train.py. Each experiment gets its own checkpoints/<name>/ AND logs/<name>/
loss_history.csv -- kept as two separate trees (not one shared dir) so checkpoints
(large, .gitignore'd, safe to delete once evaluated) and the loss history CSV
(small, worth keeping/committing even after checkpoints are cleaned up) don't get
tangled together. Colab notebooks/other machines can pick a single experiment by
name via --only.

Groups:
  original 4 (dense/sparse backbone comparison, configs/exp_*.yaml):
    dense, sparse_no_slotformer, sparse_slotformer_3l, sparse_slotformer_6l
  exp1_simple_slotformer (experiments/exp1_simple_slotformer/): the original
    single-stage downsample backbone + external SlotFormer, varying SlotFormer
    depth only (3L vs 6L)
  exp2_down_slot_up (experiments/exp2_down_slot_up/): 4-stage downsample encoder
    -> SlotFormer at the bottleneck -> M-stage upsample decoder, varying BOTH
    SlotFormer depth (3L/6L) and upsample depth (2/3/4)
  exp3_unet_slotformer (experiments/exp3_unet_slotformer/): full encoder-decoder
    with SlotFormer(3L) replacing residual blocks in every stage, varying encoder/
    decoder depth (2/3/4 layers)

See each config's own header comment for the architecture/measurement details.

Usage:
    python run_all_experiments.py --only exp1_simple_slotformer_3l
    python run_all_experiments.py --epochs 1        # quick real-data smoke run
    python run_all_experiments.py --group exp2_down_slot_up   # just that group
    python run_all_experiments.py --dry_run         # print commands, run nothing
"""
import argparse
import subprocess
import sys

EXPERIMENTS = [
    # (name, config_path, group)
    ("dense", "configs/exp_dense.yaml", "original4"),
    ("sparse_no_slotformer", "configs/exp_sparse_no_slotformer.yaml", "original4"),
    ("sparse_slotformer_3l", "configs/exp_sparse_slotformer_3l.yaml", "original4"),
    ("sparse_slotformer_6l", "configs/exp_sparse_slotformer_6l.yaml", "original4"),

    ("exp1_simple_slotformer_3l", "experiments/exp1_simple_slotformer/exp_simple_slotformer_3l.yaml", "exp1_simple_slotformer"),
    ("exp1_simple_slotformer_6l", "experiments/exp1_simple_slotformer/exp_simple_slotformer_6l.yaml", "exp1_simple_slotformer"),

    ("exp2_4down_2up_3l", "experiments/exp2_down_slot_up/exp_slotformer_4down_2up_3l.yaml", "exp2_down_slot_up"),
    ("exp2_4down_2up_6l", "experiments/exp2_down_slot_up/exp_slotformer_4down_2up_6l.yaml", "exp2_down_slot_up"),
    ("exp2_4down_3up_3l", "experiments/exp2_down_slot_up/exp_slotformer_4down_3up_3l.yaml", "exp2_down_slot_up"),
    ("exp2_4down_3up_6l", "experiments/exp2_down_slot_up/exp_slotformer_4down_3up_6l.yaml", "exp2_down_slot_up"),
    ("exp2_4down_4up_3l", "experiments/exp2_down_slot_up/exp_slotformer_4down_4up_3l.yaml", "exp2_down_slot_up"),
    ("exp2_4down_4up_6l", "experiments/exp2_down_slot_up/exp_slotformer_4down_4up_6l.yaml", "exp2_down_slot_up"),

    ("exp3_unet_2layer", "experiments/exp3_unet_slotformer/exp_unet_slotformer_2layer.yaml", "exp3_unet_slotformer"),
    ("exp3_unet_3layer", "experiments/exp3_unet_slotformer/exp_unet_slotformer_3layer.yaml", "exp3_unet_slotformer"),
    ("exp3_unet_4layer", "experiments/exp3_unet_slotformer/exp_unet_slotformer_4layer.yaml", "exp3_unet_slotformer"),
]
GROUPS = sorted({g for _, _, g in EXPERIMENTS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=[name for name, _, _ in EXPERIMENTS], default=None,
                         help="run only these experiments by name (e.g. --only exp1_simple_slotformer_3l)")
    parser.add_argument("--group", nargs="+", choices=GROUPS, default=None,
                         help=f"run only these groups (e.g. --group exp2_down_slot_up). choices: {GROUPS}")
    parser.add_argument("--epochs", type=int, default=None, help="passed through to train.py's --epochs override")
    parser.add_argument("--dry_run", action="store_true", help="print the commands instead of running them")
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.group is not None:
        experiments = [e for e in experiments if e[2] in args.group]
    if args.only is not None:
        experiments = [e for e in experiments if e[0] in args.only]

    for name, config_path, group in experiments:
        ckpt_dir = f"checkpoints/{name}"
        log_file = f"logs/{name}/loss_history.csv"
        cmd = [sys.executable, "train.py", "--config", config_path, "--ckpt_dir", ckpt_dir,
               "--exp_name", name, "--log_file", log_file]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]

        print(f"\n{'=' * 70}\n[{group}/{name}] {' '.join(cmd)}\n{'=' * 70}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[{name}] FAILED (exit code {result.returncode}) -- stopping, remaining experiments not run")
            sys.exit(result.returncode)

    print("\nall requested experiments finished." if not args.dry_run else "\n(dry run -- nothing was executed)")


if __name__ == "__main__":
    main()
