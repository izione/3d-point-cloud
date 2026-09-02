# diver_net

Sonar-based 3D diver detector: VFE -> 3D sparse conv backbone (isotropic x/y/z
downsampling, no BEV collapse) -> SlotFormer (axial slot attention, softmax or
linear + sinusoidal positional encoding) -> center heatmap / offset / box(w,l,h,
quaternion) heads, trained with a dynamic (SimOTA-style) label assignment and
a corner-distance regression cost.

The backbone itself is swappable -- see **"Experiments"** below for the full
comparison series (15 configs across 4 different backbone architectures) and
how to run them. `configs/default.yaml`'s own `BACKBONE` section is the
single proven-good baseline (1 stage, stride 2) everything else is compared
against; going deeper without care regressed precision/recall badly in an
earlier version of this backbone (see `models/backbone3d.py`'s history) --
this is why "effective resolution" (`VOXEL_SIZE * backbone total_stride`)
matters more than raw depth when changing this.

No *hard* spconv / torch_scatter dependency -- every sparse op has a fallback
implemented in plain PyTorch (`models/sparse_ops.py`), so the same code runs
on local CPU (for testing) and Colab GPU with zero extra native-extension
installs. spconv is used automatically *if* it's installed and actually
works on the current GPU (`models/backbone3d_auto.py` probes this at model
construction time) -- see "3D backbone: pure-PyTorch vs spconv" below.

## Colab setup

```python
!git clone -b slotformer https://github.com/izione/3d-point-cloud.git
%cd 3d-point-cloud
!pip install -r requirements.txt

# mount the dataset (adjust to wherever you uploaded dataset/)
from google.colab import drive
drive.mount('/content/drive')
# then edit configs/default.yaml -> DATA.ROOT to point at it
```

The model code lives on the `slotformer` branch specifically (this repo also holds
other model experiments on other branches) -- a plain `git clone` without `-b
slotformer` checks out `main` instead and won't have these files.

Confirm a GPU runtime is attached: Runtime -> Change runtime type -> T4 GPU (or better).

## Local GPU setup (own machine with an NVIDIA GPU)

The default `pip install torch` on Windows pulls a CPU-only build. Use a
dedicated venv (so this doesn't touch any other project's Python
environment) and install a CUDA build explicitly -- pick the `cuXXX` index
tag matching your driver (`nvidia-smi` shows the max CUDA version it
supports); `cu126` worked for an RTX 2070:

```bash
cd 3d-point-cloud
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# optional -- see "3D backbone: pure-PyTorch vs spconv" below
pip install spconv-cu126
```

## 3D backbone: pure-PyTorch vs spconv

`models/backbone3d.py` (pure PyTorch, always available) and
`models/backbone3d_spconv.py` (real hash-table-based sparse conv via
[spconv](https://github.com/traveller59/spconv), CUDA-only) implement the
exact same 4-stage architecture. `models/backbone3d_auto.py` picks between
them automatically every time `DiverDetector` is constructed: it tries to
import spconv AND run one real conv with it on the current GPU, and only
uses it if that actually succeeds -- a plain `import spconv` succeeding
isn't enough, since the compiled CUDA kernels can still mismatch the
installed torch/CUDA/driver combo at call time. Any failure (not installed,
import error, kernel mismatch, no CUDA at all) falls back to the
pure-PyTorch backbone silently, so this is always safe to leave installed
-- including on Colab, where it may or may not work depending on the image.
`train.py`/`test.py` print which one got picked ("3D backbone: spconv" or
"3D backbone: pure-PyTorch") right after constructing the model.

Measured with `benchmark_backbone.py` on the real sonar dataset (RTX 2070,
3000 frames, this project's actual backbone size -- stage_channels
[16,32,48,64], <0.1% voxel occupancy at VOXEL_SIZE 0.1): the pure-PyTorch
backbone is only ~1.1x faster than an equivalent dense `nn.Conv3d` backbone
(its rulebook logic is built from generic ops like `torch.unique`/
boolean-mask compaction, which force GPU-CPU syncs regardless of how few
voxels are active -- a genuine ceiling for this implementation style, not a
bug to fix further). spconv's real sparse kernels don't have that ceiling
and were ~2.7x faster than dense in the same test.

**SlotFormer's own cost, and the number that actually matters (backbone +
SlotFormer vs. dense, since dense's whole-grid receptive field makes
SlotFormer redundant there -- see benchmark_backbone.py's docstring):**
SlotFormer adds ~3.4ms/layer regardless of which backbone feeds it (3
layers/1 cycle: +10.2ms; 6 layers/2 cycles: +20.3ms -- config's default is
1 cycle). Stacked on top of each backbone (mean ms/frame, 3000-frame run):

| backbone | alone | + SlotFormer(3L, default) | + SlotFormer(6L) | vs dense |
|---|---|---|---|---|
| dense | 44.3ms | n/a | n/a | 1x |
| spconv | 16.5ms | **26.7ms** | 36.8ms | **1.66x faster** (3L) / 1.20x faster (6L) |
| pure-PyTorch | 39.7ms | 49.9ms | 60.0ms | **0.89x -- slower than dense** (3L) / 0.74x (6L) |

The takeaway: this architecture is only faster than a plain dense backbone
*with SlotFormer included* when spconv is actually available. On the
pure-PyTorch fallback (e.g. if spconv breaks on a given Colab image), the
backbone alone roughly matches dense, but adding SlotFormer on top pushes
the combination below dense -- something to know before assuming "sparse"
implies "faster" in that fallback path. Run the benchmark yourself to
compare on your own GPU/dataset:

```bash
python benchmark_backbone.py --data_root /path/to/dataset --num_frames 3000
# add --skip_slotformer for backbone-only numbers, or --slotformer_cycles 1 2 3 to test more depths
```

## Train

```bash
python train.py --config configs/default.yaml --ckpt_dir checkpoints
# resume:
python train.py --config configs/default.yaml --ckpt_dir checkpoints --resume checkpoints/default_last.pth
```

Checkpoint and loss-log filenames are prefixed with an `--exp_name` (defaults
to the config file's stem, e.g. `configs/exp_sparse_slotformer_3l.yaml` ->
`exp_sparse_slotformer_3l`) so multiple experiments can safely share one
`--ckpt_dir` without clobbering each other:
`<ckpt_dir>/<exp_name>_last.pth`, `<exp_name>_epoch_N.pth`,
`<exp_name>_step_N.pth`, `<exp_name>_loss_history.csv`.

Checkpoints save every `OPTIMIZATION.CKPT_EVERY_N_EPOCHS` epochs and every
`OPTIMIZATION.CKPT_EVERY_N_STEPS` steps (both configurable in the yaml) --
useful since Colab sessions can disconnect mid-epoch.

**Before committing to a full run**, time a handful of real steps on the
actual Colab GPU and extrapolate -- Colab GPUs vary (T4/A100/etc) and real
throughput needs to be measured, not assumed.

### GPU optimization: two code paths, pick by measuring

`models/sparse_ops.py` (`CONV_MODE`) and `models/slotformer.py`
(`ATTENTION_MODE`) each have two implementations of their core op:

- `"loop"` -- processes one kernel-offset / one slot at a time. Lower peak
  memory, no wasted padding compute. **Measured faster on CPU.**
  `slotformer.py`'s `ATTENTION_MODE` still defaults to this (only measured on
  CPU so far).
- `"vectorized"` / `"padded"` -- batches all kernel offsets (or all slots,
  padded to the same size) into one big op, trading memory/wasted-compute
  for far fewer Python-level iterations. **Measured on a real GPU (RTX 2070,
  via `benchmark_backbone.py` on the actual sonar dataset): `sparse_ops.py`'s
  `CONV_MODE="vectorized"` ran the 4-stage backbone at ~266ms/frame vs
  ~1880ms/frame for `"loop"` (7x)** -- launch-overhead dominates on GPU as
  expected, so `CONV_MODE` now defaults to `"vectorized"`. `ATTENTION_MODE`
  hasn't been benchmarked on GPU the same way yet; worth doing before a long
  run.

  `SparseConv3dDown` (the downsample used once per backbone stage) had its
  own separate k^3 Python loop for finding candidate output coords that
  `CONV_MODE` never touched -- vectorizing that too (same fix, same reason)
  took the benchmark down further, to ~44ms/frame, right on par with an
  architecture-matched dense `nn.Conv3d` backbone (~48ms/frame) despite
  <0.1% voxel occupancy in this dataset.

On a GPU box, run a few steps with each setting and compare wall-clock time
before trusting either one for a long run (or use `benchmark_backbone.py` for
the 3D backbone specifically):

```python
import models.sparse_ops as sops, models.slotformer as sf
sops.CONV_MODE = "vectorized"
sf.ATTENTION_MODE = "padded"
```
(set at the top of your training script/notebook cell, before constructing `DiverDetector`)

### SlotFormer attention: softmax vs linear (model-quality comparison)

`models/slotformer.py`'s `ATTENTION_KIND` picks the attention math used
*inside* each slot -- this is a different axis than `ATTENTION_MODE` above
(which is a speed-only toggle that doesn't change the result). `"softmax"`
is the standard scaled-dot-product attention this project was designed
around (and the current default); `"linear"` is the kernel-feature-map
attention (Katharopoulos et al., `elu(x)+1`) that FSHNet's own SlotFormer
actually uses. Same slot grouping either way -- only the in-slot attention
computation changes, so switching it doesn't affect any other part of the
model or the checkpoint's parameter shapes.

Since this changes actual model behavior (not just speed), compare val
loss / recall between the two rather than assuming one is better. Unlike
`CONV_MODE`/`ATTENTION_MODE` (which you edit in the source before running,
since they're speed-only and don't need to survive in the checkpoint),
`ATTENTION_KIND` has a proper `--attention_kind` CLI flag on both
`train.py` and `test.py`, and gets saved into every checkpoint so `test.py`
automatically re-uses whatever kind a checkpoint was actually trained with:

```bash
# two full runs, same config/ckpt_dir but different --exp_name so filenames don't clobber each other
python train.py --config configs/default.yaml --ckpt_dir checkpoints --exp_name default_softmax --attention_kind softmax
python train.py --config configs/default.yaml --ckpt_dir checkpoints --exp_name default_linear --attention_kind linear
```
```bash
# --attention_kind not needed here -- test.py reads it back out of the checkpoint
python test.py --checkpoint checkpoints/default_softmax_last.pth --split test
python test.py --checkpoint checkpoints/default_linear_last.pth --split test
```
Then compare `checkpoints/default_softmax_loss_history.csv` vs
`checkpoints/default_linear_loss_history.csv` (val rows), and the two
`test.py` reports (recall/precision/IoU sweep).

## Experiments

15 configs across 4 backbone families, all built on voxelization/VFE/detection
head that never change -- only the 3D backbone differs between them. `models/
backbone3d_auto.py`'s `build_backbone3d` picks the implementation from each
config's `BACKBONE.TYPE`.

| group | folder | backbone (`BACKBONE.TYPE`) | what varies |
|---|---|---|---|
| `original4` | `configs/exp_*.yaml` | `dense` / `auto` (single-stage sparse) | backbone family itself (dense vs sparse) x SlotFormer on/off/depth |
| `exp1_simple_slotformer` | `experiments/exp1_simple_slotformer/` | `auto` (unchanged single-stage backbone) + external SlotFormer | SlotFormer depth only (3L / 6L) |
| `exp2_down_slot_up` | `experiments/exp2_down_slot_up/` | `sparse_down_slot_up` (N-stage down -> SlotFormer at the bottleneck -> M-stage up, residual blocks kept throughout) | SlotFormer depth (3L/6L) x upsample depth (`UPSAMPLE_STAGES`: 2/3/4) -- 6 configs |
| `exp3_unet_slotformer` | `experiments/exp3_unet_slotformer/` | `sparse_slot_unet` (full encoder-decoder, always fully restores; SlotFormer(3L) replaces residual blocks in *every* stage) | encoder/decoder depth (`STAGE_CHANNELS` length: 2/3/4 layers) |

Two more backbone types exist but aren't part of the actively-compared series
above -- `sparse_slot_stages` (encoder-only, SlotFormer replaces residual
blocks per stage, no decoder) and `sparse_slot_light_unet` (SlotFormer in the
encoder only, decoder kept as plain conv -- mirrors the common "ResNet
encoder + light U-Net decoder" segmentation pattern). `configs/
exp_sparse_slot_stages.yaml` / `exp_sparse_slot_light_unet.yaml` run them
standalone via `train.py` directly; see each backbone module's own docstring
(`models/backbone3d_*.py`) for the full design rationale and how they relate
to the 4 groups above. `configs/exp_sparse_dilated_gn.yaml` (dilated residual
blocks + GroupNorm) is kept for reference but was de-prioritized after
dilation was judged risky for an object this small relative to voxel size --
see its header comment.

### Running experiments

`run_all_experiments.py` is the single entry point -- every experiment gets
its own `checkpoints/<name>/` (large `.pth` files, gitignored) and
`logs/<name>/loss_history.csv` (small, per-step + per-epoch train/val loss --
keep these even after cleaning up checkpoints, this is what you evaluate
performance from):

```bash
# one experiment by name
python run_all_experiments.py --only exp2_4down_3up_6l

# a whole series
python run_all_experiments.py --group exp2_down_slot_up   # all 6 in that series

# literally everything (15 experiments, sequentially)
python run_all_experiments.py

# print every (name, config, group) without running anything
python run_all_experiments.py --dry_run
```

`--only`/`--group` accept multiple values (`--only name1 name2`). `--epochs N`
overrides `OPTIMIZATION.NUM_EPOCHS` for a quick real-data smoke run before
committing to a full one.

To run a single experiment with more manual control than `run_all_experiments.py`
gives (custom `--resume`, `--attention_kind`, etc.), call `train.py` directly with
that experiment's config, e.g.:

```bash
python train.py --config experiments/exp2_down_slot_up/exp_slotformer_4down_3up_6l.yaml \
    --ckpt_dir checkpoints/exp2_4down_3up_6l --log_file logs/exp2_4down_3up_6l/loss_history.csv \
    --exp_name exp2_4down_3up_6l
```

**Before committing to a full run of any new experiment**: none of the 4
newer backbone families were measured on real hardware before being added
(only `exp2_4down_3up_6l` has been actually trained, locally) -- each
config's own header comment says what's measured vs. still a placeholder.
Time a handful of real steps first (see "GPU optimization" below, or the
timed cell in `colab_train.ipynb`) and adjust `BATCH_SIZE`/`LR` if needed --
LR is sqrt-scaled (`0.003 * sqrt(BATCH_SIZE/4)`) rather than linear in every
config that runs SlotFormer, since linear scaling caused the center-loss term
to spike and diverge in earlier testing.

`colab_train.ipynb` wraps all of the above for Colab: clone -> dataset ->
Drive mount -> sanity check (`smoke_test.py`, covers every `configs/exp_*.yaml`
*and* `experiments/*/*.yaml`) -> pick `EXPERIMENT_NAME` -> batch-size/speed
check -> train -> evaluate, with checkpoints+logs written to Drive using the
same `checkpoints/<name>/` + `logs/<name>/loss_history.csv` layout as above.

## Test / evaluate

```bash
python test.py --checkpoint checkpoints/default_last.pth --split test
```

Reports recall/precision at a 1m center-distance match threshold, mean
center error, and mean rotation error -- no NMS, no augmentation, matching
the project's explicit "no post-processing" requirement.
