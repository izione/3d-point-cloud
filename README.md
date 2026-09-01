# diver_net

Sonar-based 3D diver detector: VFE -> 4-stage 3D sparse conv backbone
(filter widths 16/32/48/64, isotropic x/y/z downsampling every stage, no BEV
collapse) -> SlotFormer (axial slot attention, softmax or linear +
sinusoidal positional encoding) -> center heatmap / offset / box(w,l,h,
quaternion) heads, trained with a dynamic (SimOTA-style) label assignment
and a corner-distance regression cost.

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

## Backbone registry -- adding new architectures to experiment with

`BACKBONE.TYPE` in a config selects a backbone via a name -> builder
registry (`models/backbone_registry.py`) instead of a hardcoded if/elif, so
new architectures can be added as their own module without touching
`detector.py` or any other backbone's file. Currently registered:

| `BACKBONE.TYPE` | module | what it is |
|---|---|---|
| `auto` (default) | `backbone3d_auto.py` | encoder-only multi-stage sparse conv, spconv if usable else pure-PyTorch |
| `dense` | `backbone3d_dense.py` | ordinary `nn.Conv3d` over the full voxel grid, no sparsity |
| `sparse_unet` | `backbone3d_unet.py` | sparse encoder-decoder + skip connections (see its module docstring) -- restores full input resolution (`total_stride=1`) before SlotFormer/the head, instead of relying on SlotFormer's attention to make up for a coarser encoder-only output. `configs/exp_sparse_unet.yaml` is experiment 5 in the dense/sparse/SlotFormer-depth comparison, with `SLOTFORMER.ENABLED: false` -- it's specifically testing whether the U-Net's own (local but deep) receptive field is enough on its own. |

To add another one: write an `nn.Module` matching the shared
`forward(features, coords, index_grid, grid_size, batch_size) ->
(features, coords, index_grid, grid_size)` interface (with `.out_channels`
and `.total_stride` attributes), register a `(in_channels, bcfg) -> module`
builder with `@register_backbone("your_name")`, and import that module from
`models/__init__.py` (its registration decorator has to actually run before
`build_backbone()` is called -- see that file's comment). `bcfg` is the
whole `cfg["BACKBONE"]` dict, so a new backbone can read its own extra
config keys (like `sparse_unet`'s `DECODER_BLOCKS_PER_STAGE`) without
changing the registry or `detector.py` at all.

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

## Test / evaluate

```bash
python test.py --checkpoint checkpoints/default_last.pth --split test
```

Reports recall/precision at a 1m center-distance match threshold, mean
center error, and mean rotation error -- no NMS, no augmentation, matching
the project's explicit "no post-processing" requirement.
