# diver_net

Sonar-based 3D diver detector: VFE -> isotropic sparse-conv stem -> SlotFormer
(axial slot attention, softmax or linear + sinusoidal positional encoding) ->
center heatmap / offset / box(w,l,h,quaternion) heads, trained with a dynamic
(SimOTA-style) label assignment and a corner-distance regression cost.

No spconv / torch_scatter dependency -- every sparse op is plain PyTorch
(`models/sparse_ops.py`), so the same code runs on local CPU (for testing)
and Colab GPU with zero extra native-extension installs.

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

## Train

```bash
python train.py --config configs/default.yaml --ckpt_dir checkpoints
# resume:
python train.py --config configs/default.yaml --ckpt_dir checkpoints --resume checkpoints/last.pth
```

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
  memory, no wasted padding compute. **This is what's actually been measured
  faster on local CPU** and is the current default in both files.
- `"vectorized"` / `"padded"` -- batches all kernel offsets (or all slots,
  padded to the same size) into one big op, trading memory/wasted-compute
  for far fewer Python-level iterations. This should help on GPU, where
  kernel-launch overhead is the real cost the loop version pays repeatedly
  -- but that's a theoretical expectation, not something verified on actual
  CUDA hardware here (no GPU was available to test against).

On Colab, run a few steps with each setting and compare wall-clock time
before trusting either one for a long run:

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
# two full runs, same everything else, different checkpoint dirs so they don't clobber each other
python train.py --config configs/default.yaml --ckpt_dir checkpoints_softmax --attention_kind softmax
python train.py --config configs/default.yaml --ckpt_dir checkpoints_linear --attention_kind linear
```
```bash
# --attention_kind not needed here -- test.py reads it back out of the checkpoint
python test.py --checkpoint checkpoints_softmax/last.pth --split test
python test.py --checkpoint checkpoints_linear/last.pth --split test
```
Then compare `checkpoints_softmax/loss_history.csv` vs
`checkpoints_linear/loss_history.csv` (val rows), and the two `test.py`
reports (recall/precision/IoU sweep).

## Test / evaluate

```bash
python test.py --checkpoint checkpoints/last.pth --split test
```

Reports recall/precision at a 1m center-distance match threshold, mean
center error, and mean rotation error -- no NMS, no augmentation, matching
the project's explicit "no post-processing" requirement.
