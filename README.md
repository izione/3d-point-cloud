# centerpoint

CenterPoint (Yin, Zhou & Krahenbuhl, "Center-based 3D Object Detection and
Tracking", CVPR 2021) reproduction on the sonar diver dataset: a single-class
("diver") anchor-free 3D detector.

The model code lives on the `centerpoint` branch specifically (this repo
also holds other model experiments on other branches) -- a plain `git clone`
without `-b centerpoint` checks out `main` instead and won't have these files.

See `colab_train.ipynb` for a ready-to-run Colab notebook (clone, dataset
fetch from a Google Drive `dataset.zip`, smoke test, train, evaluate).

**Breaking change (backbone now sparse, not dense)**: this version's
`model.py` uses a spconv-based sparse 3D middle encoder, replacing an earlier
dense-`nn.Conv3d` version. State dicts are NOT compatible across the two --
any checkpoint trained before this change (dense backbone) will fail to load
here (different parameter names/shapes) and must be retrained from scratch.

**Backbone**: VoxelNet(Zhou&Tuzel 2018)-style VFE stack -> a SPARSE Conv3D
middle encoder -> 2D RPN backbone -- the paper's own choice ("we largely
follow the network designs of SECOND for the backbone"): SECOND is itself
the paper that replaced VoxelNet's original dense Conv3D middle layers with
sparse convolution (spconv), so this backbone is sparse, matching the paper
(an earlier version of this file used dense `nn.Conv3d` instead purely to
avoid the spconv install -- that was NOT what the paper does, and has been
replaced). **Requires spconv** -- see the setup sections below
(`pip install spconv-cuXXX`, picking the tag matching your CUDA version).

**Head**: CenterHead (paper Sec.3.1) -- Gaussian-heatmap center classification
+ sub-voxel offset / absolute height / log-size / rotation regression, no
anchors. Target assignment follows CornerNet's Gaussian-radius formula with
the paper's own minimum-radius floor (Sec.4.1, `tau=2.0`); loss is the
standard CenterNet penalty-reduced focal loss (heatmap) + L1 (regression).
Decode is NMS-free (3x3 max-pool peak extraction), matching the paper.

**Rotation**: trained against a yaw-only-flattened GT target by default
(`dataset.zyaw_flatten`) -- matches the paper's own scope (nuScenes/Waymo
boxes are yaw-only, objects sit on a ground plane). Internally this still
uses the 6D continuous rotation representation (Zhou et al., CVPR 2019)
infrastructure rather than a raw sin/cos pair, purely so evaluation code is
shared with this project's other (full-3D-rotation) experiments -- with
rotation_x/y forced to 0 in the training target, the regressed matrix reduces
to a pure z-rotation, so this is equivalent to the paper's own yaw-only
design. Pass `--full3d` to `train.py` to instead train against the full 3D
(tilted) rotation. Evaluation GT is always full 3D either way, so AP3D
numbers are directly comparable to this project's other experiments
(VoxelNet/PointPillars head-only ablations in the sibling `voxelnet_baseline`
repo).

## Colab setup

```python
!git clone -b centerpoint https://github.com/izione/3d-point-cloud.git
%cd 3d-point-cloud
!pip install -r requirements.txt

# REQUIRED (paper-faithful sparse backbone) -- check torch's CUDA build FIRST
# and pick the matching spconv-cuXXX tag; installing the wrong one is exactly
# the `ImportError: This model requires spconv...` you'll hit at import time
# (pip install can silently succeed while the compiled kernels still don't
# match your actual CUDA/driver). See https://github.com/traveller59/spconv
# for the full tag list (cu116/cu117/cu118/cu120/cu126 as of this writing).
import torch
print(torch.__version__, torch.version.cuda)   # e.g. "2.4.0+cu121 12.1" -> use spconv-cu120
!pip install -q spconv-cu126   # <-- change the cuXXX suffix to match the line above
```

If `torch.version.cuda` is newer than spconv's newest official tag (cu126 as
of this writing -- a Colab image with cu127/cu128/cu129 has no matching
official spconv wheel, see https://github.com/traveller59/spconv/issues/775),
the fix is to reinstall torch itself against the cu126 build, THEN install
spconv-cu126 (keeps both packages on official PyPI) -- **and then restart the
Colab runtime** (Runtime > Restart session) before importing either one.
torch is a C-extension module that registers process-wide ops at import
time; those registrations can't be redone in the same already-running
process (a `pip install`-then-`import` in the same session raises
`RuntimeError: Only a single TORCH_LIBRARY can be used...`) -- reinstalling
torch only actually takes effect in a fresh process:
```python
!pip install -q torch --index-url https://download.pytorch.org/whl/cu126
!pip install -q spconv-cu126
# now: Runtime > Restart session, then re-run the notebook from the top
```
(A community wheel index also publishes a cu128 spconv build directly --
`pip install cumm-cu128 spconv-cu128 --extra-index-url https://ratharog.github.io/cumm-spconv/`
-- that works too if you'd rather keep torch's newer CUDA build, just isn't
the official distribution channel.)

`colab_train.ipynb`'s own install cell does this automatically (reads
`torch.version.cuda`, tries the matching official tag, and falls back to the
torch-reinstall-as-cu126 fix above if no official tag matches) -- use it instead
of the snippet above when running the notebook rather than copy-pasting cells.

See `colab_train.ipynb` for the full flow (spconv install, dataset download
from a Drive share link, smoke test, train, evaluate) -- the short version:

```python
# fetch dataset.zip (Person1/scene_0000/... at its top level) from a Drive
# share link and unzip it onto local disk, then point the dataset loader at it
!pip install -q gdown
!gdown --fuzzy "<your dataset.zip Drive share URL>" -O /content/dataset.zip
!unzip -q -o /content/dataset.zip -d /content/dataset_extracted
import os
os.environ["CENTERPOINT_DATA_ROOT"] = "/content/dataset_extracted"
```

Expected dataset layout under `$CENTERPOINT_DATA_ROOT`:
```
<Person1..4>/scene_XXXX/sonar/frame_*.bin
<Person1..4>/scene_XXXX/labels/frame_*.json
```

Confirm a GPU runtime is attached: Runtime -> Change runtime type -> T4 GPU (or better).

**Before committing to a full run**, always run the smoke test first:
```bash
!python smoke_test.py
```
It runs the full pipeline (dataset -> model -> loss -> backward -> decode ->
3D IoU) on a handful of real frames -- catches path/shape problems in
seconds instead of after a long dataloader startup.

## Local GPU setup (own machine with an NVIDIA GPU)

```bash
git clone -b centerpoint https://github.com/izione/3d-point-cloud.git
cd 3d-point-cloud
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126   # pick the cuXXX tag matching your driver
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# REQUIRED (paper-faithful sparse backbone) -- pick the cuXXX tag matching
# the torch build above (see https://github.com/traveller59/spconv for the
# supported tag list)
pip install spconv-cu126
python -c "import spconv; print(spconv.__version__)"
```

## Train

```bash
export CENTERPOINT_DATA_ROOT=/content/dataset_extracted   # or set it in Python before importing config, as above
python train.py                     # yaw-only rotation target (paper-faithful, default)
python train.py --full3d            # full 3D rotation target instead
python train.py --smoke             # tiny run -- CLI/pipeline wiring check only, not a real result
```

Validates on **every** val-split frame **every epoch** (no subsampling, no
"skip early epochs" -- CenterPoint's decode is anchor-free/NMS-free so this
stays cheap even for an untrained model, unlike an anchor head's unbounded
pre-NMS candidate count). Checkpoints save every epoch to `--ckpt_dir`
(default `checkpoints_centerpoint`) as `epoch_N.pth`; `loss_history.csv` in
the same directory logs per-epoch train loss and val AP3D/precision/recall at
every threshold in `config.IOU_THRESHOLDS`. Best epoch is selected by
AP3D@IoU=0.35 (printed at the end of training).

Resume after a Colab disconnect with `--resume <ckpt_dir>/epoch_N.pth` --
picks up training at epoch N+1 (see `colab_train.ipynb`'s train cell). Best-AP
tracking restarts on resume (only affects which epoch prints as "best" at the
very end of THIS run) -- every epoch's checkpoint is still saved regardless,
so nothing is lost across a resume.

**Time a handful of real steps on the actual Colab GPU before committing to a
full run** -- Colab GPUs vary (T4/A100/etc), measure rather than assume.

## Evaluate a checkpoint

```python
import torch
from dataset import VoxelDataset
from model import CenterPointVoxelNet
from eval import evaluate_full
import config

device = torch.device("cuda")
model = CenterPointVoxelNet().to(device)
ckpt = torch.load("checkpoints_centerpoint/epoch_19.pth", map_location=device)
model.load_state_dict(ckpt["model"])

test_ds = VoxelDataset("test")
results = evaluate_full(model, device, test_ds)
for thr in config.IOU_THRESHOLDS:
    ap, p, r = results[thr]
    print(f"IoU>={thr}: AP3D={ap:.4f} P={p:.4f} R={r:.4f}")
```
