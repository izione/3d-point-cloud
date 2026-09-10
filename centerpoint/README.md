# centerpoint

CenterPoint (Yin, Zhou & Krahenbuhl, "Center-based 3D Object Detection and
Tracking", CVPR 2021) reproduction on the sonar diver dataset: a single-class
("diver") anchor-free 3D detector.

**Backbone**: VoxelNet(Zhou&Tuzel 2018)-style VFE stack -> dense-grid scatter
-> Conv3D middle layers -> 2D RPN backbone -- the paper's own choice ("we
largely follow the network designs of SECOND for the backbone"), re-derived
here as plain dense `nn.Conv3d`/`nn.Conv2d` instead of a sparse-conv library,
so this needs no spconv/native-extension install and runs identically on CPU
(for testing) and Colab GPU.

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
%cd 3d-point-cloud/centerpoint
!pip install -r requirements.txt

# the raw sonar dataset is NOT in this repo (private, too large) -- upload it
# to Drive first (zip of the labeling-tool-main/dataset folder works fine),
# then mount and point CENTERPOINT_DATA_ROOT at it:
from google.colab import drive
drive.mount('/content/drive')
!unzip -q /content/drive/MyDrive/<path-to>/dataset.zip -d /content/dataset
import os
os.environ["CENTERPOINT_DATA_ROOT"] = "/content/dataset"
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
cd 3d-point-cloud/centerpoint
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126   # pick the cuXXX tag matching your driver
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Train

```bash
export CENTERPOINT_DATA_ROOT=/content/dataset   # or set it in Python before importing config, as above
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
