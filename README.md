# SonarVoxNet++

Sonar-based 3D diver detector, FCAF3D-style (arXiv 2112.00322) anchor-free
dense head on a sparse 3D U-Net backbone, with two changes that matter most
for this dataset:

1. **Gaussian-heatmap existence target** (CenterNet/CenterPoint-style) in
   place of FCAF3D's hard 0/1 classification target + separate centerness
   branch -- trained with a penalty-reduced focal loss.
2. **Corner-auxiliary loss**: an extra MSE between a Gaussian heatmap
   rendered from the predicted box's 8 corners and one rendered from the
   GT box's 8 corners, supervising box extent/orientation spatially on top
   of the direct center/size/rotation regression terms.

Box rotation is a full 6D continuous representation (Zhou et al., "On the
Continuity of Rotation Representations", CVPR 2019) rather than FCAF3D's
own yaw-only face-distance/Möbius-strip parametrization -- a diver can be
at any orientation underwater, unlike FCAF3D's original floor-constrained
indoor-scene objects (chair/sofa/bed/...).

## Layout

- `models/detector_fcaf3d.py` -- the detector itself (VFE -> sparse U-Net
  backbone -> `FCAF3DHead` -> loss/decode).
- `models/heads_fcaf3d.py`, `models/losses_fcaf3d.py`, `models/assign_fcaf3d.py`
  -- the dense head, its loss (incl. the Gaussian-heatmap focal loss and
  corner-auxiliary term), and GT assignment (FCOS-style center sampling).
- `models/rotation6d.py` -- the 6D rotation representation.
- `configs/exp_fcaf3d_single_level_heatmap_corneraux.yaml` -- this model's
  own config (chains through `exp_fcaf3d_single_level_heatmap.yaml` ->
  `exp_fcaf3d_single_level.yaml` -> `exp_fcaf3d.yaml` -> `default.yaml`).
- `train_fcaf3d.py` / `test_fcaf3d.py` -- training and evaluation entry
  points (reuse generic dataloader/checkpoint/PR-curve helpers from the
  sibling `train.py`/`test.py`, which belong to an earlier quaternion-based
  detector this project started from -- not part of this model, just where
  those shared utilities happen to live).
- `test_fcaf3d_components.py` -- unit tests for every piece above; run with
  `python test_fcaf3d_components.py`.
- `figures/` -- qualitative comparison renders (ours vs. SonarVoxNet vs. a
  faithful FCAF3D reproduction vs. two intermediate baselines) on real test
  frames.

## Running

```
pip install -r requirements.txt
python train_fcaf3d.py --config configs/exp_fcaf3d_single_level_heatmap_corneraux.yaml
python test_fcaf3d.py --checkpoint <path> --config configs/exp_fcaf3d_single_level_heatmap_corneraux.yaml
```

spconv is used automatically if it imports and actually runs on the current
GPU (`models/backbone3d_auto.py` probes this); otherwise everything falls
back to a pure-PyTorch sparse-conv implementation (`models/sparse_ops.py`),
so the same code runs on CPU too.
