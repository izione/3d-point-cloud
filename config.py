"""config.py - grid/model/training constants for the CenterPoint(Yin, Zhou &
Krahenbuhl, CVPR 2021) reproduction on our sonar diver dataset.

Backbone follows the paper's own choice of 3D backbone ("we largely follow
the network designs of second... [a] voxel feature encoder followed by a
sparse 3D backbone"): VoxelNet(Zhou&Tuzel 2018)-style VFE stack -> SPARSE
Conv3D middle layers (spconv, SECOND's own replacement for VoxelNet's
original dense Conv3D) -> 2D RPN. Requires spconv -- see README.md's Colab/
local setup (`pip install spconv-cuXXX`, pick the tag matching your CUDA).
This exact backbone shape (StackedVFE + a spconv middle encoder of the same
128->64->64->64 layer shape + RPNBackbone) is already validated end-to-end
on this same dataset in the sibling voxelnet_baseline repo's own
sparse_conv_middle.py -- ported here.

Grid/anchor-geometry constants below are that repo's own dense-pipeline
values (model/config.py, model/voxelnet_exp/ve_config.py) -- calibrated
against this exact dataset (Triband/sonar diver scenes), not re-derived here.
"""
from pathlib import Path

# --- point cloud range / voxelization (VoxelNet backbone) ---
# (x_min,y_min,z_min,x_max,y_max,z_max), meters. Comfortably wraps GT
# (x 1.4-10.5, y -3.8-3.1, z -1.55-1.13) while cropping far background noise.
POINT_CLOUD_RANGE = (0.0, -5.0, -2.5, 12.0, 5.0, 2.5)
VOXEL_SIZE = (0.1, 0.1, 0.5)  # vz=0.5 -> D'=10, matches the paper's car-config conv-middle shapes
GRID_SIZE = tuple(
    round((POINT_CLOUD_RANGE[3 + i] - POINT_CLOUD_RANGE[i]) / VOXEL_SIZE[i]) for i in range(3)
)  # (W',H',D') = (x,y,z) voxel counts
MAX_POINTS_PER_VOXEL = 35  # paper's own car-config T=35
MAX_VOXELS = 8000
INPUT_FEATURE_DIM = 7  # [x,y,z,intensity, x-vx,y-vy,z-vz] (VoxelNet paper Sec.2.1.1)

# --- CenterHead output grid (= RPN backbone's own output resolution) ---
HEAD_STRIDE = (VOXEL_SIZE[0] * 2, VOXEL_SIZE[1] * 2)  # (x,y) meters/cell = 0.2m
HEAD_GRID_SIZE = (GRID_SIZE[0] // 2, GRID_SIZE[1] // 2)  # (W'',H'')

# --- RPN backbone channels (VoxelNet Fig.4 / SECOND's dense re-derivation) ---
RPN_IN_CHANNELS = 128
RPN_BLOCK_CHANNELS = (128, 128, 256)
RPN_BLOCK_LAYERS = (4, 6, 6)
RPN_UPSAMPLE_CHANNELS = 256

# --- CenterHead target assignment (CenterPoint paper Sec.4.1 / CornerNet Law&Deng 2018) ---
GAUSSIAN_MIN_OVERLAP = 0.7
# CenterPoint paper Sec.4.1's own minimum-radius floor: "we enlarge the
# radius" for map-view sparsity (the object footprints in our sonar grid are
# small -- 0.5-1.8m at 0.2m/cell -- so the raw CornerNet formula alone gives
# a near-zero radius, effectively single-pixel supervision).
GAUSSIAN_RADIUS_TAU = 2.0

# --- CenterHead loss (CenterNet/CornerNet penalty-reduced focal loss, standard values) ---
FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0
REG_LOSS_WEIGHT = 1.0  # weight on (offset+z+dim+rot) L1 loss vs heatmap focal loss

# --- decode / NMS-free peak extraction ---
VAL_SCORE_THRESH = 0.3
VAL_NMS_IOU = 0.1
MAX_PEAKS = 100
IOU_THRESHOLDS = (0.25, 0.3, 0.35, 0.4, 0.5)
VAL_TARGET_IOU = 0.35  # checkpoint-selection criterion

# --- optimization ---
BATCH_SIZE = 4
NUM_EPOCHS = 20
LR = 0.01
LR_DECAY_EPOCH_FRAC = 0.85
LR_DECAY_FACTOR = 0.1
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
GRAD_CLIP_NORM = 35.0

# --- dataset location ---
# The raw sonar dataset itself is NOT part of this repo (private data, too
# large for git) -- point this at wherever you've uploaded/mounted it in
# Colab (see README.md's "Colab setup" section). Expected layout:
#   <DATA_ROOT>/<Person*>/<scene_*>/sonar/frame_*.bin
#   <DATA_ROOT>/<Person*>/<scene_*>/labels/frame_*.json
import os
DATA_ROOT = Path(os.environ.get("CENTERPOINT_DATA_ROOT", "/content/dataset"))
