"""Sparse 3D backbone: N-stage downsample encoder where EACH stage's receptive-field
step is a SlotFormer block instead of stacked residual SubMConv3d blocks.
BACKBONE.TYPE: sparse_slot_stages.

Why this exists: backbone3d.Sparse3DStage grows receptive field within a stage by
stacking `num_blocks_per_stage` SparseBasicBlock residual convs -- each one only a
local k^3 kernel reach, so growing receptive field this way means stacking MORE
blocks, and each block is a full sparse conv call (k^3=27 taps, plus this project's
own finding that sparse conv cost is dominated by per-call gather/launch overhead,
not raw FLOPs -- see sparse_ops.py's CONV_MODE comment). That makes "receptive field
via more residual blocks" expensive in 3D sparse conv specifically, more so than in
dense 2D ResNets. SlotFormer already solves this exact problem cheaply (global
per-axis attention, see slotformer.py) -- this backbone tries using it as the
receptive-field mechanism at EVERY stage instead of only once at a bottleneck (that's
what backbone3d_down_slot_up.py does), replacing the residual blocks entirely rather
than adding SlotFormer on top of them.

Trade-off worth measuring: SlotFormer's cost scales with active voxel count, and the
SHALLOWEST stages (right after the first downsample) have far more active voxels than
a single bottleneck ever does -- so "SlotFormer at every stage" could end up more
expensive than "residual blocks at every stage, SlotFormer only at the bottleneck",
not less. This is exactly the new experiment's open question, not an assumed win.
"""
import torch.nn as nn

from .sparse_ops import SparseConv3dDown
from .slotformer import SlotFormerBackbone


class Sparse3DSlotStage(nn.Module):
    """Down-conv (changes channels + downsamples x,y,z together) followed by a
    SlotFormer block -- receptive field growth via attention instead of local
    residual conv stacking. Channel/coord-preserving, same as the residual blocks
    it replaces (SlotFormerBackbone.forward(features, coords) -> features)."""

    def __init__(self, in_channels, out_channels, down_kernel, down_stride,
                 slot_win_size, slot_num_cycles, slot_num_heads):
        super().__init__()
        self.down = SparseConv3dDown(
            in_channels, out_channels, kernel_size=down_kernel, stride=down_stride, padding=down_kernel // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.slotformer = SlotFormerBackbone(out_channels, slot_win_size, slot_num_cycles, slot_num_heads)

    def forward(self, features, coords, index_grid, grid_size, batch_size):
        x, coords, index_grid, grid_size = self.down(features, coords, index_grid, grid_size, batch_size)
        x = self.relu(self.bn(x))
        x = self.slotformer(x, coords)
        return x, coords, index_grid, grid_size


class SparseSlotEncoderBackbone(nn.Module):
    def __init__(self, in_channels, stage_channels, down_kernel, down_stride,
                 slot_win_size=12, slot_num_cycles=1, slot_num_heads=4):
        super().__init__()
        stage_channels = list(stage_channels)
        stages = []
        c_in = in_channels
        for c_out in stage_channels:
            stages.append(Sparse3DSlotStage(c_in, c_out, down_kernel, down_stride,
                                             slot_win_size, slot_num_cycles, slot_num_heads))
            c_in = c_out
        self.stages = nn.ModuleList(stages)
        self.out_channels = stage_channels[-1]
        self.total_stride = down_stride ** len(stage_channels)

    def forward(self, features, coords, index_grid, grid_size, batch_size):
        for stage in self.stages:
            features, coords, index_grid, grid_size = stage(features, coords, index_grid, grid_size, batch_size)
        return features, coords, index_grid, grid_size
