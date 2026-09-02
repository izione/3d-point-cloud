"""Sparse 3D U-Net backbone where EVERY stage's refinement step -- both encoder AND
decoder -- is a SlotFormer(3L) block instead of stacked residual SubMConv3d blocks.
BACKBONE.TYPE: sparse_slot_unet.

This is a distinct experiment from the other two SlotFormer-backbone variants already
in this repo -- don't conflate them:
  - backbone3d_down_slot_up.py: KEEPS residual blocks in every stage, adds ONE extra
    SlotFormer at the bottleneck only, upsample depth is configurable (M<=N).
  - backbone3d_slot_stages.py: encoder ONLY (no decoder at all), residual blocks
    replaced by SlotFormer at every encoder stage.
  - THIS FILE: the full encoder-decoder ("unet") structure -- always restores to
    input resolution (M=N, no partial-restore knob) -- with residual blocks replaced
    by SlotFormer(3L) in BOTH the encoder and the decoder stages. The last encoder
    stage's own SlotFormer already serves the "bottleneck attention" role, so no
    separate bottleneck-only SlotFormer is added on top.

Cost warning: this runs SlotFormer at every encoder AND decoder stage, including the
shallowest ones on both the way down and the way back up -- those have far more
active voxels than backbone3d_down_slot_up.py's bottleneck-only SlotFormer ever sees.
This is very likely the most expensive of the three variants; measure before
committing to a long run.
"""
import torch
import torch.nn as nn

from .sparse_ops import SparseConv3dDown, SubMConv3d, SparseInverseConv3d
from .slotformer import SlotFormerBackbone


class Sparse3DSlotEncoderStage(nn.Module):
    """Down-conv followed by SlotFormer(3L) instead of residual blocks -- identical
    to backbone3d_slot_stages.Sparse3DSlotStage, duplicated here so this file only
    depends on sparse_ops/slotformer primitives."""

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


class Sparse3DSlotDecoderStage(nn.Module):
    """Inverts one Sparse3DSlotEncoderStage: SparseInverseConv3d restores the cached
    parent coords, concatenates with that stage's cached skip features, a SubMConv3d
    fuses the concatenated channels back down to skip_channels, then SlotFormer(3L)
    refines -- same position in the pipeline backbone3d_down_slot_up._DecoderStage's
    residual blocks occupied, just swapped for attention."""

    def __init__(self, in_channels, skip_channels, down_kernel, down_stride,
                 slot_win_size, slot_num_cycles, slot_num_heads):
        super().__init__()
        self.up = SparseInverseConv3d(in_channels, skip_channels, kernel_size=down_kernel, bias=False)
        self.up_bn = nn.BatchNorm1d(skip_channels)
        self.fuse = SubMConv3d(skip_channels * 2, skip_channels, kernel_size=3, bias=False)
        self.fuse_bn = nn.BatchNorm1d(skip_channels)
        self.relu = nn.ReLU(inplace=True)
        self.slotformer = SlotFormerBackbone(skip_channels, slot_win_size, slot_num_cycles, slot_num_heads)
        self.stride = down_stride
        self.padding = down_kernel // 2

    def forward(self, features, coords, index_grid, grid_size,
                skip_features, parent_coords, parent_index_grid, parent_grid_size):
        up_feat, out_coords, out_index_grid = self.up(
            features, coords, index_grid, grid_size,
            parent_coords, parent_index_grid, parent_grid_size,
            stride=self.stride, padding=self.padding,
        )
        up_feat = self.relu(self.up_bn(up_feat))
        x = torch.cat([up_feat, skip_features], dim=1)
        x, _, _ = self.fuse(x, out_coords, out_index_grid, parent_grid_size)
        x = self.relu(self.fuse_bn(x))
        x = self.slotformer(x, out_coords)
        return x, out_coords, out_index_grid, parent_grid_size


class SparseSlotUNetBackbone(nn.Module):
    def __init__(self, in_channels, stage_channels, down_kernel, down_stride,
                 decoder_out_channels=None, slot_win_size=12, slot_num_cycles=1, slot_num_heads=4):
        super().__init__()
        stage_channels = list(stage_channels)
        n = len(stage_channels)
        encoder_channels = [in_channels] + stage_channels

        self.encoder_stages = nn.ModuleList([
            Sparse3DSlotEncoderStage(encoder_channels[i], encoder_channels[i + 1], down_kernel, down_stride,
                                      slot_win_size, slot_num_cycles, slot_num_heads)
            for i in range(n)
        ])
        # Built deepest-first (inverts encoder_stages[-1] first), always all N stages --
        # this is "the unet structure" (full restore), unlike backbone3d_down_slot_up.py's
        # configurable partial upsample.
        self.decoder_stages = nn.ModuleList([
            Sparse3DSlotDecoderStage(encoder_channels[i + 1], encoder_channels[i], down_kernel, down_stride,
                                      slot_win_size, slot_num_cycles, slot_num_heads)
            for i in reversed(range(n))
        ])

        final_channels = encoder_channels[0]  # = in_channels, after the last (shallowest) decoder stage
        out_channels = decoder_out_channels or final_channels
        self._project = out_channels != final_channels
        if self._project:
            self.out_conv = SubMConv3d(final_channels, out_channels, kernel_size=3, bias=False)
            self.out_bn = nn.BatchNorm1d(out_channels)
            self.out_relu = nn.ReLU(inplace=True)

        self.out_channels = out_channels
        self.total_stride = 1  # always fully restores to input resolution

    def forward(self, features, coords, index_grid, grid_size, batch_size):
        skips = []  # (features, coords, index_grid, grid_size) cached BEFORE each encoder stage runs
        x, c, ig, gs = features, coords, index_grid, grid_size
        for stage in self.encoder_stages:
            skips.append((x, c, ig, gs))
            x, c, ig, gs = stage(x, c, ig, gs, batch_size)

        for stage in self.decoder_stages:
            skip_feat, skip_coords, skip_index_grid, skip_grid_size = skips.pop()
            x, c, ig, gs = stage(x, c, ig, gs, skip_feat, skip_coords, skip_index_grid, skip_grid_size)

        if self._project:
            x, _, _ = self.out_conv(x, c, ig, gs)
            x = self.out_relu(self.out_bn(x))

        return x, c, ig, gs
