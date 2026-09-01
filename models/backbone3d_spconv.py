"""spconv-backed version of models/backbone3d.Sparse3DBackbone -- same
interface (forward(features, coords, index_grid, grid_size, batch_size) ->
(features, coords, index_grid, grid_size), same STAGE_CHANNELS /
NUM_BLOCKS_PER_STAGE / kernel / stride config), but the actual convolutions
run through spconv's real hash-table-based sparse conv kernels instead of
models/sparse_ops.py's plain-PyTorch reimplementation.

spconv is CUDA-only and not guaranteed to import or build cleanly everywhere
(Colab especially) -- see models/backbone3d_auto.py, which probes this module
at import time and silently falls back to the pure-PyTorch
backbone3d.Sparse3DBackbone if spconv isn't usable. Nothing here should be
imported directly by detector.py; go through models.backbone_registry.build_backbone
(BACKBONE.TYPE: auto), which dispatches to backbone3d_auto.build_auto_backbone.
"""
import torch
import torch.nn as nn
import spconv.pytorch as spconv

from .sparse_ops import build_index_grid


class SpconvBasicBlock(nn.Module):
    """Two SubMConv3d + BN + ReLU with a residual connection -- spconv analog
    of sparse_ops.SparseBasicBlock. `indice_key` is shared across every block
    in a stage (see SpconvStage): all of them run SubM convs, kernel_size=3,
    stride=1 on the exact same active-coordinate set (SubM never moves it),
    so spconv can reuse one rulebook for all of them instead of rebuilding it
    per layer."""

    def __init__(self, channels, kernel_size=3, indice_key=None):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(channels, channels, kernel_size, bias=False, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = spconv.SubMConv3d(channels, channels, kernel_size, bias=False, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x.features
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features + identity))
        return out


class SpconvStage(nn.Module):
    """spconv analog of backbone3d.Sparse3DStage: one strided SparseConv3d
    (changes channels + downsamples x,y,z together) then num_blocks residual
    blocks at the new channel count."""

    def __init__(self, in_channels, out_channels, num_blocks, down_kernel=3, down_stride=2, stage_idx=0):
        super().__init__()
        self.down = spconv.SparseConv3d(
            in_channels, out_channels, down_kernel, stride=down_stride, padding=down_kernel // 2, bias=True,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        subm_key = f"backbone3d_stage{stage_idx}_subm"
        self.blocks = nn.ModuleList([
            SpconvBasicBlock(out_channels, indice_key=subm_key) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.down(x)
        x = x.replace_feature(self.relu(self.bn(x.features)))
        for block in self.blocks:
            x = block(x)
        return x


class Sparse3DBackboneSpconv(nn.Module):
    """Same architecture as backbone3d.Sparse3DBackbone (4 stages, filter
    widths 16/32/48/64 by default, isotropic x/y/z downsampling every stage),
    implemented with spconv instead of sparse_ops.py."""

    def __init__(self, in_channels, stage_channels=(16, 32, 48, 64), num_blocks_per_stage=3,
                 down_kernel=3, down_stride=2):
        super().__init__()
        stages = []
        c_in = in_channels
        for i, c_out in enumerate(stage_channels):
            stages.append(SpconvStage(c_in, c_out, num_blocks_per_stage, down_kernel, down_stride, stage_idx=i))
            c_in = c_out
        self.stages = nn.ModuleList(stages)
        self.out_channels = stage_channels[-1]
        self.total_stride = down_stride ** len(stage_channels)

    def forward(self, features, coords, index_grid, grid_size, batch_size):
        # spconv doesn't assume any particular meaning for the spatial axes -- it just
        # needs `indices` columns to line up positionally with `spatial_shape` entries,
        # so our existing (batch,x,y,z) column order carries over unchanged. int32 is
        # spconv's required index dtype (vs. torch.long everywhere else in this codebase).
        sp_coords = coords.to(dtype=torch.int32) if coords.dtype != torch.int32 else coords
        x = spconv.SparseConvTensor(features, sp_coords, list(grid_size), batch_size)
        for stage in self.stages:
            x = stage(x)
        out_coords = x.indices.long()
        out_grid_size = tuple(int(s) for s in x.spatial_shape)
        out_index_grid = build_index_grid(out_coords, batch_size, out_grid_size, device=features.device)
        return x.features, out_coords, out_index_grid, out_grid_size
