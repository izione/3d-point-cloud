"""spconv-backed version of models/backbone3d_unet.SparseUNetBackbone -- same
architecture/interface, but using spconv's native SparseConv3d/
SparseInverseConv3d kernels instead of models/sparse_ops.py's plain-PyTorch
reimplementation. Matches FCAF3D's (arXiv 2112.00322) own choice of a fast
native sparse-conv backend for its encoder-decoder neck -- they use
MinkowskiEngine, this repo's equivalent is spconv.

Uses spconv's `indice_key` mechanism for the paired inverse convs: each
decoder stage's SparseInverseConv3d shares its indice_key with the matching
encoder stage's down-conv, so it reuses that exact cached rulebook to scatter
straight back to the ORIGINAL (pre-downsample) active set, in the same
row order as that stage's own cached input tensor -- the spconv-native
equivalent of sparse_ops.SparseInverseConv3d's "restrict to the known parent
coords" behavior (see that class's docstring in sparse_ops.py for why this
project deliberately avoids a free-form generative transposed conv +
pruning, unlike FCAF3D's own neck).

spconv is CUDA-only and not guaranteed to import/build everywhere -- see
backbone3d_auto.py, which probes it and falls back to backbone3d_unet.py's
pure-PyTorch SparseUNetBackbone if spconv isn't usable."""
import torch
import torch.nn as nn
import spconv.pytorch as spconv

from .sparse_ops import build_index_grid
from .backbone3d_spconv import SpconvBasicBlock


class _SpconvUNetEncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, down_kernel, down_stride, stage_idx):
        super().__init__()
        self.down_key = f"unet_down{stage_idx}"
        self.down = spconv.SparseConv3d(
            in_channels, out_channels, down_kernel, stride=down_stride, padding=down_kernel // 2,
            bias=True, indice_key=self.down_key,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        subm_key = f"unet_stage{stage_idx}_subm"
        self.blocks = nn.ModuleList([
            SpconvBasicBlock(out_channels, indice_key=subm_key) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.down(x)
        x = x.replace_feature(self.relu(self.bn(x.features)))
        for block in self.blocks:
            x = block(x)
        return x


class _SpconvUNetDecoderStage(nn.Module):
    """Inverts one _SpconvUNetEncoderStage via its shared indice_key -- up-conv
    + skip concat + one fuse SubMConv3d, mirroring
    backbone3d_slot_light_unet._LightDecoderStage's pattern."""

    def __init__(self, in_channels, skip_channels, down_kernel, down_indice_key):
        super().__init__()
        self.up = spconv.SparseInverseConv3d(in_channels, skip_channels, down_kernel,
                                              indice_key=down_indice_key, bias=False)
        self.up_bn = nn.BatchNorm1d(skip_channels)
        self.fuse = spconv.SubMConv3d(skip_channels * 2, skip_channels, kernel_size=3, bias=False)
        self.fuse_bn = nn.BatchNorm1d(skip_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip_x):
        up = self.up(x)
        up = up.replace_feature(self.relu(self.up_bn(up.features)))
        # SparseInverseConv3d restores up's indices to exactly skip_x's own
        # (same cached rulebook, same row order) -- safe to concat features directly.
        fused = torch.cat([up.features, skip_x.features], dim=1)
        out = up.replace_feature(fused)
        out = self.fuse(out)
        out = out.replace_feature(self.relu(self.fuse_bn(out.features)))
        return out


class SparseUNetBackboneSpconv(nn.Module):
    def __init__(self, in_channels, stage_channels, num_blocks_per_stage, down_kernel=3, down_stride=2):
        super().__init__()
        stage_channels = list(stage_channels)
        n = len(stage_channels)
        assert n >= 2, ("SparseUNetBackboneSpconv needs at least 2 encoder stages -- "
                         "see backbone3d_unet.py's pure-PyTorch twin docstring for why")
        encoder_channels = [in_channels] + stage_channels
        blocks_per_stage = (num_blocks_per_stage if isinstance(num_blocks_per_stage, (list, tuple))
                             else [num_blocks_per_stage] * n)
        assert len(blocks_per_stage) == n

        self.encoder_stages = nn.ModuleList([
            _SpconvUNetEncoderStage(encoder_channels[i], encoder_channels[i + 1], blocks_per_stage[i],
                                     down_kernel, down_stride, stage_idx=i)
            for i in range(n)
        ])
        # mirrors every encoder stage EXCEPT the shallowest (i=0) -- net stride
        # stays down_stride, matching the pure-PyTorch twin.
        self.decoder_stages = nn.ModuleList([
            _SpconvUNetDecoderStage(encoder_channels[i + 1], encoder_channels[i], down_kernel,
                                     down_indice_key=self.encoder_stages[i].down_key)
            for i in reversed(range(1, n))
        ])

        self.out_channels = encoder_channels[1]
        self.total_stride = down_stride

    def forward(self, features, coords, index_grid, grid_size, batch_size):
        (fine_feat, fine_coords, fine_index_grid, fine_grid_size), _ = self._forward_impl(
            features, coords, index_grid, grid_size, batch_size)
        return fine_feat, fine_coords, fine_index_grid, fine_grid_size

    def forward_multilevel(self, features, coords, index_grid, grid_size, batch_size):
        """FCAF3D-style multi-level output: returns [fine_level, coarse_level] as
        (features, coords, index_grid, grid_size, net_stride) tuples -- the
        decoded (skip-fused, richer) fine level plus the deepest encoder
        stage's own raw (never-decoded) output, for a dense multi-level head
        (models/heads_fcaf3d.py) to run on independently, instead of only the
        single fine level DiverDetectorDETR's cross-attention key set uses.
        Only 2 levels since this backbone always has exactly one decoder
        stage (n encoder stages, n-1 decoder stages -- see __init__)."""
        fine, encoder_outputs = self._forward_impl(features, coords, index_grid, grid_size, batch_size)
        coarse = encoder_outputs[-1]
        coarse_grid_size = tuple(int(s) for s in coarse.spatial_shape)
        coarse_coords = coarse.indices.long()
        coarse_stride = 1
        for stage in self.encoder_stages:
            coarse_stride *= stage.down.stride[0]
        return [
            fine + (self.total_stride,),
            (coarse.features, coarse_coords,
             build_index_grid(coarse_coords, batch_size, coarse_grid_size, device=features.device),
             coarse_grid_size, coarse_stride),
        ]

    def _forward_impl(self, features, coords, index_grid, grid_size, batch_size):
        sp_coords = coords.to(dtype=torch.int32) if coords.dtype != torch.int32 else coords
        x = spconv.SparseConvTensor(features, sp_coords, list(grid_size), batch_size)

        skips = []  # skips[i] = SparseConvTensor INPUT to encoder_stages[i]
        encoder_outputs = []  # encoder_outputs[i] = SparseConvTensor OUTPUT of encoder_stages[i]
        for stage in self.encoder_stages:
            skips.append(x)
            x = stage(x)
            encoder_outputs.append(x)

        for stage in self.decoder_stages:
            x = stage(x, skips.pop())

        out_coords = x.indices.long()
        out_grid_size = tuple(int(s) for s in x.spatial_shape)
        out_index_grid = build_index_grid(out_coords, batch_size, out_grid_size, device=features.device)
        return (x.features, out_coords, out_index_grid, out_grid_size), encoder_outputs
