"""Sparse 3D backbone: N-stage downsample encoder (reuses backbone3d.Sparse3DStage
verbatim) -> SlotFormer (global context on the deepest, most-downsampled voxels --
fewest active voxels there, so cheapest place to run attention) -> M-stage upsample
decoder (M configurable, 0 <= M <= N, via sparse_ops.SparseInverseConv3d + skip fusion)
-> optional output projection. BACKBONE.TYPE: sparse_down_slot_up.

This is deliberately built on THIS branch's own SlotFormer-centric stack (backbone3d.py/
slotformer.py/sparse_ops.py), not on the separate `unet` branch's backbone3d_unet.py --
that branch's full encoder+decoder (no SlotFormer at all, M always = N) turned out
impractically slow (~3h+/epoch even after shrinking) and isn't where this project's
active experiments live. SparseInverseConv3d (the one primitive that branch had and this
one didn't) was ported into sparse_ops.py to build this instead.

M=0 reduces to the existing encoder-only+external-SlotFormer setup (detector.py already
does this via BACKBONE.TYPE: auto + SLOTFORMER.ENABLED: true). M=N fully restores
resolution to the encoder's own input (total_stride=1), but unlike the `unet` branch's
backbone, this one runs SlotFormer ONCE at the coarsest scale before undoing any of the
downsampling, rather than never running it at all.

SlotFormer lives INSIDE this backbone (constructed here, not in detector.py) -- any
config using BACKBONE.TYPE: sparse_down_slot_up MUST set SLOTFORMER.ENABLED: false at
the top level, or detector.py attaches a second, redundant SlotFormer after this one.
"""
import torch
import torch.nn as nn

from .backbone3d import Sparse3DStage
from .sparse_ops import SubMConv3d, SparseBasicBlock, SparseInverseConv3d
from .slotformer import SlotFormerBackbone


class _DecoderStage(nn.Module):
    """Inverts one Sparse3DStage's downsample: SparseInverseConv3d restores the
    cached parent (pre-downsample) coordinate set, concatenates with that stage's
    cached skip features (same coords -> plain index-aligned concat), a SubMConv3d
    fuses the concatenated channels back down to skip_channels, then `num_blocks`
    residual blocks refine at that width. Same shape as the `unet` branch's
    backbone3d_unet._DecoderStage, reimplemented here to avoid a cross-branch import."""

    def __init__(self, in_channels, skip_channels, num_blocks, kernel_size, stride):
        super().__init__()
        self.up = SparseInverseConv3d(in_channels, skip_channels, kernel_size=kernel_size, bias=False)
        self.up_bn = nn.BatchNorm1d(skip_channels)
        self.fuse = SubMConv3d(skip_channels * 2, skip_channels, kernel_size=3, bias=False)
        self.fuse_bn = nn.BatchNorm1d(skip_channels)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([SparseBasicBlock(skip_channels) for _ in range(num_blocks)])
        self.stride = stride
        self.padding = kernel_size // 2

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
        for block in self.blocks:
            x, _, _ = block(x, out_coords, out_index_grid, parent_grid_size)
        return x, out_coords, out_index_grid, parent_grid_size


class SparseDownSlotUpBackbone(nn.Module):
    def __init__(self, in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride,
                 upsample_stages, decoder_blocks_per_stage=None, decoder_out_channels=None,
                 slot_win_size=12, slot_num_cycles=1, slot_num_heads=4):
        super().__init__()
        stage_channels = list(stage_channels)
        n = len(stage_channels)
        assert 0 <= upsample_stages <= n, \
            f"UPSAMPLE_STAGES must be between 0 and {n} (= len(STAGE_CHANNELS)), got {upsample_stages}"
        if decoder_blocks_per_stage is None:
            decoder_blocks_per_stage = num_blocks_per_stage
        encoder_channels = [in_channels] + stage_channels

        self.encoder_stages = nn.ModuleList([
            Sparse3DStage(encoder_channels[i], encoder_channels[i + 1], num_blocks_per_stage, down_kernel, down_stride)
            for i in range(n)
        ])
        self.slotformer = SlotFormerBackbone(stage_channels[-1], slot_win_size, slot_num_cycles, slot_num_heads)
        self.num_upsample = upsample_stages
        # Invert only the M *deepest* encoder stages, deepest first. E.g. n=4, M=1
        # inverts just encoder_stages[3] and stops -- stages 0,1,2 stay downsampled,
        # matching total_stride = down_stride**(n-M) below. M=0 -> empty ModuleList
        # (no decoder at all, same as the encoder-only backbone).
        self.decoder_stages = nn.ModuleList([
            _DecoderStage(encoder_channels[i + 1], encoder_channels[i], decoder_blocks_per_stage, down_kernel, down_stride)
            for i in reversed(range(n - upsample_stages, n))
        ])

        final_channels = encoder_channels[n - upsample_stages]
        out_channels = decoder_out_channels or final_channels
        self._project = out_channels != final_channels
        if self._project:
            self.out_conv = SubMConv3d(final_channels, out_channels, kernel_size=3, bias=False)
            self.out_bn = nn.BatchNorm1d(out_channels)
            self.out_relu = nn.ReLU(inplace=True)

        self.out_channels = out_channels
        # M<N leaves total_stride > 1 -- e.g. n=4 stages of down_stride=2 with
        # UPSAMPLE_STAGES=1 undoes only the last halving, landing at stride 8 (1/8
        # resolution), not stride 1. detector.py reads this as self.stem_stride.
        self.total_stride = down_stride ** (n - upsample_stages)

    def forward(self, features, coords, index_grid, grid_size, batch_size):
        skips = []  # (features, coords, index_grid, grid_size) cached BEFORE each encoder stage runs
        x, c, ig, gs = features, coords, index_grid, grid_size
        for stage in self.encoder_stages:
            skips.append((x, c, ig, gs))
            x, c, ig, gs = stage(x, c, ig, gs, batch_size)

        x = self.slotformer(x, c)  # global context at the coarsest scale -- fewest active voxels here

        for stage in self.decoder_stages:
            skip_feat, skip_coords, skip_index_grid, skip_grid_size = skips.pop()
            x, c, ig, gs = stage(x, c, ig, gs, skip_feat, skip_coords, skip_index_grid, skip_grid_size)

        if self._project:
            x, _, _ = self.out_conv(x, c, ig, gs)
            x = self.out_relu(self.out_bn(x))

        return x, c, ig, gs
