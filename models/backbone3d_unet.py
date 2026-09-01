"""Sparse 3D U-Net backbone -- BACKBONE.TYPE: sparse_unet.

An N-stage encoder identical in shape to backbone3d.Sparse3DBackbone's
stages (SparseConv3dDown + SubMConv3d residual blocks per stage), paired
with an N-stage decoder that undoes each downsample with
sparse_ops.SparseInverseConv3d and merges the restored features with that
stage's cached (skip) features at the same resolution.

Why this exists, from the design conversation this was built out of: the
encoder-only backbones (backbone3d.py) permanently lose spatial resolution
at every downsampling stage -- going back to a 4-stage/stride-16 encoder
(see configs/default.yaml's history) tanked precision/recall, most likely
because the detection heads ended up regressing against a much coarser
effective voxel size, not because of insufficient receptive field.
SlotFormer (models/slotformer.py) compensates for a shallow encoder's
*limited context*, but it does nothing to restore resolution that a deeper
encoder would have thrown away -- those are two different problems. A U-Net
decoder is the standard fix for the resolution problem specifically: this
backbone's final output sits back at total_stride=1 (the same effective
voxel size VFE produced), so an encoder can go as deep/wide as
STAGE_CHANNELS allows for receptive field/capacity without the head ever
seeing a coarser grid than the input. AFDet (Ge & Ding, 2020) is the closest
published precedent for this combination (U-Net-style encoder-decoder
backbone directly feeding centerpoint/offset/size/orientation heads).

Trade-off worth knowing: this backbone's receptive field, however deep, is
still bounded by its conv kernel chain (local), unlike SlotFormer's
near-global reach within a couple of axial passes. Whether that matters
depends on whether the task ever needs literally scene-wide context (see
the "does a compact, contiguous target need SlotFormer" discussion) -- it's
meant to be benchmarked against sparse+SlotFormer, not assumed better.

Pure-PyTorch only for now (built on sparse_ops.py, not spconv) -- consistent
with backbone3d.py's fallback rather than backbone3d_spconv.py's. A
spconv-accelerated version could reuse spconv.SparseInverseConv3d paired to
the down-conv's indice_key, the same way backbone3d_spconv.py mirrors
backbone3d.py, but isn't implemented yet: BACKBONE.TYPE: sparse_unet always
uses this pure-PyTorch path regardless of BACKBONE.TYPE: auto's spconv
probing.
"""
import torch
import torch.nn as nn

from .backbone_registry import register_backbone
from .sparse_ops import SubMConv3d, SparseBasicBlock, SparseConv3dDown, SparseInverseConv3d


class _EncoderStage(nn.Module):
    """Down-conv (changes channels + downsamples x,y,z together) followed by
    `num_blocks` SubMConv3d residual blocks at the new channel count --
    identical shape to backbone3d.Sparse3DStage, duplicated here (rather than
    imported) so this file depends only on sparse_ops primitives."""

    def __init__(self, in_channels, out_channels, num_blocks, kernel_size, stride):
        super().__init__()
        self.down = SparseConv3dDown(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([SparseBasicBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, features, coords, index_grid, grid_size, batch_size):
        x, out_coords, out_index_grid, out_grid_size = self.down(features, coords, index_grid, grid_size, batch_size)
        x = self.relu(self.bn(x))
        for block in self.blocks:
            x, out_coords, out_index_grid = block(x, out_coords, out_index_grid, out_grid_size)
        return x, out_coords, out_index_grid, out_grid_size


class _DecoderStage(nn.Module):
    """Inverts one _EncoderStage: SparseInverseConv3d restores the cached
    parent (pre-downsample) coordinate set, the result is concatenated with
    that stage's cached skip features (same coords -> plain index-aligned
    concat, see SparseInverseConv3d's docstring), a SubMConv3d fuses the
    concatenated channels back down to skip_channels, then `num_blocks`
    residual blocks refine at that width."""

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


class SparseUNetBackbone(nn.Module):
    def __init__(self, in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride,
                 decoder_blocks_per_stage=None, decoder_out_channels=None):
        super().__init__()
        stage_channels = list(stage_channels)
        if decoder_blocks_per_stage is None:
            decoder_blocks_per_stage = num_blocks_per_stage
        # encoder_channels[i] -> encoder_channels[i+1] is what encoder stage i does;
        # e.g. in_channels=64, stage_channels=[96,128,160] -> [64, 96, 128, 160].
        encoder_channels = [in_channels] + stage_channels

        self.encoder_stages = nn.ModuleList([
            _EncoderStage(encoder_channels[i], encoder_channels[i + 1], num_blocks_per_stage, down_kernel, down_stride)
            for i in range(len(stage_channels))
        ])
        # Built in the order they're actually applied (deepest stage first,
        # inverting encoder_stages[-1] first) -- reversed(range(...)) below,
        # not reversed(ModuleList) at call time, so state_dict order matches
        # forward-pass order for readability.
        self.decoder_stages = nn.ModuleList([
            _DecoderStage(encoder_channels[i + 1], encoder_channels[i], decoder_blocks_per_stage, down_kernel, down_stride)
            for i in reversed(range(len(stage_channels)))
        ])

        final_channels = encoder_channels[0]  # = in_channels, after the last (shallowest) decoder stage
        out_channels = decoder_out_channels or max(stage_channels[0], final_channels)
        self._project = out_channels != final_channels
        if self._project:
            self.out_conv = SubMConv3d(final_channels, out_channels, kernel_size=3, bias=False)
            self.out_bn = nn.BatchNorm1d(out_channels)
            self.out_relu = nn.ReLU(inplace=True)

        self.out_channels = out_channels
        # The decoder restores all the way back to the encoder's OWN input
        # resolution (VFE's output voxels) -- unlike the encoder-only
        # backbones, nothing downstream (SlotFormer, the head, decode.py) is
        # working at a coarser effective voxel size than voxelize_batch produced.
        self.total_stride = 1

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


@register_backbone("sparse_unet")
def build_sparse_unet_backbone(in_channels, bcfg):
    """Reuses BACKBONE.STAGE_CHANNELS/NUM_BLOCKS_PER_STAGE/DOWNSAMPLE_KERNEL/
    DOWNSAMPLE_STRIDE (same meaning as for BACKBONE.TYPE: auto's encoder-only
    backbone) plus two optional sparse_unet-only knobs:
      DECODER_BLOCKS_PER_STAGE  -- residual blocks per decoder stage after
                                    the skip-fusion conv (default:
                                    NUM_BLOCKS_PER_STAGE, i.e. symmetric).
      DECODER_OUT_CHANNELS      -- final channel width handed to SlotFormer/
                                    the head (default:
                                    max(STAGE_CHANNELS[0], VFE out_channels)).
    """
    return SparseUNetBackbone(
        in_channels,
        bcfg["STAGE_CHANNELS"],
        bcfg["NUM_BLOCKS_PER_STAGE"],
        bcfg["DOWNSAMPLE_KERNEL"],
        bcfg["DOWNSAMPLE_STRIDE"],
        decoder_blocks_per_stage=bcfg.get("DECODER_BLOCKS_PER_STAGE"),
        decoder_out_channels=bcfg.get("DECODER_OUT_CHANNELS"),
    )
