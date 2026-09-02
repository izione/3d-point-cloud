"""Sparse 3D U-Net backbone mirroring the common "ResNet encoder + light U-Net
decoder" pattern (segmentation_models_pytorch-style: heavy residual/bottleneck
blocks in the encoder, plain conv in the decoder) -- except the encoder's residual
blocks are replaced by SlotFormer(3L) instead of ResNet Bottleneck blocks.
BACKBONE.TYPE: sparse_slot_light_unet.

Fourth SlotFormer-backbone variant in this repo -- don't conflate with the other
three:
  - backbone3d_down_slot_up.py: residual blocks everywhere, ONE extra SlotFormer at
    the bottleneck only, upsample depth configurable (M<=N).
  - backbone3d_slot_stages.py: encoder ONLY, no decoder at all, SlotFormer replaces
    residual blocks in every encoder stage.
  - backbone3d_slot_unet.py: full encoder-decoder, SlotFormer replaces residual
    blocks in BOTH encoder AND decoder (heaviest of the four).
  - THIS FILE: SlotFormer replaces residual blocks in the ENCODER only (same
    Sparse3DSlotEncoderStage as the other two SlotFormer-encoder variants); the
    decoder is deliberately kept light -- up-conv + skip concat + ONE fuse SubMConv3d,
    with NO residual blocks and NO SlotFormer afterward, matching how a ResNet-UNet's
    decoder is typically just plain conv (the heavy lifting stays in the encoder,
    which already has a pretrained/well-optimized representation to draw on -- here,
    the encoder's SlotFormer attention).
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


class _LightDecoderStage(nn.Module):
    """Inverts one Sparse3DSlotEncoderStage: up-conv + skip concat + ONE fuse
    SubMConv3d -- deliberately stops there (no residual blocks, no SlotFormer). This
    is the "light decoder" half of the ResNet-encoder-UNet pattern: all the
    heavy per-stage refinement lives in the encoder, the decoder just blends the
    upsampled feature with its skip connection and moves on."""

    def __init__(self, in_channels, skip_channels, down_kernel, down_stride):
        super().__init__()
        self.up = SparseInverseConv3d(in_channels, skip_channels, kernel_size=down_kernel, bias=False)
        self.up_bn = nn.BatchNorm1d(skip_channels)
        self.fuse = SubMConv3d(skip_channels * 2, skip_channels, kernel_size=3, bias=False)
        self.fuse_bn = nn.BatchNorm1d(skip_channels)
        self.relu = nn.ReLU(inplace=True)
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
        return x, out_coords, out_index_grid, parent_grid_size


class SparseSlotLightUNetBackbone(nn.Module):
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
        self.decoder_stages = nn.ModuleList([
            _LightDecoderStage(encoder_channels[i + 1], encoder_channels[i], down_kernel, down_stride)
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
