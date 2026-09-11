"""model.py - CenterPoint (Yin, Zhou & Krahenbuhl, CVPR 2021) reproduction:
VoxelNet-style VFE + a SPARSE 3D middle encoder + 2D RPN backbone. The paper's
own words: "we follow the network designs of SECOND for the backbone" --
SECOND (Yan et al. 2018) is itself the paper that replaced VoxelNet's
original dense Conv3D middle layers with sparse convolution, so a
paper-faithful CenterPoint backbone is sparse, not dense. (An earlier version
of this file used dense Conv3D instead, purely to dodge a spconv install
dependency -- that was NOT what the paper does.)

The sparse conv itself is a pure-PyTorch implementation (sparse_conv_pure.py),
not spconv -- real-world testing (2026-09) hit a hard, unresolved upstream
packaging bug in spconv's own cumm dependency on every available CUDA tag
(official PyPI cu126: cumm's bundled tensorview C++ headers are missing from
the wheel, github.com/traveller59/spconv#766 /
github.com/FindDefinition/cumm#7; the rathaROG community index built to work
around exactly this only publishes cu128/cu130, not cu126). sparse_conv_pure.py
sidesteps the whole problem: no compiled extension, so no CUDA-tag-matching
to get right at all. See its own docstring and test_sparse_conv_pure.py for
the correctness proof (exact match against nn.Conv3d).

Feeds a single CenterHead (Sec.3.1): Gaussian-heatmap classification +
sub-voxel offset / absolute height / log-size / 6D-rotation regression, no
anchors, no NMS at decode time (3x3 max-pool peak extraction instead).

This is a clean reproduction -- unlike the sibling voxelnet_baseline repo's
own model.py (which stacks many extra research-only heads/branches on the
same backbone for its own ablation program), this file has ONLY the
CenterPoint paper's own 5 regression targets (heatmap/offset/z/dim/rot) and
nothing else.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import sparse_conv_pure as spconv
import config
from sparse_ops import restrict_xy_support, yx_key


def _pool_points(pointwise: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """voxel-wise pointwise feature -> voxel-wise representative (hard max-pool).
    pointwise: (K,T,C), mask: (K,T) bool. -> (K,1,C)."""
    neg_inf = torch.finfo(pointwise.dtype).min
    masked_for_max = pointwise.masked_fill(~mask.unsqueeze(-1), neg_inf)
    aggregated, _ = masked_for_max.max(dim=1, keepdim=True)
    return aggregated.clamp_min(0.0)


class VFELayer(nn.Module):
    """FCN(Linear+BN+ReLU) -> per-voxel hard max-pool -> point-wise concat."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        assert out_channels % 2 == 0
        self.units = out_channels // 2
        self.linear = nn.Linear(in_channels, self.units)
        self.bn = nn.BatchNorm1d(self.units)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        K, T, _ = x.shape
        flat = x.reshape(K * T, -1)
        flat_mask = mask.reshape(K * T)
        pointwise = flat.new_zeros(K * T, self.units)
        if flat_mask.any():
            pointwise[flat_mask] = F.relu(self.bn(self.linear(flat[flat_mask])))
        pointwise = pointwise.reshape(K, T, self.units)

        aggregated = _pool_points(pointwise, mask)
        aggregated = aggregated.expand(-1, T, -1)

        out = torch.cat([pointwise, aggregated], dim=2)
        return out * mask.unsqueeze(-1)


class StackedVFE(nn.Module):
    """VFE-1(7,32) -> VFE-2(32,128) -> FCN(128,128)+BN+ReLU -> hard max-pool over points."""

    def __init__(self):
        super().__init__()
        self.vfe1 = VFELayer(config.INPUT_FEATURE_DIM, 32)
        self.vfe2 = VFELayer(32, 128)
        self.final_linear = nn.Linear(128, 128)
        self.final_bn = nn.BatchNorm1d(128)

    def forward(self, voxel_features: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        K, T, _ = voxel_features.shape
        mask = torch.arange(T, device=voxel_features.device)[None, :] < num_points[:, None]

        x = self.vfe1(voxel_features, mask)
        x = self.vfe2(x, mask)

        flat = x.reshape(K * T, -1)
        flat_mask = mask.reshape(K * T)
        pointwise = flat.new_zeros(K * T, 128)
        if flat_mask.any():
            pointwise[flat_mask] = F.relu(self.final_bn(self.final_linear(flat[flat_mask])))
        pointwise = pointwise.reshape(K, T, 128)

        voxelwise = _pool_points(pointwise, mask).squeeze(1)
        return voxelwise


class SparseMiddleEncoder(nn.Module):
    """SECOND-style sparse 3D middle encoder -- same 3-layer shape as the
    dense VoxelNet ConvMiddleLayers it replaces (128->64->64->64, D'(=10)
    reduced to 2 via stride(2,1,1)/stride(1,1,1,pad=(0,1,1))/stride(2,1,1)),
    computed (via sparse_conv_pure.py, a pure-PyTorch sparse conv) over only
    the active voxels instead of a dense (B,128,D,H,W) tensor. Returns a
    genuine dense (B,64*D_out,H,W) tensor (via SparseConvTensor.dense() +
    reshape) -- the 2D RPNBackbone needs a real dense tensor, so densifying
    here (once, right after the sparse 3D stage) is the natural sparse/dense
    boundary, matching SECOND's own design and ConvMiddleLayers' own reshape
    convention.

    Every layer here has stride=1 in x,y (even the two stride=(2,1,1)
    layers) -- a regular (non-submanifold) sparse conv still discovers
    neighbor-reachable x,y positions beyond the input support for a
    z-only-stride conv, so restrict_xy_support is applied after each layer to
    keep the active x,y set exactly what it should be (see sparse_ops.py)."""

    def __init__(self):
        super().__init__()
        self.conv1 = spconv.SparseConv3d(128, 64, 3, stride=(2, 1, 1), padding=(1, 1, 1),
                                          indice_key="cp_mid1")
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = spconv.SparseConv3d(64, 64, 3, stride=(1, 1, 1), padding=(0, 1, 1),
                                          indice_key="cp_mid2")
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = spconv.SparseConv3d(64, 64, 3, stride=(2, 1, 1), padding=(1, 1, 1),
                                          indice_key="cp_mid3")
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x) -> torch.Tensor:
        """x: spconv.SparseConvTensor, features=(K_total,128). -> (B,64*D_out,H,W)."""
        H, W = x.spatial_shape[1], x.spatial_shape[2]
        allowed_keys = torch.unique(yx_key(x.indices, H, W))

        x = self.conv1(x)
        x = x.replace_feature(self.relu(self.bn1(x.features)))
        x = restrict_xy_support(x, allowed_keys, H, W)

        x = self.conv2(x)
        x = x.replace_feature(self.relu(self.bn2(x.features)))
        x = restrict_xy_support(x, allowed_keys, H, W)

        x = self.conv3(x)
        x = x.replace_feature(self.relu(self.bn3(x.features)))
        x = restrict_xy_support(x, allowed_keys, H, W)

        dense = x.dense()  # (B,64,D_out,H,W), zero-filled where inactive
        B, C, D, H_, W_ = dense.shape
        return dense.reshape(B, C * D, H_, W_)


class RPNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            layers += [nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                       nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class RPNBackbone(nn.Module):
    """3-block downsample + deconv-upsample-concat neck (VoxelNet Fig.4 / SECOND's
    dense re-derivation, upsample_strides=(1,2,4) convention -- block1's deconv is
    a pure 1x1 channel projection with no spatial change, matching SECOND's own
    reference implementation, since the paper's literal Fig.4 deconv1 params
    don't actually produce a size that concats with block2/3's output)."""

    def __init__(self):
        super().__init__()
        c_in = config.RPN_IN_CHANNELS
        c1, c2, c3 = config.RPN_BLOCK_CHANNELS
        n1, n2, n3 = config.RPN_BLOCK_LAYERS
        c_up = config.RPN_UPSAMPLE_CHANNELS

        self.block1 = RPNBlock(c_in, c1, n1)
        self.block2 = RPNBlock(c1, c2, n2)
        self.block3 = RPNBlock(c2, c3, n3)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(c1, c_up, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c_up), nn.ReLU(inplace=True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(c2, c_up, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(c_up), nn.ReLU(inplace=True))
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(c3, c_up, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(c_up), nn.ReLU(inplace=True))

        self.out_channels = c_up * 3

    @staticmethod
    def _match_hw(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        top, left = (h - target_h) // 2, (w - target_w) // 2
        return x[..., top:top + target_h, left:left + target_w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        u1, u2, u3 = self.deconv1(f1), self.deconv2(f2), self.deconv3(f3)
        target_h, target_w = u1.shape[-2], u1.shape[-1]
        u2 = self._match_hw(u2, target_h, target_w)
        u3 = self._match_hw(u3, target_h, target_w)
        return torch.cat([u1, u2, u3], dim=1)


class CenterHead(nn.Module):
    """CenterPoint Sec.3.1's own 5 regression targets, nothing else: a shared
    backbone feature map feeds one heatmap head + 4 independent regression
    heads (offset/z/dim/rot), each a plain 1x1 conv."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.heatmap_head = nn.Conv2d(in_channels, 1, 1)  # raw logit, sigmoid at loss/decode
        self.offset_head = nn.Conv2d(in_channels, 2, 1)   # [dx,dy] sub-cell, cell-size units
        self.z_head = nn.Conv2d(in_channels, 1, 1)        # absolute z (m), no anchor
        self.dim_head = nn.Conv2d(in_channels, 3, 1)      # [log l, log w, log h]
        self.rot_head = nn.Conv2d(in_channels, 6, 1)      # 6D continuous rotation (rotation3d.py)

        # CenterNet/RetinaNet-standard focal-loss bias init: without this, the
        # heatmap head collapses to "everywhere background" in the first few
        # steps (thousands of negative cells per handful of positives).
        prior_prob = 0.1
        nn.init.constant_(self.heatmap_head.bias, -math.log((1 - prior_prob) / prior_prob))

    def forward(self, feat2d: torch.Tensor) -> dict:
        return {
            "heatmap": self.heatmap_head(feat2d),
            "offset": self.offset_head(feat2d),
            "z": self.z_head(feat2d),
            "dim": self.dim_head(feat2d),
            "rot": self.rot_head(feat2d),
        }


class CenterPointVoxelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = StackedVFE()
        self.middle = SparseMiddleEncoder()
        self.backbone = RPNBackbone()
        self.head = CenterHead(self.backbone.out_channels)
        self.grid_size = config.GRID_SIZE  # (W',H',D')

    def forward(self, voxel_features, num_points, coords) -> dict:
        """voxel_features: (K_total,T,7), num_points: (K_total,), coords: (K_total,4)
        [batch_idx,z,y,x] (multiple frames concatenated along K).
        -> dict of (B,C,H'',W'') tensors: heatmap(1)/offset(2)/z(1)/dim(3)/rot(6)."""
        voxelwise = self.vfe(voxel_features, num_points)  # (K_total,128)

        B = int(coords[:, 0].max().item()) + 1 if len(coords) else 1
        Wp, Hp, Dp = self.grid_size
        sparse_in = spconv.SparseConvTensor(voxelwise, coords.int(), (Dp, Hp, Wp), B)

        feat2d = self.middle(sparse_in)  # (B,64*D'',H',W') -- already reshaped, paper Sec.3.1 "reshaping"

        feat = self.backbone(feat2d)
        return self.head(feat)
