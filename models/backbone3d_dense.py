"""Dense (plain nn.Conv3d) mirror of models/backbone3d.py's Sparse3DBackbone,
architecture-matched stage for stage (same channel widths, kernel size,
stride, blocks-per-stage). Used both by benchmark_backbone.py (sparse-vs-dense
compute time) and, via DenseBackboneWrapper below, as an actual BACKBONE.TYPE:
"dense" option in the real detector -- see configs/exp_dense.yaml.
"""
import torch
import torch.nn as nn

from .sparse_ops import build_index_grid


class DenseBasicBlock3d(nn.Module):
    """Two Conv3d + BN + ReLU with a residual connection -- dense analog of
    sparse_ops.SparseBasicBlock."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.conv1 = nn.Conv3d(channels, channels, kernel_size, padding=p, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size, padding=p, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class DenseStage3d(nn.Module):
    """Dense analog of backbone3d.Sparse3DStage: one strided Conv3d
    (changes channels + downsamples x,y,z together) then num_blocks residual
    blocks at the new channel count."""

    def __init__(self, in_channels, out_channels, num_blocks, down_kernel=3, down_stride=2):
        super().__init__()
        self.down = nn.Conv3d(
            in_channels, out_channels, down_kernel, stride=down_stride, padding=down_kernel // 2, bias=True
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([DenseBasicBlock3d(out_channels) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.relu(self.bn(self.down(x)))
        for block in self.blocks:
            x = block(x)
        return x


class Dense3DBackbone(nn.Module):
    """Dense analog of backbone3d.Sparse3DBackbone -- same stage_channels /
    num_blocks_per_stage / kernel / stride, but operating on a dense
    (B, C, X, Y, Z) tensor with standard nn.Conv3d instead of the custom
    sparse ops. Input must already be a dense voxel grid (see
    benchmark_backbone.py's sparse_to_dense)."""

    def __init__(self, in_channels, stage_channels=(16, 32, 48, 64), num_blocks_per_stage=3,
                 down_kernel=3, down_stride=2):
        super().__init__()
        stages = []
        c_in = in_channels
        for c_out in stage_channels:
            stages.append(DenseStage3d(c_in, c_out, num_blocks_per_stage, down_kernel, down_stride))
            c_in = c_out
        self.stages = nn.ModuleList(stages)
        self.out_channels = stage_channels[-1]

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


def sparse_to_dense_grid(features, coords, batch_size, grid_size):
    """(N,C) sparse voxel features + (N,4) [batch,x,y,z] coords -> dense
    (B,C,X,Y,Z) grid, zero-filled everywhere there's no active voxel."""
    X, Y, Z = grid_size
    C = features.shape[1]
    dense = torch.zeros(batch_size, C, X, Y, Z, device=features.device, dtype=features.dtype)
    if coords.shape[0] > 0:
        dense[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]] = features
    return dense


def dense_grid_to_sparse(dense):
    """Inverse of sparse_to_dense_grid: (B,C,X,Y,Z) -> ((B*X*Y*Z,C) features,
    (B*X*Y*Z,4) [batch,x,y,z] coords) enumerating EVERY grid position (a dense
    backbone has no notion of "inactive" voxels once the input is a full
    grid), in the matching order so row i of one lines up with row i of the
    other."""
    B, C, X, Y, Z = dense.shape
    device = dense.device
    b_idx, x_idx, y_idx, z_idx = torch.meshgrid(
        torch.arange(B, device=device), torch.arange(X, device=device),
        torch.arange(Y, device=device), torch.arange(Z, device=device), indexing="ij",
    )
    coords = torch.stack([b_idx, x_idx, y_idx, z_idx], dim=-1).reshape(-1, 4)
    features = dense.permute(0, 2, 3, 4, 1).reshape(-1, C)
    return features, coords


class DenseBackboneWrapper(nn.Module):
    """Adapts Dense3DBackbone to the same interface as Sparse3DBackbone /
    Sparse3DBackboneSpconv (forward(features, coords, index_grid, grid_size,
    batch_size) -> (features, coords, index_grid, grid_size)) so detector.py
    can drop it into the same backbone slot with no other code changes.
    Densifies the sparse VFE output, runs ordinary nn.Conv3d over the whole
    grid, then re-flattens the (now fully "active" -- every grid position has
    a value) output back into the sparse-style (features, coords) the rest of
    the pipeline (head/assigner/loss/decode) expects. That downstream code
    doesn't distinguish "genuinely sparse" from "dense but represented the
    same way", so it works unchanged; it just runs over every grid position
    (the final grid is small -- ~1k positions at this project's config --
    not the huge full-res one)."""

    def __init__(self, in_channels, stage_channels=(16, 32, 48, 64), num_blocks_per_stage=3,
                 down_kernel=3, down_stride=2):
        super().__init__()
        self.dense_backbone = Dense3DBackbone(in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride)
        self.out_channels = self.dense_backbone.out_channels
        self.total_stride = down_stride ** len(stage_channels)

    def forward(self, features, coords, index_grid, grid_size, batch_size):
        dense_in = sparse_to_dense_grid(features, coords, batch_size, grid_size)
        dense_out = self.dense_backbone(dense_in)
        out_features, out_coords = dense_grid_to_sparse(dense_out)
        out_grid_size = tuple(dense_out.shape[2:])
        out_index_grid = build_index_grid(out_coords, batch_size, out_grid_size, device=features.device)
        return out_features, out_coords, out_index_grid, out_grid_size
