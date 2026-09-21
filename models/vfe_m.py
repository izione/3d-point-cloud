import torch
import torch.nn as nn

from .sparse_ops import scatter_mean, scatter_sum
from .vfe import PFNLayer


class MVFE(nn.Module):
    """"mVFE" from SparseVoxFormer: their modification of a HardSimpleVFE-style
    encoder to also keep per-voxel std-dev and point count (5 -> 11 raw dims in
    the paper). Here it's 10 -> 14, since this repo's base encoder
    (models/vfe.py::VFE) already augments with a center_offset the paper's base
    doesn't have -- everything else (2-layer PFN, hierarchical [pointwise,
    pooled] concat) is unchanged from VFE.

    Why this matters per the paper: raw per-voxel mean/max throws away how
    tightly the points cluster inside a voxel and how many there were --
    std-dev and point count give the network that information directly instead
    of forcing it to infer an approximation from the (fixed) VFE output alone.
    """

    def __init__(self, num_filters=(64, 64)):
        super().__init__()
        in_dim = 14
        layers = []
        for i, out_dim in enumerate(num_filters):
            layers.append(PFNLayer(in_dim, out_dim, is_last=(i == len(num_filters) - 1)))
            in_dim = out_dim * 2
        self.layers = nn.ModuleList(layers)
        self.out_channels = num_filters[-1]

    def forward(self, points, point_voxel_idx, voxel_coords, num_voxels, pc_range, voxel_size):
        xyz = points[:, :3]
        intensity = points[:, 3:4]

        voxel_mean = scatter_mean(xyz, point_voxel_idx, num_voxels)
        cluster_offset = xyz - voxel_mean[point_voxel_idx]

        vox_xyz_idx = voxel_coords[:, 1:4].float()
        pc_min = pc_range[:3].to(points.device)
        vsize = voxel_size.to(points.device)
        voxel_center = pc_min + (vox_xyz_idx + 0.5) * vsize
        center_offset = xyz - voxel_center[point_voxel_idx]

        # -- mVFE additions over VFE: per-voxel std-dev + point count --
        sq_mean = scatter_mean(xyz * xyz, point_voxel_idx, num_voxels)
        variance = (sq_mean - voxel_mean * voxel_mean).clamp(min=0)
        voxel_std = torch.sqrt(variance + 1e-8)                                    # (num_voxels,3)
        ones = torch.ones(points.shape[0], device=points.device, dtype=points.dtype)
        point_count = scatter_sum(ones, point_voxel_idx, num_voxels)               # (num_voxels,)
        log_count = torch.log1p(point_count)

        feat = torch.cat([
            xyz, intensity, cluster_offset, center_offset,
            voxel_std[point_voxel_idx], log_count[point_voxel_idx].unsqueeze(-1),
        ], dim=1)
        for layer in self.layers:
            feat = layer(feat, point_voxel_idx, num_voxels)
        return feat
