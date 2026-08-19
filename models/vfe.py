import torch
import torch.nn as nn

from .sparse_ops import scatter_max, scatter_mean


class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, point_feat, point_voxel_idx, num_voxels):
        x = self.relu(self.bn(self.linear(point_feat)))
        voxel_feat = scatter_max(x, point_voxel_idx, num_voxels)
        if self.is_last:
            return voxel_feat
        return torch.cat([x, voxel_feat[point_voxel_idx]], dim=1)


class VFE(nn.Module):
    """10D augmented point feature -> 2-layer PFN -> 64D per-voxel feature.

    10D = (x,y,z,intensity) + cluster_offset(3) + center_offset(3).
    """

    def __init__(self, num_filters=(64, 64)):
        super().__init__()
        in_dim = 10
        layers = []
        for i, out_dim in enumerate(num_filters):
            layers.append(PFNLayer(in_dim, out_dim, is_last=(i == len(num_filters) - 1)))
            in_dim = out_dim * 2  # next layer sees [own feat, pooled voxel feat] concat
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

        feat = torch.cat([xyz, intensity, cluster_offset, center_offset], dim=1)
        for layer in self.layers:
            feat = layer(feat, point_voxel_idx, num_voxels)  # point-level until the last (is_last) layer
        return feat  # (num_voxels, out_channels)
