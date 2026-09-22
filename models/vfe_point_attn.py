import torch
import torch.nn as nn

from .point_transformer import PointTransformerBlock, knn_indices_per_sample
from .sparse_ops import scatter_max


class PointAttentionVFE(nn.Module):
    """Point-level feature extraction via Point Transformer blocks (see
    models/point_transformer.py) instead of a plain PFN (models/vfe.py::VFE),
    run BEFORE voxel pooling: raw points -> k-NN self-attention refinement
    (k=16, chosen over a radius query after measuring both on a real frame
    from this dataset -- see point_transformer.py's docstring) -> masked
    max-pool per voxel (same aggregation as the existing VFE, just on
    attention-refined features instead of raw+offset features).

    Unlike VFE/MVFE, this needs to know which SAMPLE each point belongs to
    (for the k-NN search to never cross a sample boundary), so its forward()
    takes two extra arguments -- models/detector_detr.py branches on
    VFE.TYPE to pass them only when this class is selected.
    """

    def __init__(self, out_channels=64, num_blocks=1, k=16):
        super().__init__()
        self.in_proj = nn.Linear(4, out_channels)  # raw (x,y,z,intensity)
        self.blocks = nn.ModuleList([PointTransformerBlock(out_channels) for _ in range(num_blocks)])
        self.k = k
        self.out_channels = out_channels

    def forward(self, points, point_voxel_idx, voxel_coords, num_voxels, pc_range, voxel_size,
                point_batch_idx, batch_size):
        pos = points[:, :3]
        feat = self.in_proj(points)
        knn_idx = knn_indices_per_sample(pos, point_batch_idx, batch_size, self.k)
        for block in self.blocks:
            feat = block(feat, pos, knn_idx)
        return scatter_max(feat, point_voxel_idx, num_voxels)
