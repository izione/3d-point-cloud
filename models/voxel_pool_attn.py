import torch
import torch.nn as nn

from .decoder_detr import pad_tokens


def cap_points_per_voxel(point_voxel_idx: torch.Tensor, num_voxels: int, max_points: int) -> torch.Tensor:
    """Returns a LongTensor of original point indices to keep, so that no
    voxel contributes more than max_points points -- chosen uniformly at
    random per voxel (not first-max_points, so no positional bias). Needed
    because this dataset's points-per-voxel is extremely skewed (measured on
    a real batch: median 2, max 155) -- padding every voxel to the batch's
    single largest voxel wastes >98% of the attention computation on empty
    slots for a typical voxel, and can OOM. Vectorized: no Python loop over
    voxels.

    Trick: sorting by (voxel_id + random fraction) groups every voxel's
    points contiguously (the random fraction never crosses an integer
    boundary) while randomly shuffling their order WITHIN that group in the
    same pass -- so the position of each point within its now-random-order
    group is a free "randomly ranked within voxel" index.
    """
    device = point_voxel_idx.device
    n = point_voxel_idx.shape[0]
    rand_key = point_voxel_idx.double() + torch.rand(n, device=device, dtype=torch.double)
    order = torch.argsort(rand_key)
    sorted_voxel = point_voxel_idx[order]
    counts = torch.bincount(sorted_voxel, minlength=num_voxels)
    offsets = torch.cumsum(counts, dim=0) - counts
    within_voxel_rank = torch.arange(n, device=device) - offsets[sorted_voxel]
    keep = within_voxel_rank < max_points
    return order[keep]


class VoxelPoolingAttention(nn.Module):
    """Point -> voxel aggregation via cross-attention instead of max-pool: one
    learned query per voxel (shared parameter, like models/denoising.py's
    single "indicator" vector or VoxSeT's latent codes) attends over that
    voxel's own points, producing a LEARNED weighted combination instead of
    "keep only the per-channel winner and discard everything else". Reuses
    models/decoder_detr.py::pad_tokens for the variable-points-per-voxel
    grouping -- same utility the DETR decoder's cross-attention already uses
    for the analogous variable-tokens-per-sample problem.

    Points per voxel are capped to max_points first (see
    cap_points_per_voxel) -- without this, a single outlier voxel forces
    every voxel in the batch to pad to its size (measured: median 2 points/
    voxel, max 155, on a real training batch), which is both very slow and
    can OOM.

    Every KEPT voxel has at least one real point by construction (it's only
    a voxel because torch.unique found ≥1 point there, and capping never
    removes a voxel's LAST point), so the padding mask is never all-True for
    a row -- no empty-softmax risk.
    """

    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 4, max_points: int = 32):
        super().__init__()
        self.query = nn.Parameter(torch.randn(in_channels) * 0.02)
        self.cross_attn = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(in_channels)
        self.out_proj = nn.Linear(in_channels, out_channels)
        self.max_points = max_points

    def forward(self, feat: torch.Tensor, point_voxel_idx: torch.Tensor, num_voxels: int) -> torch.Tensor:
        keep = cap_points_per_voxel(point_voxel_idx, num_voxels, self.max_points)
        feat, point_voxel_idx = feat[keep], point_voxel_idx[keep]

        padded, mask = pad_tokens(feat, point_voxel_idx, num_voxels)  # (V,T_max,C), True=pad
        query = self.query[None, None, :].expand(num_voxels, 1, -1)   # (V,1,C)
        attn_out, _ = self.cross_attn(query, padded, padded, key_padding_mask=mask)  # (V,1,C)
        pooled = self.norm(attn_out.squeeze(1))
        return self.out_proj(pooled)  # (V, out_channels)
