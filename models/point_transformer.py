"""Point Transformer layer (Zhao et al., ICCV 2021, arXiv 2012.09164) -- vector
self-attention over each point's k=16 nearest neighbors, used here as a
point-level feature refinement stage BEFORE voxelization (see
models/vfe_point_attn.py), not as the paper's own 5-stage FPS-downsampling
network. We only reuse the core layer/block; voxelization is our downsampling
step, so no farthest-point-sampling pyramid is needed here.

Exact mechanism (paper Eq. 3), quoted precisely since this is meant to
reproduce it rather than approximate it:

    y_i = sum_{x_j in X(i)} rho( gamma( phi(x_i) - psi(x_j) + delta ) ) (*) ( alpha(x_j) + delta )

- X(i): k=16 nearest neighbors of point i (fixed k via k-NN, NOT a radius
  query -- measured on a real frame from this project's own dataset that a
  fixed k=16 costs LESS total compute than any tested radius, 0.2/0.3/0.5m,
  because this data's local point density is uneven enough that radius
  queries pull in far more real neighbors on average; k-NN also never hits
  the "zero neighbors" edge case a radius query can in sparse regions).
- phi, psi, alpha: independent linear projections (the paper's Q/K/V).
- delta = theta(p_i - p_j): a TRAINABLE position encoding (2-linear+ReLU MLP
  on relative 3D coordinates), added to BOTH the attention-logit branch and
  the value branch (the paper's own ablation found both matter).
- gamma: 2-linear+ReLU MLP producing a per-channel attention VECTOR (not a
  scalar) -- "vector attention", the paper's other key ablation finding.
- rho: softmax, over the neighbor axis, independently per channel.
- (*): elementwise product (not a dot product).
"""
import torch
import torch.nn as nn


def knn_indices_per_sample(points: torch.Tensor, batch_idx: torch.Tensor, batch_size: int, k: int) -> torch.Tensor:
    """points: (N,3), batch_idx: (N,) which sample each point belongs to.
    Returns (N,k) GLOBAL point indices of each point's k nearest neighbors --
    computed separately per sample so neighbors never cross a sample boundary.
    Looped per sample (not vectorized across the whole batch at once): each
    sample's own point count is small enough that a per-sample cdist is cheap,
    and this avoids ever materializing an all-samples-at-once distance matrix
    (which would need N_total^2 memory for no benefit, since cross-sample
    entries would just be masked out anyway)."""
    device = points.device
    knn_idx = torch.empty(points.shape[0], k, dtype=torch.long, device=device)
    for b in range(batch_size):
        idx = (batch_idx == b).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        pts = points[idx]
        k_b = min(k, idx.numel())
        dist = torch.cdist(pts, pts)
        _, local_knn = torch.topk(dist, k=k_b, largest=False)  # (n_b, k_b), includes self at distance 0
        if k_b < k:
            # a sample with fewer than k points: repeat the closest point to
            # pad up to k (this only fires for degenerate/near-empty frames)
            pad = local_knn[:, -1:].expand(-1, k - k_b)
            local_knn = torch.cat([local_knn, pad], dim=1)
        knn_idx[idx] = idx[local_knn]
    return knn_idx


class PointTransformerLayer(nn.Module):
    def __init__(self, channels: int, mid_channels: int = None):
        super().__init__()
        mid_channels = mid_channels or channels
        self.phi = nn.Linear(channels, mid_channels)    # query (per-point, not per-pair)
        self.psi = nn.Linear(channels, mid_channels)    # key
        self.alpha = nn.Linear(channels, mid_channels)  # value
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, mid_channels), nn.ReLU(inplace=True), nn.Linear(mid_channels, mid_channels),
        )  # theta: trainable position encoding
        self.gamma_mlp = nn.Sequential(
            nn.Linear(mid_channels, mid_channels), nn.ReLU(inplace=True), nn.Linear(mid_channels, mid_channels),
        )  # gamma: produces the per-channel attention vector
        self.out_proj = nn.Linear(mid_channels, channels)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        """x: (N,C) point features, pos: (N,3) point coordinates, knn_idx: (N,k)."""
        q = self.phi(x)                      # (N,C')
        k_all = self.psi(x)                  # (N,C')
        v_all = self.alpha(x)                # (N,C')

        k_nb = k_all[knn_idx]                 # (N,k,C')
        v_nb = v_all[knn_idx]                 # (N,k,C')
        rel_pos = pos.unsqueeze(1) - pos[knn_idx]  # (N,k,3) = p_i - p_j
        delta = self.pos_mlp(rel_pos)          # (N,k,C')

        rel = q.unsqueeze(1) - k_nb + delta    # (N,k,C')
        attn_logits = self.gamma_mlp(rel)      # (N,k,C')
        attn = torch.softmax(attn_logits, dim=1)

        y = (attn * (v_nb + delta)).sum(dim=1)  # (N,C')
        return self.out_proj(y)


class PointTransformerBlock(nn.Module):
    """Residual block wrapping the layer (paper Fig. 4a): Linear down ->
    PointTransformerLayer -> Linear up -> residual add."""

    def __init__(self, channels: int, mid_channels: int = None):
        super().__init__()
        mid_channels = mid_channels or channels
        self.linear_down = nn.Linear(channels, mid_channels)
        self.pt_layer = PointTransformerLayer(mid_channels, mid_channels)
        self.linear_up = nn.Linear(mid_channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        identity = x
        y = self.linear_down(x)
        y = self.pt_layer(y, pos, knn_idx)
        y = self.linear_up(y)
        return self.norm(identity + y)
