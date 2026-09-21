"""DSVT (Dynamic Sparse Voxel Transformer with Rotated Sets, Wang et al. CVPR
2023, arXiv 2301.06051) -- the actual module SparseVoxFormer uses for its
"Deep Fusion Module" sparse-token refinement stage. This is a from-scratch
implementation, since the official repo's op is a custom CUDA kernel; the
math here follows the paper directly rather than approximating it with the
codebase's pre-existing SlotFormerBackbone (models/slotformer.py, from a
different paper, FSHNet -- kept available as REFINEMENT.TYPE: slotformer for
comparison, see models/detector_detr.py).

Core idea, in the paper's own terms:
- Voxels are grouped into non-overlapping 3D **windows** (window_shape).
- Within each window, voxels are partitioned into fixed-size **sets** of
  exactly tau voxels each (dynamic_set_partition below) -- NOT fixed-shape
  sub-windows. This equalizes compute across sets regardless of how unevenly
  voxels are actually distributed, which is DSVT's whole efficiency point
  (every set is the same size, so attention batches perfectly with no
  padding/masking). Windows with a leftover remainder get some voxels
  duplicated into more than one set to fill the last set to tau; the paper
  found (their Table 8) that masking those duplicates out actually hurts
  performance, so they're simply left in.
- A DSVT **block** is 2 layers: one sorts voxels X-axis-major before forming
  sets, the next sorts Y-axis-major ("rotated sets") -- this is what lets
  information cross between two sets that shared no voxels in the previous
  layer, since their set boundaries were computed along a different axis
  order (comparable in spirit to Swin's shifted windows, but for irregular
  sparse sets rather than a regular grid).
- Consecutive blocks use different window sizes ("hybrid window design").

Two simplifications from the paper, made explicit rather than silently
approximated:
- Positional encoding reuses this codebase's existing sinusoidal 3D
  coordinate embedding (models/decoder_detr.py::token_positional_embedding)
  rather than the paper's own (Swin-style) relative position encoding.
- The paper's separate "attention-style 3D pooling" module (for downsampling
  BETWEEN backbone stages in their multi-stage DSVT-V variant) isn't
  implemented -- this DSVT sits at a single fixed resolution here, same role
  SlotFormerBackbone already played, with no stage changes of its own.
"""
import torch
import torch.nn as nn

from .decoder_detr import token_positional_embedding
from .sparse_ops import scatter_mean

_KEY_SCALE = 10000  # comfortably exceeds any real coordinate/window-bucket value in this project


def dynamic_set_partition(coords: torch.Tensor, window_shape, tau: int, x_major: bool):
    """coords: (N,4) [batch,x,y,z] integer voxel coords. Returns (num_sets, tau)
    long tensor of ORIGINAL voxel indices (0..N-1) -- gather features/coords
    with this to get (num_sets, tau, *) uniformly-shaped sets ready for batched
    attention. Some indices repeat (see module docstring) when a window's
    voxel count doesn't divide evenly by tau; every real voxel appears at
    least once (see the paper's Eq., reproduced in the loop below)."""
    device = coords.device
    n = coords.shape[0]
    wx, wy, wz = window_shape
    window_bucket = torch.stack([
        coords[:, 0], coords[:, 1] // wx, coords[:, 2] // wy, coords[:, 3] // wz,
    ], dim=1)
    _, window_id = torch.unique(window_bucket, dim=0, return_inverse=True)

    # sort by (x,y,z) or (y,x,z) priority, THEN stable-sort by window_id -- the
    # second stable sort preserves the first sort's relative order within each
    # window, giving exactly "sorted by axis-major order within each window"
    # without ever combining window_id and coordinates into one arithmetic key
    # (which would risk integer overflow for no benefit).
    x, y, z = coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()
    secondary_key = (x * _KEY_SCALE + y) * _KEY_SCALE + z if x_major else (y * _KEY_SCALE + x) * _KEY_SCALE + z
    perm1 = torch.argsort(secondary_key, stable=True)
    perm2 = torch.argsort(window_id[perm1], stable=True)
    sort_perm = perm1[perm2]  # (N,) -- sorted_voxel[i] = coords[sort_perm[i]]

    sorted_window_id = window_id[sort_perm]
    _, counts = torch.unique_consecutive(sorted_window_id, return_counts=True)
    num_windows = counts.shape[0]
    offsets = torch.cumsum(counts, dim=0) - counts  # each window's start position in the sorted array

    num_sets_per_window = (counts + tau - 1) // tau           # ceil(N_w / tau)
    total_slots_per_window = num_sets_per_window * tau

    total_slots = int(total_slots_per_window.sum().item())
    slot_offsets = torch.cumsum(total_slots_per_window, dim=0) - total_slots_per_window
    window_of_slot = torch.repeat_interleave(torch.arange(num_windows, device=device), total_slots_per_window)
    local_slot = torch.arange(total_slots, device=device) - slot_offsets[window_of_slot]

    n_w = counts[window_of_slot].float()
    t_w = total_slots_per_window[window_of_slot].float()
    local_src = torch.floor(local_slot.float() / t_w * n_w).long()  # paper's Eq.: q_k^j formula
    global_src_in_sorted = offsets[window_of_slot] + local_src
    original_voxel_index = sort_perm[global_src_in_sorted]

    assert total_slots % tau == 0  # every window's slot block is itself a multiple of tau by construction
    return original_voxel_index.reshape(-1, tau)


class DSVTLayer(nn.Module):
    """One "set attention" layer: dynamic_set_partition -> batched MHSA (every
    set has exactly tau voxels, so this is one ordinary batched attention call,
    no padding/masking) -> post-norm residual -> 2-layer GELU MLP -> post-norm
    residual (post-norm + this MHSA->MLP order matches the paper's own
    description, not this codebase's usual pre-norm convention elsewhere)."""

    def __init__(self, channels, num_heads, tau, x_major, ffn_ratio=4, dropout=0.0):
        super().__init__()
        self.tau = tau
        self.x_major = x_major
        self.self_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * ffn_ratio), nn.GELU(), nn.Linear(channels * ffn_ratio, channels),
        )

    def forward(self, features, coords, window_shape, pc_range, effective_voxel_size):
        n, channels = features.shape
        if n == 0:
            return features
        set_idx = dynamic_set_partition(coords, window_shape, self.tau, self.x_major)  # (S,tau)

        set_feat = features[set_idx]                                                   # (S,tau,C)
        set_coords = coords[set_idx.reshape(-1)]                                        # (S*tau,4)
        pe = token_positional_embedding(set_coords, channels, pc_range, effective_voxel_size).reshape(set_idx.shape[0], self.tau, channels)

        qk = set_feat + pe
        attn_out, _ = self.self_attn(qk, qk, set_feat)
        set_feat = self.norm1(set_feat + attn_out)
        set_feat = self.norm2(set_feat + self.mlp(set_feat))

        # scatter back to per-voxel features; a voxel that landed in more than
        # one set (the "duplicated to fill the last set" case) gets the mean
        # of its occurrences' outputs -- a reasonable, order-independent
        # resolution the paper's own text doesn't fully spell out.
        return scatter_mean(set_feat.reshape(-1, channels), set_idx.reshape(-1), n)


class DSVTBlock(nn.Module):
    """[X-axis layer] -> [Y-axis layer] ("rotated sets"), same window_shape
    for both -- the window size itself only changes BETWEEN blocks, see
    DSVTBackbone's hybrid window alternation."""

    def __init__(self, channels, num_heads, tau, ffn_ratio=4):
        super().__init__()
        self.layer_x = DSVTLayer(channels, num_heads, tau, x_major=True, ffn_ratio=ffn_ratio)
        self.layer_y = DSVTLayer(channels, num_heads, tau, x_major=False, ffn_ratio=ffn_ratio)

    def forward(self, features, coords, window_shape, pc_range, effective_voxel_size):
        features = self.layer_x(features, coords, window_shape, pc_range, effective_voxel_size)
        features = self.layer_y(features, coords, window_shape, pc_range, effective_voxel_size)
        return features


class DSVTBackbone(nn.Module):
    def __init__(self, channels, window_shapes, tau, num_blocks, num_heads, ffn_ratio=4):
        super().__init__()
        assert len(window_shapes) >= 1
        self.window_shapes = [tuple(w) for w in window_shapes]
        self.blocks = nn.ModuleList([DSVTBlock(channels, num_heads, tau, ffn_ratio) for _ in range(num_blocks)])

    def forward(self, features, coords, pc_range, effective_voxel_size):
        for i, block in enumerate(self.blocks):
            window_shape = self.window_shapes[i % len(self.window_shapes)]  # hybrid window alternation
            features = block(features, coords, window_shape, pc_range, effective_voxel_size)
        return features
