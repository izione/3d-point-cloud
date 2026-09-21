"""DETR-style decoder over sparse voxel tokens (SparseVoxFormer's core idea:
feed sparse tokens straight into a transformer decoder instead of first
collapsing them into a dense BEV map). The sparse backbone + SlotFormer
refinement that produce the tokens this decoder attends to are unchanged --
see models/detector_detr.py for how they're wired together.

Per-sample token counts vary (different scenes have different numbers of
active voxels), so cross-attention needs a padded (B, T_max, C) key/value
layout with a padding mask -- built once here rather than per-layer.
"""
import torch
import torch.nn as nn

from .slotformer import sinusoidal_pe

PE_TEMPERATURE = 10000


def pad_tokens(features: torch.Tensor, batch_idx: torch.Tensor, batch_size: int):
    """features: (N,C), batch_idx: (N,) which sample each token belongs to.
    Returns (padded (B,T_max,C), key_padding_mask (B,T_max) bool, True=pad)."""
    device = features.device
    counts = torch.bincount(batch_idx, minlength=batch_size)
    t_max = max(int(counts.max().item()), 1)
    order = torch.argsort(batch_idx)
    sorted_idx = batch_idx[order]
    offsets = torch.cumsum(counts, dim=0) - counts
    within = torch.arange(order.shape[0], device=device) - offsets[sorted_idx]

    padded = torch.zeros(batch_size, t_max, features.shape[1], device=device, dtype=features.dtype)
    mask = torch.ones(batch_size, t_max, dtype=torch.bool, device=device)  # True = padding
    padded[sorted_idx, within] = features[order]
    mask[sorted_idx, within] = False
    return padded, mask


def token_positional_embedding(coords: torch.Tensor, channels: int, pc_range: torch.Tensor,
                                effective_voxel_size: torch.Tensor) -> torch.Tensor:
    """coords: (N,4) [batch,x,y,z] integer voxel coords -> (N,channels) sinusoidal
    PE of the token's world-space center, summed over the 3 axes (matches
    SlotFormer's own convention -- see slotformer.py's SFLayer._positional_encoding)."""
    world = pc_range[:3].to(coords.device) + (coords[:, 1:4].float() + 0.5) * effective_voxel_size.to(coords.device)
    pe = torch.zeros(coords.shape[0], channels, device=coords.device, dtype=torch.float32)
    for axis in range(3):
        pe = pe + sinusoidal_pe(world[:, axis], channels, PE_TEMPERATURE)
    return pe


class DetrDecoderLayer(nn.Module):
    def __init__(self, channels, num_heads, ffn_ratio=4, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * ffn_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels * ffn_ratio, channels),
        )

    def forward(self, query, query_pos, key, key_pos, key_padding_mask):
        # self-attention among queries (pre-norm, residual)
        q = self.norm1(query)
        qk = q + query_pos
        attn_out, _ = self.self_attn(qk, qk, q)
        query = query + attn_out

        # cross-attention: queries -> sparse tokens
        q = self.norm2(query)
        attn_out, _ = self.cross_attn(q + query_pos, key + key_pos, key, key_padding_mask=key_padding_mask)
        query = query + attn_out

        query = query + self.ffn(self.norm3(query))
        return query


class DetrDecoder(nn.Module):
    def __init__(self, channels, num_queries, num_layers, num_heads, ffn_ratio=4):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, channels) * 0.02)
        # learnable initial reference point per query, in [0,1]^3 (normalized
        # over the point-cloud range) via sigmoid -- keeps every query's
        # starting guess inside the actual scene volume regardless of init scale.
        self.query_ref_raw = nn.Parameter(torch.randn(num_queries, 3) * 0.5)
        self.layers = nn.ModuleList([DetrDecoderLayer(channels, num_heads, ffn_ratio) for _ in range(num_layers)])

    def reference_points(self, pc_range: torch.Tensor) -> torch.Tensor:
        """(num_queries, 3) world-space reference points."""
        norm = torch.sigmoid(self.query_ref_raw)
        return pc_range[:3] + norm * (pc_range[3:] - pc_range[:3])

    def forward(self, key, key_pos, key_padding_mask, pc_range: torch.Tensor, batch_size: int):
        """key/key_pos: (B,T_max,C). Returns (query_feat (B,Q,C), ref_points (B,Q,3))."""
        channels = self.query_embed.shape[1]
        query = self.query_embed[None, :, :].expand(batch_size, -1, -1).clone()

        ref_points = self.reference_points(pc_range)  # (Q,3), shared init across the batch
        query_pos = torch.zeros(batch_size, self.num_queries, channels, device=key.device)
        for axis in range(3):
            query_pos = query_pos + sinusoidal_pe(ref_points[:, axis], channels, PE_TEMPERATURE)[None, :, :]

        for layer in self.layers:
            query = layer(query, query_pos, key, key_pos, key_padding_mask)

        return query, ref_points[None, :, :].expand(batch_size, -1, -1)
