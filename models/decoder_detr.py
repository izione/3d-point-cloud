"""DETR-style decoder over sparse voxel tokens (SparseVoxFormer's core idea:
feed sparse tokens straight into a transformer decoder instead of first
collapsing them into a dense BEV map). The sparse backbone + SlotFormer
refinement that produce the tokens this decoder attends to are unchanged --
see models/detector_detr.py for how they're wired together (including how the
learned "matching" queries here get concatenated with the denoising queries
built by models/denoising.py before being passed to forward()).

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
    Returns (padded (B,T_max,C), key_padding_mask (B,T_max) bool, True=pad).

    A sample with ZERO tokens (e.g. a sonar frame with no in-range points --
    SonarDiverDataset/collate_fn can produce this) would otherwise get an
    all-True mask row; nn.MultiheadAttention's softmax over an all-masked row
    is NaN (confirmed: every value in that row is -inf before softmax), which
    then poisons that whole training step's loss via the batch-wide mean.
    Fixed by leaving slot 0 unmasked for any such sample -- it's still zeros
    (harmless, already-zero-initialized `padded`), just no longer *entirely*
    masked, so attention over it returns 0 instead of NaN."""
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
    mask[counts == 0, 0] = False  # avoid an all-True (empty-softmax/NaN) row for zero-token samples
    mask[sorted_idx, within] = False
    return padded, mask


def token_positional_embedding(coords: torch.Tensor, channels: int, pc_range: torch.Tensor,
                                effective_voxel_size: torch.Tensor) -> torch.Tensor:
    """coords: (N,4) [batch,x,y,z] integer voxel coords -> (N,channels) sinusoidal
    PE of the token's world-space center, summed over the 3 axes (matches
    SlotFormer's own convention -- see slotformer.py's SFLayer._positional_encoding)."""
    world = pc_range[:3].to(coords.device) + (coords[:, 1:4].float() + 0.5) * effective_voxel_size.to(coords.device)
    return ref_point_positional_embedding(world, channels)


def ref_point_positional_embedding(ref_points: torch.Tensor, channels: int) -> torch.Tensor:
    """ref_points: (...,3) world coordinates -> (...,channels) sinusoidal PE,
    summed over the 3 axes. Same primitive used for both the sparse-token keys
    (token_positional_embedding above) and any set of query reference points
    (matching queries' learned points, or denoising queries' noised GT
    centers) -- a query and a key at the same world location get the same PE,
    which is the point of encoding position this way instead of a per-slot
    learned embedding."""
    shape = ref_points.shape[:-1]
    pe = torch.zeros(*shape, channels, device=ref_points.device, dtype=torch.float32)
    for axis in range(3):
        pe = pe + sinusoidal_pe(ref_points[..., axis].reshape(-1), channels, PE_TEMPERATURE).reshape(*shape, channels)
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

    def forward(self, query, query_pos, key, key_pos, key_padding_mask, self_attn_mask=None):
        # self-attention among queries (pre-norm, residual). self_attn_mask (if
        # given) blocks matching<->denoising and cross-group denoising leakage
        # -- see models/denoising.py::build_attention_mask.
        q = self.norm1(query)
        qk = q + query_pos
        attn_out, _ = self.self_attn(qk, qk, q, attn_mask=self_attn_mask)
        query = query + attn_out

        # cross-attention: queries -> sparse tokens (every query, matching or
        # denoising, can freely see all of that sample's sparse tokens)
        q = self.norm2(query)
        attn_out, _ = self.cross_attn(q + query_pos, key + key_pos, key, key_padding_mask=key_padding_mask)
        query = query + attn_out

        query = query + self.ffn(self.norm3(query))
        return query


class DetrDecoder(nn.Module):
    def __init__(self, channels, num_queries, num_layers, num_heads, ffn_ratio=4):
        super().__init__()
        self.num_queries = num_queries
        self.channels = channels
        self.query_embed = nn.Parameter(torch.randn(num_queries, channels) * 0.02)
        # learnable initial reference point per query, in [0,1]^3 (normalized
        # over the point-cloud range) via sigmoid -- keeps every query's
        # starting guess inside the actual scene volume regardless of init scale.
        self.query_ref_raw = nn.Parameter(torch.randn(num_queries, 3) * 0.5)
        # learnable initial log-size / 6D-rotation anchor per query, added to
        # that head's raw output the same way query_ref_raw is added to the
        # center offset (see heads_detr.py::SetPredictionHead.forward()).
        # Without this, a matching query's size head had NO reference to
        # correct from -- it had to predict absolute log-size from scratch --
        # and empirically never learned anything: a real 20-epoch run's size
        # loss matched a trivial "always predict the dataset's mean size"
        # baseline the entire time. Denoising queries never had this problem
        # since their content embedding already encodes a noised size/rotation
        # to correct (models/denoising.py); this gives matching queries the
        # same kind of starting point. Small random init (not zero) so
        # different queries can specialize, same reasoning as query_embed's
        # own init. Rotation anchor starts at identity ([1,0,0,0,1,0] is the
        # sixd encoding of the identity matrix -- see rotation6d.py) plus a
        # small perturbation for the same reason.
        self.size_anchor_raw = nn.Parameter(torch.randn(num_queries, 3) * 0.1)
        self.rot_anchor_raw = nn.Parameter(
            torch.tensor([1., 0., 0., 0., 1., 0.]).expand(num_queries, 6).clone()
            + torch.randn(num_queries, 6) * 0.1
        )
        self.layers = nn.ModuleList([DetrDecoderLayer(channels, num_heads, ffn_ratio) for _ in range(num_layers)])

    def matching_reference_points(self, pc_range: torch.Tensor) -> torch.Tensor:
        """(num_queries, 3) world-space reference points for the learned queries."""
        norm = torch.sigmoid(self.query_ref_raw)
        return pc_range[:3] + norm * (pc_range[3:] - pc_range[:3])

    def matching_log_size_anchor(self) -> torch.Tensor:
        """(num_queries, 3) learned log-size anchor for the learned queries."""
        return self.size_anchor_raw

    def matching_rotation_anchor(self) -> torch.Tensor:
        """(num_queries, 6) learned 6D-rotation anchor for the learned queries."""
        return self.rot_anchor_raw

    def matching_content(self, batch_size: int) -> torch.Tensor:
        """(B, num_queries, C) learned content embedding, expanded per sample."""
        return self.query_embed[None, :, :].expand(batch_size, -1, -1).clone()

    def forward(self, query_content, query_pos, key, key_pos, key_padding_mask, self_attn_mask=None,
                return_intermediate=False):
        """query_content/query_pos: (B,Q,C) -- already-built queries (matching
        only, or matching+denoising concatenated; see models/detector_detr.py).
        key/key_pos: (B,T_max,C). Returns query_feat (B,Q,C) by default.

        return_intermediate=True additionally returns every layer's output
        (including the last) as a list -- used for DETR's per-layer auxiliary
        loss: supervising only the final layer leaves early layers/backbone
        getting gradient through the full decoder stack, which in practice
        stalled box regression (center/size loss flat for 10+ epochs) while
        classification and rotation still moved -- see
        models/detector_detr.py::loss()."""
        query = query_content
        intermediate = []
        for layer in self.layers:
            query = layer(query, query_pos, key, key_pos, key_padding_mask, self_attn_mask)
            if return_intermediate:
                intermediate.append(query)
        if return_intermediate:
            return query, intermediate
        return query
