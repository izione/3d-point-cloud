"""sparse_conv_pure.py - pure-PyTorch sparse 3D convolution, no spconv/cumm or
any other compiled-extension dependency.

Why this exists: CenterPoint's paper-faithful backbone needs sparse 3D conv
(see model.py), but both spconv wheel sources hit hard failures in real
testing (2026-09):
  - official PyPI spconv-cu126/cumm-cu126: cumm's own bundled tensorview C++
    headers are missing from the published wheel -- confirmed via the actual
    compiler error ("fatal error: tensorview/pybind_utils.h: No such file or
    directory"), a known open upstream packaging bug (traveller59/spconv#766,
    FindDefinition/cumm#7).
  - the rathaROG community index (built specifically to work around exactly
    this): only publishes cu128/cu130 tags, not cu126, so pip silently fell
    back to the same broken official cu126 wheel anyway.
Rather than keep chasing wheel/CUDA-version combinations outside our control,
this implements the sparse conv operator directly in plain PyTorch. Same
precedent as the sibling `slotformer` branch's own models/sparse_ops.py
("every sparse op has a fallback implemented in plain PyTorch... so the same
code runs... with or without spconv installed") -- here it's just the ONLY
implementation, not a fallback, since a packaging bug in a specific CUDA tag
is exactly the class of problem a pure-PyTorch implementation sidesteps
entirely.

Correctness over speed: our per-frame active-voxel counts are in the
thousands, not millions, so an unfold/unique/index_add_-based approach is
plenty fast for this dataset's training loop; see
test_sparse_conv_pure.py for a from-scratch-vs-dense-nn.Conv3d equivalence
check (the actual correctness proof for this file).

API deliberately mirrors just the slice of spconv.pytorch that model.py
needs (SparseConvTensor, SparseConv3d) so model.py's import line is the only
thing that changes to switch between them.
"""
import torch
import torch.nn as nn


class SparseConvTensor:
    """features: (N,C). indices: (N,4) int64 [batch,z,y,x]. spatial_shape: (D,H,W)."""

    def __init__(self, features: torch.Tensor, indices: torch.Tensor, spatial_shape, batch_size: int):
        self.features = features
        self.indices = indices
        self.spatial_shape = tuple(int(s) for s in spatial_shape)
        self.batch_size = int(batch_size)

    def replace_feature(self, features: torch.Tensor) -> "SparseConvTensor":
        return SparseConvTensor(features, self.indices, self.spatial_shape, self.batch_size)

    def dense(self) -> torch.Tensor:
        D, H, W = self.spatial_shape
        C = self.features.shape[1]
        out = self.features.new_zeros(self.batch_size, C, D, H, W)
        if self.indices.shape[0]:
            b, z, y, x = self.indices.unbind(dim=1)
            out[b, :, z, y, x] = self.features
        return out


class SparseConv3d(nn.Module):
    """Regular (non-submanifold) sparse 3D conv, kernel_size fixed to 3 (the
    only size this project needs), arbitrary per-axis stride/padding,
    bias-free. Output coordinates are derived from valid strided
    reachability from the input's active set -- exactly spconv's own
    SparseConv3d semantics (the active set can grow/shrink/shift, unlike a
    submanifold conv). A stride=1 axis that should NOT grow its active set
    (this project's x,y axes) still needs restrict_xy_support applied
    afterward, same as with real spconv -- this class does not do that
    itself.

    Implementation: for each of the up to 27 kernel taps (kz,ky,kx in
    0..2), vectorized over all active input rows at once: compute each
    row's candidate output coordinate via the standard conv index relation
    `in = out*stride - padding + k` inverted, keep only rows landing on an
    exact (integer, in-bounds) output cell, matmul the input feature by
    that tap's (out,in) weight slice, then torch.unique the encoded output
    coordinates once across ALL taps to merge every tap's contribution
    into its output cell via index_add_ (this is the "coalesce duplicate
    sparse-COO entries by summing" step; multiple taps/input rows landing
    on the same output cell is the normal, expected case, not a bug)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride=(1, 1, 1), padding=(1, 1, 1), bias: bool = False, indice_key: str = None):
        super().__init__()
        assert kernel_size == 3, "only kernel_size=3 implemented (all this project needs)"
        assert not bias, "bias=True not implemented (unused by this project)"
        self.stride = tuple(stride)
        self.padding = tuple(padding)
        self.out_channels = out_channels
        # weight[kz,ky,kx]: (out,in) -- one dense matmul weight per kernel tap
        self.weight = nn.Parameter(torch.empty(3, 3, 3, out_channels, in_channels))
        nn.init.kaiming_uniform_(self.weight.reshape(-1, in_channels), a=5 ** 0.5)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        coords, feats = x.indices, x.features
        D_in, H_in, W_in = x.spatial_shape
        sz, sy, sx = self.stride
        pz, py, px = self.padding
        D_out = (D_in + 2 * pz - 3) // sz + 1
        H_out = (H_in + 2 * py - 3) // sy + 1
        W_out = (W_in + 2 * px - 3) // sx + 1
        empty_out = SparseConvTensor(feats.new_zeros(0, self.out_channels),
                                      coords.new_zeros(0, 4), (D_out, H_out, W_out), x.batch_size)
        if coords.shape[0] == 0:
            return empty_out

        b_in, z_in, y_in, x_in = coords.unbind(dim=1)
        all_keys, all_contribs = [], []
        for kz in range(3):
            zc = z_in + pz - kz
            z_out = torch.div(zc, sz, rounding_mode="floor")
            z_ok = (zc - z_out * sz == 0) & (z_out >= 0) & (z_out < D_out)
            for ky in range(3):
                yc = y_in + py - ky
                y_out = torch.div(yc, sy, rounding_mode="floor")
                y_ok = (yc - y_out * sy == 0) & (y_out >= 0) & (y_out < H_out)
                for kx in range(3):
                    xc = x_in + px - kx
                    x_out = torch.div(xc, sx, rounding_mode="floor")
                    x_ok = (xc - x_out * sx == 0) & (x_out >= 0) & (x_out < W_out)

                    valid = z_ok & y_ok & x_ok
                    if not bool(valid.any()):
                        continue
                    idx = valid.nonzero(as_tuple=True)[0]
                    contrib = feats[idx] @ self.weight[kz, ky, kx].t()
                    key = ((b_in[idx] * D_out + z_out[idx]) * H_out + y_out[idx]) * W_out + x_out[idx]
                    all_keys.append(key)
                    all_contribs.append(contrib)

        if not all_keys:
            return empty_out

        keys_cat = torch.cat(all_keys)
        contrib_cat = torch.cat(all_contribs, dim=0)
        uniq_keys, inverse = torch.unique(keys_cat, return_inverse=True)
        out_feats = contrib_cat.new_zeros(uniq_keys.shape[0], self.out_channels)
        out_feats.index_add_(0, inverse, contrib_cat)

        rem = uniq_keys
        x_out_u = rem % W_out
        rem = torch.div(rem, W_out, rounding_mode="floor")
        y_out_u = rem % H_out
        rem = torch.div(rem, H_out, rounding_mode="floor")
        z_out_u = rem % D_out
        b_out_u = torch.div(rem, D_out, rounding_mode="floor")
        out_coords = torch.stack([b_out_u, z_out_u, y_out_u, x_out_u], dim=1)

        return SparseConvTensor(out_feats, out_coords, (D_out, H_out, W_out), x.batch_size)
