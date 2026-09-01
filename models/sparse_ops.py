"""Minimal sparse 3D conv primitives built entirely on core PyTorch.

No spconv / torch_scatter dependency: both need a CUDA-matched compiled
extension and are a common source of broken Colab setups. Everything here
uses only torch.scatter_reduce_ / index_add_ / advanced indexing, so it runs
identically on CPU (for local testing) and CUDA (Colab), with no extra installs.

A "sparse tensor" here is just:
  features: (N, C) float
  coords:   (N, 4) long, columns = [batch_idx, x_idx, y_idx, z_idx]
  index_grid: (B, X, Y, Z) long, dense lookup -> row index into features, or -1
"""
import torch
import torch.nn as nn


def build_index_grid(coords: torch.Tensor, batch_size: int, grid_size, device=None) -> torch.Tensor:
    X, Y, Z = grid_size
    device = device or coords.device
    grid = torch.full((batch_size, X, Y, Z), -1, dtype=torch.long, device=device)
    if coords.shape[0] > 0:
        grid[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = torch.arange(
            coords.shape[0], device=device
        )
    return grid


# Two implementations, same signature/output: vectorized batches all k^3
# kernel offsets into one gather+einsum (fewer, bigger GPU kernel launches --
# helps when launch overhead dominates, i.e. on GPU); loop processes one
# offset at a time (smaller peak memory, no k^3 blowup -- measured faster on
# CPU, where there's no launch-overhead penalty to amortize away).
CONV_MODE = "vectorized"  # "vectorized" or "loop" -- measured on a real GPU (RTX 2070, via
                          # benchmark_backbone.py on the actual sonar dataset): "loop" took
                          # ~1880ms/frame for this backbone, "vectorized" ~266ms/frame (7x) --
                          # launch-overhead dominates on GPU as expected, so "vectorized" is now
                          # the default. "loop" is still what's faster on CPU-only setups.
                          # (SparseConv3dDown's output-coord candidate search had its own,
                          # separate k^3 Python loop that CONV_MODE never touched -- vectorizing
                          # that too, see SparseConv3dDown.forward below, took the same benchmark
                          # down to ~44ms/frame, on par with an equivalent dense nn.Conv3d backbone
                          # (~48ms/frame) despite <0.1% voxel occupancy.)


def _sparse_conv_core(in_features, in_coords, in_index_grid, in_grid_size, out_coords,
                       weight, bias, stride, padding):
    if CONV_MODE == "loop":
        return _sparse_conv_core_loop(in_features, in_coords, in_index_grid, in_grid_size, out_coords, weight, bias, stride, padding)
    return _sparse_conv_core_vectorized(in_features, in_coords, in_index_grid, in_grid_size, out_coords, weight, bias, stride, padding)


def _sparse_conv_core_loop(in_features, in_coords, in_index_grid, in_grid_size, out_coords,
                            weight, bias, stride, padding):
    k = weight.shape[0]
    Nout = out_coords.shape[0]
    Cout = weight.shape[-2]
    out = torch.zeros(Nout, Cout, device=in_features.device, dtype=in_features.dtype)
    X, Y, Z = in_grid_size
    for kx in range(k):
        for ky in range(k):
            for kz in range(k):
                ix = out_coords[:, 1] * stride + kx - padding
                iy = out_coords[:, 2] * stride + ky - padding
                iz = out_coords[:, 3] * stride + kz - padding
                in_bounds = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
                if not in_bounds.any():
                    continue
                b = out_coords[in_bounds, 0]
                row_idx = in_index_grid[b, ix[in_bounds], iy[in_bounds], iz[in_bounds]]
                has_nb = row_idx >= 0
                if not has_nb.any():
                    continue
                dest = in_bounds.clone()
                dest[in_bounds] = has_nb
                gathered = in_features[row_idx[has_nb]]
                w = weight[kx, ky, kz]
                out[dest] = out[dest] + gathered @ w.t()
    if bias is not None:
        out = out + bias
    return out


def _sparse_conv_core_vectorized(in_features, in_coords, in_index_grid, in_grid_size, out_coords,
                                  weight, bias, stride, padding):
    """Vectorized over all k^3 kernel offsets at once (one gather + one einsum)
    instead of a Python loop -- far fewer, larger GPU kernel launches, which
    matters a lot more than raw FLOPs here since every offset's per-op cost
    is tiny relative to launch overhead."""
    k = weight.shape[0]
    device = in_features.device
    Nout = out_coords.shape[0]
    Cout, Cin = weight.shape[-2], weight.shape[-1]
    if Nout == 0:
        return torch.zeros(0, Cout, device=device, dtype=in_features.dtype)

    ar = torch.arange(k, device=device)
    offsets = torch.stack(torch.meshgrid(ar, ar, ar, indexing="ij"), dim=-1).reshape(-1, 3)  # (K,3)
    K = offsets.shape[0]
    X, Y, Z = in_grid_size

    ix = out_coords[None, :, 1] * stride + offsets[:, 0:1] - padding  # (K,Nout)
    iy = out_coords[None, :, 2] * stride + offsets[:, 1:2] - padding
    iz = out_coords[None, :, 3] * stride + offsets[:, 2:3] - padding
    ib = out_coords[None, :, 0].expand(K, Nout)

    in_bounds = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
    row_idx = in_index_grid[ib, ix.clamp(0, X - 1), iy.clamp(0, Y - 1), iz.clamp(0, Z - 1)]  # (K,Nout)
    valid = in_bounds & (row_idx >= 0)
    safe_row_idx = row_idx.clamp(min=0)

    gathered = in_features[safe_row_idx.reshape(-1)].reshape(K, Nout, Cin)
    gathered = gathered * valid.unsqueeze(-1).to(gathered.dtype)

    w = weight.reshape(K, Cout, Cin)
    out = torch.einsum("kod,knd->no", w, gathered)
    if bias is not None:
        out = out + bias
    return out


class SubMConv3d(nn.Module):
    """Submanifold conv: output lives on the exact same active coords as input."""

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        k = kernel_size
        self.weight = nn.Parameter(torch.empty(k, k, k, out_channels, in_channels))
        nn.init.kaiming_uniform_(self.weight.reshape(-1, in_channels), a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.k = k
        self.padding = k // 2

    def forward(self, features, coords, index_grid, grid_size):
        out = _sparse_conv_core(
            features, coords, index_grid, grid_size, coords,
            self.weight, self.bias, stride=1, padding=self.padding,
        )
        return out, coords, index_grid  # active set unchanged


class SparseConv3dDown(nn.Module):
    """Strided conv that can shrink the active set (used for the stem downsample)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True):
        super().__init__()
        k = kernel_size
        self.weight = nn.Parameter(torch.empty(k, k, k, out_channels, in_channels))
        nn.init.kaiming_uniform_(self.weight.reshape(-1, in_channels), a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.k = k
        self.stride = stride
        self.padding = padding

    @staticmethod
    def output_grid_size(grid_size, kernel_size, stride, padding):
        return tuple((g + 2 * padding - kernel_size) // stride + 1 for g in grid_size)

    def forward(self, features, coords, index_grid, grid_size, batch_size):
        out_grid_size = self.output_grid_size(grid_size, self.k, self.stride, self.padding)
        device = features.device
        # candidate output coords: invert in_coord = out*stride + koff - padding for every
        # active input voxel and every kernel offset, keep integer/in-range results. Batches
        # all k^3 offsets into one vectorized pass instead of a 27-iteration Python loop --
        # on GPU that loop was k^3 sequential small launches *every* downsample stage, which
        # (per benchmark_backbone.py against the real dataset on an RTX 2070) was a big chunk
        # of why the sparse backbone was slower than an equivalent dense nn.Conv3d one despite
        # <0.1% voxel occupancy; this arithmetic is cheap enough that batching it costs nothing
        # extra on CPU either.
        k = self.k
        if coords.shape[0] == 0:
            out_coords = torch.zeros((0, 4), dtype=torch.long, device=device)
        else:
            ar = torch.arange(k, device=device)
            offsets = torch.stack(torch.meshgrid(ar, ar, ar, indexing="ij"), dim=-1).reshape(-1, 3)  # (K,3)

            numer_x = coords[None, :, 1] - offsets[:, 0:1] + self.padding  # (K,N)
            numer_y = coords[None, :, 2] - offsets[:, 1:2] + self.padding
            numer_z = coords[None, :, 3] - offsets[:, 2:3] + self.padding
            div = (numer_x % self.stride == 0) & (numer_y % self.stride == 0) & (numer_z % self.stride == 0)

            ox = numer_x // self.stride
            oy = numer_y // self.stride
            oz = numer_z // self.stride
            ob = coords[None, :, 0].expand_as(ox)

            in_range = (
                (ox >= 0) & (ox < out_grid_size[0]) &
                (oy >= 0) & (oy < out_grid_size[1]) &
                (oz >= 0) & (oz < out_grid_size[2])
            )
            valid = (div & in_range).reshape(-1)

            cand = torch.stack([ob, ox, oy, oz], dim=-1).reshape(-1, 4)[valid]
            out_coords = torch.zeros((0, 4), dtype=torch.long, device=device) if cand.shape[0] == 0 \
                else torch.unique(cand, dim=0)

        out_features = _sparse_conv_core(
            features, coords, index_grid, grid_size, out_coords,
            self.weight, self.bias, stride=self.stride, padding=self.padding,
        )
        out_index_grid = build_index_grid(out_coords, batch_size, out_grid_size, device=device)
        return out_features, out_coords, out_index_grid, out_grid_size


class SparseBasicBlock(nn.Module):
    """Two SubMConv3d + BN + ReLU with a residual connection (ResNet-style)."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = SubMConv3d(channels, channels, kernel_size, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = SubMConv3d(channels, channels, kernel_size, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features, coords, index_grid, grid_size):
        identity = features
        out, _, _ = self.conv1(features, coords, index_grid, grid_size)
        out = self.relu(self.bn1(out))
        out, _, _ = self.conv2(out, coords, index_grid, grid_size)
        out = self.bn2(out)
        out = self.relu(out + identity)
        return out, coords, index_grid


# ---- scatter helpers (replace what torch_scatter would provide) ----

def scatter_sum(src, index, dim_size):
    shape = (dim_size,) + src.shape[1:]
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


def scatter_mean(src, index, dim_size):
    s = scatter_sum(src, index, dim_size)
    ones = torch.ones(src.shape[0], device=src.device, dtype=src.dtype)
    cnt = scatter_sum(ones, index, dim_size).clamp(min=1)
    return s / cnt.view(-1, *([1] * (s.dim() - 1)))


def scatter_max(src, index, dim_size):
    shape = (dim_size,) + src.shape[1:]
    init = torch.full(shape, float("-inf"), dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    # out-of-place scatter_reduce (no trailing "_"): scatter_max is called repeatedly
    # inside the VFE's PFN stack, and mutating a tensor in place after it already has
    # autograd history attached corrupts the saved state scatter_reduce's backward needs.
    out = init.scatter_reduce(0, idx, src, reduce="amax", include_self=True)
    out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    return out


def scatter_softmax(logits, index, dim_size):
    """Softmax over groups defined by `index`, along dim 0. logits: (N,) or (N,H)."""
    max_per_group = scatter_max(logits, index, dim_size)
    shifted = logits - max_per_group[index]
    exp = shifted.exp()
    denom = scatter_sum(exp, index, dim_size).clamp(min=1e-12)
    return exp / denom[index]
