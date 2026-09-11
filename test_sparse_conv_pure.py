"""test_sparse_conv_pure.py - correctness proof for sparse_conv_pure.SparseConv3d:
when EVERY cell of a dense volume is marked active, sparse conv with the same
weights/stride/padding must produce EXACTLY the same output as nn.Conv3d
(zero-padding at the boundary is the same either way -- a coordinate that
falls outside the grid simply has no row in the active set, contributing 0,
which is exactly nn.Conv3d's own zero-padding). Also checks a partially-active
(genuinely sparse) input against a dense reference computed by zeroing the
inactive cells first (still must match at active OUTPUT cells the sparse path
actually returns, since output values there only depend on which INPUT cells
are truly active, not on which output cells get computed).
"""
import torch
import torch.nn as nn

from sparse_conv_pure import SparseConv3d, SparseConvTensor


def dense_to_sparse(dense: torch.Tensor, active_mask: torch.Tensor):
    """dense: (B,C,D,H,W). active_mask: (B,D,H,W) bool. -> (coords(N,4), feats(N,C))."""
    b, z, y, x = active_mask.nonzero(as_tuple=True)
    feats = dense[b, :, z, y, x]
    coords = torch.stack([b, z, y, x], dim=1)
    return coords, feats


def check_case(name, B, C_in, C_out, D, H, W, stride, padding, active_frac):
    torch.manual_seed(0)
    dense_in = torch.randn(B, C_in, D, H, W)
    active_mask = torch.rand(B, D, H, W) < active_frac
    coords, feats = dense_to_sparse(dense_in, active_mask)

    sconv = SparseConv3d(C_in, C_out, 3, stride=stride, padding=padding)
    with torch.no_grad():
        dconv = nn.Conv3d(C_in, C_out, 3, stride=stride, padding=padding, bias=False)
        # dconv.weight: (out,in,kD,kH,kW). sconv.weight: (kD,kH,kW,out,in) -- same values, different layout.
        dconv.weight.copy_(sconv.weight.detach().permute(3, 4, 0, 1, 2))

    dense_in_masked = dense_in * active_mask.unsqueeze(1)
    with torch.no_grad():
        dense_out = dconv(dense_in_masked)  # (B,C_out,D',H',W')
        sparse_out = sconv(SparseConvTensor(feats, coords, (D, H, W), B))

    sparse_dense = sparse_out.dense()
    if sparse_dense.shape != dense_out.shape:
        print(f"[{name}] FAIL shape mismatch: sparse={tuple(sparse_dense.shape)} dense={tuple(dense_out.shape)}")
        return False

    # sparse_out only has rows for output cells reachable from an ACTIVE input cell;
    # everywhere else sparse_dense is exactly 0 by construction (dense() zero-fills).
    # dense_out is nonzero only where at least one active input cell falls in that
    # cell's receptive field too (since inactive cells were zeroed before conv) --
    # so the two should be equal everywhere, not just at "active" output cells.
    ok = torch.allclose(sparse_dense, dense_out, atol=1e-5, rtol=1e-4)
    max_err = (sparse_dense - dense_out).abs().max().item()
    print(f"[{name}] {'OK' if ok else 'FAIL'}  max_abs_err={max_err:.2e}  "
          f"active_frac={active_frac}  n_active_in={coords.shape[0]}  n_active_out={sparse_out.indices.shape[0]}")
    return ok


def main():
    results = []
    # fully dense (active_frac=1.0) -- must match nn.Conv3d exactly, this is the
    # core correctness proof
    results.append(check_case("full-dense, conv1 shape (stride 2,1,1 pad 1,1,1)",
                               B=2, C_in=8, C_out=6, D=10, H=6, W=7,
                               stride=(2, 1, 1), padding=(1, 1, 1), active_frac=1.0))
    results.append(check_case("full-dense, conv2 shape (stride 1,1,1 pad 0,1,1)",
                               B=2, C_in=6, C_out=6, D=5, H=6, W=7,
                               stride=(1, 1, 1), padding=(0, 1, 1), active_frac=1.0))
    results.append(check_case("full-dense, conv3 shape (stride 2,1,1 pad 1,1,1)",
                               B=2, C_in=6, C_out=6, D=3, H=6, W=7,
                               stride=(2, 1, 1), padding=(1, 1, 1), active_frac=1.0))
    results.append(check_case("full-dense, stride (1,1,1) pad (1,1,1) all-preserve",
                               B=1, C_in=4, C_out=5, D=6, H=6, W=6,
                               stride=(1, 1, 1), padding=(1, 1, 1), active_frac=1.0))
    # genuinely sparse (partial active) -- correctness still holds since the dense
    # reference is computed on the SAME zeroed-out input, not a fully dense one
    results.append(check_case("sparse 30% active, conv1 shape",
                               B=2, C_in=8, C_out=6, D=10, H=6, W=7,
                               stride=(2, 1, 1), padding=(1, 1, 1), active_frac=0.3))
    results.append(check_case("sparse 5% active, conv1 shape",
                               B=3, C_in=8, C_out=6, D=10, H=8, W=9,
                               stride=(2, 1, 1), padding=(1, 1, 1), active_frac=0.05))
    results.append(check_case("sparse 1% active (very sparse), conv3 shape",
                               B=2, C_in=6, C_out=6, D=3, H=20, W=20,
                               stride=(2, 1, 1), padding=(1, 1, 1), active_frac=0.01))

    # gradient flows correctly
    torch.manual_seed(0)
    coords, feats = dense_to_sparse(torch.randn(1, 4, 6, 6, 6), torch.rand(1, 6, 6, 6) < 0.5)
    feats.requires_grad_(True)
    sconv = SparseConv3d(4, 3, 3, stride=(2, 1, 1), padding=(1, 1, 1))
    out = sconv(SparseConvTensor(feats, coords, (6, 6, 6), 1))
    loss = out.features.sum()
    loss.backward()
    grad_ok = feats.grad is not None and torch.isfinite(feats.grad).all()
    print(f"[gradient check] {'OK' if grad_ok else 'FAIL'}")
    results.append(grad_ok)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
