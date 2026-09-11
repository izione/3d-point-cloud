"""sparse_ops.py - small sparse-tensor utility shared by model.py's sparse
middle encoder. Ported from the sibling voxelnet_baseline repo's own
sparse_ops.py (only the one function this package actually needs).
"""
import torch


def yx_key(coords: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Encode a sparse tensor's (batch,y,x) columns (indices [0,2,3] in this
    package's [batch,z,y,x] coords convention) into one integer per row, for
    fast set-membership tests via torch.isin -- used by restrict_xy_support."""
    b, y, x = coords[:, 0].long(), coords[:, 2].long(), coords[:, 3].long()
    return (b * H + y) * W + x


def restrict_xy_support(x, allowed_keys: torch.Tensor, H: int, W: int):
    """Drop every row of sparse tensor `x` (a spconv.SparseConvTensor) whose
    (batch,y,x) isn't in allowed_keys.

    Needed because spconv's SparseConv3d (unlike SubMConv3d) doesn't restrict
    a stride==1 axis to its input support the way a true submanifold conv
    would: for a z-only-stride conv (stride=(2,1,1), x,y meant to stay
    untouched), spconv still discovers every neighbor-reachable (y,x) the k^3
    kernel touches, "dilating" the active set outward in x,y at every stage
    even though nothing is being downsampled there. Call with `allowed_keys`
    computed ONCE from the voxel grid's original (pre-any-z-down-stage)
    active set (x,y never change across these stages), after every
    z-only-stride stage."""
    keys = yx_key(x.indices, H, W)
    mask = torch.isin(keys, allowed_keys)
    return x.__class__(x.features[mask], x.indices[mask], x.spatial_shape, x.batch_size)
