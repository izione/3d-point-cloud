"""6D continuous rotation representation (Zhou et al., "On the Continuity of
Rotation Representations in Neural Networks", CVPR 2019), used by the DETR-style
head (models/heads_detr.py) instead of this repo's existing quaternion
convention (models/box_utils.py). Ported from
voxelnet_baseline/model/rotation3d.py rather than imported cross-repo, so this
module has no dependency outside 3d-point-cloud.

Why not quaternion here: a quaternion has a q = -q double-cover ambiguity that
needs a special sign-invariant geodesic loss (see box_utils.quat_geodesic_loss)
to train correctly. 6D regresses the rotation matrix's first two columns
directly and reconstructs an orthonormal frame via Gram-Schmidt -- no
discontinuity, so training is a bit more forgiving. GT boxes stay quaternion
in data/dataset.py; conversion to a rotation matrix (box_utils.quat_to_rotmat)
happens only where a 6D prediction needs to be compared against one.
"""
import torch

# Same fixed local corner-sign convention as box_utils.py's _SIGNS, duplicated
# rather than imported so this module doesn't reach into box_utils' private name.
_CORNER_SIGNS = torch.tensor(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=torch.float32,
)


def sixd_to_matrix(ortho6d: torch.Tensor) -> torch.Tensor:
    """(...,6) -> (...,3,3) via Gram-Schmidt orthogonalization of the first two
    (unnormalized) columns; the third is their cross product."""
    a1, a2 = ortho6d[..., 0:3], ortho6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    a2_proj = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2_proj, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns are the local axes


def matrix_to_sixd(R: torch.Tensor) -> torch.Tensor:
    """(...,3,3) -> (...,6): concatenate the first two columns."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def corners_from_matrix(center: torch.Tensor, size: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """center:(N,3), size:(N,3) [length,width,height], R:(N,3,3) -> corners (N,8,3)."""
    signs = _CORNER_SIGNS.to(center.device)
    local = signs[None, :, :] * (size[:, None, :] / 2)
    return torch.einsum("nij,nkj->nki", R, local) + center[:, None, :]


def matrix_geodesic_loss(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Angular distance in [0, pi] between two rotation matrices, per row.
    R1/R2: (...,3,3). trace(R1^T R2) = 1 + 2*cos(theta)."""
    rel = torch.einsum("...ij,...ik->...jk", R1, R2)  # R1^T @ R2
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = ((trace - 1) / 2).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_theta)
