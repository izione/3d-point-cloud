import torch

# Fixed local corner ordering (8,3): every ± combination of (l/2, w/2, h/2).
# Predicted and GT corners MUST use this same template so index i always means
# "the same relative corner" on both sides -- that's what makes a plain
# per-index L2 distance a meaningful (and rotation-sensitive) cost.
_SIGNS = torch.tensor(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=torch.float32,
)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """q: (...,4) as (w,x,y,z), normalized. Returns (...,3,3)."""
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def box_to_corners(center: torch.Tensor, size: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """center: (N,3), size: (N,3) [length,width,height], quat: (N,4) [w,x,y,z].
    Returns corners (N,8,3) in the fixed order given by _SIGNS."""
    device = center.device
    signs = _SIGNS.to(device)                       # (8,3)
    local = signs[None, :, :] * (size[:, None, :] / 2)   # (N,8,3)
    R = quat_to_rotmat(quat)                         # (N,3,3)
    world = torch.einsum("nij,nkj->nki", R, local) + center[:, None, :]
    return world


def quat_geodesic_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """Double-cover-safe angular distance in [0, pi], per row. q_*: (N,4)."""
    q_pred = quat_normalize(q_pred)
    q_gt = quat_normalize(q_gt)
    dot = (q_pred * q_gt).sum(dim=-1).abs().clamp(max=1.0 - 1e-7)
    return 2 * torch.acos(dot)


def corner_distance_cost(pred_center, pred_size, pred_quat, gt_center, gt_size, gt_quat):
    """Mean per-corner L2 distance between two (broadcast-compatible) boxes.
    All inputs (N,3)/(N,3)/(N,4); returns (N,) cost in meters."""
    pred_corners = box_to_corners(pred_center, pred_size, pred_quat)
    gt_corners = box_to_corners(gt_center, gt_size, gt_quat)
    return (pred_corners - gt_corners).norm(dim=-1).mean(dim=-1)
