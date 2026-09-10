"""rotation3d.py - 3D rotation representation utilities.

CenterHead regresses full 3D rotation as a 6D continuous representation
(Zhou et al., "On the Continuity of Rotation Representations in Neural
Networks", CVPR 2019) instead of a raw Euler angle or quaternion: Euler has
gimbal-lock/wraparound discontinuities, quaternion has the q=-q double-cover
problem. The 6D representation regresses the rotation matrix's first two
columns (6 numbers) and recovers the third via Gram-Schmidt -- no
discontinuities anywhere on SO(3), which makes it a much more stable
regression target.

Axis convention: local x=length, local y=width, local z=height
(world_col = R @ local_col) -- matches this dataset's label convention.

Ported unchanged from the sibling voxelnet_baseline repo's rotation3d.py.
"""
import numpy as np
import torch


def euler_to_matrix(rotation_x_deg, rotation_y_deg, rotation_z_deg) -> np.ndarray:
    """(3,3), Rz . Ry . Rx, world_col = R @ local_col."""
    rx, ry, rz = np.radians([rotation_x_deg, rotation_y_deg, rotation_z_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """(3,3) -> (6,): R's first two columns concatenated."""
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def sixd_to_matrix_np(ortho6d: np.ndarray) -> np.ndarray:
    """(6,) -> (3,3) via Gram-Schmidt orthogonalization (numpy, decode/eval path)."""
    a1, a2 = ortho6d[0:3], ortho6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2_proj = a2 - np.dot(b1, a2) * b1
    b2 = a2_proj / (np.linalg.norm(a2_proj) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def sixd_to_matrix_torch(ortho6d: torch.Tensor) -> torch.Tensor:
    """(...,6) -> (...,3,3), batched Gram-Schmidt (train/decode path)."""
    a1 = ortho6d[..., 0:3]
    a2 = ortho6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    a2_proj = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2_proj, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)
