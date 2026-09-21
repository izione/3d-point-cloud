"""DN-DETR-style query denoising, matching SparseVoxFormer's training recipe
("auxiliary queries from GT cuboid centers with noise injection, training
only"). Alongside the normal learned "matching" queries (models/decoder_detr.py
:: DetrDecoder.matching_content/matching_reference_points), each training step
adds extra queries built directly FROM noised GT boxes: their target is simply
"reconstruct the box this noise came from", so no Hungarian matching is needed
for them (models/losses_detr.py::compute_denoising_loss) -- this gives the
decoder+head a much more direct training signal for the box regression than
matching queries get early on, when their reference points/content are still
essentially random.

These denoising queries must never leak the GT they're built from into the
real matching queries (that information isn't available at inference), so
they share the decoder's self-attention with a block-diagonal mask
(build_attention_mask): matching queries only attend to each other, and each
noise "group" only attends within itself.
"""
import math

import torch
import torch.nn as nn

from .rotation6d import matrix_to_sixd
from .box_utils import quat_to_rotmat


def _skew(v: torch.Tensor) -> torch.Tensor:
    """(...,3) -> (...,3,3) skew-symmetric cross-product matrix."""
    zeros = torch.zeros_like(v[..., 0])
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    row0 = torch.stack([zeros, -z, y], dim=-1)
    row1 = torch.stack([z, zeros, -x], dim=-1)
    row2 = torch.stack([-y, x, zeros], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _random_rotation_perturbation(shape, max_angle_rad: float, device) -> torch.Tensor:
    """Random small rotation matrices via Rodrigues' formula, one per leading
    index in `shape`. Returns (*shape,3,3)."""
    axis = torch.nn.functional.normalize(torch.randn(*shape, 3, device=device), dim=-1, eps=1e-8)
    angle = torch.rand(*shape, 1, device=device) * max_angle_rad
    K = _skew(axis)
    eye = torch.eye(3, device=device).expand(*shape, 3, 3)
    sin_a, cos_a = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    return eye + sin_a * K + (1 - cos_a) * (K @ K)


def build_attention_mask(num_matching: int, group_size: int, num_groups: int, device) -> torch.Tensor:
    """(Q_total,Q_total) additive mask (0=attend, -inf=blocked): the matching
    block only attends within itself, and each denoising group only attends
    within itself -- no cross-block attention at all."""
    total = num_matching + num_groups * group_size
    mask = torch.full((total, total), float("-inf"), device=device)
    mask[:num_matching, :num_matching] = 0.0
    for g in range(num_groups):
        s = num_matching + g * group_size
        e = s + group_size
        mask[s:e, s:e] = 0.0
    return mask


class QueryDenoising(nn.Module):
    def __init__(self, channels, num_groups=5, center_noise_scale=0.4, size_noise_scale=0.4, rot_noise_deg=15.0):
        super().__init__()
        self.channels = channels
        self.num_groups = num_groups
        self.center_noise_scale = center_noise_scale
        self.size_noise_scale = size_noise_scale
        self.rot_noise_rad = math.radians(rot_noise_deg)
        # a single learned "this is a denoising query" indicator (there's only
        # one object class here, so no label-noise/class-embedding is needed --
        # every denoising query is a "this really is a diver" positive sample)
        # plus a small MLP encoding the noised box itself (center_noise(3),
        # noised log-size(3), noised 6D rotation(6) = 12 dims) into the query's
        # content embedding, so the decoder actually has something box-specific
        # to condition on beyond just the reference point.
        self.indicator = nn.Parameter(torch.randn(channels) * 0.02)
        self.box_embed = nn.Sequential(
            nn.Linear(12, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels),
        )

    def build(self, gt_boxes_list: list, pc_range: torch.Tensor, device):
        """Returns None if every sample in the batch has zero GT boxes (nothing
        to denoise that step), else (dn_content (B,G*Mmax,C), dn_ref (B,G*Mmax,3),
        dn_valid (B,G*Mmax) bool, dn_targets dict of (B,G*Mmax,*), group_size=Mmax)."""
        batch_size = len(gt_boxes_list)
        m_list = [g.shape[0] for g in gt_boxes_list]
        m_max = max(m_list)
        if m_max == 0:
            return None
        G = self.num_groups

        gt_padded = torch.zeros(batch_size, m_max, 10, device=device)
        valid = torch.zeros(batch_size, m_max, dtype=torch.bool, device=device)
        for b, gt in enumerate(gt_boxes_list):
            m = gt.shape[0]
            if m > 0:
                gt_padded[b, :m] = gt.to(device)
                valid[b, :m] = True

        # replicate across G independently-noised groups
        gt_g = gt_padded[:, None, :, :].expand(batch_size, G, m_max, 10)
        valid_g = valid[:, None, :].expand(batch_size, G, m_max)

        true_center, true_size, true_quat = gt_g[..., :3], gt_g[..., 3:6], gt_g[..., 6:10]
        true_R = quat_to_rotmat(true_quat)  # (B,G,Mmax,3,3)

        center_noise = (torch.rand_like(true_center) * 2 - 1) * self.center_noise_scale * (true_size / 2)
        noised_center = true_center + center_noise
        noised_center = torch.max(torch.min(noised_center, pc_range[3:]), pc_range[:3])

        size_factor = 1 + (torch.rand_like(true_size) * 2 - 1) * self.size_noise_scale
        noised_size = (true_size * size_factor).clamp(min=0.05)
        noised_log_size = torch.log(noised_size)

        R_perturb = _random_rotation_perturbation(true_R.shape[:-2], self.rot_noise_rad, device)
        noised_R = R_perturb @ true_R
        noised_sixd = matrix_to_sixd(noised_R)

        box_feat = torch.cat([center_noise, noised_log_size, noised_sixd], dim=-1)  # (B,G,Mmax,12)
        content = self.indicator[None, None, None, :] + self.box_embed(box_feat)

        # flatten (G,Mmax) -> one query dimension
        dn_content = content.reshape(batch_size, G * m_max, self.channels)
        dn_ref = noised_center.reshape(batch_size, G * m_max, 3)
        dn_valid = valid_g.reshape(batch_size, G * m_max)
        dn_targets = {
            "center": true_center.reshape(batch_size, G * m_max, 3),
            "log_size": torch.log(true_size.clamp(min=1e-3)).reshape(batch_size, G * m_max, 3),
            "rot_matrix": true_R.reshape(batch_size, G * m_max, 3, 3),
        }
        return dn_content, dn_ref, dn_valid, dn_targets, m_max
