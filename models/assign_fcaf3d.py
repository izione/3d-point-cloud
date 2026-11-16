"""FCAF3D-style (arXiv 2112.00322) multi-level anchor-free GT assignment: for
each GT box, pick the coarsest ("last") feature level whose voxels cover at
least MIN_LOCATIONS of it, falling back to the finest level if none qualify;
then mark only the voxels within CENTER_SAMPLE_RADIUS (as a fraction of the
box's own half-size) of that GT's center as positive (FCOS/FCAF3D's "center
sampling"). This project's own dataset is far smaller-scale than FCAF3D's
indoor scenes, so MIN_LOCATIONS/CENTER_SAMPLE_RADIUS are much smaller than
the paper's own Nloc=33/18 -- see configs/exp_fcaf3d.yaml's ASSIGN section.

Ties (two GT boxes both wanting the same voxel) go to the smaller-volume GT,
same as FCOS/FCAF3D."""
import torch

from .box_utils import quat_to_rotmat


def assign_multilevel(level_world_centers, level_batch_idx, gt_boxes_list, batch_size,
                       min_locations=6, center_sample_radius=0.5):
    """level_world_centers: list[L] of (N_l,3) world-space voxel centers, fine-to-coarse.
    level_batch_idx: list[L] of (N_l,) which sample each voxel belongs to.
    gt_boxes_list: list[B] of (Mb,10) [center(3),size(3),quat(4)].
    Returns list[L] of dicts, each with (N_l,*) tensors: pos_mask (bool),
    center, log_size, rot_matrix, centerness -- only meaningful where
    pos_mask is True, zero-filled elsewhere."""
    num_levels = len(level_world_centers)
    device = level_world_centers[0].device

    targets = []
    for level_centers in level_world_centers:
        n = level_centers.shape[0]
        targets.append({
            "pos_mask": torch.zeros(n, dtype=torch.bool, device=device),
            "gt_volume": torch.full((n,), float("inf"), device=device),
            "center": torch.zeros(n, 3, device=device),
            "log_size": torch.zeros(n, 3, device=device),
            "rot_matrix": torch.eye(3, device=device)[None].expand(n, -1, -1).clone(),
            "centerness": torch.zeros(n, device=device),
        })

    for b, gt_boxes in enumerate(gt_boxes_list):
        if gt_boxes.shape[0] == 0:
            continue
        gt_boxes = gt_boxes.to(device)
        for m in range(gt_boxes.shape[0]):
            center, size, quat = gt_boxes[m, :3], gt_boxes[m, 3:6], gt_boxes[m, 6:10]
            R = quat_to_rotmat(quat.unsqueeze(0))[0]
            half = size / 2
            volume = size.prod()

            level_idx_in_box = []
            level_counts = []
            for lvl in range(num_levels):
                mask_b = level_batch_idx[lvl] == b
                idx_b = mask_b.nonzero(as_tuple=True)[0]
                if idx_b.numel() == 0:
                    level_idx_in_box.append(idx_b)
                    level_counts.append(0)
                    continue
                local = (level_world_centers[lvl][idx_b] - center[None, :]) @ R
                inside = (local.abs() <= half[None, :]).all(dim=-1)
                level_idx_in_box.append(idx_b[inside])
                level_counts.append(int(inside.sum().item()))

            chosen = None
            for lvl in range(num_levels - 1, -1, -1):
                if level_counts[lvl] >= min_locations:
                    chosen = lvl
                    break
            if chosen is None:
                best = max(range(num_levels), key=lambda lvl: level_counts[lvl])
                if level_counts[best] > 0:
                    chosen = best
            if chosen is None:
                continue  # no level has any voxel inside this box at all

            idx_in_box = level_idx_in_box[chosen]
            local = (level_world_centers[chosen][idx_in_box] - center[None, :]) @ R
            sample_half = half * center_sample_radius
            center_sampled = (local.abs() <= sample_half[None, :]).all(dim=-1)
            pos_idx = idx_in_box[center_sampled]
            if pos_idx.numel() == 0:
                pos_idx = idx_in_box  # center sampling too strict for this box -- use every covering voxel

            tgt = targets[chosen]
            better = volume < tgt["gt_volume"][pos_idx]
            win_idx = pos_idx[better]
            if win_idx.numel() == 0:
                continue

            tgt["pos_mask"][win_idx] = True
            tgt["gt_volume"][win_idx] = volume
            tgt["center"][win_idx] = center
            tgt["log_size"][win_idx] = torch.log(size.clamp(min=1e-3))
            tgt["rot_matrix"][win_idx] = R

            local_win = (level_world_centers[chosen][win_idx] - center[None, :]) @ R
            d_lo = (half[None, :] + local_win).clamp(min=1e-6)
            d_hi = (half[None, :] - local_win).clamp(min=1e-6)
            ratio = torch.minimum(d_lo, d_hi) / torch.maximum(d_lo, d_hi)
            tgt["centerness"][win_idx] = ratio.clamp(min=1e-6).prod(dim=-1).pow(1.0 / 3.0)

    return targets
