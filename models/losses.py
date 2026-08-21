import torch
import torch.nn.functional as F

from .box_utils import box_to_corners, quat_geodesic_loss


def focal_heatmap_loss(pred_logit, gt_heatmap, alpha=2, beta=4):
    """Penalty-reduced focal loss (CornerNet/CenterNet). pred_logit, gt_heatmap: (N,).
    gt_heatmap==1 marks the (dynamically-assigned) positives exactly; everywhere
    else is a soft Gaussian target used only to reduce the negative penalty."""
    pred = torch.sigmoid(pred_logit).clamp(min=1e-4, max=1 - 1e-4)
    pos_mask = gt_heatmap.eq(1)
    pos_loss = -torch.log(pred) * (1 - pred).pow(alpha) * pos_mask
    neg_loss = -torch.log(1 - pred) * pred.pow(alpha) * (1 - gt_heatmap).pow(beta) * (~pos_mask)
    num_pos = pos_mask.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def _voxel_world_centers(coords, pc_range, effective_voxel_size):
    device = coords.device
    return pc_range[:3].to(device) + (coords[:, 1:4].float() + 0.5) * effective_voxel_size.to(device)


def render_gaussian_targets(coords, points_per_batch, pc_range, effective_voxel_size, sigma):
    """points_per_batch: list[B] of (Mi,3) world points (GT centers or corners).
    Returns (N,) = max over that batch item's points of exp(-dist^2/2sigma^2)."""
    device = coords.device
    voxel_world = _voxel_world_centers(coords, pc_range, effective_voxel_size)
    heat = torch.zeros(coords.shape[0], device=device)
    for b, pts in enumerate(points_per_batch):
        if pts.shape[0] == 0:
            continue
        pts = pts.to(device)
        bmask = coords[:, 0] == b
        if not bmask.any():
            continue
        vw = voxel_world[bmask]                                    # (Nb,3)
        dist2 = ((vw[:, None, :] - pts[None, :, :]) ** 2).sum(-1)   # (Nb,M)
        g = torch.exp(-dist2 / (2 * sigma ** 2))
        heat[bmask] = torch.maximum(heat[bmask], g.max(dim=1).values)
    return heat


def render_predicted_corner_map(coords, pos_mask, pred_center, pred_size, pred_quat, pc_range, effective_voxel_size, sigma):
    pos_idx = pos_mask.nonzero(as_tuple=True)[0]
    if pos_idx.numel() == 0:
        return torch.zeros(coords.shape[0], device=coords.device)
    corners = box_to_corners(pred_center[pos_idx], pred_size[pos_idx], pred_quat[pos_idx])  # (P,8,3)
    batch_of_pos = coords[pos_idx, 0]
    n_batch = int(coords[:, 0].max().item()) + 1 if coords.shape[0] > 0 else 0
    corners_per_batch = []
    for b in range(n_batch):
        bm = batch_of_pos == b
        corners_per_batch.append(corners[bm].reshape(-1, 3) if bm.any() else torch.zeros(0, 3, device=coords.device))
    return render_gaussian_targets(coords, corners_per_batch, pc_range, effective_voxel_size, sigma)


def render_gt_corner_map(coords, gt_boxes_list, pc_range, effective_voxel_size, sigma):
    corners_per_batch = []
    for gt_boxes in gt_boxes_list:
        if gt_boxes.shape[0] == 0:
            corners_per_batch.append(torch.zeros(0, 3))
        else:
            c = box_to_corners(gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6:10])
            corners_per_batch.append(c.reshape(-1, 3))
    return render_gaussian_targets(coords, corners_per_batch, pc_range, effective_voxel_size, sigma)


def compute_total_loss(pred, coords, gt_boxes_list, assign_result, loss_cfg, pc_range, effective_voxel_size):
    weights = loss_cfg["WEIGHTS"]
    alpha, beta = loss_cfg["FOCAL_ALPHA"], loss_cfg["FOCAL_BETA"]
    center_sigma_m = loss_cfg["CENTER_SIGMA_M"]
    pos_mask = assign_result["pos_mask"]
    device = coords.device

    gt_centers_per_batch = [g[:, :3] for g in gt_boxes_list]
    center_heat_target = render_gaussian_targets(coords, gt_centers_per_batch, pc_range, effective_voxel_size, center_sigma_m)
    center_heat_target = torch.where(pos_mask, torch.ones_like(center_heat_target), center_heat_target)
    center_loss = focal_heatmap_loss(pred["center_logit"].squeeze(-1), center_heat_target, alpha, beta)

    if pos_mask.any():
        voxel_world = _voxel_world_centers(coords, pc_range, effective_voxel_size)
        target_offset = assign_result["target_center"] - voxel_world
        offset_loss = F.l1_loss(pred["offset"][pos_mask], target_offset[pos_mask])

        pred_log_size = pred["box"][:, :3]
        target_log_size = torch.log(assign_result["target_size"].clamp(min=1e-3))
        size_loss = F.l1_loss(pred_log_size[pos_mask], target_log_size[pos_mask])

        pred_quat = pred["box"][:, 3:7]
        rot_loss = quat_geodesic_loss(pred_quat[pos_mask], assign_result["target_quat"][pos_mask]).mean()
    else:
        offset_loss = pred["offset"].sum() * 0.0
        size_loss = pred["box"][:, :3].sum() * 0.0
        rot_loss = pred["box"][:, 3:7].sum() * 0.0

    gt_corner_heat = render_gt_corner_map(coords, gt_boxes_list, pc_range, effective_voxel_size, center_sigma_m)
    pred_corner_heat = render_predicted_corner_map(
        coords, pos_mask, assign_result["pred_center"], assign_result["pred_size"], assign_result["pred_quat"],
        pc_range, effective_voxel_size, center_sigma_m,
    )
    corner_loss = F.mse_loss(pred_corner_heat, gt_corner_heat)

    total = (
        weights["center"] * center_loss +
        weights["corner_aux"] * corner_loss +
        weights["offset"] * offset_loss +
        weights["size"] * size_loss +
        weights["rotation"] * rot_loss
    )
    return {
        "total": total, "center": center_loss, "corner_aux": corner_loss,
        "offset": offset_loss, "size": size_loss, "rotation": rot_loss,
    }
