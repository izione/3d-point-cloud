import torch
import torch.nn.functional as F

from .rotation6d import sixd_to_matrix, matrix_geodesic_loss
from .box_utils import quat_to_rotmat


def sigmoid_focal_loss(logit: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Standard RetinaNet-style focal loss for a HARD 0/1 target (existence:
    matched query -> 1, unmatched -> 0). Not the same as heads.py's
    focal_heatmap_loss, which is CornerNet-style and expects a soft Gaussian
    target -- the dense head's heatmap has one, a query's existence doesn't."""
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_t).pow(gamma) * ce).mean()


def compute_detr_loss(pred: dict, gt_boxes_list: list, matches: list, loss_cfg: dict) -> dict:
    """pred: dict of (B,Q,*) tensors from SetPredictionHead. gt_boxes_list: list[B]
    of (Mb,10). matches: list[B] of (query_idx, gt_idx) from HungarianMatcher."""
    weights = loss_cfg["WEIGHTS"]
    device = pred["exist_logit"].device
    batch_size, num_queries = pred["exist_logit"].shape[:2]

    exist_target = torch.zeros(batch_size, num_queries, device=device)
    center_terms, size_terms, rot_terms = [], [], []

    for b, (query_idx, gt_idx) in enumerate(matches):
        if query_idx.numel() == 0:
            continue
        exist_target[b, query_idx] = 1.0

        gt_boxes = gt_boxes_list[b].to(device)
        gt_center = gt_boxes[gt_idx, :3]
        gt_size = gt_boxes[gt_idx, 3:6]
        gt_R = quat_to_rotmat(gt_boxes[gt_idx, 6:10])

        pred_center = pred["center"][b, query_idx]
        pred_log_size = pred["log_size"][b, query_idx]
        pred_R = sixd_to_matrix(pred["sixd"][b, query_idx])

        center_terms.append(F.l1_loss(pred_center, gt_center, reduction="none").sum(-1))
        size_terms.append(F.l1_loss(pred_log_size, torch.log(gt_size.clamp(min=1e-3)), reduction="none").sum(-1))
        rot_terms.append(matrix_geodesic_loss(pred_R, gt_R))

    cls_loss = sigmoid_focal_loss(pred["exist_logit"].squeeze(-1), exist_target,
                                   loss_cfg.get("CLS_ALPHA", 0.25), loss_cfg.get("CLS_GAMMA", 2.0))

    if center_terms:
        center_loss = torch.cat(center_terms).mean()
        size_loss = torch.cat(size_terms).mean()
        rot_loss = torch.cat(rot_terms).mean()
    else:
        center_loss = pred["center"].sum() * 0.0
        size_loss = pred["log_size"].sum() * 0.0
        rot_loss = pred["sixd"].sum() * 0.0

    total = (weights["cls"] * cls_loss + weights["center"] * center_loss +
             weights["size"] * size_loss + weights["rotation"] * rot_loss)
    return {"total": total, "cls": cls_loss, "center": center_loss, "size": size_loss, "rotation": rot_loss}


def compute_denoising_loss(pred_dn: dict, targets: dict, valid_mask: torch.Tensor, loss_cfg: dict) -> dict:
    """Loss for the query-denoising queries (models/denoising.py) -- no
    matching needed, each denoising query's target is simply the (unnoised)
    GT box it was built from. `valid_mask` (B,G*Mmax) marks real GT slots vs.
    padding (samples with fewer GT boxes than that batch's max)."""
    weights = loss_cfg["WEIGHTS"]
    exist_target = valid_mask.float()
    cls_loss = sigmoid_focal_loss(pred_dn["exist_logit"].squeeze(-1), exist_target,
                                   loss_cfg.get("CLS_ALPHA", 0.25), loss_cfg.get("CLS_GAMMA", 2.0))

    if valid_mask.any():
        pred_center = pred_dn["center"][valid_mask]
        pred_log_size = pred_dn["log_size"][valid_mask]
        pred_R = sixd_to_matrix(pred_dn["sixd"][valid_mask])
        center_loss = F.l1_loss(pred_center, targets["center"][valid_mask])
        size_loss = F.l1_loss(pred_log_size, targets["log_size"][valid_mask])
        rot_loss = matrix_geodesic_loss(pred_R, targets["rot_matrix"][valid_mask]).mean()
    else:
        center_loss = pred_dn["center"].sum() * 0.0
        size_loss = pred_dn["log_size"].sum() * 0.0
        rot_loss = pred_dn["sixd"].sum() * 0.0

    total = (weights["cls"] * cls_loss + weights["center"] * center_loss +
             weights["size"] * size_loss + weights["rotation"] * rot_loss)
    return {"total": total, "cls": cls_loss, "center": center_loss, "size": size_loss, "rotation": rot_loss}
