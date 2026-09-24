"""FCAF3D-style (arXiv 2112.00322) loss: focal classification (over every
location, positive or negative) + IoU regression + geodesic rotation + BCE
centerness (both summed over positive locations only and normalized by
Npos) -- see models/assign_fcaf3d.py for how locations become positive.
Reuses losses_detr.py's sigmoid_focal_loss (same hard 0/1 target convention)
rather than duplicating it; that one averages over every location instead of
dividing by Npos the way the paper's own formula does, a minor, standard
(RetinaNet-style) deviation."""
import torch
import torch.nn.functional as F

from .losses_detr import sigmoid_focal_loss
from .rotation6d import sixd_to_matrix, matrix_geodesic_loss


def differentiable_axis_aligned_iou_3d(center1: torch.Tensor, size1: torch.Tensor,
                                        center2: torch.Tensor, size2: torch.Tensor) -> torch.Tensor:
    """Batched, differentiable twin of box_utils.axis_aligned_iou_3d (that one
    calls .item(), breaking autograd, and only handles one box pair at a
    time). center/size: (N,3) each. Returns (N,) IoU in [0,1]."""
    lo1, hi1 = center1 - size1 / 2, center1 + size1 / 2
    lo2, hi2 = center2 - size2 / 2, center2 + size2 / 2
    overlap = (torch.minimum(hi1, hi2) - torch.maximum(lo1, lo2)).clamp(min=0)
    inter_vol = overlap.prod(dim=-1)
    union_vol = (size1.prod(dim=-1) + size2.prod(dim=-1) - inter_vol).clamp(min=1e-9)
    return inter_vol / union_vol


def compute_fcaf3d_loss(level_preds: list, level_targets: list, loss_cfg: dict) -> dict:
    """level_preds: list[L] of dicts of (N_l,*) tensors from FCAF3DHead.
    level_targets: list[L] of dicts from assign_multilevel. Returns a loss
    dict with 'total' plus per-term keys, summed across all levels together."""
    weights = loss_cfg["WEIGHTS"]
    all_exist_logit, all_exist_target = [], []
    iou_terms, center_terms, size_terms, rot_terms, cntr_terms = [], [], [], [], []

    for pred, tgt in zip(level_preds, level_targets):
        pos = tgt["pos_mask"]
        all_exist_logit.append(pred["exist_logit"].squeeze(-1))
        all_exist_target.append(pos.float())
        if not pos.any():
            continue

        pred_center = pred["center"][pos]
        # An unstable early-training step can push log_size to a large value;
        # exp() of that overflows float32 (~3.4e38, so anything past ~35
        # cubed already overflows), which then turns the IoU loss's
        # size1.prod() into inf and, a step later, nan -- hit this for real
        # (iou term spiked to 34 then the whole loss went nan the next step).
        # Clamped to +-15 nats (~3.3M x smaller/larger than 1), far outside
        # any real object size, so this never affects a well-behaved model.
        pred_log_size = pred["log_size"][pos].clamp(min=-15.0, max=15.0)
        pred_size = pred_log_size.exp()
        pred_R = sixd_to_matrix(pred["sixd"][pos])
        gt_center = tgt["center"][pos]
        gt_log_size = tgt["log_size"][pos]
        gt_size = gt_log_size.exp()
        gt_R = tgt["rot_matrix"][pos]

        iou = differentiable_axis_aligned_iou_3d(pred_center, pred_size, gt_center, gt_size)
        iou_terms.append(1.0 - iou)
        center_terms.append(F.l1_loss(pred_center, gt_center, reduction="none").sum(-1))
        size_terms.append(F.l1_loss(pred_log_size, gt_log_size, reduction="none").sum(-1))
        rot_terms.append(matrix_geodesic_loss(pred_R, gt_R))
        cntr_terms.append(F.binary_cross_entropy_with_logits(
            pred["centerness_logit"][pos].squeeze(-1), tgt["centerness"][pos], reduction="none"))

    exist_logit = torch.cat(all_exist_logit)
    exist_target = torch.cat(all_exist_target)
    cls_loss = sigmoid_focal_loss(exist_logit, exist_target,
                                   loss_cfg.get("CLS_ALPHA", 0.25), loss_cfg.get("CLS_GAMMA", 2.0))

    n_pos = int(exist_target.sum().item())
    if n_pos > 0:
        iou_loss = torch.cat(iou_terms).sum() / n_pos
        center_loss = torch.cat(center_terms).sum() / n_pos
        size_loss = torch.cat(size_terms).sum() / n_pos
        rot_loss = torch.cat(rot_terms).sum() / n_pos
        cntr_loss = torch.cat(cntr_terms).sum() / n_pos
    else:
        zero = exist_logit.sum() * 0.0
        iou_loss = center_loss = size_loss = rot_loss = cntr_loss = zero

    total = (weights["cls"] * cls_loss + weights["iou"] * iou_loss + weights["center"] * center_loss +
             weights["size"] * size_loss + weights["rotation"] * rot_loss + weights["centerness"] * cntr_loss)
    # Same defensive guard as models/matcher.py's cost-matrix clamp: a
    # destabilized step can still produce inf/nan here despite the log_size
    # clamp above (e.g. via center/rotation), and without this one bad step
    # poisons every parameter for the rest of a multi-hour run instead of
    # just being a wasted step.
    total = torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=0.0)
    return {"total": total, "cls": cls_loss, "iou": iou_loss, "center": center_loss, "size": size_loss,
            "rotation": rot_loss, "centerness": cntr_loss, "n_pos": n_pos}
