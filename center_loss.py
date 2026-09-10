"""center_loss.py - CenterPoint/CenterNet-style loss, pairs with
heatmap_targets.py's targets.

Classification: Gaussian-penalty-reduced focal loss (Law & Deng, CornerNet
2018; Zhou et al., CenterNet 2019) -- negative cells near a positive peak are
penalized less (by the Gaussian value itself) than far negatives, instead of
being treated as flatly "wrong."

Regression (offset/z/dim/rot): plain L1 loss at the single peak cell only --
no anchor, so no residual concept, values are regressed directly.
"""
import torch
import torch.nn.functional as F

import config


def gaussian_focal_loss(pred_logit: torch.Tensor, target: torch.Tensor,
                         alpha: float = None, beta: float = None) -> torch.Tensor:
    """pred_logit/target: (B,1,H,W). target is a 0-1 Gaussian heatmap (peak=1).
    CenterNet-standard normalization: divide by the number of positive (peak)
    cells, floor at 1.

    Uses F.logsigmoid rather than sigmoid+clamp+log: as predictions become
    confident (late in training), sigmoid output saturates near the clamp
    boundary and gradient there is exactly zero, silently freezing learning;
    logsigmoid's internal log-sum-exp stays numerically stable with nonzero
    gradient throughout."""
    alpha = config.FOCAL_ALPHA if alpha is None else alpha
    beta = config.FOCAL_BETA if beta is None else beta

    pred = torch.sigmoid(pred_logit)
    pos_mask = (target == 1).float()
    neg_mask = (target < 1).float()
    neg_weight = torch.pow(1 - target, beta)

    log_p = F.logsigmoid(pred_logit)
    log_1mp = F.logsigmoid(-pred_logit)

    pos_loss = -log_p * torch.pow(1 - pred, alpha) * pos_mask
    neg_loss = -log_1mp * torch.pow(pred, alpha) * neg_weight * neg_mask

    n_pos = pos_mask.sum().clamp_min(1)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def center_voxelnet_loss(heatmap_pred, offset_pred, z_pred, dim_pred, rot_pred,
                          heatmap_target, reg_mask, offset_target, z_target, dim_target, rot_target,
                          reg_weight: float = None):
    """*_pred: model output (B,C,H,W). *_target/reg_mask: cached targets
    (B,H,W,C)/(B,H,W) -- permuted here to match pred's axis order.
    Returns (total_loss, stats_dict)."""
    reg_weight = config.REG_LOSS_WEIGHT if reg_weight is None else reg_weight
    hm_loss = gaussian_focal_loss(heatmap_pred, heatmap_target)

    def to_bhwc(x):
        return x.permute(0, 2, 3, 1)

    n_pos = reg_mask.float().sum().clamp_min(1)
    reg_cell_w = reg_mask.unsqueeze(-1).float()

    offset_loss = (F.l1_loss(to_bhwc(offset_pred), offset_target, reduction="none") * reg_cell_w).sum() / n_pos
    z_loss = (F.l1_loss(to_bhwc(z_pred), z_target, reduction="none") * reg_cell_w).sum() / n_pos
    dim_loss = (F.l1_loss(to_bhwc(dim_pred), dim_target, reduction="none") * reg_cell_w).sum() / n_pos
    rot_loss = (F.l1_loss(to_bhwc(rot_pred), rot_target, reduction="none") * reg_cell_w).sum() / n_pos

    reg_loss = offset_loss + z_loss + dim_loss + rot_loss
    total = hm_loss + reg_weight * reg_loss

    stats = {"hm_loss": hm_loss.item(), "reg_loss": reg_loss.item(),
              "offset_loss": offset_loss.item(), "z_loss": z_loss.item(),
              "dim_loss": dim_loss.item(), "rot_loss": rot_loss.item(),
              "n_pos": int(reg_mask.sum().item())}
    return total, stats
