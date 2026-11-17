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
from .losses import focal_heatmap_loss
from .assign_fcaf3d import render_point_heatmap, render_point_heatmap_channels
from .rotation6d import sixd_to_matrix, matrix_geodesic_loss, rotation_from_raw

# Same fixed local corner-sign convention as box_utils.py's _SIGNS, duplicated
# rather than imported (see models/backbone3d_unet_spconv.py's own precedent
# for why: keeps this module from reaching into box_utils' private name).
_CORNER_SIGNS = torch.tensor(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=torch.float32)


def _corners_from_matrix(center: torch.Tensor, size: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """center/size: (N,3), R: (N,3,3) -> corners (N,8,3). Same fixed corner
    order as box_utils.box_to_corners, but takes a rotation MATRIX directly
    (predicted boxes already have one via sixd_to_matrix) instead of a
    quaternion, avoiding a pointless matrix->quat->matrix round trip."""
    signs = _CORNER_SIGNS.to(center.device)                 # (8,3)
    local = signs[None, :, :] * (size[:, None, :] / 2)      # (N,8,3)
    return torch.einsum("nij,nkj->nki", R, local) + center[:, None, :]


def differentiable_axis_aligned_iou_3d(center1: torch.Tensor, size1: torch.Tensor,
                                        center2: torch.Tensor, size2: torch.Tensor) -> torch.Tensor:
    """Batched, differentiable twin of box_utils.axis_aligned_iou_3d (that one
    calls .item(), breaking autograd, and only handles one box pair at a
    time). center/size: (N,3) each. Returns (N,) IoU in [0,1].

    Kept only as the SINGLE_LEVEL=false / legacy reference point -- rotation
    is ignored entirely here, which is exactly the gap
    differentiable_rotated_iou_3d_gwd below closes (see its docstring); the
    rotated version is what compute_fcaf3d_loss actually uses now."""
    lo1, hi1 = center1 - size1 / 2, center1 + size1 / 2
    lo2, hi2 = center2 - size2 / 2, center2 + size2 / 2
    overlap = (torch.minimum(hi1, hi2) - torch.maximum(lo1, lo2)).clamp(min=0)
    inter_vol = overlap.prod(dim=-1)
    union_vol = (size1.prod(dim=-1) + size2.prod(dim=-1) - inter_vol).clamp(min=1e-9)
    return inter_vol / union_vol


def differentiable_rotated_iou_3d_gwd(center1: torch.Tensor, size1: torch.Tensor, R1: torch.Tensor,
                                       center2: torch.Tensor, size2: torch.Tensor, R2: torch.Tensor,
                                       tau: float = 1.0) -> torch.Tensor:
    """Rotation-AWARE differentiable pseudo-IoU, via the Gaussian Wasserstein
    Distance (GWD) surrogate (Yang et al., "Rethinking Rotated Object
    Detection with Gaussian Wasserstein Distance Loss", arXiv 2101.11952) --
    an exact differentiable rotated-3D-box IoU needs convex polygon clipping
    (no closed form, awkward to batch/autograd); GWD sidesteps that by
    representing each oriented box as an equivalent Gaussian
    N(center, R diag((size/2)^2) R^T) and using the closed-form 2-Wasserstein
    distance between two Gaussians, which IS smooth everywhere (unlike exact
    rotated IoU, which is flat/non-differentiable whenever boxes don't
    overlap at all).

    center/size: (N,3), R: (N,3,3) rotation matrices (box_utils/rotation6d
    convention). Returns (N,) pseudo-IoU in (0,1] (1.0 only at an exact
    match, when tau=1 -- see tau below), via loss=1-pseudo_iou so this drops
    straight into compute_fcaf3d_loss's existing `iou_terms`.

    W2^2(N1,N2) = ||mu1-mu2||^2 + Tr(S1) + Tr(S2) - 2*Tr((S2^0.5 S1 S2^0.5)^0.5)
    Tr(S_i) = sum((size_i/2)^2) directly (S_i's eigenvalues ARE (size_i/2)^2,
    since S_i = R_i diag(a_i) R_i^T with R_i orthonormal) -- no decomposition
    needed for those. S2^0.5 = R2 diag(sqrt(a2)) R2^T is similarly closed-form.
    The middle product's eigenvalues (for the cross trace term) genuinely do
    need `eigvalsh` (R1 != R2 in general so S2^0.5 S1 S2^0.5 isn't diagonal in
    either box's own frame) -- eigenvalues only (not eigenvectors), which is
    the cheap, numerically-stable half of an eigendecomposition and is enough
    since only the trace of the matrix's own square root is needed."""
    mu_sq = (center1 - center2).pow(2).sum(-1)

    a1 = (size1 / 2).pow(2)  # (N,3), S1's eigenvalues in its own R1 frame
    a2 = (size2 / 2).pow(2)
    tr1, tr2 = a1.sum(-1), a2.sum(-1)

    sqrt_a2 = a2.clamp(min=0).sqrt()
    S1 = R1 @ torch.diag_embed(a1) @ R1.transpose(-1, -2)
    S2_half = R2 @ torch.diag_embed(sqrt_a2) @ R2.transpose(-1, -2)
    M = S2_half @ S1 @ S2_half
    M = 0.5 * (M + M.transpose(-1, -2))  # enforce exact symmetry (fp roundoff) before eigvalsh
    eigvals = torch.linalg.eigvalsh(M).clamp(min=0)
    cross_tr = eigvals.sqrt().sum(-1)

    w2_sq = (mu_sq + tr1 + tr2 - 2 * cross_tr).clamp(min=0)
    return 1.0 / (tau + w2_sq.sqrt())


def compute_fcaf3d_loss(level_preds: list, level_targets: list, loss_cfg: dict, gaussian_heatmap: bool = False,
                         level_world_centers: list = None, level_batch_idx: list = None,
                         corner_aux_sigma: float = None, corner_identity: bool = False,
                         rotation_mode: str = "6d") -> dict:
    """level_preds: list[L] of dicts of (N_l,*) tensors from FCAF3DHead.
    level_targets: list[L] of dicts from assign_multilevel. Returns a loss
    dict with 'total' plus per-term keys, summed across all levels together.

    If the head was built with predict_box=False (see heads_fcaf3d.py --
    configs/exp_fcaf3d_stage1_lite.yaml), level_preds won't have 'log_size'/
    'sixd' keys at all; this function then skips iou/size/rotation entirely
    (stage 2's pose_refine.py owns those instead) rather than computing a
    loss against predictions that don't exist.

    corner_aux_sigma (needs level_world_centers/level_batch_idx too, and
    gaussian_heatmap=True, and level_targets to carry 'gt_corner_heatmap'
    from assign_multilevel's own gaussian_sigma): the "corner auxiliary" term
    ported from this project's original pre-FCAF3D dense head
    (models/losses.py::compute_total_loss's corner_loss) -- an extra MSE
    between a Gaussian heatmap rendered from the GT boxes' 8 corners and one
    rendered from the PREDICTED boxes' 8 corners (at positive locations
    only), supervising box EXTENT/orientation spatially instead of only via
    the center/size/rotation regression terms above."""
    weights = loss_cfg["WEIGHTS"]
    predict_box = "log_size" in level_preds[0]
    all_exist_logit, all_exist_target = [], []
    iou_terms, center_terms, size_terms, rot_terms, cntr_terms = [], [], [], [], []
    pred_corner_heats, gt_corner_heats = [], []

    for lvl, (pred, tgt) in enumerate(zip(level_preds, level_targets)):
        pos = tgt["pos_mask"]
        all_exist_logit.append(pred["exist_logit"].squeeze(-1))
        all_exist_target.append(tgt["exist_heatmap"] if gaussian_heatmap else pos.float())
        if not pos.any():
            continue

        pred_center = pred["center"][pos]
        gt_center = tgt["center"][pos]
        center_terms.append(F.l1_loss(pred_center, gt_center, reduction="none").sum(-1))
        if not gaussian_heatmap:
            cntr_terms.append(F.binary_cross_entropy_with_logits(
                pred["centerness_logit"][pos].squeeze(-1), tgt["centerness"][pos], reduction="none"))

        if not predict_box:
            continue
        # An unstable early-training step can push log_size to a large value;
        # exp() of that overflows float32 (~3.4e38, so anything past ~35
        # cubed already overflows), which then turns the IoU loss's
        # size1.prod() into inf and, a step later, nan -- hit this for real
        # (iou term spiked to 34 then the whole loss went nan the next step).
        # Clamped to +-15 nats (~3.3M x smaller/larger than 1), far outside
        # any real object size, so this never affects a well-behaved model.
        pred_log_size = pred["log_size"][pos].clamp(min=-15.0, max=15.0)
        pred_size = pred_log_size.exp()
        pred_R = rotation_from_raw(pred["sixd"][pos], rotation_mode)
        gt_log_size = tgt["log_size"][pos]
        gt_size = gt_log_size.exp()
        gt_R = tgt["rot_matrix"][pos]

        rotated_iou = differentiable_rotated_iou_3d_gwd(pred_center, pred_size, pred_R, gt_center, gt_size, gt_R)
        iou_terms.append(1.0 - rotated_iou)
        size_terms.append(F.l1_loss(pred_log_size, gt_log_size, reduction="none").sum(-1))
        rot_terms.append(matrix_geodesic_loss(pred_R, gt_R))

        if corner_aux_sigma is not None:
            pred_corners = _corners_from_matrix(pred_center, pred_size, pred_R)  # (Npos,8,3)
            batch_of_pos = level_batch_idx[lvl][pos]                            # (Npos,)
            full_bidx = level_batch_idx[lvl]
            n_batch = int(full_bidx.max().item()) + 1 if full_bidx.numel() > 0 else 0
            if corner_identity:
                # RTM3D-style: keep each of the 8 corners' identity separate
                # (8 independent heatmap channels) instead of pooling them all
                # into one combined max-heatmap -- see
                # assign_fcaf3d.render_gt_corner_heatmap_identity's docstring.
                # Uses render_point_heatmap_channels (one broadcast per batch
                # item, all 8 channels at once) -- NOT a per-channel python
                # loop, which earlier measurably stalled training near this
                # GPU's VRAM ceiling (8x the kernel launches for the same FLOPs).
                pred_corners_per_batch = [pred_corners[batch_of_pos == b] for b in range(n_batch)]  # list of (Mb,8,3)
                pred_heat = render_point_heatmap_channels(level_world_centers[lvl], full_bidx,
                                                           pred_corners_per_batch, corner_aux_sigma)  # (N_lvl,8)
                pred_corner_heats.append(pred_heat.transpose(0, 1))         # (8,N_lvl)
                gt_corner_heats.append(tgt["gt_corner_heatmap_identity"])   # (8,N_lvl)
            else:
                pred_corners_per_batch = [pred_corners[batch_of_pos == b].reshape(-1, 3) for b in range(n_batch)]
                pred_heat = render_point_heatmap(level_world_centers[lvl], full_bidx,
                                                  pred_corners_per_batch, corner_aux_sigma)
                pred_corner_heats.append(pred_heat)
                gt_corner_heats.append(tgt["gt_corner_heatmap"])

    exist_logit = torch.cat(all_exist_logit)
    exist_target = torch.cat(all_exist_target)
    if gaussian_heatmap:
        cls_loss = focal_heatmap_loss(exist_logit, exist_target,
                                       loss_cfg.get("HEATMAP_ALPHA", 2), loss_cfg.get("HEATMAP_BETA", 4))
        n_pos = int(exist_target.eq(1.0).sum().item())
    else:
        cls_loss = sigmoid_focal_loss(exist_logit, exist_target,
                                       loss_cfg.get("CLS_ALPHA", 0.25), loss_cfg.get("CLS_GAMMA", 2.0))
        n_pos = int(exist_target.sum().item())
    zero = exist_logit.sum() * 0.0
    if n_pos > 0:
        center_loss = torch.cat(center_terms).sum() / n_pos
        cntr_loss = torch.cat(cntr_terms).sum() / n_pos if not gaussian_heatmap else zero
        if predict_box:
            iou_loss = torch.cat(iou_terms).sum() / n_pos
            size_loss = torch.cat(size_terms).sum() / n_pos
            rot_loss = torch.cat(rot_terms).sum() / n_pos
        else:
            iou_loss = size_loss = rot_loss = zero
    else:
        iou_loss = center_loss = size_loss = rot_loss = cntr_loss = zero

    if corner_aux_sigma is not None and pred_corner_heats:
        cat_dim = 1 if corner_identity else 0  # identity mode: each entry is (8,N_lvl), concat along N
        corner_loss = F.mse_loss(torch.cat(pred_corner_heats, dim=cat_dim), torch.cat(gt_corner_heats, dim=cat_dim))
    else:
        corner_loss = zero

    total = weights["cls"] * cls_loss + weights["center"] * center_loss
    if not gaussian_heatmap:
        total = total + weights["centerness"] * cntr_loss
    if predict_box:
        total = total + weights["iou"] * iou_loss + weights["size"] * size_loss + weights["rotation"] * rot_loss
    if corner_aux_sigma is not None:
        total = total + weights.get("corner_aux", 0.0) * corner_loss
    # Same defensive guard as models/matcher.py's cost-matrix clamp: a
    # destabilized step can still produce inf/nan here despite the log_size
    # clamp above (e.g. via center/rotation), and without this one bad step
    # poisons every parameter for the rest of a multi-hour run instead of
    # just being a wasted step.
    total = torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=0.0)
    return {"total": total, "cls": cls_loss, "iou": iou_loss, "center": center_loss, "size": size_loss,
            "rotation": rot_loss, "centerness": cntr_loss, "corner_aux": corner_loss, "n_pos": n_pos}


