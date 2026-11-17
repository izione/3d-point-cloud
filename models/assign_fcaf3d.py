"""FCAF3D-style (arXiv 2112.00322) multi-level anchor-free GT assignment: for
each GT box, pick the coarsest ("last") feature level whose voxels cover at
least MIN_LOCATIONS of it, falling back to the finest level if none qualify;
then mark only the voxels within CENTER_SAMPLE_RADIUS (as a fraction of the
box's own half-size) of that GT's center as positive (FCOS/FCAF3D's "center
sampling"). This project's own dataset is far smaller-scale than FCAF3D's
indoor scenes, so MIN_LOCATIONS/CENTER_SAMPLE_RADIUS are much smaller than
the paper's own Nloc=33/18 -- see configs/exp_fcaf3d.yaml's ASSIGN section.

Ties (two GT boxes both wanting the same voxel) go to the smaller-volume GT,
same as FCOS/FCAF3D.

TR3D (arXiv 2302.02858) nearest-point fallback: FCAF3D's own assignment
requires at least one voxel strictly INSIDE the box at some level, which
silently drops the GT entirely (zero positives anywhere, no gradient at all
for that object) whenever a thin/small/sparsely-sampled box has no covering
voxel -- TR3D's paper cites exactly this as the reason some thin objects
(e.g. whiteboards) got missed. nearest_fallback_k turns this on: when a GT
box has no in-box voxel at ANY level, instead of dropping it, the K nearest
voxels (by raw center distance, no containment check) at the FINEST level
are assigned positive instead. Keeps the normal in-box path completely
unchanged for every GT that already gets covered -- this only recovers
objects that were previously silently dropped."""
import torch

from .box_utils import quat_to_rotmat, box_to_corners


def render_point_heatmap(level_centers, level_batch_idx, points_per_batch, sigma):
    """CenterNet/CenterPoint-style soft Gaussian heatmap target, rendered from
    an arbitrary set of world-space target points per batch item -- adapted
    from models/losses.py::render_gaussian_targets (this project's original
    pre-FCAF3D dense head already used exactly this mechanism; reused here
    rather than reimplemented, just taking already-known world centers
    directly instead of re-deriving them from grid coords + pc_range).
    General enough to render either GT centers (1 point/box -> the existence
    heatmap) or GT/predicted box corners (8 points/box -> the corner
    auxiliary heatmap, see compute_fcaf3d_loss's CORNER_AUX term).

    level_centers: (N,3) world-space voxel centers. level_batch_idx: (N,).
    points_per_batch: list[B] of (Mi,3) world points. Returns (N,) in [0,1]:
    for each voxel, the max over every point in its own batch sample of
    exp(-dist^2 / (2*sigma^2)) -- NOT clamped to 1 at any "true" positive
    here; callers that need that (e.g. the existence heatmap, at assigned
    positives) do it themselves by overwriting afterward."""
    heat = torch.zeros(level_centers.shape[0], device=level_centers.device)
    for b, pts in enumerate(points_per_batch):
        if pts.shape[0] == 0:
            continue
        bmask = level_batch_idx == b
        if not bmask.any():
            continue
        pts = pts.to(level_centers.device)
        vw = level_centers[bmask]
        dist2 = ((vw[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        g = torch.exp(-dist2 / (2 * sigma ** 2))
        heat[bmask] = torch.maximum(heat[bmask], g.max(dim=1).values)
    return heat


def render_center_heatmap(level_centers, level_batch_idx, gt_boxes_list, sigma):
    """render_point_heatmap specialized to GT box centers (1 point/box) --
    see that function's docstring for the shared mechanism."""
    return render_point_heatmap(level_centers, level_batch_idx,
                                 [gt_boxes[:, :3] for gt_boxes in gt_boxes_list], sigma)


def render_gt_corner_heatmap(level_centers, level_batch_idx, gt_boxes_list, sigma):
    """render_point_heatmap specialized to GT box CORNERS (8 points/box) --
    the "corner auxiliary" target (models/losses.py's original corner_aux
    term, ported here): an extra supervision signal on box EXTENT, not just
    center, rendered the same Gaussian-heatmap way. gt_boxes_list: list[B] of
    (Mb,10) [center(3),size(3),quat(4)] -- box_to_corners needs quat
    directly, which GT already stores natively (no matrix round-trip)."""
    corners_per_batch = []
    for gt_boxes in gt_boxes_list:
        if gt_boxes.shape[0] == 0:
            corners_per_batch.append(torch.zeros(0, 3, device=level_centers.device))
        else:
            c = box_to_corners(gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6:10])  # (Mb,8,3)
            corners_per_batch.append(c.reshape(-1, 3))
    return render_point_heatmap(level_centers, level_batch_idx, corners_per_batch, sigma)


def render_point_heatmap_channels(level_centers, level_batch_idx, points_per_batch, sigma):
    """Multi-channel variant of render_point_heatmap: each batch item's target
    points carry an extra channel dimension (e.g. "which of the 8 box
    corners"), and ALL channels are computed in one broadcast per batch item.
    (An earlier version called render_point_heatmap once per channel in a
    python loop -- same total FLOPs, but 8x the kernel launches/allocator
    calls, which measurably stalled training near this GPU's 8GB ceiling.
    This does the whole (Nb, Mi, C) distance tensor in one shot per batch
    item instead.) points_per_batch: list[B] of (Mi, C, 3). Returns (N, C)."""
    # Every batch item's tensor is (Mi,C,3) -- C (e.g. 8 corners) is defined
    # even when Mi=0 (no GT boxes in that frame), so just read it off the
    # first entry rather than hunting for a non-empty one (which can fail
    # outright on a validation batch where EVERY frame happens to have no GT).
    assert len(points_per_batch) > 0, "points_per_batch must have at least one (possibly empty) batch item"
    n_channels = points_per_batch[0].shape[1]
    heat = torch.zeros(level_centers.shape[0], n_channels, device=level_centers.device)
    for b, pts in enumerate(points_per_batch):
        if pts.shape[0] == 0:
            continue
        bmask = level_batch_idx == b
        if not bmask.any():
            continue
        pts = pts.to(level_centers.device)                                       # (Mi,C,3)
        vw = level_centers[bmask]                                                 # (Nb,3)
        dist2 = ((vw[:, None, None, :] - pts[None, :, :, :]) ** 2).sum(-1)         # (Nb,Mi,C)
        g = torch.exp(-dist2 / (2 * sigma ** 2))
        heat[bmask] = torch.maximum(heat[bmask], g.max(dim=1).values)             # (Nb,C)
    return heat


def render_gt_corner_heatmap_identity(level_centers, level_batch_idx, gt_boxes_list, sigma):
    """RTM3D-style ablation of render_gt_corner_heatmap: that function pools
    all 8 corners of every GT box into ONE combined max-heatmap, so a voxel
    near ANY corner of ANY box lights up the same channel -- corner IDENTITY
    (which of the 8 box-local corners it is) is thrown away. RTM3D (Li et al.,
    ECCV 2020) instead predicts its 9 box keypoints (8 vertices + center) as
    SEPARATE heatmap channels, one per keypoint identity. This mirrors that,
    via render_point_heatmap_channels (box_utils._SIGNS order, shared with
    rotation6d._CORNER_SIGNS so GT and predicted corners stay index-aligned).
    Returns (8, N)."""
    corners_per_batch = []
    for gt_boxes in gt_boxes_list:
        if gt_boxes.shape[0] == 0:
            corners_per_batch.append(torch.zeros(0, 8, 3, device=level_centers.device))
        else:
            corners_per_batch.append(box_to_corners(gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6:10]))  # (Mb,8,3)
    heat = render_point_heatmap_channels(level_centers, level_batch_idx, corners_per_batch, sigma)  # (N,8)
    return heat.transpose(0, 1)  # (8,N) -- matches the combined version's (N,) convention, channel-major


def assign_multilevel(level_world_centers, level_batch_idx, gt_boxes_list, batch_size,
                       min_locations=6, center_sample_radius=0.5, nearest_fallback_k=0,
                       gaussian_sigma=None, corner_identity=False):
    """level_world_centers: list[L] of (N_l,3) world-space voxel centers, fine-to-coarse.
    level_batch_idx: list[L] of (N_l,) which sample each voxel belongs to.
    gt_boxes_list: list[B] of (Mb,10) [center(3),size(3),quat(4)].
    Returns list[L] of dicts, each with (N_l,*) tensors: pos_mask (bool),
    center, log_size, rot_matrix, centerness -- only meaningful where
    pos_mask is True, zero-filled elsewhere."""
    num_levels = len(level_world_centers)
    device = level_world_centers[0].device

    targets = []
    for level_centers, level_bidx in zip(level_world_centers, level_batch_idx):
        n = level_centers.shape[0]
        tgt = {
            "pos_mask": torch.zeros(n, dtype=torch.bool, device=device),
            "gt_volume": torch.full((n,), float("inf"), device=device),
            "center": torch.zeros(n, 3, device=device),
            "log_size": torch.zeros(n, 3, device=device),
            "rot_matrix": torch.eye(3, device=device)[None].expand(n, -1, -1).clone(),
            "centerness": torch.zeros(n, device=device),
        }
        if gaussian_sigma is not None:
            tgt["exist_heatmap"] = render_center_heatmap(level_centers, level_bidx, gt_boxes_list, gaussian_sigma)
            if corner_identity:
                tgt["gt_corner_heatmap_identity"] = render_gt_corner_heatmap_identity(
                    level_centers, level_bidx, gt_boxes_list, gaussian_sigma)
            else:
                tgt["gt_corner_heatmap"] = render_gt_corner_heatmap(level_centers, level_bidx, gt_boxes_list, gaussian_sigma)
        targets.append(tgt)

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

            used_nearest_fallback = False
            if chosen is None and nearest_fallback_k > 0:
                # TR3D-style rescue: no level has any voxel inside this box at
                # all -- instead of dropping the GT, take the K nearest fine-
                # level voxels (by raw center distance) regardless of containment.
                fine_mask_b = level_batch_idx[0] == b
                fine_idx_b = fine_mask_b.nonzero(as_tuple=True)[0]
                if fine_idx_b.numel() > 0:
                    dists = (level_world_centers[0][fine_idx_b] - center[None, :]).norm(dim=-1)
                    k = min(nearest_fallback_k, fine_idx_b.numel())
                    nearest = dists.topk(k, largest=False).indices
                    pos_idx = fine_idx_b[nearest]
                    chosen = 0
                    used_nearest_fallback = True
            if chosen is None:
                continue  # no level has any voxel inside this box at all, and no fallback configured

            if not used_nearest_fallback:
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
            if gaussian_sigma is not None:
                tgt["exist_heatmap"][win_idx] = 1.0

            local_win = (level_world_centers[chosen][win_idx] - center[None, :]) @ R
            d_lo = (half[None, :] + local_win).clamp(min=1e-6)
            d_hi = (half[None, :] - local_win).clamp(min=1e-6)
            ratio = torch.minimum(d_lo, d_hi) / torch.maximum(d_lo, d_hi)
            tgt["centerness"][win_idx] = ratio.clamp(min=1e-6).prod(dim=-1).pow(1.0 / 3.0)

    return targets
