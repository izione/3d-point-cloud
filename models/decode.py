import torch

from .box_utils import quat_normalize


def find_local_peaks(scores, coords, index_grid, grid_size, threshold):
    """A voxel is a peak if its score is >= every one of its 26 neighbors
    and above `threshold` -- the same NMS-free local-max trick CenterNet
    uses (3x3x3 max-pool), reimplemented on our sparse index_grid so no
    separate post-processing/NMS step is needed."""
    X, Y, Z = grid_size
    n = coords.shape[0]
    is_peak = torch.ones(n, dtype=torch.bool, device=scores.device)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nb = coords.clone()
                nb[:, 1] += dx
                nb[:, 2] += dy
                nb[:, 3] += dz
                valid = (nb[:, 1] >= 0) & (nb[:, 1] < X) & (nb[:, 2] >= 0) & (nb[:, 2] < Y) & (nb[:, 3] >= 0) & (nb[:, 3] < Z)
                nb_idx = torch.full((n,), -1, dtype=torch.long, device=scores.device)
                if valid.any():
                    v = nb[valid]
                    nb_idx[valid] = index_grid[v[:, 0], v[:, 1], v[:, 2], v[:, 3]]
                has_nb = nb_idx >= 0
                nb_scores = torch.zeros_like(scores)
                nb_scores[has_nb] = scores[nb_idx[has_nb]]
                is_peak &= scores >= nb_scores
    return is_peak & (scores > threshold)


def decode_detections(pred, coords, index_grid, grid_size, pc_range, effective_voxel_size, score_threshold=0.1):
    """Returns a list[B] of dicts with 'center','size','quat','score' tensors
    for the detections in each batch item. No NMS -- local-max peaks only."""
    device = coords.device
    scores = torch.sigmoid(pred["center_logit"].squeeze(-1))
    peak_mask = find_local_peaks(scores, coords, index_grid, grid_size, score_threshold)

    voxel_world = pc_range[:3].to(device) + (coords[:, 1:4].float() + 0.5) * effective_voxel_size.to(device)
    center = voxel_world + pred["offset"]
    size = pred["box"][:, :3].exp()
    quat = quat_normalize(pred["box"][:, 3:7])

    batch_size = int(coords[:, 0].max().item()) + 1 if coords.shape[0] > 0 else 0
    results = []
    for b in range(batch_size):
        m = peak_mask & (coords[:, 0] == b)
        results.append({
            "center": center[m].detach().cpu(),
            "size": size[m].detach().cpu(),
            "quat": quat[m].detach().cpu(),
            "score": scores[m].detach().cpu(),
        })
    return results
