"""eval.py - AP3D evaluation, VOC2010-style all-point AP, exact 3D OBB-OBB
IoU (Monte-Carlo). Matches the sibling voxelnet_baseline repo's own
eval_voxelnet.py definitions so numbers are directly comparable across the
whole ablation program.
"""
import numpy as np
from shapely.geometry import Polygon

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import config
import rotation3d
import heatmap_targets as ht

_LOCAL_CORNERS_UNIT = np.array([
    [-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1],
    [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1],
], dtype=np.float64)


def obb_corners(center: np.ndarray, dims: np.ndarray, R: np.ndarray) -> np.ndarray:
    """(8,3) world corners -- center(3,), dims(3,)=[l,w,h], R(3,3) local->world."""
    return _LOCAL_CORNERS_UNIT * (dims / 2) @ R.T + center


def point_in_obb(points: np.ndarray, center: np.ndarray, dims: np.ndarray, R: np.ndarray) -> np.ndarray:
    """points: (N,3) -> (N,) bool."""
    local = (points - center) @ R
    half = dims / 2
    return np.all(np.abs(local) <= half, axis=1)


def iou_3d_obb(center_a, dims_a, R_a, center_b, dims_b, R_b,
                n_samples: int = 8000, rng: np.random.Generator = None) -> float:
    """Exact 3D OBB-OBB IoU via Monte-Carlo: box volumes are exact (l*w*h),
    only the intersection volume is Monte-Carlo-estimated over the union AABB
    (much lower variance than sampling the whole IoU)."""
    if not (np.all(np.isfinite(center_a)) and np.all(np.isfinite(dims_a))
            and np.all(np.isfinite(center_b)) and np.all(np.isfinite(dims_b))):
        return 0.0  # guards against exp() overflow in early, undertrained dim_head outputs

    rng = rng or np.random.default_rng()
    vol_a = float(np.prod(dims_a))
    vol_b = float(np.prod(dims_b))
    if vol_a <= 0 or vol_b <= 0:
        return 0.0

    corners_a = obb_corners(center_a, dims_a, R_a)
    corners_b = obb_corners(center_b, dims_b, R_b)
    lo = np.minimum(corners_a.min(axis=0), corners_b.min(axis=0))
    hi = np.maximum(corners_a.max(axis=0), corners_b.max(axis=0))
    if np.any(hi <= lo) or not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return 0.0
    sample_vol = float(np.prod(hi - lo))

    pts = rng.uniform(lo, hi, size=(n_samples, 3))
    in_a = point_in_obb(pts, center_a, dims_a, R_a)
    in_b = point_in_obb(pts, center_b, dims_b, R_b)
    inter_frac = float(np.mean(in_a & in_b))
    inter_vol = inter_frac * sample_vol
    union_vol = vol_a + vol_b - inter_vol
    return float(inter_vol / union_vol) if union_vol > 0 else 0.0


def gt_obb_from_row(row: np.ndarray):
    """row: (13,) [x,y,z,l,w,h,theta_z_rad,6D(6)] -> (center(3,), dims(3,), R(3,3))."""
    center = row[0:3]
    dims = row[3:6]
    R = rotation3d.sixd_to_matrix_np(row[7:13])
    return center, dims, R


def pred_obb_from_box(box: dict):
    """decode_center_boxes() box dict -> (center(3,), dims(3,), R(3,3))."""
    center = np.array([box["x"], box["y"], box["z"]])
    dims = np.array([box["l"], box["w"], box["h"]])
    return center, dims, box["R"]


def rotated_rect_corners(x: float, y: float, l: float, w: float, theta: float) -> np.ndarray:
    """(4,2) -- center(x,y), length l (local x), width w (local y), theta (rad, z-rotation)."""
    hl, hw = l / 2, w / 2
    local = np.array([[-hl, -hw], [-hl, hw], [hl, hw], [hl, -hw]])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([x, y])


def rotated_nms(boxes: list, iou_thresh: float = 0.1) -> list:
    """Greedy NMS on shapely BEV polygon IoU (z-yaw footprint approximation)."""
    boxes = sorted(boxes, key=lambda b: -b["score"])
    polys = [Polygon(rotated_rect_corners(b["x"], b["y"], b["l"], b["w"], b["theta"])) for b in boxes]
    keep = []
    suppressed = [False] * len(boxes)
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(boxes[i])
        if not polys[i].is_valid or polys[i].area <= 0:
            continue
        for j in range(i + 1, len(boxes)):
            if suppressed[j] or not polys[j].is_valid or polys[j].area <= 0:
                continue
            inter = polys[i].intersection(polys[j]).area
            union = polys[i].area + polys[j].area - inter
            iou = inter / union if union > 0 else 0.0
            if iou > iou_thresh:
                suppressed[j] = True
    return keep


def compute_ap(detections: list, n_gt: int):
    """VOC2010-style all-point AP. detections: list of (confidence, is_tp)."""
    if n_gt == 0 or not detections:
        return 0.0, 0.0, 0.0
    confs = np.array([d[0] for d in detections])
    is_tp = np.array([d[1] for d in detections])
    order = np.argsort(-confs)
    is_tp = is_tp[order]

    tp_cum = np.cumsum(is_tp)
    fp_cum = np.cumsum(~is_tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, float(precision[-1]) if len(precision) else 0.0, float(recall[-1]) if len(recall) else 0.0


def _score_frame(boxes: list, gt_boxes: np.ndarray, detections: dict, score_thresh: float, nms_iou: float):
    """Filter `boxes` (already decoded at the lowest threshold) by score_thresh
    -> NMS -> GT IoU matching. Appends (conf, is_tp) into detections[iou_thr]."""
    filtered = [b for b in boxes if b["score"] >= score_thresh]
    filtered = rotated_nms(filtered, iou_thresh=nms_iou)

    gt_list = [gt_obb_from_row(row) for row in gt_boxes]
    pred_list = [pred_obb_from_box(b) for b in filtered]
    pred_conf = [b["score"] for b in filtered]

    n_p, n_g = len(pred_list), len(gt_list)
    iou_mat = np.zeros((n_p, n_g))
    rng = np.random.default_rng(0)  # fixed per-frame seed -- eval reproducibility
    for pi, (pc, pd, pR) in enumerate(pred_list):
        for gi, (gc, gd, gR) in enumerate(gt_list):
            iou_mat[pi, gi] = iou_3d_obb(pc, pd, pR, gc, gd, gR, rng=rng)

    order = np.argsort(-np.array(pred_conf)) if n_p else np.array([], dtype=int)
    for iou_thr in config.IOU_THRESHOLDS:
        matched_gt = set()
        for pi in order:
            candidate_gt = [gi for gi in range(n_g) if gi not in matched_gt]
            is_tp = False
            if candidate_gt:
                ious = iou_mat[pi, candidate_gt]
                best_local = int(np.argmax(ious))
                best_gt, best_iou = candidate_gt[best_local], ious[best_local]
                if best_iou >= iou_thr:
                    matched_gt.add(best_gt)
                    is_tp = True
            detections[iou_thr].append((pred_conf[pi], is_tp))


def evaluate_full(model, device, val_dataset, score_thresh: float = None, nms_iou: float = None):
    """Runs the model over EVERY frame in val_dataset (no subsampling --
    CenterPoint's decode is anchor-free/NMS-free so this is cheap even for an
    undertrained model, unlike an anchor head's unbounded pre-NMS candidate
    count). Returns {iou_thr: (ap,precision,recall)}."""
    import torch
    from dataset import collate_fn
    from torch.utils.data import DataLoader

    score_thresh = config.VAL_SCORE_THRESH if score_thresh is None else score_thresh
    nms_iou = config.VAL_NMS_IOU if nms_iou is None else nms_iou

    model.eval()
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
    detections = {thr: [] for thr in config.IOU_THRESHOLDS}
    n_gt_total = 0

    iterable = tqdm(loader, desc="val (full)", unit="frame") if HAS_TQDM else loader
    with torch.no_grad():
        for batch in iterable:
            voxel_features = batch["voxel_features"].to(device)
            num_points = batch["num_points"].to(device)
            coords = batch["coords"].to(device)
            gt_boxes = batch["gt_boxes"][0].numpy()
            n_gt_total += len(gt_boxes)

            pred = model(voxel_features, num_points, coords)
            hm_np = torch.sigmoid(pred["heatmap"][0]).cpu().numpy()
            boxes = ht.decode_center_boxes(
                hm_np, pred["offset"][0].permute(1, 2, 0).cpu().numpy(),
                pred["z"][0].permute(1, 2, 0).cpu().numpy(), pred["dim"][0].permute(1, 2, 0).cpu().numpy(),
                pred["rot"][0].permute(1, 2, 0).cpu().numpy(), score_thresh=score_thresh)
            _score_frame(boxes, gt_boxes, detections, score_thresh, nms_iou)

    model.train()
    return {thr: compute_ap(detections[thr], n_gt_total) for thr in config.IOU_THRESHOLDS}
