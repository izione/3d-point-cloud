"""heatmap_targets.py - CenterPoint-style target generation and decode.

Grid is CenterHead's own output resolution (config.HEAD_GRID_SIZE/HEAD_STRIDE
-- the RPN backbone's own output, not a re-derived head-specific grid).

Regression targets supervise ONLY the single grid cell the GT center rounds
to (CenterPoint standard convention): offset (sub-cell correction), z
(absolute, no anchor), dim (log l/w/h), rot (6D continuous rotation).
"""
import math

import numpy as np

import config
import rotation3d


def gaussian_radius(height: float, width: float, min_overlap: float = None, tau: float = None) -> float:
    """CornerNet (Law & Deng 2018) standard formula: largest radius r such
    that a Gaussian peak of that radius overlaps the true box by at least
    min_overlap IoU (min of the 3 corner-aligned cases, per the reference
    implementation).

    tau: CenterPoint (Yin et al. 2021) Sec.4.1's own minimum-radius floor
    (paper default 2.0) -- "the object distribution is sparser in map-view
    than image-view, making the supervisory signal too sparse". Our diver
    boxes are small (0.5-1.8m) at this grid's 0.2m/cell, so the raw formula
    alone gives near-zero radii (effectively single-point supervision)
    without this floor."""
    min_overlap = config.GAUSSIAN_MIN_OVERLAP if min_overlap is None else min_overlap
    tau = config.GAUSSIAN_RADIUS_TAU if tau is None else tau

    a1, b1 = 1, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0))
    r1 = (b1 + sq1) / 2

    a2, b2 = 4, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0))
    r2 = (b2 + sq2) / 2

    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0))
    r3 = (b3 + sq3) / 2

    return max(min(r1, r2, r3), tau)


def draw_gaussian(heatmap: np.ndarray, center_col: float, center_row: float, radius: float):
    """In-place draw of a Gaussian peak at (center_row,center_col) onto
    heatmap(H,W) -- takes the max with existing values (CenterNet convention:
    where two objects' Gaussians overlap, the larger one wins)."""
    radius = max(int(round(radius)), 0)
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    gaussian = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma + 1e-9))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0

    x, y = int(round(center_col)), int(round(center_row))
    height, width = heatmap.shape
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def build_heatmap_targets(objects: list, min_overlap: float = None) -> dict:
    """objects: raw label dicts (centroid/dimensions/rotation_x/y/z).

    Returns (all on grid (H,W)=config.HEAD_GRID_SIZE[::-1]):
      heatmap: (1,H,W) float32, 0-1
      reg_mask: (H,W) bool -- exactly the cell(s) to supervise regression on
      offset: (H,W,2) float32 [dx,dy] sub-cell correction (cell-size units)
      z: (H,W,1) float32 absolute z (m)
      dim: (H,W,3) float32 [log(l),log(w),log(h)]
      rot: (H,W,6) float32 -- full 3D rotation (rotation3d.matrix_to_6d),
      absolute GT value (no anchor, so no residual to take).

    Note (deliberate scope limit, matches CenterPoint's own approximation):
    the Gaussian radius uses only the object's local length/width in BEV --
    it's a soft heuristic for how widely to spread positive supervision, not
    an exact footprint; only the cell's regression TARGETS (dim/rot) reflect
    the true tilted 3D shape."""
    W, H = config.HEAD_GRID_SIZE
    sx, sy = config.HEAD_STRIDE
    x0, y0 = config.POINT_CLOUD_RANGE[0], config.POINT_CLOUD_RANGE[1]

    heatmap = np.zeros((1, H, W), dtype=np.float32)
    reg_mask = np.zeros((H, W), dtype=bool)
    offset = np.zeros((H, W, 2), dtype=np.float32)
    z_t = np.zeros((H, W, 1), dtype=np.float32)
    dim_t = np.zeros((H, W, 3), dtype=np.float32)
    rot_t = np.zeros((H, W, 6), dtype=np.float32)

    for o in objects:
        c, d = o["centroid"], o["dimensions"]
        gx_cell = (c["x"] - x0) / sx - 0.5
        gy_cell = (c["y"] - y0) / sy - 0.5
        col, row = int(round(gx_cell)), int(round(gy_cell))
        if not (0 <= col < W and 0 <= row < H):
            continue  # rare out-of-range center -- POINT_CLOUD_RANGE comfortably wraps GT

        l_cells, w_cells = d["length"] / sx, d["width"] / sy
        radius = gaussian_radius(w_cells, l_cells, min_overlap)
        draw_gaussian(heatmap[0], gx_cell, gy_cell, radius)

        reg_mask[row, col] = True
        offset[row, col] = [gx_cell - col, gy_cell - row]
        z_t[row, col, 0] = c["z"]
        dim_t[row, col] = [np.log(d["length"]), np.log(d["width"]), np.log(d["height"])]
        rx, ry, rz = o.get("rotation_x", 0.0), o.get("rotation_y", 0.0), o.get("rotation_z", 0.0)
        R = rotation3d.euler_to_matrix(rx, ry, rz)
        rot_t[row, col] = rotation3d.matrix_to_6d(R)

    return {"heatmap": heatmap, "reg_mask": reg_mask, "offset": offset, "z": z_t, "dim": dim_t, "rot": rot_t}


def decode_center_boxes(heatmap_pred, offset_pred, z_pred, dim_pred, rot_pred,
                         score_thresh: float = None, max_peaks: int = None) -> list:
    """heatmap_pred: (1,H,W) sigmoid probability (already activated).
    offset/z/dim_pred: (H,W,C). rot_pred: (H,W,6). All numpy -- convert torch
    tensors with .cpu().numpy() before calling.

    3x3 max-pool peak-finding (NMS-free, CenterPoint's own decode) -- a cell
    is a candidate only if it's the max within its own 3x3 neighborhood.
    Returns list of dict {score,x,y,z,l,w,h,theta,R} -- theta is R's z-yaw
    component (BEV-NMS-compatible approximation); R(3,3) is the real rotation
    matrix, used directly for 3D IoU evaluation."""
    score_thresh = config.VAL_SCORE_THRESH if score_thresh is None else score_thresh
    max_peaks = config.MAX_PEAKS if max_peaks is None else max_peaks

    hm = heatmap_pred[0]
    H, W = hm.shape
    padded = np.full((H + 2, W + 2), -1.0, dtype=hm.dtype)
    padded[1:-1, 1:-1] = hm
    is_peak = np.ones((H, W), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            is_peak &= hm >= padded[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]

    rows, cols = np.where(is_peak & (hm >= score_thresh))
    if len(rows) > max_peaks:
        top = np.argsort(-hm[rows, cols])[:max_peaks]
        rows, cols = rows[top], cols[top]

    sx, sy = config.HEAD_STRIDE
    x0, y0 = config.POINT_CLOUD_RANGE[0], config.POINT_CLOUD_RANGE[1]

    boxes = []
    for row, col in zip(rows, cols):
        dx, dy = offset_pred[row, col]
        gx_cell, gy_cell = col + dx, row + dy
        x = x0 + (gx_cell + 0.5) * sx
        y = y0 + (gy_cell + 0.5) * sy
        z = float(z_pred[row, col, 0])
        l, w, h = np.exp(dim_pred[row, col])
        R = rotation3d.sixd_to_matrix_np(rot_pred[row, col])
        theta = float(np.arctan2(R[1, 0], R[0, 0]))
        boxes.append({"score": float(hm[row, col]), "x": float(x), "y": float(y), "z": z,
                      "l": float(l), "w": float(w), "h": float(h), "theta": theta, "R": R})
    return boxes
