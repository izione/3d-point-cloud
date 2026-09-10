"""voxelize.py - point cloud (N,4)[x,y,z,intensity] -> hard voxel representation.

VoxelNet paper Sec.2.3 (Efficient Implementation) procedure: shuffle points,
keep up to T per voxel (random sampling), augment each point with its
voxel's centroid offset. `np.unique(..., axis=0)` groups voxel coordinates
the same way the paper's O(1) hash-table lookup would -- just a different
(vectorized-numpy) implementation, fast enough at this dataset's per-frame
point count (~6-9k) without needing a CUDA hash table.

Ported unchanged from the sibling voxelnet_baseline repo's voxelize.py
(only voxelize_polar, unused here, is dropped).
"""
import numpy as np


def voxelize(points: np.ndarray, pc_range, voxel_size, max_points_per_voxel: int, max_voxels: int):
    """points: (N,4) float32 [x,y,z,intensity].

    Returns:
      voxel_features: (K,T,4) float32, zero-padded
      voxel_coords:   (K,3) int64 [z_idx,y_idx,x_idx] (D,H,W order, matches conv3d tensor axes)
      num_points:     (K,) int64, actual point count per voxel (<=T)
    K = number of actual non-empty voxels (<=max_voxels), varies per frame.
    """
    pc_range = np.asarray(pc_range, dtype=np.float32)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    grid_size = np.round((pc_range[3:6] - pc_range[0:3]) / voxel_size).astype(np.int64)

    xyz = points[:, :3]
    in_range = np.all((xyz >= pc_range[0:3]) & (xyz < pc_range[3:6]), axis=1)
    points = points[in_range]
    if len(points) == 0:
        return (np.zeros((0, max_points_per_voxel, 4), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.int64))

    perm = np.random.permutation(len(points))
    points = points[perm]

    idx_xyz = np.floor((points[:, :3] - pc_range[0:3]) / voxel_size).astype(np.int64)
    idx_xyz = np.clip(idx_xyz, 0, grid_size - 1)
    idx_zyx = idx_xyz[:, [2, 1, 0]]

    voxel_coords, inverse, counts = np.unique(idx_zyx, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    if len(voxel_coords) > max_voxels:
        keep_voxel = np.zeros(len(voxel_coords), dtype=bool)
        keep_voxel[:max_voxels] = True
        keep_point = keep_voxel[inverse]
        points, inverse = points[keep_point], inverse[keep_point]
        voxel_coords, counts = voxel_coords[:max_voxels], counts[:max_voxels]

    K = len(voxel_coords)
    num_points = np.minimum(counts, max_points_per_voxel).astype(np.int64)

    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    slot_in_voxel = np.arange(len(order)) - np.searchsorted(sorted_inverse, sorted_inverse, side="left")
    slot = np.empty(len(order), dtype=np.int64)
    slot[order] = slot_in_voxel

    voxel_features = np.zeros((K, max_points_per_voxel, 4), dtype=np.float32)
    valid = slot < max_points_per_voxel
    voxel_features[inverse[valid], slot[valid]] = points[valid]

    return voxel_features, voxel_coords, num_points


def augment_with_centroid_offset(voxel_features: np.ndarray, num_points: np.ndarray) -> np.ndarray:
    """(K,T,4) -> (K,T,7): [x,y,z,intensity, x-vx,y-vy,z-vz]. vx,vy,vz = mean
    (x,y,z) over the voxel's valid points (VoxelNet paper Sec.2.1.1's
    "relative offset w.r.t. the centroid")."""
    K, T, _ = voxel_features.shape
    mask = np.arange(T)[None, :] < num_points[:, None]
    xyz = voxel_features[:, :, :3]
    denom = np.maximum(num_points, 1).astype(np.float32)[:, None]
    centroid = (xyz * mask[:, :, None]).sum(axis=1) / denom
    offset = (xyz - centroid[:, None, :]) * mask[:, :, None]
    out = np.concatenate([voxel_features, offset], axis=2)
    out *= mask[:, :, None]
    return out.astype(np.float32)
