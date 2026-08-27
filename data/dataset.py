import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _label_path_for(sonar_path: Path) -> Path:
    return sonar_path.parent.parent / "labels" / (sonar_path.stem + ".json")


def _list_scene_frames(root: Path, scene_ids: list) -> list:
    frames = []
    for scene_id in scene_ids:
        sonar_dir = root / scene_id / "sonar"
        if not sonar_dir.is_dir():
            raise FileNotFoundError(f"expected sonar dir at {sonar_dir}")
        frames.extend(sorted(sonar_dir.glob("frame_*.bin")))
    return frames


class SonarDiverDataset(Dataset):
    """Reads raw sonar .bin frames + JSON labels for the 4-person scene split.

    TEST is a genuine held-out set of whole scenes (DATA.TEST_SCENES) - no frame
    from a test scene ever appears in train/val, so it measures generalization to
    an unseen scene, not just an unseen frame. TRAIN/VAL are instead a frame-level
    random split of DATA.TRAINVAL_SCENES pooled together (deterministic, seeded by
    DATA.SPLIT_SEED): every non-test scene contributes frames to both train and
    val, so val loss reflects "how well is training going" across the full pose/
    scene diversity, rather than "how well does it generalize to a couple of
    held-out scenes" (that's what TEST is for). DATA.VAL_FRAME_RATIO controls the
    split fraction (default 0.1).

    Each item returns raw points (already filtered to the point-cloud range) and
    GT boxes as (cx, cy, cz, length, width, height, qw, qx, qy, qz). Voxelization
    and VFE feature construction happen later (collate_fn + model), not here, so
    this class stays a thin, easily-testable IO layer.
    """

    def __init__(self, cfg: dict, split: str):
        assert split in ("train", "val", "test")
        self.cfg = cfg
        self.split = split
        self.root = Path(cfg["DATA"]["ROOT"])
        self.pc_range = np.array(cfg["DATA"]["POINT_CLOUD_RANGE"], dtype=np.float32)

        if split == "test":
            self.samples = _list_scene_frames(self.root, cfg["DATA"]["TEST_SCENES"])
        else:
            all_frames = _list_scene_frames(self.root, cfg["DATA"]["TRAINVAL_SCENES"])
            all_frames.sort(key=lambda p: str(p))  # deterministic order before shuffling
            rng = random.Random(cfg["DATA"].get("SPLIT_SEED", 42))
            rng.shuffle(all_frames)
            val_ratio = cfg["DATA"].get("VAL_FRAME_RATIO", 0.1)
            n_val = round(len(all_frames) * val_ratio)
            self.samples = all_frames[:n_val] if split == "val" else all_frames[n_val:]

        if len(self.samples) == 0:
            raise RuntimeError(f"no frames found for split={split}")

    def __len__(self):
        return len(self.samples)

    def _load_points(self, sonar_path: Path) -> np.ndarray:
        data = np.fromfile(sonar_path, dtype=np.float32)
        if data.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        pts = data.reshape(-1, 4)
        mask = (
            (pts[:, 0] >= self.pc_range[0]) & (pts[:, 0] <= self.pc_range[3]) &
            (pts[:, 1] >= self.pc_range[1]) & (pts[:, 1] <= self.pc_range[4]) &
            (pts[:, 2] >= self.pc_range[2]) & (pts[:, 2] <= self.pc_range[5])
        )
        return pts[mask]

    def _load_gt_boxes(self, sonar_path: Path) -> np.ndarray:
        label_path = _label_path_for(sonar_path)
        if not label_path.is_file():
            return np.zeros((0, 10), dtype=np.float32)
        with open(label_path) as f:
            label = json.load(f)
        objs = label.get("objects", [])
        boxes = np.zeros((len(objs), 10), dtype=np.float32)
        for i, obj in enumerate(objs):
            c, d, q = obj["centroid"], obj["dimensions"], obj["quaternion"]
            boxes[i] = [
                c["x"], c["y"], c["z"],
                d["length"], d["width"], d["height"],
                q["w"], q["x"], q["y"], q["z"],
            ]
        return boxes

    def __getitem__(self, idx):
        sonar_path = self.samples[idx]
        points = self._load_points(sonar_path)
        gt_boxes = self._load_gt_boxes(sonar_path)
        return {
            "points": torch.from_numpy(points),
            "gt_boxes": torch.from_numpy(gt_boxes),
            "frame_id": str(sonar_path.relative_to(self.root)),
        }


def voxelize_points(points: torch.Tensor, pc_range: torch.Tensor, voxel_size: torch.Tensor, grid_size: torch.Tensor):
    """points: (N,4) -> voxel_coords_xyz (N,3) long, clipped to grid."""
    xyz = points[:, :3]
    coords = torch.floor((xyz - pc_range[:3]) / voxel_size).long()
    coords = torch.clamp(coords, torch.zeros(3, dtype=torch.long, device=coords.device), grid_size - 1)
    return coords


def voxelize_batch(points, point_batch_idx, pc_range, voxel_size, grid_size):
    """The part of batch prep that benefits from running on GPU: per-point coord
    computation + torch.unique. Call this *after* moving points/point_batch_idx
    to the training device -- keeping it out of collate_fn is what lets a CPU
    DataLoader worker just concatenate tensors (cheap) instead of doing this
    unique() on CPU, which was the actual bottleneck (measured: GPU sat at ~33%
    util / dataloading-bound, not compute-bound)."""
    voxel_coords_xyz = voxelize_points(points, pc_range, voxel_size, grid_size)
    voxel_key = torch.cat([point_batch_idx.unsqueeze(1), voxel_coords_xyz], dim=1)
    uniq_voxel_coords, point_voxel_idx = torch.unique(voxel_key, dim=0, return_inverse=True)
    return uniq_voxel_coords, point_voxel_idx


def collate_fn(batch):
    """Cheap, CPU-side: just concatenate points/gt_boxes/frame_ids and tag each
    point with which sample it came from. No coordinate math, no torch.unique --
    see voxelize_batch() for the GPU-side part of batch prep."""
    all_points = []
    all_batch_idx = []
    gt_boxes_list = []
    frame_ids = []

    for b, sample in enumerate(batch):
        pts = sample["points"]
        if pts.shape[0] > 0:
            all_points.append(pts)
            all_batch_idx.append(torch.full((pts.shape[0],), b, dtype=torch.long))
        gt_boxes_list.append(sample["gt_boxes"])
        frame_ids.append(sample["frame_id"])

    points = torch.cat(all_points, dim=0) if all_points else torch.zeros((0, 4))
    point_batch_idx = torch.cat(all_batch_idx, dim=0) if all_batch_idx else torch.zeros((0,), dtype=torch.long)

    return {
        "points": points,                    # (Ntot, 4)
        "point_batch_idx": point_batch_idx,  # (Ntot,)
        "gt_boxes": gt_boxes_list,            # list[B] of (Nobj_b, 10)
        "frame_ids": frame_ids,
        "batch_size": len(batch),
    }
