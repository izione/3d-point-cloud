import json
import os
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

    TRAIN/VAL/TEST are all genuine held-out sets of WHOLE SCENES (DATA.TRAIN_SCENES/
    VAL_SCENES/TEST_SCENES) -- no frame from a val or test scene ever appears in
    train, so every split measures generalization to unseen scenes, not just
    unseen frames within an otherwise-seen scene. This is the canonical split
    from the sibling eugene/SonarVoxNet project's tools/prepare_data.py (see
    configs/splits/scene_split.json for the reference copy) -- both projects use
    the exact same scene assignment. Replaced 2026-09-26 a previous frame-level
    random split of a pooled TRAINVAL_SCENES set (every non-test scene
    contributing frames to both train and val); every result from before that
    date was measured under the OLD split and isn't directly comparable to runs
    after it.

    Each item returns raw points (already filtered to the point-cloud range) and
    GT boxes as (cx, cy, cz, length, width, height, qw, qx, qy, qz). Voxelization
    and VFE feature construction happen later (collate_fn + model), not here, so
    this class stays a thin, easily-testable IO layer.
    """

    _SPLIT_KEY = {"train": "TRAIN_SCENES", "val": "VAL_SCENES", "test": "TEST_SCENES"}

    def __init__(self, cfg: dict, split: str):
        assert split in ("train", "val", "test")
        self.cfg = cfg
        self.split = split
        self.root = Path(cfg["DATA"]["ROOT"])
        self.pc_range = np.array(cfg["DATA"]["POINT_CLOUD_RANGE"], dtype=np.float32)

        self.samples = _list_scene_frames(self.root, cfg["DATA"][self._SPLIT_KEY[split]])
        self.samples.sort(key=lambda p: str(p))  # deterministic order regardless of filesystem enumeration order

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
        boxes = []
        for obj in objs:
            if "quaternion" not in obj:
                # some auto_track.py-propagated labels have centroid/dimensions/
                # rotations but no quaternion (a bug in that script's writer --
                # it never computes one) -- drop just this object rather than
                # crash the whole frame/batch on it.
                continue
            c, d, q = obj["centroid"], obj["dimensions"], obj["quaternion"]
            boxes.append([
                c["x"], c["y"], c["z"],
                d["length"], d["width"], d["height"],
                q["w"], q["x"], q["y"], q["z"],
            ])
        return np.array(boxes, dtype=np.float32).reshape(-1, 10)

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
