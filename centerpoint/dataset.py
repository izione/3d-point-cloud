"""dataset.py - sonar diver point-cloud dataset, on-the-fly dense voxelization.

Reads directly from <config.DATA_ROOT>/<Person*>/<scene_*>/{sonar,labels}
(same raw layout and scene-level TRAIN/VAL/TEST split as the sibling
voxelnet_baseline repo's sonar_diver_dataset.py) -- this data is NOT part of
this git repo (private, too large); see README.md's "Colab setup" for how to
get it onto the Colab instance and point config.DATA_ROOT at it.

No caching: voxelize() is fast enough per-frame that on-the-fly is fine for
this dataset's size (~37k frames total across train/val/test).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import config
from voxelize import augment_with_centroid_offset, voxelize
import rotation3d

TEST_SCENES = [
    "Person2/scene_0035", "Person3/scene_0043", "Person4/scene_0055", "Person3/scene_0067",
    "Person4/scene_0077", "Person4/scene_0088", "Person2/scene_0089", "Person4/scene_0093",
]

VAL_SCENES = [
    "Person2/scene_0009", "Person3/scene_0019", "Person1/scene_0049", "Person3/scene_0064",
    "Person4/scene_0075", "Person4/scene_0090", "Person3/scene_0092",
]

TRAIN_SCENES = [
    "Person2/scene_0050", "Person2/scene_0091", "Person2/scene_0079", "Person2/scene_0078",
    "Person1/scene_0068", "Person1/scene_0063", "Person3/scene_0022", "Person3/scene_0080",
    "Person4/scene_0074", "Person1/scene_0036", "Person1/scene_0042", "Person3/scene_0048",
    "Person3/scene_0072", "Person4/scene_0021", "Person3/scene_0082", "Person1/scene_0000",
    "Person1/scene_0044", "Person3/scene_0070", "Person2/scene_0076", "Person4/scene_0065",
    "Person4/scene_0081",
]

_SPLIT_SCENES = {"train": TRAIN_SCENES, "val": VAL_SCENES, "test": TEST_SCENES}


def _list_scene_frames(root: Path, scene_ids: list) -> list:
    frames = []
    for scene_id in scene_ids:
        sonar_dir = root / scene_id / "sonar"
        if not sonar_dir.is_dir():
            raise FileNotFoundError(
                f"expected sonar dir at {sonar_dir} -- check config.DATA_ROOT / "
                f"$CENTERPOINT_DATA_ROOT points at the dataset root")
        frames.extend(sorted(sonar_dir.glob("frame_*.bin")))
    return frames


def _label_path_for(sonar_path: Path) -> Path:
    return sonar_path.parent.parent / "labels" / (sonar_path.stem + ".json")


def _load_gt_objects_full(sonar_path: Path) -> list:
    """-> list of dicts {centroid, dimensions, rotation_x, rotation_y, rotation_z}
    (degrees) -- the label JSON's nested "rotations":{x,y,z} flattened to this
    flat-key convention (heatmap_targets.py's o.get("rotation_x", 0.0))."""
    label_path = _label_path_for(sonar_path)
    if not label_path.is_file():
        return []
    with open(label_path) as f:
        label = json.load(f)
    objects = []
    for o in label.get("objects", []):
        if "centroid" not in o or "dimensions" not in o:
            continue
        r = o.get("rotations", {})
        objects.append({
            "centroid": o["centroid"], "dimensions": o["dimensions"],
            "rotation_x": r.get("x", 0.0), "rotation_y": r.get("y", 0.0), "rotation_z": r.get("z", 0.0),
        })
    return objects


def zyaw_flatten(objects: list) -> list:
    """Zero out rotation_x/rotation_y on a copy of each object -- reduces the
    GT to a yaw-only rotation, matching the CenterPoint paper's own scope
    (nuScenes/Waymo boxes are yaw-only, objects sit on a ground plane).
    Eval GT stays full-3D always -- only the TRAINING target is flattened."""
    out = []
    for o in objects:
        o2 = dict(o)
        o2["rotation_x"] = 0.0
        o2["rotation_y"] = 0.0
        out.append(o2)
    return out


def gt_boxes_from_objects(objects: list) -> np.ndarray:
    """-> (M,13) float32 [x,y,z,l,w,h,theta_z_rad, 6D-rotation(6)] -- always
    built from the FULL (unflattened) 3D rotation, regardless of what the
    model was trained against -- this is the eval-time GT contract."""
    out = []
    for o in objects:
        c, d = o["centroid"], o["dimensions"]
        rx, ry, rz = o.get("rotation_x", 0.0), o.get("rotation_y", 0.0), o.get("rotation_z", 0.0)
        theta = np.radians(rz)
        R = rotation3d.euler_to_matrix(rx, ry, rz)
        six = rotation3d.matrix_to_6d(R)
        out.append(np.concatenate([
            [c["x"], c["y"], c["z"], d["length"], d["width"], d["height"], theta], six,
        ]))
    if not out:
        return np.zeros((0, 13), dtype=np.float32)
    return np.stack(out).astype(np.float32)


class VoxelDataset(Dataset):
    def __init__(self, split: str, pc_range=None, voxel_size=None,
                 max_points_per_voxel=None, max_voxels=None):
        assert split in ("train", "val", "test")
        self.root = config.DATA_ROOT
        self.samples = _list_scene_frames(self.root, _SPLIT_SCENES[split])
        assert len(self.samples) > 0, f"no frames found for split={split}"
        self.pc_range = pc_range or config.POINT_CLOUD_RANGE
        self.voxel_size = voxel_size or config.VOXEL_SIZE
        self.max_points_per_voxel = max_points_per_voxel or config.MAX_POINTS_PER_VOXEL
        self.max_voxels = max_voxels or config.MAX_VOXELS

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sonar_path = self.samples[idx]
        frame_id = str(sonar_path.relative_to(self.root))

        data = np.fromfile(sonar_path, dtype=np.float32)
        points = np.zeros((0, 4), dtype=np.float32) if data.size == 0 else data.reshape(-1, 4)
        points = points[~np.isnan(points).any(axis=1)]

        voxel_xyzr, coords, num_points = voxelize(
            points, self.pc_range, self.voxel_size, self.max_points_per_voxel, self.max_voxels)
        voxel_features = augment_with_centroid_offset(voxel_xyzr, num_points)

        gt_objects = _load_gt_objects_full(sonar_path)
        gt_boxes = gt_boxes_from_objects(gt_objects)

        return {
            "voxel_features": torch.from_numpy(voxel_features),
            "num_points": torch.from_numpy(num_points),
            "coords": torch.from_numpy(coords),
            "gt_boxes": torch.from_numpy(gt_boxes),
            "gt_objects": gt_objects,
            "points": points,
            "frame_id": frame_id,
        }


def collate_fn(batch: list) -> dict:
    voxel_features, num_points, coords, gt_boxes, gt_objects, points, frame_ids = [], [], [], [], [], [], []
    for b_idx, item in enumerate(batch):
        voxel_features.append(item["voxel_features"])
        num_points.append(item["num_points"])
        k = item["coords"].shape[0]
        batch_col = torch.full((k, 1), b_idx, dtype=torch.int64)
        coords.append(torch.cat([batch_col, item["coords"]], dim=1))
        gt_boxes.append(item["gt_boxes"])
        gt_objects.append(item["gt_objects"])
        points.append(item["points"])
        frame_ids.append(item["frame_id"])

    return {
        "voxel_features": torch.cat(voxel_features, dim=0) if voxel_features
            else torch.zeros(0, config.MAX_POINTS_PER_VOXEL, 7),
        "num_points": torch.cat(num_points, dim=0) if num_points else torch.zeros(0, dtype=torch.int64),
        "coords": torch.cat(coords, dim=0) if coords else torch.zeros(0, 4, dtype=torch.int64),
        "gt_boxes": gt_boxes,
        "gt_objects": gt_objects,
        "points": points,
        "frame_ids": frame_ids,
        "batch_size": len(batch),
    }
