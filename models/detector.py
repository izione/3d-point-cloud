import torch
import torch.nn as nn

from .vfe import VFE
from .stem import Stem
from .slotformer import SlotFormerBackbone
from .heads import DetectionHead
from .assign import DynamicLabelAssigner
from .losses import compute_total_loss
from .sparse_ops import build_index_grid


class DiverDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        pc_range = cfg["DATA"]["POINT_CLOUD_RANGE"]
        voxel_size = cfg["DATA"]["VOXEL_SIZE"]
        self.register_buffer("pc_range", torch.tensor(pc_range, dtype=torch.float32))
        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32))
        self.grid_size = tuple(round((pc_range[3 + i] - pc_range[i]) / voxel_size[i]) for i in range(3))
        self.stem_stride = cfg["STEM"]["DOWNSAMPLE_STRIDE"]

        self.vfe = VFE(num_filters=cfg["VFE"]["NUM_FILTERS"])
        s = cfg["STEM"]
        self.stem = Stem(s["IN_CHANNELS"], s["OUT_CHANNELS"], s["NUM_BASIC_BLOCKS"], s["DOWNSAMPLE_KERNEL"], s["DOWNSAMPLE_STRIDE"])
        bcfg = cfg["BACKBONE"]
        self.backbone = SlotFormerBackbone(bcfg["FEATURE_DIM"], bcfg["WIN_SIZE"], bcfg["NUM_CYCLES"], bcfg["NUM_HEADS"])
        self.head = DetectionHead(bcfg["FEATURE_DIM"], cfg["DENSE_HEAD"])
        self.assigner = DynamicLabelAssigner(cfg["ASSIGNER"], self.pc_range, self.voxel_size * self.stem_stride)

    def _to_device(self, batch, device):
        return {
            "points": batch["points"].to(device),
            "point_voxel_idx": batch["point_voxel_idx"].to(device),
            "voxel_coords": batch["voxel_coords"].to(device),
            "gt_boxes": [g.to(device) for g in batch["gt_boxes"]],
            "batch_size": batch["batch_size"],
        }

    def forward(self, batch, device):
        b = self._to_device(batch, device)
        num_voxels = b["voxel_coords"].shape[0]
        vfe_out = self.vfe(b["points"], b["point_voxel_idx"], b["voxel_coords"], num_voxels, self.pc_range, self.voxel_size)
        index_grid = build_index_grid(b["voxel_coords"], b["batch_size"], self.grid_size, device=device)
        stem_feat, stem_coords, stem_idx_grid, stem_grid_size = self.stem(
            vfe_out, b["voxel_coords"], index_grid, self.grid_size, b["batch_size"]
        )
        bb_out = self.backbone(stem_feat, stem_coords)
        pred = self.head(bb_out, stem_coords, stem_idx_grid, stem_grid_size)
        return pred, stem_coords, b["gt_boxes"]

    def loss(self, batch, device):
        pred, stem_coords, gt_boxes_list = self.forward(batch, device)
        assign_result = self.assigner.assign(pred, stem_coords, gt_boxes_list)
        eff_voxel_size = self.voxel_size * self.stem_stride
        losses = compute_total_loss(
            pred, stem_coords, gt_boxes_list, assign_result, self.cfg["LOSS"], self.pc_range, eff_voxel_size
        )
        return losses, pred, stem_coords, assign_result
