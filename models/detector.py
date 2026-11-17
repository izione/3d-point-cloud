import torch
import torch.nn as nn

from .vfe import VFE
from .backbone3d_auto import build_backbone3d
from .slotformer import SlotFormerBackbone
from .heads import DetectionHead
from .assign import DynamicLabelAssigner
from .losses import compute_total_loss
from .sparse_ops import build_index_grid
from data.dataset import voxelize_batch


class DiverDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        pc_range = cfg["DATA"]["POINT_CLOUD_RANGE"]
        voxel_size = cfg["DATA"]["VOXEL_SIZE"]
        self.register_buffer("pc_range", torch.tensor(pc_range, dtype=torch.float32))
        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32))
        self.grid_size = tuple(round((pc_range[3 + i] - pc_range[i]) / voxel_size[i]) for i in range(3))

        self.vfe = VFE(num_filters=cfg["VFE"]["NUM_FILTERS"])
        bcfg = cfg["BACKBONE"]
        self.backbone = build_backbone3d(
            self.vfe.out_channels, bcfg["STAGE_CHANNELS"], bcfg["NUM_BLOCKS_PER_STAGE"],
            bcfg["DOWNSAMPLE_KERNEL"], bcfg["DOWNSAMPLE_STRIDE"], bcfg.get("TYPE", "auto"),
            block_dilations=bcfg.get("BLOCK_DILATIONS"), norm_type=bcfg.get("NORM_TYPE", "batch"), bcfg=bcfg,
        )
        # total downsample factor from all 4 backbone stages (x,y,z all strided every
        # stage) -- kept as `stem_stride` since test.py/visualize*.py already read
        # model.stem_stride to get the effective voxel size at the backbone's output.
        self.stem_stride = self.backbone.total_stride
        scfg = cfg["SLOTFORMER"]
        # ENABLED lets an experiment skip SlotFormer entirely (e.g. the dense backbone
        # already has a whole-grid receptive field, so SlotFormer's job is redundant
        # there -- see README's backbone benchmark section).
        self.use_slotformer = scfg.get("ENABLED", True)
        if self.use_slotformer:
            self.slot_backbone = SlotFormerBackbone(self.backbone.out_channels, scfg["WIN_SIZE"], scfg["NUM_CYCLES"], scfg["NUM_HEADS"])
        self.head = DetectionHead(self.backbone.out_channels, cfg["DENSE_HEAD"])
        self.assigner = DynamicLabelAssigner(cfg["ASSIGNER"], self.pc_range, self.voxel_size * self.stem_stride)

    def _to_device(self, batch, device):
        # points/point_batch_idx move first; voxel_coords/point_voxel_idx are
        # then computed *on device* (voxelize_batch does the coord math +
        # torch.unique) instead of on the CPU dataloader worker -- see
        # data/dataset.py for why (measured CPU-dataloading-bound, not
        # compute-bound, especially with few CPU cores available).
        points = batch["points"].to(device, non_blocking=True)
        point_batch_idx = batch["point_batch_idx"].to(device, non_blocking=True)
        voxel_coords, point_voxel_idx = voxelize_batch(points, point_batch_idx, self.pc_range, self.voxel_size, torch.tensor(self.grid_size, device=device))
        return {
            "points": points,
            "point_voxel_idx": point_voxel_idx,
            "voxel_coords": voxel_coords,
            "gt_boxes": [g.to(device) for g in batch["gt_boxes"]],
            "batch_size": batch["batch_size"],
        }

    def forward(self, batch, device):
        b = self._to_device(batch, device)
        num_voxels = b["voxel_coords"].shape[0]
        vfe_out = self.vfe(b["points"], b["point_voxel_idx"], b["voxel_coords"], num_voxels, self.pc_range, self.voxel_size)
        index_grid = build_index_grid(b["voxel_coords"], b["batch_size"], self.grid_size, device=device)
        bb_feat, bb_coords, bb_idx_grid, bb_grid_size = self.backbone(
            vfe_out, b["voxel_coords"], index_grid, self.grid_size, b["batch_size"]
        )
        sf_feat = self.slot_backbone(bb_feat, bb_coords) if self.use_slotformer else bb_feat
        pred = self.head(sf_feat, bb_coords, bb_idx_grid, bb_grid_size)
        return pred, bb_coords, b["gt_boxes"]

    def loss(self, batch, device):
        pred, stem_coords, gt_boxes_list = self.forward(batch, device)
        assign_result = self.assigner.assign(pred, stem_coords, gt_boxes_list)
        eff_voxel_size = self.voxel_size * self.stem_stride
        losses = compute_total_loss(
            pred, stem_coords, gt_boxes_list, assign_result, self.cfg["LOSS"], self.pc_range, eff_voxel_size
        )
        return losses, pred, stem_coords, assign_result
