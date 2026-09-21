import torch
import torch.nn as nn

from .vfe import VFE
from .backbone3d_auto import build_backbone3d
from .slotformer import SlotFormerBackbone
from .decoder_detr import DetrDecoder, pad_tokens, token_positional_embedding
from .heads_detr import SetPredictionHead
from .matcher import HungarianMatcher
from .losses_detr import compute_detr_loss
from .sparse_ops import build_index_grid
from data.dataset import voxelize_batch


class DiverDetectorDETR(nn.Module):
    """Same VFE -> sparse 3D backbone -> SlotFormer trunk as models/detector.py's
    DiverDetector (unchanged, reused as-is), but with a DETR-style query decoder
    + set-prediction head in place of the dense per-voxel CenterPoint-style head
    -- see the SparseVoxFormer adaptation plan for why. Only the head after the
    trunk differs; keep both trunks in sync if one changes."""

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
        self.stem_stride = self.backbone.total_stride
        scfg = cfg["SLOTFORMER"]
        self.use_slotformer = scfg.get("ENABLED", True)
        if self.use_slotformer:
            self.slot_backbone = SlotFormerBackbone(self.backbone.out_channels, scfg["WIN_SIZE"], scfg["NUM_CYCLES"], scfg["NUM_HEADS"])

        dcfg = cfg["DETR_HEAD"]
        channels = self.backbone.out_channels
        self.decoder = DetrDecoder(channels, dcfg["NUM_QUERIES"], dcfg["NUM_DECODER_LAYERS"], dcfg["NUM_HEADS"])
        self.head = SetPredictionHead(channels, dcfg.get("HIDDEN_DIM"), dcfg.get("NUM_HIDDEN_LAYERS", 1))

        mcfg = cfg["MATCHER"]
        self.matcher = HungarianMatcher(mcfg["COST_CLS"], mcfg["COST_CENTER"], mcfg["COST_SIZE"], mcfg["COST_ROT"])

    def _to_device(self, batch, device):
        # same pattern as models/detector.py::DiverDetector._to_device -- voxel
        # coords are computed on-device (voxelize_batch), not in the CPU dataloader
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
        bb_feat, bb_coords, _, _ = self.backbone(vfe_out, b["voxel_coords"], index_grid, self.grid_size, b["batch_size"])
        sf_feat = self.slot_backbone(bb_feat, bb_coords) if self.use_slotformer else bb_feat

        eff_voxel_size = self.voxel_size * self.stem_stride
        channels = sf_feat.shape[1]
        key_pad, key_padding_mask = pad_tokens(sf_feat, bb_coords[:, 0], b["batch_size"])
        pos = token_positional_embedding(bb_coords, channels, self.pc_range, eff_voxel_size)
        key_pos, _ = pad_tokens(pos, bb_coords[:, 0], b["batch_size"])

        query_feat, ref_points = self.decoder(key_pad, key_pos, key_padding_mask, self.pc_range, b["batch_size"])
        pred = self.head(query_feat, ref_points)
        return pred, b["gt_boxes"]

    def loss(self, batch, device):
        pred, gt_boxes_list = self.forward(batch, device)
        matches = self.matcher.match(pred, gt_boxes_list)
        losses = compute_detr_loss(pred, gt_boxes_list, matches, self.cfg["DETR_LOSS"])
        return losses, pred, gt_boxes_list, matches

    @torch.no_grad()
    def decode(self, pred: dict, score_threshold: float = 0.1) -> list:
        """No local-max/NMS step needed -- each query IS a candidate box, so this
        is just a per-query threshold + rotation decode. Returns list[B] of dicts
        with 'center'/'size'/'rot_matrix'/'score' (rot_matrix, not quat -- see
        models/rotation6d.py; a test/eval script comparing against quaternion GT
        should convert GT via box_utils.quat_to_rotmat rather than the other way)."""
        from .rotation6d import sixd_to_matrix
        scores = torch.sigmoid(pred["exist_logit"].squeeze(-1))  # (B,Q)
        sizes = pred["log_size"].exp()
        rot_matrices = sixd_to_matrix(pred["sixd"])

        results = []
        for b in range(scores.shape[0]):
            keep = scores[b] >= score_threshold
            results.append({
                "center": pred["center"][b][keep].detach().cpu(),
                "size": sizes[b][keep].detach().cpu(),
                "rot_matrix": rot_matrices[b][keep].detach().cpu(),
                "score": scores[b][keep].detach().cpu(),
            })
        return results
