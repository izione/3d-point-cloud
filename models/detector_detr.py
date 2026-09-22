import torch
import torch.nn as nn

from .vfe import VFE
from .vfe_m import MVFE
from .vfe_point_attn import PointAttentionVFE
from .backbone3d_auto import build_backbone3d
from .slotformer import SlotFormerBackbone
from .dsvt import DSVTBackbone
from .decoder_detr import DetrDecoder, pad_tokens, token_positional_embedding, ref_point_positional_embedding
from .heads_detr import SetPredictionHead
from .matcher import HungarianMatcher
from .losses_detr import compute_detr_loss, compute_denoising_loss
from .denoising import QueryDenoising, build_attention_mask
from .sparse_ops import build_index_grid
from data.dataset import voxelize_batch


class DiverDetectorDETR(nn.Module):
    """Same VFE -> sparse 3D backbone -> SlotFormer trunk as models/detector.py's
    DiverDetector (unchanged, reused as-is), but with a DETR-style query decoder
    + set-prediction head in place of the dense per-voxel CenterPoint-style head
    -- see the SparseVoxFormer adaptation plan for why. Only the head after the
    trunk differs; keep both trunks in sync if one changes.

    Training-only query denoising (models/denoising.py) adds extra queries
    built from noised GT boxes alongside the normal learned ("matching")
    queries -- forward() runs both through the same decoder call (with an
    attention mask keeping them from leaking into each other) and returns the
    matching-only predictions plus an optional denoising bundle; loss() adds
    both loss terms together. At eval time (self.training=False) there is no
    denoising, so forward()'s dn_info is always None and pred is exactly the
    matching queries' output -- decode()/test_detr.py don't need to know
    denoising exists at all.

    forward() returns (pred, gt_boxes_list, dn_info, aux_info): aux_info is
    (aux_preds, aux_preds_dn), the same-shaped prediction dicts for every
    decoder layer BEFORE the last one (aux_preds_dn is None when dn_info is
    None). loss() runs the matcher+loss independently on each of these too --
    see its docstring comment for why this auxiliary supervision is needed.
    Callers that only want the final prediction (decode(), test_detr.py) can
    ignore aux_info entirely.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        pc_range = cfg["DATA"]["POINT_CLOUD_RANGE"]
        voxel_size = cfg["DATA"]["VOXEL_SIZE"]
        self.register_buffer("pc_range", torch.tensor(pc_range, dtype=torch.float32))
        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32))
        self.grid_size = tuple(round((pc_range[3 + i] - pc_range[i]) / voxel_size[i]) for i in range(3))

        # Point-to-voxel feature extraction: SparseVoxFormer's "mVFE" (std-dev
        # + point count on top of the base VFE, models/vfe_m.py) by default;
        # VFE.TYPE: vfe falls back to the plain base encoder; VFE.TYPE:
        # point_attn instead refines points via Point Transformer local
        # self-attention before pooling (models/vfe_point_attn.py) -- needs
        # point_batch_idx/batch_size at call time (see forward()), unlike the
        # other two, since its k-NN search must not cross a sample boundary.
        self.vfe_type = cfg["VFE"].get("TYPE", "mvfe")
        if self.vfe_type == "point_attn":
            pcfg = cfg["VFE"].get("POINT_ATTN", {})
            self.vfe = PointAttentionVFE(
                pcfg.get("OUT_CHANNELS", 128), pcfg.get("POINT_CHANNELS", 64),
                pcfg.get("NUM_BLOCKS", 1), pcfg.get("K", 16),
            )
        else:
            vfe_cls = MVFE if self.vfe_type == "mvfe" else VFE
            self.vfe = vfe_cls(num_filters=cfg["VFE"]["NUM_FILTERS"])
        bcfg = cfg["BACKBONE"]
        self.backbone = build_backbone3d(
            self.vfe.out_channels, bcfg["STAGE_CHANNELS"], bcfg["NUM_BLOCKS_PER_STAGE"],
            bcfg["DOWNSAMPLE_KERNEL"], bcfg["DOWNSAMPLE_STRIDE"], bcfg.get("TYPE", "auto"),
            block_dilations=bcfg.get("BLOCK_DILATIONS"), norm_type=bcfg.get("NORM_TYPE", "batch"), bcfg=bcfg,
        )
        self.stem_stride = self.backbone.total_stride
        channels = self.backbone.out_channels

        # sparse-token refinement stage after the backbone: this codebase's
        # existing SlotFormer (a different paper, FSHNet) by default, or
        # SparseVoxFormer's own DSVT (models/dsvt.py) when REFINEMENT.TYPE:
        # dsvt is set -- see dsvt.py's docstring for why these aren't the same
        # architecture despite playing the same role.
        rcfg = cfg.get("REFINEMENT", {})
        self.refinement_type = rcfg.get("TYPE", "slotformer")
        if self.refinement_type == "slotformer":
            scfg = cfg["SLOTFORMER"]
            self.use_slotformer = scfg.get("ENABLED", True)
            if self.use_slotformer:
                self.slot_backbone = SlotFormerBackbone(channels, scfg["WIN_SIZE"], scfg["NUM_CYCLES"], scfg["NUM_HEADS"])
        elif self.refinement_type == "dsvt":
            self.dsvt_backbone = DSVTBackbone(
                channels, rcfg["WINDOW_SHAPES"], rcfg["TAU"], rcfg["NUM_BLOCKS"], rcfg["NUM_HEADS"],
            )
        else:
            raise ValueError(f"unknown REFINEMENT.TYPE: {self.refinement_type!r} (expected 'slotformer' or 'dsvt')")

        dcfg = cfg["DETR_HEAD"]
        self.decoder = DetrDecoder(channels, dcfg["NUM_QUERIES"], dcfg["NUM_DECODER_LAYERS"], dcfg["NUM_HEADS"])
        self.head = SetPredictionHead(channels, dcfg.get("HIDDEN_DIM"), dcfg.get("NUM_HIDDEN_LAYERS", 1))

        mcfg = cfg["MATCHER"]
        self.matcher = HungarianMatcher(mcfg["COST_CLS"], mcfg["COST_CENTER"], mcfg["COST_SIZE"], mcfg["COST_ROT"])

        ncfg = cfg.get("DENOISING", {})
        self.use_denoising = ncfg.get("ENABLED", False)
        if self.use_denoising:
            self.denoising = QueryDenoising(
                channels, ncfg.get("NUM_GROUPS", 5), ncfg.get("CENTER_NOISE_SCALE", 0.4),
                ncfg.get("SIZE_NOISE_SCALE", 0.4), ncfg.get("ROT_NOISE_DEG", 15.0),
            )

    def _to_device(self, batch, device):
        # same pattern as models/detector.py::DiverDetector._to_device -- voxel
        # coords are computed on-device (voxelize_batch), not in the CPU dataloader
        points = batch["points"].to(device, non_blocking=True)
        point_batch_idx = batch["point_batch_idx"].to(device, non_blocking=True)
        voxel_coords, point_voxel_idx = voxelize_batch(points, point_batch_idx, self.pc_range, self.voxel_size, torch.tensor(self.grid_size, device=device))
        return {
            "points": points,
            "point_batch_idx": point_batch_idx,
            "point_voxel_idx": point_voxel_idx,
            "voxel_coords": voxel_coords,
            "gt_boxes": [g.to(device) for g in batch["gt_boxes"]],
            "batch_size": batch["batch_size"],
        }

    def forward(self, batch, device):
        b = self._to_device(batch, device)
        num_voxels = b["voxel_coords"].shape[0]
        if self.vfe_type == "point_attn":
            vfe_out = self.vfe(b["points"], b["point_voxel_idx"], b["voxel_coords"], num_voxels, self.pc_range,
                                self.voxel_size, b["point_batch_idx"], b["batch_size"])
        else:
            vfe_out = self.vfe(b["points"], b["point_voxel_idx"], b["voxel_coords"], num_voxels, self.pc_range, self.voxel_size)
        index_grid = build_index_grid(b["voxel_coords"], b["batch_size"], self.grid_size, device=device)
        bb_feat, bb_coords, _, _ = self.backbone(vfe_out, b["voxel_coords"], index_grid, self.grid_size, b["batch_size"])
        eff_voxel_size = self.voxel_size * self.stem_stride

        if self.refinement_type == "dsvt":
            sf_feat = self.dsvt_backbone(bb_feat, bb_coords, self.pc_range, eff_voxel_size)
        else:
            sf_feat = self.slot_backbone(bb_feat, bb_coords) if self.use_slotformer else bb_feat

        channels = sf_feat.shape[1]
        pos = token_positional_embedding(bb_coords, channels, self.pc_range, eff_voxel_size)
        # pad_tokens' sort/scatter bookkeeping only depends on bb_coords[:, 0] (identical for
        # feat and pos), so pad both in one call by concatenating along the channel dim first
        # instead of redoing that bookkeeping twice.
        padded_both, key_padding_mask = pad_tokens(torch.cat([sf_feat, pos], dim=1), bb_coords[:, 0], b["batch_size"])
        key_pad, key_pos = padded_both[..., :channels], padded_both[..., channels:]

        num_matching = self.decoder.num_queries
        match_content = self.decoder.matching_content(b["batch_size"])
        match_ref = self.decoder.matching_reference_points(self.pc_range)[None, :, :].expand(b["batch_size"], -1, -1)

        dn_bundle = None
        if self.training and self.use_denoising:
            built = self.denoising.build(b["gt_boxes"], self.pc_range, device)
            if built is not None:
                dn_content, dn_ref, dn_valid, dn_targets, group_size = built
                query_content = torch.cat([match_content, dn_content], dim=1)
                ref_points_all = torch.cat([match_ref, dn_ref], dim=1)
                attn_mask = build_attention_mask(num_matching, group_size, self.denoising.num_groups, device)
                dn_bundle = (dn_valid, dn_targets)
            else:
                query_content, ref_points_all, attn_mask = match_content, match_ref, None
        else:
            query_content, ref_points_all, attn_mask = match_content, match_ref, None

        query_pos = ref_point_positional_embedding(ref_points_all, channels)
        # return_intermediate=True gets every decoder layer's output (not just
        # the last) so loss() can apply an auxiliary loss at each one -- see
        # DetrDecoder.forward()'s docstring for why.
        _, intermediate = self.decoder(query_content, query_pos, key_pad, key_pos, key_padding_mask, attn_mask,
                                        return_intermediate=True)
        all_preds = [self.head(feat, ref_points_all) for feat in intermediate]
        pred_all = all_preds[-1]

        if dn_bundle is not None:
            dn_valid, dn_targets = dn_bundle
            pred = {k: v[:, :num_matching] for k, v in pred_all.items()}
            pred_dn = {k: v[:, num_matching:] for k, v in pred_all.items()}
            aux_preds = [{k: v[:, :num_matching] for k, v in p.items()} for p in all_preds[:-1]]
            aux_preds_dn = [{k: v[:, num_matching:] for k, v in p.items()} for p in all_preds[:-1]]
            return pred, b["gt_boxes"], (pred_dn, dn_valid, dn_targets), (aux_preds, aux_preds_dn)
        return pred_all, b["gt_boxes"], None, (all_preds[:-1], None)

    def loss(self, batch, device):
        pred, gt_boxes_list, dn_info, (aux_preds, aux_preds_dn) = self.forward(batch, device)
        matches = self.matcher.match(pred, gt_boxes_list)
        losses = compute_detr_loss(pred, gt_boxes_list, matches, self.cfg["DETR_LOSS"])

        if dn_info is not None:
            pred_dn, dn_valid, dn_targets = dn_info
            dn_losses = compute_denoising_loss(pred_dn, dn_targets, dn_valid, self.cfg["DETR_LOSS"])
        else:
            zero = pred["center"].sum() * 0.0
            dn_losses = {"total": zero, "cls": zero, "center": zero, "size": zero, "rotation": zero}

        dn_weight = self.cfg.get("DENOISING", {}).get("LOSS_WEIGHT", 1.0)
        total = losses["total"] + dn_weight * dn_losses["total"]

        # Auxiliary loss: independently re-run the matcher + loss for every
        # earlier decoder layer's output too (standard DETR trick, see
        # DetrDecoder.forward()'s docstring). Each layer gets its own
        # Hungarian assignment since its predictions differ from the final
        # layer's -- same as the paper's own auxiliary loss.
        aux_weight = self.cfg["DETR_LOSS"].get("AUX_LOSS_WEIGHT", 1.0)
        for aux_pred in aux_preds:
            aux_matches = self.matcher.match(aux_pred, gt_boxes_list)
            aux_losses = compute_detr_loss(aux_pred, gt_boxes_list, aux_matches, self.cfg["DETR_LOSS"])
            total = total + aux_weight * aux_losses["total"]
        if aux_preds_dn is not None:
            for aux_pred_dn in aux_preds_dn:
                aux_dn_losses = compute_denoising_loss(aux_pred_dn, dn_targets, dn_valid, self.cfg["DETR_LOSS"])
                total = total + aux_weight * dn_weight * aux_dn_losses["total"]

        losses = {
            **losses, "total": total,
            "dn_cls": dn_losses["cls"], "dn_center": dn_losses["center"],
            "dn_size": dn_losses["size"], "dn_rotation": dn_losses["rotation"],
        }
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
