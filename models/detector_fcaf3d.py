"""FCAF3D-style (arXiv 2112.00322) dense anchor-free detector: same VFE +
sparse U-Net backbone trunk as DiverDetectorDETR (models/detector_detr.py),
but with a dense multi-level head (models/heads_fcaf3d.py) + multi-level GT
assignment (models/assign_fcaf3d.py) in place of the DETR decoder + Hungarian
matcher + query denoising. No SlotFormer/DSVT refinement stage either -- the
U-Net backbone's own encoder-decoder IS the "neck" that plays that role in
the paper; there's no separate token-mixing transformer stage in FCAF3D."""
import torch
import torch.nn as nn

from .vfe import VFE
from .vfe_m import MVFE
from .vfe_point_attn import PointAttentionVFE
from .backbone3d_unet_spconv import SparseUNetBackboneSpconv
from .backbone3d_spconv import Sparse3DBackboneSpconv
from .heads_fcaf3d import FCAF3DHead
from .assign_fcaf3d import assign_multilevel
from .losses_fcaf3d import compute_fcaf3d_loss
from .rotation6d import rotation_from_raw
from .box_utils import oriented_iou_3d_sampled
from .sparse_ops import build_index_grid, local_max_mask_3d
from data.dataset import voxelize_batch


class DiverDetectorFCAF3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        pc_range = cfg["DATA"]["POINT_CLOUD_RANGE"]
        voxel_size = cfg["DATA"]["VOXEL_SIZE"]
        self.register_buffer("pc_range", torch.tensor(pc_range, dtype=torch.float32))
        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32))
        self.grid_size = tuple(round((pc_range[3 + i] - pc_range[i]) / voxel_size[i]) for i in range(3))

        # mirrors models/detector_detr.py's own VFE.TYPE branching -- point_attn
        # (Point Transformer k-NN self-attention + cross-attention voxel
        # pooling, models/vfe_point_attn.py) needs point_batch_idx/batch_size
        # at call time, unlike mvfe/vfe, so forward() branches on this too.
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
        # "sparse_single_stage" (models/backbone3d_spconv.py::Sparse3DBackboneSpconv):
        # a plain stack of stride-2-downsample-then-residual-blocks stages, NO
        # decoder/skip connections at all -- for use with FCAF3D_HEAD.SINGLE_LEVEL:
        # true, where the U-Net's whole second encoder stage + decoder existed
        # ONLY to reconstruct a fine level the head already gets from the first
        # stage in this mode; that machinery is dead weight once multi-level
        # assignment isn't used (see exp_fcaf3d_single_level.yaml's own ablation),
        # and its extra capacity measurably encouraged faster overfitting on a
        # center-only stage1 (see the session's own exp_fcaf3d_stage1_lite result).
        self.backbone_type = bcfg.get("TYPE", "sparse_unet")
        assert self.backbone_type in ("sparse_unet", "sparse_single_stage"), \
            "DiverDetectorFCAF3D needs BACKBONE.TYPE: sparse_unet or sparse_single_stage"
        stage_channels = bcfg["STAGE_CHANNELS"]
        if self.backbone_type == "sparse_unet":
            # forward_multilevel() always returns exactly 2 CONCEPTUAL levels
            # (fully-decoded fine + deepest-encoder-output-as-coarse) regardless
            # of how many encoder/decoder stages the U-Net itself has inside --
            # SparseUNetBackboneSpconv's own forward()/total_stride are already
            # correct for any depth >= 2 (checked: the decoded "fine" output
            # always lands back at stride=down_stride no matter how deep the
            # encoder goes, since there's always exactly one decoder stage per
            # encoder stage below the top), so this only needs >= 2, not == 2.
            assert len(stage_channels) >= 2, "sparse_unet needs at least 2 stages"
            self.backbone = SparseUNetBackboneSpconv(
                self.vfe.out_channels, stage_channels, bcfg["NUM_BLOCKS_PER_STAGE"],
                bcfg["DOWNSAMPLE_KERNEL"], bcfg["DOWNSAMPLE_STRIDE"],
                atrous_dilations=bcfg.get("ATROUS_DILATIONS"),
            )
        else:
            self.backbone = Sparse3DBackboneSpconv(
                self.vfe.out_channels, stage_channels, bcfg["NUM_BLOCKS_PER_STAGE"],
                bcfg["DOWNSAMPLE_KERNEL"], bcfg["DOWNSAMPLE_STRIDE"],
            )
        channels = self.backbone.out_channels  # = stage_channels[0] (unet) or stage_channels[-1] (single-stage)
        if self.backbone_type == "sparse_unet":
            # FCAF3D's head shares weights across levels, so every level must be
            # the same width; the coarse level (stage_channels[-1]) gets a linear
            # projection down/up to match before going through the shared head.
            self.level_proj = nn.ModuleList([
                nn.Identity(),
                nn.Identity() if stage_channels[-1] == channels else nn.Linear(stage_channels[-1], channels),
            ])
        else:
            self.level_proj = nn.ModuleList([nn.Identity()])  # only ever one level

        hcfg = cfg["FCAF3D_HEAD"]
        self.rotation_mode = hcfg.get("ROTATION_MODE", "6d")
        self.head = FCAF3DHead(channels, hcfg.get("HIDDEN_DIM"), hcfg.get("NUM_HIDDEN_LAYERS", 1),
                                predict_box=hcfg.get("PREDICT_BOX", True), rotation_mode=self.rotation_mode)
        # ablation switch (see configs/exp_fcaf3d_single_level.yaml): drop the
        # coarse level entirely and run every GT box through fine-level
        # assignment only, to measure whether multi-level assignment actually
        # earns its keep on this single-class (diver-only) dataset, where box
        # size varies far less than FCAF3D's original indoor-scene classes.
        self.single_level = hcfg.get("SINGLE_LEVEL", False)

        acfg = cfg["ASSIGN"]
        self.min_locations = acfg.get("MIN_LOCATIONS", 6)
        self.center_sample_radius = acfg.get("CENTER_SAMPLE_RADIUS", 0.5)
        self.nearest_fallback_k = acfg.get("NEAREST_FALLBACK_K", 0)

        # CenterPoint/CenterNet-style soft Gaussian existence target +
        # penalty-reduced focal loss, replacing the hard-0/1 existence target
        # and the decoupled centerness branch -- this project's own original
        # pre-FCAF3D dense head (models/losses.py::focal_heatmap_loss) already
        # used exactly this mechanism; reused here rather than reinvented.
        # Centerness is still built in the head (so state_dict shape is
        # unaffected by toggling this) but unused in both loss and score
        # whenever this is on.
        self.gaussian_heatmap = hcfg.get("GAUSSIAN_HEATMAP", False)
        self.gaussian_sigma = hcfg.get("GAUSSIAN_SIGMA_M", 0.5)
        # "Corner auxiliary" loss (models/losses.py's original corner_loss,
        # ported): an extra Gaussian-heatmap MSE between predicted and GT box
        # CORNERS, supervising extent/orientation spatially. Only meaningful
        # alongside GAUSSIAN_HEATMAP (reuses its sigma unless overridden).
        self.corner_aux = hcfg.get("CORNER_AUX", False)
        # RTM3D-style ablation of corner_aux: keep each of the 8 box corners'
        # identity separate (8 heatmap channels) instead of pooling all 8 into
        # one combined max-heatmap -- see assign_fcaf3d's own docstring.
        self.corner_aux_identity = hcfg.get("CORNER_AUX_IDENTITY", False)

    def _to_device(self, batch, device):
        points = batch["points"].to(device, non_blocking=True)
        point_batch_idx = batch["point_batch_idx"].to(device, non_blocking=True)
        voxel_coords, point_voxel_idx = voxelize_batch(
            points, point_batch_idx, self.pc_range, self.voxel_size, torch.tensor(self.grid_size, device=device))
        return {
            "points": points, "point_batch_idx": point_batch_idx, "point_voxel_idx": point_voxel_idx,
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
        if self.backbone_type == "sparse_unet":
            levels = self.backbone.forward_multilevel(vfe_out, b["voxel_coords"], index_grid, self.grid_size, b["batch_size"])
            if self.single_level:
                levels = levels[:1]
        else:
            feat, coords, lvl_index_grid, lvl_grid_size = self.backbone(
                vfe_out, b["voxel_coords"], index_grid, self.grid_size, b["batch_size"])
            levels = [(feat, coords, lvl_index_grid, lvl_grid_size, self.backbone.total_stride)]

        level_preds, level_world_centers, level_batch_idx = [], [], []
        for (feat, coords, lvl_index_grid, lvl_grid_size, stride), proj in zip(levels, self.level_proj):
            eff_voxel_size = self.voxel_size * stride
            world_centers = self.pc_range[:3] + (coords[:, 1:4].float() + 0.5) * eff_voxel_size
            pred = self.head(proj(feat), coords, lvl_index_grid, lvl_grid_size, world_centers)
            if self.gaussian_heatmap:
                # SonarVoxNet-style local-max peak picking (see
                # sparse_ops.local_max_mask_3d's own docstring for why this
                # replaces plain nms_radius for this mode): the Gaussian
                # heatmap target means several adjacent active voxels around
                # one true object center all get high (but not identical)
                # exist scores, and a flat-radius NMS can leave more than one
                # of them standing if they happen to spread out past
                # nms_radius -- grid-adjacency peak picking catches these
                # since it only ever compares a voxel against its own
                # immediate neighbors, regardless of how wide the whole blob is.
                exist_score = torch.sigmoid(pred["exist_logit"].squeeze(-1))
                pred["is_local_max"] = local_max_mask_3d(exist_score, coords, lvl_index_grid, lvl_grid_size)
            level_preds.append(pred)
            level_world_centers.append(world_centers)
            level_batch_idx.append(coords[:, 0])

        return level_preds, level_world_centers, level_batch_idx, b["gt_boxes"], b["batch_size"]

    def loss(self, batch, device):
        level_preds, level_world_centers, level_batch_idx, gt_boxes_list, batch_size = self.forward(batch, device)
        level_targets = assign_multilevel(level_world_centers, level_batch_idx, gt_boxes_list, batch_size,
                                           self.min_locations, self.center_sample_radius, self.nearest_fallback_k,
                                           gaussian_sigma=self.gaussian_sigma if self.gaussian_heatmap else None,
                                           corner_identity=self.corner_aux_identity if self.corner_aux else False)
        losses = compute_fcaf3d_loss(level_preds, level_targets, self.cfg["FCAF3D_LOSS"],
                                      gaussian_heatmap=self.gaussian_heatmap,
                                      level_world_centers=level_world_centers, level_batch_idx=level_batch_idx,
                                      corner_aux_sigma=self.gaussian_sigma if self.corner_aux else None,
                                      corner_identity=self.corner_aux_identity if self.corner_aux else False,
                                      rotation_mode=self.rotation_mode)
        return losses, level_preds, level_batch_idx, gt_boxes_list

    @torch.no_grad()
    def decode(self, level_preds: list, level_batch_idx: list, batch_size: int, score_threshold: float = 0.1,
               nms_radius: float = None, nms_iou_threshold: float = None) -> list:
        """Score = exist_prob * centerness (FCAF3D's own inference-time combination,
        see the paper: "scores are multiplied by centerness just before NMS").

        gaussian_heatmap mode additionally pre-filters candidates to
        pred["is_local_max"] (see sparse_ops.local_max_mask_3d) -- a
        SonarVoxNet-style grid-adjacency peak pick, computed once in
        forward() -- before score_threshold/NMS ever run. Needed because a
        Gaussian heatmap target makes several adjacent active voxels around
        one true center all score highly, and the official evaluator
        (benchmark_eval.py) only credits ONE true positive per GT box
        (greedy, highest-score-first matching) -- every extra overlapping
        detection past the first is counted as a false positive and drags
        precision (hence AP) down at every recall level beyond it, even
        though it's "the same" real detection. nms_radius/nms_iou_threshold
        below still run afterward as an optional extra cleanup, but
        is_local_max is what actually removes most of the duplicates, since
        it compares each voxel only against its own immediate neighbors
        rather than a flat Euclidean radius that two peaks of the same wide
        blob can both sit outside of.

        nms_radius=None (default) does no NMS at all -- every active voxel
        across both levels that clears score_threshold is kept independently,
        same as this project's other decode()s (see
        models/detector_detr.py::decode's own note that a query/location IS
        the candidate there, no peak-finding needed -- not true here, since
        FCAF3D's dense per-location prediction means several nearby voxels
        routinely fire for the same diver). Measured on the val split: at
        score_threshold=0.1 alone, recall=0.98 but precision=0.17 (n_det ~6x
        n_gt); raising score_threshold alone trades recall away fast (0.65
        recall by threshold=0.3). Passing nms_radius applies greedy
        center-distance NMS instead (same distance-based matching convention
        as test.py's own MATCH_DIST_THRESHOLD_M, simpler than a rotated-IoU
        NMS): process candidates score-descending, keep one only if its
        center is farther than nms_radius from every already-kept center in
        that sample.

        nms_iou_threshold: FCAF3D-faithful alternative -- the paper's own
        NMS is rotated-3D-IoU based (threshold 0.5), not distance based (see
        the WebFetch-verified paper quote: "the scores p_hat are multiplied
        by 3D centerness c_hat just before NMS", then standard IoU-threshold
        NMS). Reuses box_utils.oriented_iou_3d_sampled (same rotation-aware
        IoU already used for the AP metric) rather than the GWD pseudo-IoU
        (that one's only guaranteed monotonic, not an exact IoU, and training
        loss code -- no need to pay its own looser-approximation cost at
        inference when we already have a validated exact-enough IoU). Mutually
        exclusive with nms_radius -- if both are given, IoU NMS takes
        priority."""
        results = [{"center": [], "center_offset": [], "size": [], "rot_matrix": [], "score": []} for _ in range(batch_size)]
        for pred, batch_idx in zip(level_preds, level_batch_idx):
            if self.gaussian_heatmap:
                # the heatmap value itself already encodes "how central" a
                # location is -- no separate centerness factor needed (see
                # GAUSSIAN_HEATMAP's own docstring in __init__).
                scores = torch.sigmoid(pred["exist_logit"].squeeze(-1))
            else:
                scores = torch.sigmoid(pred["exist_logit"].squeeze(-1)) * torch.sigmoid(pred["centerness_logit"].squeeze(-1))
            if "log_size" in pred:
                sizes = pred["log_size"].exp()
                rot_matrices = rotation_from_raw(pred["sixd"], self.rotation_mode)
            else:
                # predict_box=False (stage-1-lite, see heads_fcaf3d.py) -- no
                # size/rotation prediction exists at all here; these are pure
                # placeholders (median GT box size, identity rotation) for a
                # caller that only cares about this decode()'s center/score
                # (a stage-2 pose_refine.py is expected to fill in the real
                # size/rotation from these centers).
                n = scores.shape[0]
                sizes = pred["center"].new_tensor([0.85, 1.1, 1.4]).expand(n, 3)
                rot_matrices = torch.eye(3, device=pred["center"].device)[None].expand(n, -1, -1)
            keep = scores >= score_threshold
            if "is_local_max" in pred:
                keep = keep & pred["is_local_max"]
            for b in range(batch_size):
                mask = keep & (batch_idx == b)
                results[b]["center"].append(pred["center"][mask].detach().cpu())
                results[b]["center_offset"].append(pred["center_offset"][mask].detach().cpu())
                results[b]["size"].append(sizes[mask].detach().cpu())
                results[b]["rot_matrix"].append(rot_matrices[mask].detach().cpu())
                results[b]["score"].append(scores[mask].detach().cpu())
        for r in results:
            r["center"] = torch.cat(r["center"], dim=0) if r["center"] else torch.zeros(0, 3)
            r["center_offset"] = torch.cat(r["center_offset"], dim=0) if r["center_offset"] else torch.zeros(0, 3)
            r["size"] = torch.cat(r["size"], dim=0) if r["size"] else torch.zeros(0, 3)
            r["rot_matrix"] = torch.cat(r["rot_matrix"], dim=0) if r["rot_matrix"] else torch.zeros(0, 3, 3)
            r["score"] = torch.cat(r["score"], dim=0) if r["score"] else torch.zeros(0)

        if nms_iou_threshold is not None:
            for r in results:
                n = r["center"].shape[0]
                if n == 0:
                    continue
                order = torch.argsort(r["score"], descending=True)
                kept = []
                for i in order.tolist():
                    is_dup = any(
                        oriented_iou_3d_sampled(r["center"][i], r["size"][i], r["rot_matrix"][i],
                                                 r["center"][j], r["size"][j], r["rot_matrix"][j]) > nms_iou_threshold
                        for j in kept
                    )
                    if not is_dup:
                        kept.append(i)
                kept = torch.as_tensor(kept, dtype=torch.long)
                r["center"] = r["center"][kept]
                r["center_offset"] = r["center_offset"][kept]
                r["size"] = r["size"][kept]
                r["rot_matrix"] = r["rot_matrix"][kept]
                r["score"] = r["score"][kept]
        elif nms_radius is not None:
            for r in results:
                n = r["center"].shape[0]
                if n == 0:
                    continue
                order = torch.argsort(r["score"], descending=True)
                kept = []
                for i in order.tolist():
                    ci = r["center"][i]
                    if all((ci - r["center"][j]).norm().item() > nms_radius for j in kept):
                        kept.append(i)
                kept = torch.as_tensor(kept, dtype=torch.long)
                r["center"] = r["center"][kept]
                r["center_offset"] = r["center_offset"][kept]
                r["size"] = r["size"][kept]
                r["rot_matrix"] = r["rot_matrix"][kept]
                r["score"] = r["score"][kept]
        return results
