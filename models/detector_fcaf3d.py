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
from .backbone3d_unet_spconv import SparseUNetBackboneSpconv
from .heads_fcaf3d import FCAF3DHead
from .assign_fcaf3d import assign_multilevel
from .losses_fcaf3d import compute_fcaf3d_loss
from .rotation6d import sixd_to_matrix
from .sparse_ops import build_index_grid
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

        vfe_type = cfg["VFE"].get("TYPE", "mvfe")
        vfe_cls = MVFE if vfe_type == "mvfe" else VFE
        self.vfe = vfe_cls(num_filters=cfg["VFE"]["NUM_FILTERS"])

        bcfg = cfg["BACKBONE"]
        assert bcfg.get("TYPE") == "sparse_unet", \
            "DiverDetectorFCAF3D needs BACKBONE.TYPE: sparse_unet (the only backbone with forward_multilevel())"
        stage_channels = bcfg["STAGE_CHANNELS"]
        assert len(stage_channels) == 2, "forward_multilevel() only supports exactly 2 levels right now"
        self.backbone = SparseUNetBackboneSpconv(
            self.vfe.out_channels, stage_channels, bcfg["NUM_BLOCKS_PER_STAGE"],
            bcfg["DOWNSAMPLE_KERNEL"], bcfg["DOWNSAMPLE_STRIDE"],
        )
        channels = self.backbone.out_channels  # = stage_channels[0], the fine level's width
        # FCAF3D's head shares weights across levels, so every level must be
        # the same width; the coarse level (stage_channels[-1]) gets a linear
        # projection down/up to match before going through the shared head.
        self.level_proj = nn.ModuleList([
            nn.Identity(),
            nn.Identity() if stage_channels[-1] == channels else nn.Linear(stage_channels[-1], channels),
        ])

        hcfg = cfg["FCAF3D_HEAD"]
        self.head = FCAF3DHead(channels, hcfg.get("HIDDEN_DIM"), hcfg.get("NUM_HIDDEN_LAYERS", 1))

        acfg = cfg["ASSIGN"]
        self.min_locations = acfg.get("MIN_LOCATIONS", 6)
        self.center_sample_radius = acfg.get("CENTER_SAMPLE_RADIUS", 0.5)

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
        vfe_out = self.vfe(b["points"], b["point_voxel_idx"], b["voxel_coords"], num_voxels, self.pc_range, self.voxel_size)
        index_grid = build_index_grid(b["voxel_coords"], b["batch_size"], self.grid_size, device=device)
        levels = self.backbone.forward_multilevel(vfe_out, b["voxel_coords"], index_grid, self.grid_size, b["batch_size"])

        level_preds, level_world_centers, level_batch_idx = [], [], []
        for (feat, coords, lvl_index_grid, lvl_grid_size, stride), proj in zip(levels, self.level_proj):
            eff_voxel_size = self.voxel_size * stride
            world_centers = self.pc_range[:3] + (coords[:, 1:4].float() + 0.5) * eff_voxel_size
            pred = self.head(proj(feat), coords, lvl_index_grid, lvl_grid_size, world_centers)
            level_preds.append(pred)
            level_world_centers.append(world_centers)
            level_batch_idx.append(coords[:, 0])

        return level_preds, level_world_centers, level_batch_idx, b["gt_boxes"], b["batch_size"]

    def loss(self, batch, device):
        level_preds, level_world_centers, level_batch_idx, gt_boxes_list, batch_size = self.forward(batch, device)
        level_targets = assign_multilevel(level_world_centers, level_batch_idx, gt_boxes_list, batch_size,
                                           self.min_locations, self.center_sample_radius)
        losses = compute_fcaf3d_loss(level_preds, level_targets, self.cfg["FCAF3D_LOSS"])
        return losses, level_preds, level_batch_idx, gt_boxes_list

    @torch.no_grad()
    def decode(self, level_preds: list, level_batch_idx: list, batch_size: int, score_threshold: float = 0.1,
               nms_radius: float = None) -> list:
        """Score = exist_prob * centerness (FCAF3D's own inference-time combination,
        see the paper: "scores are multiplied by centerness just before NMS").

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
        that sample."""
        results = [{"center": [], "size": [], "rot_matrix": [], "score": []} for _ in range(batch_size)]
        for pred, batch_idx in zip(level_preds, level_batch_idx):
            scores = torch.sigmoid(pred["exist_logit"].squeeze(-1)) * torch.sigmoid(pred["centerness_logit"].squeeze(-1))
            sizes = pred["log_size"].exp()
            rot_matrices = sixd_to_matrix(pred["sixd"])
            keep = scores >= score_threshold
            for b in range(batch_size):
                mask = keep & (batch_idx == b)
                results[b]["center"].append(pred["center"][mask].detach().cpu())
                results[b]["size"].append(sizes[mask].detach().cpu())
                results[b]["rot_matrix"].append(rot_matrices[mask].detach().cpu())
                results[b]["score"].append(scores[mask].detach().cpu())
        for r in results:
            r["center"] = torch.cat(r["center"], dim=0) if r["center"] else torch.zeros(0, 3)
            r["size"] = torch.cat(r["size"], dim=0) if r["size"] else torch.zeros(0, 3)
            r["rot_matrix"] = torch.cat(r["rot_matrix"], dim=0) if r["rot_matrix"] else torch.zeros(0, 3, 3)
            r["score"] = torch.cat(r["score"], dim=0) if r["score"] else torch.zeros(0)

        if nms_radius is not None:
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
                r["size"] = r["size"][kept]
                r["rot_matrix"] = r["rot_matrix"][kept]
                r["score"] = r["score"][kept]
        return results
