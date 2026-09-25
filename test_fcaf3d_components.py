"""Standalone correctness checks for the FCAF3D-style dense-head components
(models/backbone3d_unet_spconv.py::forward_multilevel, heads_fcaf3d.py,
assign_fcaf3d.py, losses_fcaf3d.py, detector_fcaf3d.py) -- same "prove it
before trusting it" practice as test_detr_components.py. Run directly:
`python test_fcaf3d_components.py`. CUDA-only (spconv).
"""
import torch


def check_differentiable_iou_matches_eval_only():
    from models.losses_fcaf3d import differentiable_axis_aligned_iou_3d
    from models.box_utils import axis_aligned_iou_3d

    torch.manual_seed(0)
    n = 20
    center1, size1 = torch.randn(n, 3) * 3, torch.rand(n, 3) * 2 + 0.2
    center2, size2 = torch.randn(n, 3) * 3, torch.rand(n, 3) * 2 + 0.2
    center1.requires_grad_(True)

    batched = differentiable_axis_aligned_iou_3d(center1, size1, center2, size2)
    for i in range(n):
        single = axis_aligned_iou_3d(center1[i].detach(), size1[i], center2[i], size2[i])
        assert abs(batched[i].item() - single) < 1e-5, \
            f"row {i}: batched={batched[i].item():.6f} vs eval-only={single:.6f}"

    batched.sum().backward()
    assert center1.grad is not None and torch.isfinite(center1.grad).all()
    print(f"[ok] differentiable_axis_aligned_iou_3d: matches box_utils.axis_aligned_iou_3d "
          f"on all {n} pairs, gradients flow")


def check_assign_multilevel_basic():
    from models.assign_fcaf3d import assign_multilevel

    torch.manual_seed(0)
    # fine level: a dense grid of voxels covering [0,4]x[0,4]x[0,4] at 0.5 spacing
    # (plenty of voxels -> qualifies a box for the coarse level if big enough)
    ar = torch.arange(0, 4, 0.5)
    gx, gy, gz = torch.meshgrid(ar, ar, ar, indexing="ij")
    fine_centers = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    fine_batch_idx = torch.zeros(fine_centers.shape[0], dtype=torch.long)

    # coarse level: much sparser (2.0 spacing) over the same volume
    ar_c = torch.arange(0, 4, 2.0)
    cx, cy, cz = torch.meshgrid(ar_c, ar_c, ar_c, indexing="ij")
    coarse_centers = torch.stack([cx, cy, cz], dim=-1).reshape(-1, 3)
    coarse_batch_idx = torch.zeros(coarse_centers.shape[0], dtype=torch.long)

    # a SMALL box (should stay on the fine level -- too few coarse voxels inside)
    # and a LARGE box (should get promoted to the coarse level)
    small = torch.zeros(1, 10); small[0, :3] = torch.tensor([1.0, 1.0, 1.0])
    small[0, 3:6] = 0.4; small[0, 6] = 1.0
    # large enough (half-size 2.0, extent [0,4]^3) to contain BOTH coarse grid
    # values (0 and 2) along every axis -> 2^3=8 covering coarse voxels, well
    # over min_locations=4
    large = torch.zeros(1, 10); large[0, :3] = torch.tensor([2.0, 2.0, 2.0])
    large[0, 3:6] = 4.0; large[0, 6] = 1.0
    gt_boxes_list = [torch.cat([small, large], dim=0)]

    targets = assign_multilevel([fine_centers, coarse_centers], [fine_batch_idx, coarse_batch_idx],
                                 gt_boxes_list, batch_size=1, min_locations=4, center_sample_radius=0.6)
    fine_tgt, coarse_tgt = targets

    assert fine_tgt["pos_mask"].any(), "the small box should have positive voxels on the FINE level"
    assert coarse_tgt["pos_mask"].any(), "the large box should have positive voxels on the COARSE level"
    # the small box's positives should be centered near (1,1,1), not (2,2,2)
    small_pos_centers = fine_centers[fine_tgt["pos_mask"]]
    assert torch.allclose(small_pos_centers.mean(0), torch.tensor([1.0, 1.0, 1.0]), atol=0.6)
    assert torch.allclose(fine_tgt["log_size"][fine_tgt["pos_mask"]][0], torch.log(torch.tensor(0.4)).expand(3), atol=1e-4)
    assert torch.allclose(coarse_tgt["log_size"][coarse_tgt["pos_mask"]][0], torch.log(torch.tensor(4.0)).expand(3), atol=1e-4)
    assert (fine_tgt["centerness"][fine_tgt["pos_mask"]] > 0).all()
    print(f"[ok] assign_multilevel: small box -> fine level ({fine_tgt['pos_mask'].sum().item()} positives), "
          f"large box -> coarse level ({coarse_tgt['pos_mask'].sum().item()} positives), correct targets")


def check_fcaf3d_head_shapes():
    from models.heads_fcaf3d import FCAF3DHead
    from models.sparse_ops import build_index_grid

    torch.manual_seed(0)
    channels, n = 16, 30
    grid_size = (10, 10, 10)
    coords = torch.stack([
        torch.zeros(n, dtype=torch.long),
        torch.randint(0, 10, (n,)), torch.randint(0, 10, (n,)), torch.randint(0, 10, (n,)),
    ], dim=1)
    coords = torch.unique(coords, dim=0)
    n = coords.shape[0]
    feat = torch.randn(n, channels, requires_grad=True)
    index_grid = build_index_grid(coords, batch_size=1, grid_size=grid_size)
    world_centers = torch.randn(n, 3)

    head = FCAF3DHead(channels)
    out = head(feat, coords, index_grid, grid_size, world_centers)
    for k, shape in [("exist_logit", 1), ("center", 3), ("log_size", 3), ("sixd", 6), ("centerness_logit", 1)]:
        assert out[k].shape == (n, shape), f"{k}: expected ({n},{shape}), got {tuple(out[k].shape)}"
    for v in out.values():
        assert torch.isfinite(v).all()

    total = sum(v.sum() for v in out.values())
    total.backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    n_grad = sum(1 for p in head.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    n_total = sum(1 for p in head.parameters())
    assert n_grad == n_total, f"only {n_grad}/{n_total} FCAF3DHead params got gradients (summed all 5 branches' outputs)"
    print(f"[ok] FCAF3DHead: correct per-branch shapes ({n} voxels), finite outputs, "
          f"gradients flow to input features and all {n_total} params")


def check_backbone_forward_multilevel():
    from models.backbone3d_auto import spconv_usable
    if not spconv_usable():
        print("[skip] forward_multilevel: spconv not usable on this machine")
        return

    from models.backbone3d_unet_spconv import SparseUNetBackboneSpconv
    from models.sparse_ops import build_index_grid

    torch.manual_seed(0)
    device = torch.device("cuda")
    in_channels, stage_channels = 8, [16, 32]
    down_stride = 2
    n = 300
    grid_size = (20, 20, 10)
    coords = torch.stack([
        torch.zeros(n, dtype=torch.long),
        torch.randint(0, grid_size[0], (n,)),
        torch.randint(0, grid_size[1], (n,)),
        torch.randint(0, grid_size[2], (n,)),
    ], dim=1)
    coords = torch.unique(coords, dim=0).to(device)
    features = torch.randn(coords.shape[0], in_channels, device=device, requires_grad=True)
    index_grid = build_index_grid(coords, batch_size=1, grid_size=grid_size, device=device)

    backbone = SparseUNetBackboneSpconv(in_channels, stage_channels, num_blocks_per_stage=2,
                                         down_kernel=3, down_stride=down_stride).to(device)
    levels = backbone.forward_multilevel(features, coords, index_grid, grid_size, batch_size=1)
    assert len(levels) == 2, f"expected 2 levels, got {len(levels)}"

    (fine_feat, fine_coords, _, _, fine_stride) = levels[0]
    (coarse_feat, coarse_coords, _, _, coarse_stride) = levels[1]
    assert fine_feat.shape[1] == stage_channels[0]
    assert coarse_feat.shape[1] == stage_channels[1]
    assert fine_stride == down_stride
    assert coarse_stride == down_stride * down_stride
    assert coarse_coords.shape[0] <= fine_coords.shape[0], \
        "the coarse level should never have MORE active voxels than the fine level"
    assert torch.isfinite(fine_feat).all() and torch.isfinite(coarse_feat).all()

    (fine_feat + 0).sum().backward(retain_graph=True)  # cheap way to confirm the fine path is differentiable
    assert features.grad is not None and torch.isfinite(features.grad).all()
    print(f"[ok] SparseUNetBackboneSpconv.forward_multilevel: 2 levels, correct channel widths "
          f"({stage_channels[0]}/{stage_channels[1]}) and strides ({fine_stride}/{coarse_stride}), "
          f"coarse has fewer-or-equal voxels ({coarse_coords.shape[0]} <= {fine_coords.shape[0]}), gradients flow")


def check_decode_nms():
    """Real training run showed precision=0.17 at score_threshold=0.1 (recall
    0.98) because several nearby voxels routinely fire for the same diver --
    raising score_threshold alone traded away recall too fast (0.65 recall by
    threshold=0.3). Confirms decode()'s greedy center-distance NMS actually
    collapses near-duplicate detections of the same object while leaving
    genuinely separate objects alone."""
    from models.detector_fcaf3d import DiverDetectorFCAF3D
    from config_utils import load_config

    torch.manual_seed(0)
    cfg = load_config("configs/exp_fcaf3d.yaml")
    model = DiverDetectorFCAF3D(cfg)

    # two tight clusters (should each collapse to 1 after NMS) representing
    # two real, well-separated divers, plus their scores so the highest-score
    # detection in each cluster is the one that should survive.
    centers = torch.tensor([
        [1.0, 0.0, 0.0], [1.1, 0.05, -0.05], [0.95, -0.1, 0.0],   # cluster A (diver 1)
        [8.0, 0.0, 0.0], [8.05, 0.1, 0.0],                          # cluster B (diver 2)
    ])
    scores = torch.tensor([0.9, 0.6, 0.5, 0.8, 0.7])
    pred = {
        "exist_logit": torch.logit(scores.clamp(1e-4, 1 - 1e-4)).unsqueeze(-1),
        "center": centers,
        "log_size": torch.zeros(5, 3),
        "sixd": torch.tensor([1., 0, 0, 0, 1, 0]).expand(5, 6).clone(),
        "centerness_logit": torch.logit(torch.ones(5) * 0.999).unsqueeze(-1),  # ~1.0, so score ~= exist_prob
    }
    batch_idx = torch.zeros(5, dtype=torch.long)

    no_nms = model.decode([pred], [batch_idx], batch_size=1, score_threshold=0.0, nms_radius=None)
    assert no_nms[0]["center"].shape[0] == 5, "sanity: no NMS should keep all 5 raw detections"

    nms = model.decode([pred], [batch_idx], batch_size=1, score_threshold=0.0, nms_radius=1.0)
    assert nms[0]["center"].shape[0] == 2, \
        f"expected NMS to collapse the 2 clusters to 2 detections, got {nms[0]['center'].shape[0]}"
    # the surviving detections must be the HIGHEST-scoring one from each cluster
    kept_centers = nms[0]["center"]
    assert any(torch.allclose(c, torch.tensor([1.0, 0.0, 0.0]), atol=1e-4) for c in kept_centers), \
        "NMS should keep cluster A's highest-score detection (center [1,0,0], score 0.9), not a lower-score neighbor"
    assert any(torch.allclose(c, torch.tensor([8.0, 0.0, 0.0]), atol=1e-4) for c in kept_centers), \
        "NMS should keep cluster B's highest-score detection (center [8,0,0], score 0.8), not a lower-score neighbor"
    print(f"[ok] decode() NMS: 5 raw detections (2 clusters) -> 2 after NMS, "
          f"each cluster's highest-score detection survives")


def check_full_fcaf3d_model_forward_backward():
    from config_utils import load_config
    from models.detector_fcaf3d import DiverDetectorFCAF3D

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[skip] full FCAF3D model forward/backward: needs CUDA (spconv)")
        return

    cfg = load_config("configs/exp_fcaf3d.yaml")
    model = DiverDetectorFCAF3D(cfg).to(device)
    pc_range = cfg["DATA"]["POINT_CLOUD_RANGE"]

    def rand_points(n):
        pts = torch.rand(n, 4)
        pts[:, 0] = pc_range[0] + pts[:, 0] * (pc_range[3] - pc_range[0])
        pts[:, 1] = pc_range[1] + pts[:, 1] * (pc_range[4] - pc_range[1])
        pts[:, 2] = pc_range[2] + pts[:, 2] * (pc_range[5] - pc_range[2])
        return pts

    points_list = [rand_points(300), rand_points(120)]
    batch_idx_list = [torch.full((300,), 0, dtype=torch.long), torch.full((120,), 1, dtype=torch.long)]
    gt0 = torch.zeros(3, 10); gt0[:, 3:6] = 1.0; gt0[:, 6] = 1.0
    gt0[:, :3] = rand_points(3)[:, :3]
    gt1 = torch.zeros(0, 10)

    batch = {
        "points": torch.cat(points_list, dim=0),
        "point_batch_idx": torch.cat(batch_idx_list, dim=0),
        "gt_boxes": [gt0, gt1],
        "frame_ids": ["synthetic_0", "synthetic_1"],
        "batch_size": 2,
    }

    losses, level_preds, level_batch_idx, gt_boxes_list = model.loss(batch, device)
    assert torch.isfinite(losses["total"]), f"total loss is not finite: {losses['total']}"
    losses["total"].backward()

    n_params_with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    n_params_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[ok] full FCAF3D model forward/backward: total_loss={losses['total'].item():.4f}, n_pos={losses['n_pos']}, "
          f"{n_params_with_grad}/{n_params_total} params got gradients")

    decoded = model.decode(level_preds, level_batch_idx, batch_size=2, score_threshold=0.0)
    assert len(decoded) == 2
    print(f"[ok] decode(): sample0 kept {decoded[0]['center'].shape[0]} locations, "
          f"sample1 kept {decoded[1]['center'].shape[0]} (score_threshold=0.0 keeps every location)")


if __name__ == "__main__":
    check_differentiable_iou_matches_eval_only()
    check_assign_multilevel_basic()
    check_fcaf3d_head_shapes()
    check_backbone_forward_multilevel()
    check_decode_nms()
    check_full_fcaf3d_model_forward_backward()
    print("\nall FCAF3D-component checks passed.")
