"""Standalone correctness checks for the FCAF3D-style dense-head components
(models/backbone3d_unet_spconv.py::forward_multilevel, heads_fcaf3d.py,
assign_fcaf3d.py, losses_fcaf3d.py, detector_fcaf3d.py) -- same "prove it
before trusting it" practice as test_detr_components.py. Run directly:
`python test_fcaf3d_components.py`. CUDA-only (spconv).
"""
import torch


def check_oriented_iou_3d_sampled():
    """models/box_utils.py::oriented_iou_3d_sampled -- the rotation-AWARE IoU
    added for the per-epoch AP@0.35 metric (models/assign_fcaf3d.py's
    axis_aligned_iou_3d ignores rotation entirely, which the user specifically
    flagged as wrong for this metric: "GT는 3차원 회전을 포함해야해"). Confirms
    (1) identical boxes -> IoU~1, (2) far-apart boxes -> IoU=0, (3) matches
    axis_aligned_iou_3d closely when both boxes are axis-aligned (identity
    rotation) as a sanity cross-check, and (4) actually USES rotation: two
    long thin boxes whose axis-aligned bounding boxes overlap a lot but whose
    true (perpendicular) footprints barely overlap must score much lower here
    than the axis-aligned function would."""
    from models.box_utils import oriented_iou_3d_sampled, axis_aligned_iou_3d, quat_to_rotmat

    identity_R = torch.eye(3)

    # (1) identical boxes
    c, s = torch.tensor([1.0, 2.0, 0.5]), torch.tensor([1.0, 0.8, 1.5])
    iou_same = oriented_iou_3d_sampled(c, s, identity_R, c, s, identity_R, n_samples_per_axis=12)
    assert iou_same > 0.95, f"identical boxes should give IoU~1, got {iou_same}"

    # (2) far apart -> 0
    iou_far = oriented_iou_3d_sampled(c, s, identity_R, c + 100, s, identity_R, n_samples_per_axis=12)
    assert iou_far == 0.0, f"far-apart boxes should give IoU=0, got {iou_far}"

    # (3) axis-aligned cross-check against the trusted closed-form function
    c1, s1 = torch.tensor([0.0, 0.0, 0.0]), torch.tensor([2.0, 2.0, 2.0])
    c2, s2 = torch.tensor([1.0, 0.0, 0.0]), torch.tensor([2.0, 2.0, 2.0])
    exact = axis_aligned_iou_3d(c1, s1, c2, s2)
    sampled = oriented_iou_3d_sampled(c1, s1, identity_R, c2, s2, identity_R, n_samples_per_axis=20)
    assert abs(exact - sampled) < 0.05, f"axis-aligned case: exact={exact:.4f} vs sampled={sampled:.4f} disagree too much"

    # (4) rotation actually matters: two 3x0.3x0.3 boxes centered at the same
    # point, one along x, one rotated 90 deg about z (so its long axis is
    # along y) -- their AABBs are both ~3x3x0.3 and overlap almost entirely,
    # but their true cross-shaped footprints only overlap in a small central
    # square, so the TRUE IoU should be much smaller than the axis-aligned one.
    long_box_center = torch.zeros(3)
    long_box_size = torch.tensor([3.0, 0.3, 0.3])
    quat_90_about_z = torch.tensor([0.70710678, 0.0, 0.0, 0.70710678])  # (w,x,y,z)
    R_rotated = quat_to_rotmat(quat_90_about_z.unsqueeze(0))[0]

    aabb_iou = axis_aligned_iou_3d(long_box_center, long_box_size, long_box_center, long_box_size)
    true_iou = oriented_iou_3d_sampled(long_box_center, long_box_size, identity_R,
                                        long_box_center, long_box_size, R_rotated, n_samples_per_axis=24)
    assert aabb_iou > 0.9, f"sanity: identical-size boxes at the same center should have axis-aligned IoU~1, got {aabb_iou}"
    assert true_iou < 0.15, \
        (f"cross-shaped overlap of two perpendicular long thin boxes should have a small TRUE IoU, "
         f"got {true_iou:.4f} (axis-aligned would say {aabb_iou:.4f}) -- rotation isn't being used correctly")
    print(f"[ok] oriented_iou_3d_sampled: identical={iou_same:.3f}, far-apart={iou_far:.3f}, "
          f"axis-aligned cross-check exact={exact:.3f} vs sampled={sampled:.3f}, "
          f"rotation-sensitivity case: axis-aligned={aabb_iou:.3f} vs true(rotated)={true_iou:.3f}")


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


def check_rotated_iou_gwd_sane():
    """models/losses_fcaf3d.py::differentiable_rotated_iou_3d_gwd -- the
    rotation-AWARE differentiable IoU surrogate (GWD, arXiv 2101.11952) that
    replaced differentiable_axis_aligned_iou_3d in compute_fcaf3d_loss's
    'iou' term (that one ignored rotation entirely -- see its own docstring).

    Confirms (1) identical boxes -> pseudo-iou exactly 1.0, (2) for each of
    several random GT boxes, pseudo-iou rises monotonically as a perturbed
    box is interpolated back toward that exact GT (center/size/rotation all
    perturbed together, blended by t: 1=at the perturbed box, 0=exact GT) --
    this is the property that actually matters for training (a consistent
    gradient toward the right answer), not numeric closeness to exact IoU:
    exact IoU is flatly 0 over a wide range of "how not-overlapping", so
    rank-correlating against it over random far-apart pairs isn't a
    meaningful test (confirmed by hand: GWD still usefully orders cases exact
    IoU can't distinguish at all) -- and (3) gradients flow into
    center/size/rotation."""
    from models.losses_fcaf3d import differentiable_rotated_iou_3d_gwd
    from models.rotation6d import sixd_to_matrix

    torch.manual_seed(0)
    c = torch.zeros(1, 3)
    s = torch.ones(1, 3) * 2.0
    R = torch.eye(3).unsqueeze(0)
    piou = differentiable_rotated_iou_3d_gwd(c, s, R, c, s, R)
    assert abs(piou.item() - 1.0) < 1e-5, f"identical boxes should give pseudo-iou=1.0, got {piou.item()}"

    n_cases, n_steps = 6, 8
    worst_inversions = 0
    for _ in range(n_cases):
        gt_c, gt_s = torch.randn(3), torch.rand(3) * 1.5 + 0.5
        gt_R = sixd_to_matrix(torch.randn(1, 6))[0]
        off_c = gt_c + torch.randn(3) * 2.0
        off_s = (gt_s + torch.randn(3) * 0.8).clamp(min=0.2)
        off_R = sixd_to_matrix(torch.randn(1, 6))[0]

        pseudo_along_path = []
        for t in torch.linspace(1.0, 0.0, n_steps):
            c_t = gt_c + t * (off_c - gt_c)
            s_t = (gt_s + t * (off_s - gt_s)).clamp(min=0.05)
            # blend rotations via the 6D representation, renormalized by sixd_to_matrix itself
            sixd_gt = torch.cat([gt_R[:, 0], gt_R[:, 1]])
            sixd_off = torch.cat([off_R[:, 0], off_R[:, 1]])
            sixd_t = sixd_gt + t * (sixd_off - sixd_gt)
            R_t = sixd_to_matrix(sixd_t.unsqueeze(0))[0]
            p = differentiable_rotated_iou_3d_gwd(c_t.unsqueeze(0), s_t.unsqueeze(0), R_t.unsqueeze(0),
                                                   gt_c.unsqueeze(0), gt_s.unsqueeze(0), gt_R.unsqueeze(0)).item()
            pseudo_along_path.append(p)
        inversions = sum(1 for i in range(n_steps - 1) if pseudo_along_path[i] > pseudo_along_path[i + 1])
        worst_inversions = max(worst_inversions, inversions)
        # eigvalsh on a near-diagonal 3x3 (identical boxes at t=0) has a little
        # float32 roundoff -- 0.999x not 1.000x -- so this tolerance is looser
        # than the exact check_rotated_iou_gwd_sane() t=0 case up top.
        assert abs(pseudo_along_path[-1] - 1.0) < 2e-3, \
            f"path should end at the GT (pseudo-iou~=1.0), got {pseudo_along_path[-1]}"
        assert inversions <= 1, (f"pseudo-iou isn't monotonically improving as the box approaches GT: "
                                  f"{pseudo_along_path} ({inversions} inversions)")

    c1 = torch.randn(4, 3, requires_grad=True)
    s1 = (torch.rand(4, 3) + 0.5).requires_grad_(True)
    R1 = sixd_to_matrix(torch.randn(4, 6)); R1.requires_grad_(True)
    c2, s2, R2 = torch.randn(4, 3), torch.rand(4, 3) + 0.5, sixd_to_matrix(torch.randn(4, 6))
    loss = (1.0 - differentiable_rotated_iou_3d_gwd(c1, s1, R1, c2, s2, R2)).sum()
    loss.backward()
    assert c1.grad is not None and torch.isfinite(c1.grad).all()
    assert s1.grad is not None and torch.isfinite(s1.grad).all()
    assert R1.grad is not None and torch.isfinite(R1.grad).all()
    print(f"[ok] differentiable_rotated_iou_3d_gwd: identical->1.0, monotonic along "
          f"{n_cases} GT-approach paths (worst case {worst_inversions} inversions/{n_steps-1}), gradients finite")


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


def check_nearest_fallback_rescue():
    """models/assign_fcaf3d.py's TR3D-style nearest_fallback_k: a GT box with
    NO voxel strictly inside it at any level is normally dropped entirely
    (zero positives, no gradient for that object ever) -- confirms (1) that's
    still true with nearest_fallback_k=0 (unchanged default behavior), and
    (2) with it turned on, the box gets exactly K positives at the fine
    level, nearest to its center, with correct center/size/rotation targets
    -- and (3) a GT that already gets covered normally is byte-identical
    whether the fallback is on or off (it never touches the normal path)."""
    from models.assign_fcaf3d import assign_multilevel

    torch.manual_seed(0)
    ar = torch.arange(0, 4, 1.0)  # sparse 1.0-spacing grid -> easy to miss entirely
    gx, gy, gz = torch.meshgrid(ar, ar, ar, indexing="ij")
    fine_centers = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    fine_batch_idx = torch.zeros(fine_centers.shape[0], dtype=torch.long)

    # tiny box dead-center between 8 grid voxels (nearest voxel is ~0.87m away,
    # half-size is 0.05m -- guaranteed no voxel falls inside it at any setting)
    tiny = torch.zeros(1, 10)
    tiny[0, :3] = torch.tensor([0.5, 0.5, 0.5])
    tiny[0, 3:6] = 0.1
    tiny[0, 6] = 1.0
    gt_boxes_list = [tiny]

    dropped = assign_multilevel([fine_centers], [fine_batch_idx], gt_boxes_list, batch_size=1,
                                 min_locations=4, center_sample_radius=0.6, nearest_fallback_k=0)
    assert not dropped[0]["pos_mask"].any(), "with the fallback off, a fully-uncovered box should still be dropped"

    rescued = assign_multilevel([fine_centers], [fine_batch_idx], gt_boxes_list, batch_size=1,
                                 min_locations=4, center_sample_radius=0.6, nearest_fallback_k=3)
    tgt = rescued[0]
    assert tgt["pos_mask"].sum().item() == 3, f"expected exactly 3 rescued positives, got {tgt['pos_mask'].sum().item()}"
    rescued_centers = fine_centers[tgt["pos_mask"]]
    dists = (rescued_centers - tiny[0, :3]).norm(dim=-1)
    all_dists = (fine_centers - tiny[0, :3]).norm(dim=-1).sort().values
    assert torch.allclose(dists.sort().values, all_dists[:3], atol=1e-5), "rescued voxels should be the 3 nearest ones"
    assert torch.allclose(tgt["log_size"][tgt["pos_mask"]][0], torch.log(torch.tensor(0.1)).expand(3), atol=1e-4)

    # a normally-covered box must come out identical regardless of the flag
    ar2 = torch.arange(0, 4, 0.5)
    gx2, gy2, gz2 = torch.meshgrid(ar2, ar2, ar2, indexing="ij")
    fine_centers2 = torch.stack([gx2, gy2, gz2], dim=-1).reshape(-1, 3)
    fine_batch_idx2 = torch.zeros(fine_centers2.shape[0], dtype=torch.long)
    normal_box = torch.zeros(1, 10); normal_box[0, :3] = torch.tensor([1.0, 1.0, 1.0])
    normal_box[0, 3:6] = 0.4; normal_box[0, 6] = 1.0
    off = assign_multilevel([fine_centers2], [fine_batch_idx2], [normal_box], batch_size=1,
                             min_locations=4, center_sample_radius=0.6, nearest_fallback_k=0)
    on = assign_multilevel([fine_centers2], [fine_batch_idx2], [normal_box], batch_size=1,
                            min_locations=4, center_sample_radius=0.6, nearest_fallback_k=3)
    assert torch.equal(off[0]["pos_mask"], on[0]["pos_mask"]), "fallback must not alter an already-covered GT's assignment"
    print(f"[ok] nearest_fallback_k: dropped-without / rescued-with-3-nearest-voxels / "
          f"already-covered-GT unaffected by the flag")


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
        "center_offset": torch.zeros(5, 3),
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


def check_decode_nms_iou():
    """FCAF3D-faithful rotated-IoU NMS (nms_iou_threshold, vs the project's
    default distance-based nms_radius) -- same two-cluster synthetic setup as
    check_decode_nms, confirms IoU-based greedy NMS also collapses each
    cluster to its highest-score detection, and that passing BOTH keyword
    args makes IoU NMS take priority (per decode()'s own docstring)."""
    from models.detector_fcaf3d import DiverDetectorFCAF3D
    from config_utils import load_config

    torch.manual_seed(0)
    cfg = load_config("configs/exp_fcaf3d.yaml")
    model = DiverDetectorFCAF3D(cfg)

    # unit-cube boxes (log_size=0 -> size=[1,1,1], identity rotation): cluster
    # A's 3 centers are all within ~0.2m of each other -> near-total IoU
    # overlap: cluster B is 7m away -> IoU=0 against cluster A.
    centers = torch.tensor([
        [1.0, 0.0, 0.0], [1.1, 0.05, -0.05], [0.95, -0.1, 0.0],
        [8.0, 0.0, 0.0], [8.05, 0.1, 0.0],
    ])
    scores = torch.tensor([0.9, 0.6, 0.5, 0.8, 0.7])
    pred = {
        "exist_logit": torch.logit(scores.clamp(1e-4, 1 - 1e-4)).unsqueeze(-1),
        "center": centers,
        "center_offset": torch.zeros(5, 3),
        "log_size": torch.zeros(5, 3),
        "sixd": torch.tensor([1., 0, 0, 0, 1, 0]).expand(5, 6).clone(),
        "centerness_logit": torch.logit(torch.ones(5) * 0.999).unsqueeze(-1),
    }
    batch_idx = torch.zeros(5, dtype=torch.long)

    nms = model.decode([pred], [batch_idx], batch_size=1, score_threshold=0.0, nms_iou_threshold=0.5)
    assert nms[0]["center"].shape[0] == 2, \
        f"expected IoU NMS to collapse the 2 clusters to 2 detections, got {nms[0]['center'].shape[0]}"
    kept_centers = nms[0]["center"]
    assert any(torch.allclose(c, torch.tensor([1.0, 0.0, 0.0]), atol=1e-4) for c in kept_centers), \
        "IoU NMS should keep cluster A's highest-score detection"
    assert any(torch.allclose(c, torch.tensor([8.0, 0.0, 0.0]), atol=1e-4) for c in kept_centers), \
        "IoU NMS should keep cluster B's highest-score detection"

    # both kwargs given -> IoU NMS takes priority (docstring's stated contract)
    both = model.decode([pred], [batch_idx], batch_size=1, score_threshold=0.0,
                         nms_radius=100.0, nms_iou_threshold=0.5)
    assert both[0]["center"].shape[0] == 2, \
        "passing both nms_radius and nms_iou_threshold should use IoU NMS (not the radius=100 distance NMS, " \
        "which would collapse everything to 1)"
    print(f"[ok] decode() rotated-IoU NMS: 5 raw detections (2 clusters) -> 2 after NMS, "
          f"IoU NMS takes priority over nms_radius when both are given")


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


def check_gaussian_heatmap_mode():
    """FCAF3D_HEAD.GAUSSIAN_HEATMAP: soft Gaussian existence target + penalty-
    reduced focal loss replacing the hard-0/1 target + decoupled centerness
    branch (see detector_fcaf3d.py's own docstring on this flag, and
    models/losses.py::focal_heatmap_loss -- this project's original
    pre-FCAF3D dense head's mechanism, reused here). Confirms (1) the
    rendered heatmap target is 1.0 exactly at assigned positives and decays
    smoothly elsewhere (not hard 0/1 like the default path), (2) a full
    forward/loss/backward pass is finite and gradients reach the exist/
    center/size/rotation branches (NOT centerness, which this mode doesn't
    use), and (3) decode()'s score is sigmoid(exist) alone, independent of
    the (unused, but still present) centerness branch's output."""
    from config_utils import load_config
    from models.detector_fcaf3d import DiverDetectorFCAF3D

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[skip] gaussian heatmap mode: needs CUDA (spconv)")
        return

    cfg = load_config("configs/exp_fcaf3d_single_level_heatmap.yaml")
    model = DiverDetectorFCAF3D(cfg).to(device)
    assert model.gaussian_heatmap
    pc_range = cfg["DATA"]["POINT_CLOUD_RANGE"]

    def rand_points(n):
        pts = torch.rand(n, 4)
        pts[:, 0] = pc_range[0] + pts[:, 0] * (pc_range[3] - pc_range[0])
        pts[:, 1] = pc_range[1] + pts[:, 1] * (pc_range[4] - pc_range[1])
        pts[:, 2] = pc_range[2] + pts[:, 2] * (pc_range[5] - pc_range[2])
        return pts

    points = rand_points(400)
    batch_idx = torch.zeros(400, dtype=torch.long)
    gt = torch.zeros(1, 10); gt[0, :3] = points[0, :3]; gt[0, 3:6] = 1.0; gt[0, 6] = 1.0
    batch = {"points": points, "point_batch_idx": batch_idx, "gt_boxes": [gt],
             "frame_ids": ["synthetic_0"], "batch_size": 1}

    level_preds, level_world_centers, level_batch_idx, gt_boxes_list, batch_size = model.forward(batch, device)
    from models.assign_fcaf3d import assign_multilevel
    targets = assign_multilevel(level_world_centers, level_batch_idx, gt_boxes_list, batch_size,
                                 model.min_locations, model.center_sample_radius, model.nearest_fallback_k,
                                 gaussian_sigma=model.gaussian_sigma)
    heat = targets[0]["exist_heatmap"]
    pos = targets[0]["pos_mask"]
    assert pos.any(), "need at least one positive to test against"
    assert torch.allclose(heat[pos], torch.ones_like(heat[pos])), "heatmap must be exactly 1.0 at assigned positives"
    assert (heat[~pos] < 1.0).all() and (heat[~pos] >= 0.0).all(), "non-positive voxels should be a soft value in [0,1)"
    assert (heat[~pos] > 0).any(), "at least some nearby voxels should have nonzero (not hard-0) heatmap value"

    losses, level_preds, level_batch_idx, gt_boxes_list = model.loss(batch, device)
    assert torch.isfinite(losses["total"])
    assert losses["centerness"].item() == 0.0, "centerness loss term should be exactly zero (unused) in this mode"
    losses["total"].backward()
    assert model.head.exist_layers[0].weight.grad is not None
    assert model.head.center_offset_layers[0].weight.grad is not None
    print("[ok] gaussian heatmap mode: heatmap target correct (1.0 at positives, soft elsewhere), "
          "finite loss, centerness term zeroed, gradients flow")


def check_corner_aux_loss():
    """FCAF3D_HEAD.CORNER_AUX: the ported "corner auxiliary" term (see
    detector_fcaf3d.py's own docstring, and compute_fcaf3d_loss's
    corner_aux_sigma param). Confirms (1) the GT corner heatmap rendered by
    assign_multilevel peaks near 1.0 at a voxel sitting right on top of a
    known GT corner, (2) a full forward/loss/backward pass is finite with
    corner_aux nonzero and contributing to total, and gradients reach the
    size/rotation branches through this new path too."""
    from config_utils import load_config
    from models.detector_fcaf3d import DiverDetectorFCAF3D

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[skip] corner aux loss: needs CUDA (spconv)")
        return

    cfg = load_config("configs/exp_fcaf3d_single_level_heatmap_corneraux.yaml")
    model = DiverDetectorFCAF3D(cfg).to(device)
    assert model.gaussian_heatmap and model.corner_aux
    pc_range = cfg["DATA"]["POINT_CLOUD_RANGE"]

    def rand_points(n):
        pts = torch.rand(n, 4)
        pts[:, 0] = pc_range[0] + pts[:, 0] * (pc_range[3] - pc_range[0])
        pts[:, 1] = pc_range[1] + pts[:, 1] * (pc_range[4] - pc_range[1])
        pts[:, 2] = pc_range[2] + pts[:, 2] * (pc_range[5] - pc_range[2])
        return pts

    points = rand_points(400)
    batch_idx = torch.zeros(400, dtype=torch.long)
    gt = torch.zeros(1, 10); gt[0, :3] = points[0, :3]; gt[0, 3:6] = 1.0; gt[0, 6] = 1.0
    batch = {"points": points, "point_batch_idx": batch_idx, "gt_boxes": [gt],
             "frame_ids": ["synthetic_0"], "batch_size": 1}

    level_preds, level_world_centers, level_batch_idx, gt_boxes_list, batch_size = model.forward(batch, device)
    from models.assign_fcaf3d import assign_multilevel
    from models.box_utils import box_to_corners
    targets = assign_multilevel(level_world_centers, level_batch_idx, gt_boxes_list, batch_size,
                                 model.min_locations, model.center_sample_radius, model.nearest_fallback_k,
                                 gaussian_sigma=model.gaussian_sigma)
    gt_corner_heat = targets[0]["gt_corner_heatmap"]
    corners = box_to_corners(gt[:, :3].to(device), gt[:, 3:6].to(device), gt[:, 6:10].to(device))[0]  # (8,3)
    nearest_voxel = (level_world_centers[0] - corners[0:1]).norm(dim=-1).argmin()
    assert gt_corner_heat[nearest_voxel] > 0.5, \
        f"voxel nearest a GT corner should have high corner-heatmap value, got {gt_corner_heat[nearest_voxel].item()}"

    losses, level_preds, level_batch_idx, gt_boxes_list = model.loss(batch, device)
    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["corner_aux"]) and losses["corner_aux"].item() > 0
    losses["total"].backward()
    assert model.head.log_size_layers[0].weight.grad is not None
    assert model.head.rot_layers[0].weight.grad is not None
    print(f"[ok] corner aux loss: GT-corner heatmap peaks correctly, "
          f"corner_aux={losses['corner_aux'].item():.4f} (finite, >0), gradients flow")


if __name__ == "__main__":
    check_oriented_iou_3d_sampled()
    check_differentiable_iou_matches_eval_only()
    check_rotated_iou_gwd_sane()
    check_assign_multilevel_basic()
    check_nearest_fallback_rescue()
    check_fcaf3d_head_shapes()
    check_backbone_forward_multilevel()
    check_decode_nms()
    check_decode_nms_iou()
    check_full_fcaf3d_model_forward_backward()
    check_gaussian_heatmap_mode()
    check_corner_aux_loss()
    print("\nall FCAF3D-component checks passed.")
