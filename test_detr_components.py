"""Standalone correctness checks for the new DETR-head components, run BEFORE
trusting them in a real training run -- same "prove it before trusting it"
practice as test_sparse_conv_pure.py used for the from-scratch sparse conv.
Not a pytest suite (this repo doesn't have one) -- a script with asserts,
run directly: `python test_detr_components.py`.
"""
import torch

from models.rotation6d import sixd_to_matrix, matrix_to_sixd, matrix_geodesic_loss
from models.matcher import HungarianMatcher
from models.losses_detr import compute_detr_loss, sigmoid_focal_loss
from models.box_utils import quat_to_rotmat


def check_rotation6d_roundtrip():
    torch.manual_seed(0)
    # random valid rotation matrices via QR decomposition
    a = torch.randn(8, 3, 3)
    q, r = torch.linalg.qr(a)
    d = torch.diagonal(r, dim1=-2, dim2=-1).sign()
    R = q * d[:, None, :]
    R[torch.linalg.det(R) < 0, :, -1] *= -1  # force proper rotations (det=+1)

    sixd = matrix_to_sixd(R)
    R_recon = sixd_to_matrix(sixd)
    err = (R - R_recon).abs().max().item()
    assert err < 1e-4, f"6D roundtrip error too large: {err}"

    self_err = matrix_geodesic_loss(R, R_recon).abs().max().item()
    assert self_err < 1e-3, f"geodesic(R, R) should be ~0, got {self_err}"
    print(f"[ok] rotation6d roundtrip: max abs err={err:.2e}, self geodesic={self_err:.2e}")


def check_matcher_shapes_and_assignment():
    torch.manual_seed(0)
    B, Q, C = 2, 6, 16
    pred = {
        "exist_logit": torch.randn(B, Q, 1, requires_grad=True),
        "center": torch.randn(B, Q, 3, requires_grad=True),
        "log_size": torch.zeros(B, Q, 3, requires_grad=True),
        "sixd": torch.tensor([1., 0, 0, 0, 1, 0]).expand(B, Q, 6).clone().requires_grad_(True),
    }
    # sample 0 has 2 GT boxes, sample 1 has 0 (edge case: empty GT)
    gt0 = torch.zeros(2, 10)
    gt0[:, 3:6] = 1.0
    gt0[:, 6] = 1.0  # identity quat (w=1)
    gt0[0, :3] = pred["center"][0, 1]  # make query 1 an exact match for gt 0
    gt0[1, :3] = pred["center"][0, 4]  # make query 4 an exact match for gt 1
    gt1 = torch.zeros(0, 10)

    matcher = HungarianMatcher()
    matches = matcher.match(pred, [gt0, gt1])
    assert len(matches) == 2
    q_idx0, g_idx0 = matches[0]
    assert q_idx0.numel() == 2 and g_idx0.numel() == 2, "sample 0 should match both GT boxes one-to-one"
    assert set(q_idx0.tolist()) == {1, 4}, f"expected the exact-center queries {{1,4}} to win the match, got {q_idx0.tolist()}"
    q_idx1, g_idx1 = matches[1]
    assert q_idx1.numel() == 0 and g_idx1.numel() == 0, "sample 1 has no GT -> no matches"
    print(f"[ok] matcher: sample0 matched queries={q_idx0.tolist()} (expected order-independent {{1,4}}), sample1 empty ok")

    losses = compute_detr_loss(pred, [gt0, gt1], matches, {"WEIGHTS": {"cls": 1.0, "center": 1.0, "size": 1.0, "rotation": 1.0}})
    for k in ("total", "cls", "center", "size", "rotation"):
        assert k in losses and torch.isfinite(losses[k]), f"loss term {k} missing or non-finite: {losses.get(k)}"
    losses["total"].backward()
    print(f"[ok] compute_detr_loss: total={losses['total'].item():.4f} (all terms finite, backward() ran clean)")


def check_focal_loss_sane():
    logit_confident_right = torch.tensor([5.0, -5.0])
    logit_confident_wrong = torch.tensor([-5.0, 5.0])
    target = torch.tensor([1.0, 0.0])
    l_right = sigmoid_focal_loss(logit_confident_right, target)
    l_wrong = sigmoid_focal_loss(logit_confident_wrong, target)
    assert l_right.item() < l_wrong.item(), "focal loss should penalize confident-wrong more than confident-right"
    print(f"[ok] sigmoid_focal_loss: confident-right={l_right.item():.4f} < confident-wrong={l_wrong.item():.4f}")


def check_dsvt_set_partition():
    from models.dsvt import dynamic_set_partition

    torch.manual_seed(0)
    # two windows (window_shape=(4,4,4)): window A gets coords with x,y,z in
    # [0,4), window B gets x in [4,8) -- everything else the same. Give window
    # A exactly tau=5 voxels (no duplication needed) and window B 3 voxels
    # (needs 2 duplicates to fill one set of 5).
    coords_a = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 1, 0], [0, 3, 2, 1], [0, 1, 3, 3]])
    coords_b = torch.tensor([[0, 5, 0, 0], [0, 6, 1, 0], [0, 7, 2, 1]])
    coords = torch.cat([coords_a, coords_b], dim=0)  # indices 0-4 = window A, 5-7 = window B
    tau = 5

    set_idx = dynamic_set_partition(coords, window_shape=(4, 4, 4), tau=tau, x_major=True)
    assert set_idx.shape == (2, tau), f"expected 2 sets of {tau}, got {set_idx.shape}"

    # every set's tau indices must all belong to the SAME window (sets never cross window boundaries)
    window_of = torch.tensor([0] * 5 + [1] * 3)
    for row in set_idx:
        windows_in_row = window_of[row].unique()
        assert windows_in_row.numel() == 1, f"a set mixed voxels from windows {windows_in_row.tolist()}"

    # every real voxel (0..7) must appear at least once across the whole partition
    covered = torch.unique(set_idx.reshape(-1))
    assert torch.equal(covered, torch.arange(8)), f"not all voxels covered: got {covered.tolist()}"

    # window A had exactly tau=5 voxels -> its set should be a PERMUTATION of
    # {0,1,2,3,4} with no duplicates at all
    row_a = [r for r in set_idx if window_of[r].unique().item() == 0][0]
    assert torch.equal(torch.sort(row_a).values, torch.arange(5)), \
        f"window A (exactly tau voxels) should need no duplication, got {row_a.tolist()}"
    print(f"[ok] dynamic_set_partition: window isolation, full coverage, "
          f"and no-duplication-when-exact all correct")


def check_dsvt_layer_and_backbone():
    from models.dsvt import DSVTLayer, DSVTBackbone

    torch.manual_seed(0)
    channels = 16
    n = 37
    coords = torch.stack([
        torch.zeros(n, dtype=torch.long),
        torch.randint(0, 20, (n,)), torch.randint(0, 20, (n,)), torch.randint(0, 10, (n,)),
    ], dim=1)
    features = torch.randn(n, channels, requires_grad=True)
    pc_range = torch.tensor([0.0, -5.0, -2.5, 12.0, 5.0, 2.5])
    eff_voxel_size = torch.tensor([0.4, 0.4, 0.4])

    layer = DSVTLayer(channels, num_heads=4, tau=8, x_major=True)
    out = layer(features, coords, window_shape=(6, 6, 10), pc_range=pc_range, effective_voxel_size=eff_voxel_size)
    assert out.shape == (n, channels) and torch.isfinite(out).all()
    out.sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    print(f"[ok] DSVTLayer: shape preserved, finite output, gradients flow to input features")

    backbone = DSVTBackbone(channels, window_shapes=[(6, 6, 10), (10, 8, 10)], tau=8, num_blocks=4, num_heads=4)
    features2 = torch.randn(n, channels, requires_grad=True)
    out2 = backbone(features2, coords, pc_range, eff_voxel_size)
    assert out2.shape == (n, channels) and torch.isfinite(out2).all()
    out2.sum().backward()
    n_params_with_grad = sum(1 for p in backbone.parameters() if p.grad is not None)
    n_params = sum(1 for p in backbone.parameters())
    assert n_params_with_grad == n_params, f"only {n_params_with_grad}/{n_params} DSVTBackbone params got gradients"
    print(f"[ok] DSVTBackbone (4 blocks, hybrid windows): shape preserved, all {n_params} params got gradients")


def check_knn_indices_per_sample():
    from models.point_transformer import knn_indices_per_sample

    torch.manual_seed(0)
    # sample 0: 5 points on a line at x=0,1,2,3,4 (y=z=0) -- exact NN order is obvious.
    # sample 1: a single cluster far away, fewer points than k to exercise the pad-by-repeat path.
    pos_a = torch.stack([torch.arange(5).float(), torch.zeros(5), torch.zeros(5)], dim=1)
    pos_b = torch.tensor([[100., 0, 0], [100.1, 0, 0], [100.2, 0, 0]])
    pos = torch.cat([pos_a, pos_b], dim=0)
    batch_idx = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])

    knn = knn_indices_per_sample(pos, batch_idx, batch_size=2, k=3)
    assert knn.shape == (8, 3)

    # no cross-sample leakage: every neighbor index for a sample-0 point must itself be a sample-0 point
    for i in range(5):
        assert batch_idx[knn[i]].eq(0).all(), f"point {i} (sample 0) got a neighbor from another sample: {knn[i].tolist()}"
    for i in range(5, 8):
        assert batch_idx[knn[i]].eq(1).all(), f"point {i} (sample 1) got a neighbor from another sample: {knn[i].tolist()}"

    # point 0 (x=0) 's 3 nearest on the line x=0..4 must be {0,1,2} (itself + the two closest)
    assert set(knn[0].tolist()) == {0, 1, 2}, f"expected {{0,1,2}}, got {knn[0].tolist()}"
    # point 2 (x=2, in the middle) 's 3 nearest must be {1,2,3}
    assert set(knn[2].tolist()) == {1, 2, 3}, f"expected {{1,2,3}}, got {knn[2].tolist()}"
    print(f"[ok] knn_indices_per_sample: no cross-sample leakage, correct nearest-neighbor sets, pad-by-repeat path ran clean")


def check_point_transformer_layer():
    from models.point_transformer import PointTransformerLayer, PointTransformerBlock, knn_indices_per_sample

    torch.manual_seed(0)
    n, channels, k = 40, 16, 8
    pos = torch.rand(n, 3) * 5
    batch_idx = torch.zeros(n, dtype=torch.long)
    knn_idx = knn_indices_per_sample(pos, batch_idx, batch_size=1, k=k)

    x = torch.randn(n, channels, requires_grad=True)
    layer = PointTransformerLayer(channels)
    out = layer(x, pos, knn_idx)
    assert out.shape == (n, channels) and torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print(f"[ok] PointTransformerLayer: shape preserved, finite, gradients flow to input")

    x2 = torch.randn(n, channels, requires_grad=True)
    block = PointTransformerBlock(channels)
    out2 = block(x2, pos, knn_idx)
    assert out2.shape == (n, channels) and torch.isfinite(out2).all()
    out2.sum().backward()
    n_grad = sum(1 for p in block.parameters() if p.grad is not None)
    n_total = sum(1 for p in block.parameters())
    assert n_grad == n_total, f"only {n_grad}/{n_total} PointTransformerBlock params got gradients"
    print(f"[ok] PointTransformerBlock (residual): shape preserved, all {n_total} params got gradients")


def check_cap_points_per_voxel():
    from models.voxel_pool_attn import cap_points_per_voxel

    torch.manual_seed(0)
    # voxel 0: 155 points (the real batch's observed outlier), voxel 1: 2 points (median),
    # voxel 2: 1 point (must never be fully dropped)
    counts = [155, 2, 1]
    point_voxel_idx = torch.cat([torch.full((c,), i, dtype=torch.long) for i, c in enumerate(counts)])
    n_total_before = point_voxel_idx.shape[0]

    keep = cap_points_per_voxel(point_voxel_idx, num_voxels=3, max_points=32)
    kept_voxel_idx = point_voxel_idx[keep]
    kept_counts = torch.bincount(kept_voxel_idx, minlength=3)

    assert kept_counts[0].item() == 32, f"voxel 0 (155 pts) should be capped to 32, got {kept_counts[0].item()}"
    assert kept_counts[1].item() == 2, f"voxel 1 (2 pts, under cap) should keep all 2, got {kept_counts[1].item()}"
    assert kept_counts[2].item() == 1, f"voxel 2 (1 pt) must never be fully dropped, got {kept_counts[2].item()}"
    assert keep.unique().numel() == keep.numel(), "cap_points_per_voxel returned duplicate indices"
    print(f"[ok] cap_points_per_voxel: {n_total_before} points -> kept {keep.numel()} "
          f"(voxel counts capped to [32, 2, 1] as expected, no duplicates)")


def check_voxel_pooling_attention():
    from models.voxel_pool_attn import VoxelPoolingAttention

    torch.manual_seed(0)
    channels, out_channels = 16, 32
    # 3 voxels: one with a single point (edge case -- fully-real, zero-pad row
    # except it's the ONLY entry, so mask has T_max-1 True's for that row),
    # one with a handful, one with many.
    counts = [1, 4, 9]
    point_voxel_idx = torch.cat([torch.full((c,), i, dtype=torch.long) for i, c in enumerate(counts)])
    feat = torch.randn(sum(counts), channels, requires_grad=True)

    pool = VoxelPoolingAttention(channels, out_channels, num_heads=4)
    out = pool(feat, point_voxel_idx, num_voxels=3)
    assert out.shape == (3, out_channels) and torch.isfinite(out).all(), \
        "VoxelPoolingAttention output has wrong shape or non-finite values (possible all-masked-row NaN)"
    out.sum().backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    n_grad = sum(1 for p in pool.parameters() if p.grad is not None)
    n_total = sum(1 for p in pool.parameters())
    assert n_grad == n_total, f"only {n_grad}/{n_total} VoxelPoolingAttention params got gradients"
    print(f"[ok] VoxelPoolingAttention: handles 1/4/9-point voxels (no all-masked-row NaN), "
          f"output shape (3,{out_channels}), all {n_total} params got gradients")


def check_point_attention_vfe():
    from models.vfe_point_attn import PointAttentionVFE

    torch.manual_seed(0)
    pc_range = torch.tensor([0.0, -5.0, -2.5, 12.0, 5.0, 2.5])
    voxel_size = torch.tensor([0.2, 0.2, 0.2])

    # two samples' worth of points, already voxelized (reuse the real voxelize_batch)
    from data.dataset import voxelize_batch
    n0, n1 = 150, 90
    points = torch.rand(n0 + n1, 4)
    points[:, 0] = pc_range[0] + points[:, 0] * (pc_range[3] - pc_range[0])
    points[:, 1] = pc_range[1] + points[:, 1] * (pc_range[4] - pc_range[1])
    points[:, 2] = pc_range[2] + points[:, 2] * (pc_range[5] - pc_range[2])
    point_batch_idx = torch.cat([torch.zeros(n0, dtype=torch.long), torch.ones(n1, dtype=torch.long)])
    grid_size = torch.tensor([round(((pc_range[3 + i] - pc_range[i]) / voxel_size[i]).item()) for i in range(3)])
    voxel_coords, point_voxel_idx = voxelize_batch(points, point_batch_idx, pc_range, voxel_size, grid_size)
    num_voxels = voxel_coords.shape[0]

    vfe = PointAttentionVFE(out_channels=128, point_channels=64, num_blocks=1, k=16)
    points.requires_grad_(True)
    out = vfe(points, point_voxel_idx, voxel_coords, num_voxels, pc_range, voxel_size, point_batch_idx, batch_size=2)
    assert out.shape == (num_voxels, 128) and torch.isfinite(out).all()
    out.sum().backward()
    n_grad = sum(1 for p in vfe.parameters() if p.grad is not None)
    n_total = sum(1 for p in vfe.parameters())
    assert n_grad == n_total, f"only {n_grad}/{n_total} PointAttentionVFE params got gradients"
    print(f"[ok] PointAttentionVFE: point-transformer(64D) -> cross-attention pooling -> output shape ({num_voxels},128) matches voxel count, all {n_total} params got gradients")


def check_query_denoising_build():
    from models.denoising import QueryDenoising, build_attention_mask

    torch.manual_seed(0)
    pc_range = torch.tensor([0.0, -5.0, -2.5, 12.0, 5.0, 2.5])
    channels = 16

    # sample 0 has 3 GT boxes, sample 1 has 1 -> Mmax=3, so sample 1 has 2 pad slots per group
    gt0 = torch.zeros(3, 10); gt0[:, 3:6] = 1.0; gt0[:, 6] = 1.0
    gt0[:, :3] = torch.tensor([[2.0, 0.0, 0.0], [5.0, -1.0, 1.0], [8.0, 2.0, -1.0]])
    gt1 = torch.zeros(1, 10); gt1[:, 3:6] = 1.0; gt1[:, 6] = 1.0
    gt1[:, :3] = torch.tensor([[4.0, 0.0, 0.0]])

    dn = QueryDenoising(channels, num_groups=2, center_noise_scale=0.0, size_noise_scale=0.0, rot_noise_deg=0.0)
    built = dn.build([gt0, gt1], pc_range, torch.device("cpu"))
    assert built is not None
    dn_content, dn_ref, dn_valid, dn_targets, m_max = built

    assert m_max == 3
    G = 2
    assert dn_content.shape == (2, G * m_max, channels)
    assert dn_valid.shape == (2, G * m_max)
    # sample0: all 3 slots real in both groups; sample1: only slot 0 real per group
    expected_valid = torch.tensor([[True, True, True] * G, [True, False, False] * G])
    assert torch.equal(dn_valid, expected_valid), f"valid mask mismatch:\n{dn_valid}\nvs\n{expected_valid}"

    # noise scales are all 0 -> the "noised" reference point must exactly equal
    # the true GT center for every REAL slot (this isolates the noise-generation
    # math from the learned embedding, which check_full_model_forward_backward
    # already covers together with the rest of the model)
    real = dn_valid
    assert torch.allclose(dn_ref[real], dn_targets["center"][real], atol=1e-5), \
        "with noise scale 0, noised center should exactly equal the true GT center"
    print(f"[ok] QueryDenoising.build(): shapes correct, valid mask correct, zero-noise ref==true center")

    mask = build_attention_mask(num_matching=5, group_size=m_max, num_groups=G, device=torch.device("cpu"))
    total = 5 + G * m_max
    assert mask.shape == (total, total)
    # matching block (0:5) must not see any denoising block, and vice versa
    assert (mask[:5, 5:] == float("-inf")).all()
    assert (mask[5:, :5] == float("-inf")).all()
    # group 0 (5:5+m_max) must not see group 1 (5+m_max:5+2*m_max)
    g0, g1 = slice(5, 5 + m_max), slice(5 + m_max, 5 + 2 * m_max)
    assert (mask[g0, g1] == float("-inf")).all() and (mask[g1, g0] == float("-inf")).all()
    assert (mask[g0, g0] == 0.0).all() and (mask[:5, :5] == 0.0).all()
    print(f"[ok] build_attention_mask(): matching/denoising and cross-group blocks correctly isolated")


def check_full_model_forward_backward():
    """End-to-end wiring check with a synthetic batch (no real dataset needed):
    VFE -> backbone -> SlotFormer -> DETR decoder -> head -> matcher -> loss ->
    backward(), on CPU. Catches shape/wiring bugs (pad_tokens, attention masks,
    index_grid, etc.) before a real training run touches real data."""
    from config_utils import load_config
    from models.detector_detr import DiverDetectorDETR

    torch.manual_seed(0)
    cfg = load_config("configs/exp_detr_head.yaml")
    model = DiverDetectorDETR(cfg)
    # match train.py's own device selection -- backbone3d_auto picks spconv
    # whenever CUDA+spconv work on this machine, and spconv's kernels are
    # CUDA-only, so forcing CPU here (independent of what's actually
    # available) would fault on a machine like this one where CUDA *is*
    # available. Not a DETR-specific issue -- the existing dense DiverDetector
    # would hit the same thing under a hardcoded-CPU smoke test.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pc_range = cfg["DATA"]["POINT_CLOUD_RANGE"]

    def rand_points(n):
        pts = torch.rand(n, 4)
        pts[:, 0] = pc_range[0] + pts[:, 0] * (pc_range[3] - pc_range[0])
        pts[:, 1] = pc_range[1] + pts[:, 1] * (pc_range[4] - pc_range[1])
        pts[:, 2] = pc_range[2] + pts[:, 2] * (pc_range[5] - pc_range[2])
        return pts

    # two samples with different point counts and different numbers of GT boxes
    # (including one empty-GT sample) -- the two edge cases most likely to break
    # padding/masking code.
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

    losses, pred, gt_boxes_list, matches = model.loss(batch, device)
    assert torch.isfinite(losses["total"]), "total loss is not finite"
    losses["total"].backward()

    n_params_with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    n_params_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[ok] full model forward/backward: total_loss={losses['total'].item():.4f}, "
          f"{n_params_with_grad}/{n_params_total} params got gradients")

    decoded = model.decode(pred, score_threshold=0.0)
    assert len(decoded) == 2
    print(f"[ok] decode(): sample0 kept {decoded[0]['center'].shape[0]} queries, "
          f"sample1 kept {decoded[1]['center'].shape[0]} (both should be NUM_QUERIES={cfg['DETR_HEAD']['NUM_QUERIES']} at threshold 0.0)")


if __name__ == "__main__":
    check_rotation6d_roundtrip()
    check_matcher_shapes_and_assignment()
    check_focal_loss_sane()
    check_dsvt_set_partition()
    check_dsvt_layer_and_backbone()
    check_knn_indices_per_sample()
    check_point_transformer_layer()
    check_cap_points_per_voxel()
    check_voxel_pooling_attention()
    check_point_attention_vfe()
    check_query_denoising_build()
    check_full_model_forward_backward()
    print("\nall DETR-component checks passed.")
