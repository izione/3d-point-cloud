"""Picks the fastest working 3D backbone implementation at model-construction
time: spconv (real hash-table-based sparse conv, CUDA-only) if it imports AND
actually runs on this machine, otherwise the pure-PyTorch fallback in
backbone3d.py that's guaranteed to work anywhere torch runs (including CPU
and any Colab image, which is exactly the situation spconv can't be trusted
to install cleanly on -- see README).

Both implementations share the same constructor signature and forward()
interface (features, coords, index_grid, grid_size, batch_size) ->
(features, coords, index_grid, grid_size), so detector.py doesn't need to
know or care which one it got.
"""
import torch

from .backbone3d import Sparse3DBackbone


def spconv_usable() -> bool:
    """True only if spconv both imports AND successfully runs one real conv
    on this device -- import can succeed while the compiled CUDA kernels
    still don't match the installed torch/CUDA/driver combo, so a plain
    `import spconv` isn't enough of a check."""
    if not torch.cuda.is_available():
        return False  # spconv's prebuilt wheels are CUDA-only; don't even try on CPU
    try:
        import spconv.pytorch as spconv  # noqa: F401
        device = "cuda"
        feats = torch.rand(4, 3, device=device)
        coords = torch.zeros(4, 4, dtype=torch.int32, device=device)
        x = spconv.SparseConvTensor(feats, coords, spatial_shape=[4, 4, 4], batch_size=1)
        probe = spconv.SubMConv3d(3, 3, 3, indice_key="_backbone3d_auto_probe").to(device)
        _ = probe(x)
        return True
    except Exception as e:
        print(f"spconv not usable ({type(e).__name__}: {e}) -- falling back to the pure-PyTorch sparse backbone")
        return False


def build_backbone3d(in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride, backbone_type="auto"):
    """backbone_type: "auto" (spconv if usable here, else the pure-PyTorch
    fallback -- the normal case) or "dense" (nn.Conv3d over the full voxel
    grid, no sparsity at all -- see configs/exp_dense.yaml)."""
    if backbone_type == "dense":
        from .backbone3d_dense import DenseBackboneWrapper
        print("3D backbone: dense (nn.Conv3d)")
        return DenseBackboneWrapper(in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride)
    if backbone_type not in ("auto", None):
        raise ValueError(f"unknown BACKBONE.TYPE: {backbone_type!r} (expected 'auto' or 'dense')")
    if spconv_usable():
        from .backbone3d_spconv import Sparse3DBackboneSpconv
        print("3D backbone: spconv")
        return Sparse3DBackboneSpconv(in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride)
    print("3D backbone: pure-PyTorch (models/sparse_ops.py)")
    return Sparse3DBackbone(in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride)
