"""Registers the two "plain" backbone choices -- BACKBONE.TYPE: auto and
BACKBONE.TYPE: dense -- with models/backbone_registry.py. See that module
for how a config's BACKBONE.TYPE turns into a constructed backbone; this
file only supplies these two.

auto picks the fastest working 3D backbone implementation at
model-construction time: spconv (real hash-table-based sparse conv,
CUDA-only) if it imports AND actually runs on this machine, otherwise the
pure-PyTorch fallback in backbone3d.py that's guaranteed to work anywhere
torch runs (including CPU and any Colab image, which is exactly the
situation spconv can't be trusted to install cleanly on -- see README).
dense runs an ordinary nn.Conv3d over the full voxel grid instead (see
backbone3d_dense.py).

Both sparse implementations share the same constructor signature and
forward() interface (features, coords, index_grid, grid_size, batch_size) ->
(features, coords, index_grid, grid_size), so detector.py doesn't need to
know or care which one it got -- and neither does any other registered
backbone (e.g. backbone3d_unet.py's sparse_unet).
"""
import torch

from .backbone3d import Sparse3DBackbone
from .backbone_registry import register_backbone


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


@register_backbone("auto")
def build_auto_backbone(in_channels, bcfg):
    """BACKBONE.TYPE: auto -- spconv if it's usable here, else the
    pure-PyTorch fallback (models/sparse_ops.py). The normal/default choice."""
    stage_channels = bcfg["STAGE_CHANNELS"]
    num_blocks_per_stage = bcfg["NUM_BLOCKS_PER_STAGE"]
    down_kernel = bcfg["DOWNSAMPLE_KERNEL"]
    down_stride = bcfg["DOWNSAMPLE_STRIDE"]
    if spconv_usable():
        from .backbone3d_spconv import Sparse3DBackboneSpconv
        print("3D backbone: spconv")
        return Sparse3DBackboneSpconv(in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride)
    print("3D backbone: pure-PyTorch (models/sparse_ops.py)")
    return Sparse3DBackbone(in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride)


@register_backbone("dense")
def build_dense_backbone(in_channels, bcfg):
    """BACKBONE.TYPE: dense -- nn.Conv3d over the full voxel grid, no
    sparsity at all (see configs/exp_dense.yaml)."""
    from .backbone3d_dense import DenseBackboneWrapper
    print("3D backbone: dense (nn.Conv3d)")
    return DenseBackboneWrapper(
        in_channels, bcfg["STAGE_CHANNELS"], bcfg["NUM_BLOCKS_PER_STAGE"],
        bcfg["DOWNSAMPLE_KERNEL"], bcfg["DOWNSAMPLE_STRIDE"],
    )


def build_backbone3d(in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride, backbone_type="auto"):
    """Deprecated: old fixed-positional-argument entry point, kept only in
    case something outside this repo still imports it directly. detector.py
    now goes through models.backbone_registry.build_backbone(in_channels,
    bcfg) instead, which dispatches on BACKBONE.TYPE to whatever's registered
    (see that module) rather than a hardcoded if/elif here."""
    from .backbone_registry import build_backbone
    bcfg = {
        "TYPE": backbone_type, "STAGE_CHANNELS": stage_channels, "NUM_BLOCKS_PER_STAGE": num_blocks_per_stage,
        "DOWNSAMPLE_KERNEL": down_kernel, "DOWNSAMPLE_STRIDE": down_stride,
    }
    return build_backbone(in_channels, bcfg)
