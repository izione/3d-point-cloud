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


def build_backbone3d(in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride, backbone_type="auto",
                      block_dilations=None, norm_type="batch", bcfg=None):
    """backbone_type: "auto" (spconv if usable here, else the pure-PyTorch
    fallback -- the normal case), "dense" (nn.Conv3d over the full voxel
    grid, no sparsity at all -- see configs/exp_dense.yaml), "sparse_dilated_gn"
    (pure-PyTorch backbone only -- spconv's SubMConv3d doesn't take the dilation/
    norm_type knobs this repo added, see sparse_ops.py's SubMConv3d/make_norm1d --
    forces the pure-PyTorch path regardless of spconv availability so those knobs
    actually take effect; see configs/exp_sparse_dilated_gn.yaml), or
    "sparse_down_slot_up" (N-stage downsample -> SlotFormer -> M-stage upsample,
    SlotFormer built INSIDE the backbone -- see backbone3d_down_slot_up.py and
    configs/exp_sparse_down4_slot_up.yaml; reads UPSAMPLE_STAGES/DECODER_*/
    SLOTFORMER_* from `bcfg`, the whole cfg["BACKBONE"] dict)."""
    if backbone_type == "dense":
        from .backbone3d_dense import DenseBackboneWrapper
        print("3D backbone: dense (nn.Conv3d)")
        return DenseBackboneWrapper(in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride)
    if backbone_type == "sparse_dilated_gn":
        print(f"3D backbone: pure-PyTorch, dilated+{norm_type}norm (models/sparse_ops.py) "
              f"block_dilations={block_dilations}")
        return Sparse3DBackbone(in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride,
                                 block_dilations=block_dilations, norm_type=norm_type)
    if backbone_type == "sparse_down_slot_up":
        from .backbone3d_down_slot_up import SparseDownSlotUpBackbone
        bcfg = bcfg or {}
        upsample_stages = bcfg["UPSAMPLE_STAGES"]
        print(f"3D backbone: pure-PyTorch, {len(stage_channels)}-stage down -> SlotFormer -> "
              f"{upsample_stages}-stage up (models/backbone3d_down_slot_up.py)")
        return SparseDownSlotUpBackbone(
            in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride, upsample_stages,
            decoder_blocks_per_stage=bcfg.get("DECODER_BLOCKS_PER_STAGE"),
            decoder_out_channels=bcfg.get("DECODER_OUT_CHANNELS"),
            slot_win_size=bcfg.get("SLOTFORMER_WIN_SIZE", 12),
            slot_num_cycles=bcfg.get("SLOTFORMER_NUM_CYCLES", 1),
            slot_num_heads=bcfg.get("SLOTFORMER_NUM_HEADS", 4),
        )
    if backbone_type == "sparse_slot_stages":
        from .backbone3d_slot_stages import SparseSlotEncoderBackbone
        bcfg = bcfg or {}
        print(f"3D backbone: pure-PyTorch, {len(stage_channels)}-stage down, SlotFormer AT EVERY STAGE "
              f"instead of residual blocks (models/backbone3d_slot_stages.py) -- NUM_BLOCKS_PER_STAGE ignored")
        return SparseSlotEncoderBackbone(
            in_channels, stage_channels, down_kernel, down_stride,
            slot_win_size=bcfg.get("SLOTFORMER_WIN_SIZE", 12),
            slot_num_cycles=bcfg.get("SLOTFORMER_NUM_CYCLES", 1),
            slot_num_heads=bcfg.get("SLOTFORMER_NUM_HEADS", 4),
        )
    if backbone_type == "sparse_slot_unet":
        from .backbone3d_slot_unet import SparseSlotUNetBackbone
        bcfg = bcfg or {}
        print(f"3D backbone: pure-PyTorch, full {len(stage_channels)}-stage U-Net, SlotFormer "
              f"(both encoder AND decoder) instead of residual blocks (models/backbone3d_slot_unet.py) -- "
              f"NUM_BLOCKS_PER_STAGE ignored")
        return SparseSlotUNetBackbone(
            in_channels, stage_channels, down_kernel, down_stride,
            decoder_out_channels=bcfg.get("DECODER_OUT_CHANNELS"),
            slot_win_size=bcfg.get("SLOTFORMER_WIN_SIZE", 12),
            slot_num_cycles=bcfg.get("SLOTFORMER_NUM_CYCLES", 1),
            slot_num_heads=bcfg.get("SLOTFORMER_NUM_HEADS", 4),
        )
    if backbone_type == "sparse_slot_light_unet":
        from .backbone3d_slot_light_unet import SparseSlotLightUNetBackbone
        bcfg = bcfg or {}
        print(f"3D backbone: pure-PyTorch, {len(stage_channels)}-stage U-Net, SlotFormer in the "
              f"ENCODER only + light (plain-conv, no residual/attention) decoder "
              f"(models/backbone3d_slot_light_unet.py) -- NUM_BLOCKS_PER_STAGE ignored")
        return SparseSlotLightUNetBackbone(
            in_channels, stage_channels, down_kernel, down_stride,
            decoder_out_channels=bcfg.get("DECODER_OUT_CHANNELS"),
            slot_win_size=bcfg.get("SLOTFORMER_WIN_SIZE", 12),
            slot_num_cycles=bcfg.get("SLOTFORMER_NUM_CYCLES", 1),
            slot_num_heads=bcfg.get("SLOTFORMER_NUM_HEADS", 4),
        )
    if backbone_type not in ("auto", None):
        raise ValueError(
            f"unknown BACKBONE.TYPE: {backbone_type!r} "
            f"(expected 'auto', 'dense', 'sparse_dilated_gn', 'sparse_down_slot_up', "
            f"'sparse_slot_stages', 'sparse_slot_unet', or 'sparse_slot_light_unet')"
        )
    if spconv_usable():
        from .backbone3d_spconv import Sparse3DBackboneSpconv
        print("3D backbone: spconv")
        return Sparse3DBackboneSpconv(in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride)
    print("3D backbone: pure-PyTorch (models/sparse_ops.py)")
    return Sparse3DBackbone(in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride)
