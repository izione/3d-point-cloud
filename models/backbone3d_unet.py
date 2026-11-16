"""Sparse 3D U-Net backbone: N-stage plain-residual-block encoder (same
Sparse3DStage building block as backbone3d.py's single-stage Sparse3DBackbone)
paired with a decoder that upsamples most of the way back, fusing each
encoder stage's own (skip-connected) features instead of handing only the
coarsest, deepest feature map to whatever comes after (SlotFormer/DSVT
refinement, then the DETR decoder). BACKBONE.TYPE: sparse_unet.

Motivated by FCAF3D (arXiv 2112.00322): its "neck" is exactly this same
encoder-decoder-with-skip-connections idea, upsampling with generative sparse
transposed convolutions and a probability-based pruning layer to control the
sparsity explosion a raw transposed conv causes. We don't need that pruning
machinery here -- SparseInverseConv3d (see its own docstring in sparse_ops.py)
deliberately restricts each upsample step to the ENCODER's own already-known
active positions at that level. This project's sparsity comes from real
voxelized points, not something that needs "inventing" candidate positions
the way FCAF3D's general-purpose scene reconstruction does, so there's no
uncontrolled active-set growth to prune in the first place.

The decoder stops ONE level short of full input resolution (N-1 upsamples for
N encoder stages), so the backbone's net total_stride still equals the plain
single-stage backbone's own down_stride (e.g. 2) -- this keeps
REFINEMENT.WINDOW_SHAPES/TAU and every positional-embedding call downstream
(all tuned around that effective voxel size) valid without retuning. Also
matches this project's own history (configs/default.yaml's BACKBONE comment):
a previous 4-stage/stride-16 backbone with NO decoder at all tanked
precision/recall -- the working theory here is that it was the missing
decoder (losing fine-resolution detail permanently), not stage depth itself,
that caused that regression; this backbone tests that theory directly by
keeping the depth but restoring resolution via skip connections.

Never collapses to a 2D/BEV representation -- skip connections carry full 3D
(x,y,z) sparse coordinates throughout, same as every other backbone in this
project.

Pure-PyTorch only (sparse_ops.py), like BACKBONE.TYPE: sparse_dilated_gn --
SparseInverseConv3d has no spconv-native equivalent wired up in this repo
yet, so this type always uses the pure-PyTorch path regardless of spconv
availability (see backbone3d_auto.py)."""
import torch.nn as nn

from .backbone3d import Sparse3DStage
from .backbone3d_slot_light_unet import _LightDecoderStage


class SparseUNetBackbone(nn.Module):
    def __init__(self, in_channels, stage_channels, num_blocks_per_stage, down_kernel=3, down_stride=2,
                 block_dilations=None, norm_type="batch"):
        super().__init__()
        stage_channels = list(stage_channels)
        n = len(stage_channels)
        assert n >= 2, ("SparseUNetBackbone needs at least 2 encoder stages (one stays un-decoded so the net "
                         "stride matches the single-stage backbone's) -- use BACKBONE.TYPE: auto for one stage")
        encoder_channels = [in_channels] + stage_channels
        blocks_per_stage = (num_blocks_per_stage if isinstance(num_blocks_per_stage, (list, tuple))
                             else [num_blocks_per_stage] * n)
        assert len(blocks_per_stage) == n

        self.encoder_stages = nn.ModuleList([
            Sparse3DStage(encoder_channels[i], encoder_channels[i + 1], blocks_per_stage[i],
                          down_kernel, down_stride, block_dilations=block_dilations, norm_type=norm_type)
            for i in range(n)
        ])
        # mirrors every encoder stage EXCEPT the shallowest (i=0) -- n-1 decoder
        # stages, leaving the net stride at down_stride instead of fully
        # restoring to the input's own resolution (see module docstring)
        self.decoder_stages = nn.ModuleList([
            _LightDecoderStage(encoder_channels[i + 1], encoder_channels[i], down_kernel, down_stride)
            for i in reversed(range(1, n))
        ])

        self.out_channels = encoder_channels[1]  # stage_channels[0]'s width, restored via skip connections
        self.total_stride = down_stride

    def forward(self, features, coords, index_grid, grid_size, batch_size):
        skips = []  # skips[i] = (features, coords, index_grid, grid_size) INPUT to encoder_stages[i]
        x, c, ig, gs = features, coords, index_grid, grid_size
        for stage in self.encoder_stages:
            skips.append((x, c, ig, gs))
            x, c, ig, gs = stage(x, c, ig, gs, batch_size)

        # n-1 decoder stages pop skips[-1], skips[-2], ..., skips[1] off this
        # LIFO stack in turn -- skips[0] (the backbone's own input) is never
        # popped, since the decoder deliberately doesn't go that far (see
        # module docstring on why the net stride stays down_stride).
        for stage in self.decoder_stages:
            skip_feat, skip_coords, skip_index_grid, skip_grid_size = skips.pop()
            x, c, ig, gs = stage(x, c, ig, gs, skip_feat, skip_coords, skip_index_grid, skip_grid_size)

        return x, c, ig, gs
