"""FCAF3D-style (arXiv 2112.00322) anchor-free dense head: per-location
classification + box regression + centerness, with weights SHARED across
every backbone feature level (models/backbone3d_unet_spconv.py's
forward_multilevel()). Applied independently to each level's sparse tensor --
unlike models/heads_detr.py's SetPredictionHead, there is no query/decoder
here: every ACTIVE VOXEL at every level is itself a candidate prediction.

Box parametrization stays this project's own (center offset + log-size + 6D
rotation, models/rotation6d.py) rather than FCAF3D's face-distance/Mobius-strip
encoding -- that encoding assumes a single heading angle (objects sitting on a
floor), which doesn't generalize to this project's full 3D orientation (a
diver can be at any orientation underwater). The defining FCAF3D
characteristics kept faithfully: dense per-location prediction, centerness,
weight sharing across levels, and (in models/assign_fcaf3d.py) multi-level
GT assignment with center sampling."""
import torch
import torch.nn as nn

from .heads_detr import FOCAL_BIAS_INIT
from .sparse_ops import SubMConv3d


class FCAF3DHead(nn.Module):
    def __init__(self, channels, hidden_dim=None, num_hidden_layers=1):
        super().__init__()
        hidden_dim = hidden_dim or channels

        def mlp(out_dim, bias_init=None):
            layers = []
            in_dim = channels
            for _ in range(num_hidden_layers):
                layers += [SubMConv3d(in_dim, hidden_dim, kernel_size=3, bias=True), nn.ReLU(inplace=True)]
                in_dim = hidden_dim
            final = SubMConv3d(in_dim, out_dim, kernel_size=1, bias=True)
            if bias_init is not None:
                with torch.no_grad():
                    final.bias.fill_(bias_init)
            return nn.ModuleList(layers + [final])

        self.exist_layers = mlp(1, bias_init=FOCAL_BIAS_INIT)
        self.center_offset_layers = mlp(3)
        self.log_size_layers = mlp(3)
        self.rot_layers = mlp(6)
        self.centerness_layers = mlp(1)

    @staticmethod
    def _run(layers, feat, coords, index_grid, grid_size):
        x = feat
        for layer in layers:
            if isinstance(layer, nn.ReLU):
                x = layer(x)
            else:
                x, _, _ = layer(x, coords, index_grid, grid_size)
        return x

    def forward(self, feat, coords, index_grid, grid_size, world_centers):
        """feat: (N,C) one level's sparse features. coords: (N,4) that level's
        voxel coords (for the SubMConv3d ops, which need the active set/grid
        context but never move it). world_centers: (N,3) that level's voxels'
        own world-space centers (models/decoder_detr.py::token_positional_embedding's
        world-coordinate math, reused as the reference point every location
        regresses its box center offset from -- same role ref_points plays in
        heads_detr.py). Returns per-location predictions, dict of (N,*)."""
        return {
            "exist_logit": self._run(self.exist_layers, feat, coords, index_grid, grid_size),
            "center": world_centers + self._run(self.center_offset_layers, feat, coords, index_grid, grid_size),
            "log_size": self._run(self.log_size_layers, feat, coords, index_grid, grid_size),
            "sixd": self._run(self.rot_layers, feat, coords, index_grid, grid_size),
            "centerness_logit": self._run(self.centerness_layers, feat, coords, index_grid, grid_size),
        }
