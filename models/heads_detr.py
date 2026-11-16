import math

import torch
import torch.nn as nn

FOCAL_BIAS_INIT = -math.log((1 - 0.01) / 0.01)  # same convention as heads.py's DetectionHead


class SetPredictionHead(nn.Module):
    """Per-query MLP heads on top of the DETR decoder's output. Box
    parameterization is (center(3), log_size(3), sixd_rotation(6)) -- same
    center/log-size convention as heads.py's DetectionHead, but 6D rotation
    (models/rotation6d.py) instead of quaternion, per the project's existing
    preference for 6D elsewhere (voxelnet_baseline/model/rotation3d.py)."""

    def __init__(self, channels, hidden_dim=None, num_hidden_layers=1):
        super().__init__()
        hidden_dim = hidden_dim or channels

        def mlp(out_dim, bias_init=None):
            layers = []
            in_dim = channels
            for _ in range(num_hidden_layers):
                layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
                in_dim = hidden_dim
            final = nn.Linear(in_dim, out_dim)
            if bias_init is not None:
                with torch.no_grad():
                    final.bias.fill_(bias_init)
            layers.append(final)
            return nn.Sequential(*layers)

        self.exist_head = mlp(1, bias_init=FOCAL_BIAS_INIT)
        self.center_offset_head = mlp(3)
        self.log_size_head = mlp(3)
        self.rot_head = mlp(6)

    def forward(self, query_feat: torch.Tensor, ref_points: torch.Tensor,
                log_size_anchor: torch.Tensor, rot_anchor: torch.Tensor) -> dict:
        """query_feat: (B,Q,C). ref_points/log_size_anchor/rot_anchor: (B,Q,3)/(B,Q,3)/(B,Q,6)
        -- per-query anchors (matching queries: DetrDecoder's learned
        parameters; denoising queries: that query's own noised box, see
        models/denoising.py) that each head predicts a correction from,
        instead of size/rotation being regressed as an absolute value with no
        reference point the way center already wasn't."""
        return {
            "exist_logit": self.exist_head(query_feat),               # (B,Q,1)
            "center": ref_points + self.center_offset_head(query_feat),  # (B,Q,3), absolute world coords
            "log_size": log_size_anchor + self.log_size_head(query_feat),  # (B,Q,3)
            "sixd": rot_anchor + self.rot_head(query_feat),           # (B,Q,6)
        }
