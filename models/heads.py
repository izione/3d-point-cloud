import math

import torch.nn as nn

from .sparse_ops import SubMConv3d

FOCAL_BIAS_INIT = -math.log((1 - 0.01) / 0.01)


class SeparateHead(nn.Module):
    """num_conv-1 x [SubMConv3d(k=3)+BN+ReLU] followed by one SubMConv3d(k=1) output layer."""

    def __init__(self, in_channels, out_channels, num_conv=2, bias_init=None):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_conv - 1):
            self.convs.append(SubMConv3d(in_channels, in_channels, kernel_size=3, bias=False))
            self.bns.append(nn.BatchNorm1d(in_channels))
        self.final = SubMConv3d(in_channels, out_channels, kernel_size=1, bias=True)
        if bias_init is not None:
            import torch
            with torch.no_grad():
                self.final.bias.fill_(bias_init)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features, coords, index_grid, grid_size):
        x = features
        for conv, bn in zip(self.convs, self.bns):
            x, _, _ = conv(x, coords, index_grid, grid_size)
            x = self.relu(bn(x))
        out, _, _ = self.final(x, coords, index_grid, grid_size)
        return out


class DetectionHead(nn.Module):
    """Center heatmap (1ch, focal-loss bias init) + offset (3ch) + box (7ch:
    log(l),log(w),log(h),qw,qx,qy,qz -- size stays in log-scale until decode,
    quaternion stays unnormalized until decode/loss)."""

    def __init__(self, in_channels, head_cfg):
        super().__init__()
        c = head_cfg["CENTER_HEAD"]
        self.center_head = SeparateHead(in_channels, c["OUTPUT_CHANNELS"], c["NUM_CONV"], bias_init=FOCAL_BIAS_INIT)
        o = head_cfg["OFFSET_HEAD"]
        self.offset_head = SeparateHead(in_channels, o["OUTPUT_CHANNELS"], o["NUM_CONV"])
        b = head_cfg["BOX_HEAD"]
        self.box_head = SeparateHead(in_channels, b["OUTPUT_CHANNELS"], b["NUM_CONV"])

    def forward(self, features, coords, index_grid, grid_size):
        center_logit = self.center_head(features, coords, index_grid, grid_size)
        offset = self.offset_head(features, coords, index_grid, grid_size)
        box = self.box_head(features, coords, index_grid, grid_size)
        return {
            "center_logit": center_logit,   # (N,1)
            "offset": offset,               # (N,3)
            "box": box,                     # (N,7) = log_size(3) + quat_raw(4)
        }
