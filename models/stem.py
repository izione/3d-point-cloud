import torch.nn as nn

from .sparse_ops import SubMConv3d, SparseBasicBlock, SparseConv3dDown


class Stem(nn.Module):
    """SubMConv3d channel expansion + residual refinement + one isotropic
    stride-2 downsample (deliberately isotropic on x,y,z -- FSHNet's original
    downsamples z only, which reintroduces the ground-plane bias we're avoiding)."""

    def __init__(self, in_channels=64, out_channels=128, num_blocks=3, down_kernel=3, down_stride=2):
        super().__init__()
        self.expand = SubMConv3d(in_channels, out_channels, kernel_size=3)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([SparseBasicBlock(out_channels) for _ in range(num_blocks)])
        self.down = SparseConv3dDown(
            out_channels, out_channels, kernel_size=down_kernel, stride=down_stride, padding=down_kernel // 2
        )

    def forward(self, features, coords, index_grid, grid_size, batch_size):
        x, coords, index_grid = self.expand(features, coords, index_grid, grid_size)
        x = self.relu(self.bn(x))
        for block in self.blocks:
            x, coords, index_grid = block(x, coords, index_grid, grid_size)
        x, coords, index_grid, grid_size = self.down(x, coords, index_grid, grid_size, batch_size)
        return x, coords, index_grid, grid_size
