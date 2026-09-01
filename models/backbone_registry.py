"""Name -> builder registry for 3D backbones.

Before this, adding a new backbone architecture meant editing
backbone3d_auto.build_backbone3d's if/elif chain (and its fixed positional
argument list) every time. That doesn't scale once "architectures to try"
becomes a real experiment matrix (sparse alone / +SlotFormer(3L) /
+SlotFormer(6L) / sparse U-Net / ...) -- this registry lets each backbone
module register itself once, by name, and BACKBONE.TYPE in a config selects
among all of them with no changes needed here or in detector.py.

To add a new backbone:
  1. Write an nn.Module with the shared interface every registered backbone
     must expose:
         module(features, coords, index_grid, grid_size, batch_size)
           -> (features, coords, index_grid, grid_size)
         module.out_channels   -- int, channel count of the returned `features`
         module.total_stride   -- int, ratio between the voxel size
                                  detector.py voxelized the point cloud at and
                                  the EFFECTIVE voxel size of the coords this
                                  backbone hands to SlotFormer/the head. 1
                                  means "the backbone restored full input
                                  resolution" (e.g. a U-Net decoder); the
                                  plain multi-stage backbones use
                                  down_stride ** num_stages. detector.py reads
                                  this as `self.stem_stride` to compute the
                                  effective voxel size used everywhere
                                  downstream (decode/assign/loss).
  2. In that same module, register a (in_channels: int, bcfg: dict) -> module
     builder function with @register_backbone("your_name"). `bcfg` is the
     *whole* cfg["BACKBONE"] dict, not a fixed positional signature, so a
     backbone with its own extra config knobs (e.g. sparse_unet's
     DECODER_BLOCKS_PER_STAGE) can read them without changing this file.
  3. Import that module somewhere it actually runs before build_backbone() is
     called -- see the bottom of this file.
  4. Set BACKBONE.TYPE: your_name in a config. Done.
"""
from typing import Callable, Dict

import torch.nn as nn

_REGISTRY: Dict[str, Callable[[int, dict], nn.Module]] = {}


def register_backbone(name: str):
    """Decorator for a (in_channels, bcfg) -> nn.Module builder function."""
    def _decorator(build_fn):
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not build_fn:
            raise ValueError(
                f"BACKBONE.TYPE {name!r} is already registered (to {existing!r}) -- "
                f"pick a different name for {build_fn!r}"
            )
        _REGISTRY[name] = build_fn
        return build_fn
    return _decorator


def build_backbone(in_channels: int, bcfg: dict) -> nn.Module:
    """bcfg is cfg["BACKBONE"] as a whole -- each registered builder pulls out
    whatever keys it needs (see that builder's own module for which ones)."""
    name = bcfg.get("TYPE", "auto")
    try:
        build_fn = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown BACKBONE.TYPE: {name!r} -- registered backbones: {registered_backbones()}"
        ) from None
    return build_fn(in_channels, bcfg)


def registered_backbones():
    return sorted(_REGISTRY)


# NOTE: this module intentionally does NOT import backbone3d_auto/
# backbone3d_unet/etc itself (that would make this file import the very
# modules that import register_backbone from it -- a circular import that
# happens to work today only because of import-order luck). Instead,
# models/__init__.py imports every backbone module once, for its
# registration side effect, so build_backbone() always sees the full
# registry regardless of which module first triggers `import models.*`.
# Add a new backbone module there too, or it'll raise "unknown BACKBONE.TYPE"
# even though the file exists.
