# Import every backbone module once so its @register_backbone(...) calls run
# before anything (detector.py, smoke_test.py, benchmark scripts, ...) calls
# backbone_registry.build_backbone(). This runs as soon as ANY submodule of
# `models` is imported (e.g. `from models.detector import DiverDetector`),
# since Python always executes a package's __init__.py first -- so this is
# the one place a new backbone module needs to be added for its BACKBONE.TYPE
# name to actually be selectable from a config, on top of writing the module
# itself (see models/backbone_registry.py's module docstring for the rest of
# the "add a new backbone" steps).
from . import backbone3d_auto  # noqa: F401  registers "auto", "dense"
from . import backbone3d_unet  # noqa: F401  registers "sparse_unet"
