"""Tiny config-loading helper: a yaml file may set a top-level BASE_CONFIG
(path relative to its own directory) to inherit from another config instead
of repeating every field. Used by configs/exp_*.yaml so each experiment file
only states what actually differs from configs/default.yaml -- keeping e.g.
the scene lists in one place instead of copy-pasted per experiment.
"""
from pathlib import Path

import yaml


def load_config(path) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base_name = cfg.pop("BASE_CONFIG", None)
    if base_name is None:
        return cfg
    base_cfg = load_config(path.parent / base_name)
    return _deep_merge(base_cfg, cfg)


def _deep_merge(base: dict, override: dict) -> dict:
    """override wins; nested dicts merge key-by-key, everything else
    (lists, scalars) is replaced wholesale."""
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged
