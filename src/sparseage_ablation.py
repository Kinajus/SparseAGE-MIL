"""Generate ablation YAML configurations."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ABLATIONS: dict[str, dict[str, Any]] = {
    "feature_only_k8": {
        "model": {
            "use_spatial": False,
            "spatial_weight": 0.0,
            "cluster_weight": 0.0,
            "boundary_weight": 0.0,
            "topk": 8,
        }
    },
    "spatial_only_k8": {
        "model": {"use_spatial": True, "cluster_weight": 0.0, "boundary_weight": 0.0, "topk": 8}
    },
    "cluster_boundary_k8": {
        "model": {"use_spatial": True, "cluster_weight": 0.15, "boundary_weight": 0.10, "topk": 8}
    },
    "k4": {"model": {"topk": 4}},
    "k8": {"model": {"topk": 8}},
    "k16": {"model": {"topk": 16}},
    "k32": {"model": {"topk": 32}},
    "multi_k_4_8_16": {"model": {"topk": [4, 8, 16]}},
    "no_topology_aux": {"train": {"topology_weight": 0.0}, "model": {"topology_target_dim": 0}},
    "include_self_false": {"model": {"include_self": False}},
    "pcgrad": {"train": {"gradient_strategy": "pcgrad"}},
    "uncertainty_off": {"model": {"use_uncertainty_weighting": False}},
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a dict copy."""

    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def generate_ablation_configs(
    template: dict[str, Any],
    output_dir: str | Path,
    *,
    ablations: dict[str, dict[str, Any]] | None = None,
) -> list[Path]:
    """Write one YAML config per ablation and return paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ablations = ablations or DEFAULT_ABLATIONS
    paths: list[Path] = []
    for name, overrides in ablations.items():
        cfg = deep_update(template, overrides)
        cfg["name"] = f"{template.get('name', 'sparseage')}_{name}"
        path = output_dir / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        paths.append(path)
    return paths
