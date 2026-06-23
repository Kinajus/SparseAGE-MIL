"""Utilities for exporting sparse attention/topology selections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor


def _to_numpy(value: Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def export_sparse_topology(
    *,
    slide_id: str,
    coords: Tensor | np.ndarray,
    attention: Tensor | np.ndarray,
    topk_indices: Tensor | np.ndarray,
    topk_weights: Tensor | np.ndarray | None,
    output_dir: str | Path,
    top_fraction: float = 0.10,
    clusters: Tensor | np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Export selected patches and sparse edges as CSV files.

    The output can be overlaid on an H&E thumbnail in QuPath, napari, or a
    custom plotting script. ``attention`` may be [1, N], [N], or [B, 1, N] for a
    single slide.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coords_np = _to_numpy(coords).reshape(-1, coords.shape[-1])[:, :2]
    attn_np = _to_numpy(attention).reshape(-1)
    indices_np = _to_numpy(topk_indices)
    if indices_np.ndim == 3:
        indices_np = indices_np[0]
    weights_np = None if topk_weights is None else _to_numpy(topk_weights)
    if weights_np is not None and weights_np.ndim == 3:
        weights_np = weights_np[0]
    clusters_np = None if clusters is None else _to_numpy(clusters).reshape(-1)

    n = len(attn_np)
    keep = max(1, int(round(n * top_fraction)))
    selected = np.zeros(n, dtype=bool)
    selected[np.argsort(-attn_np)[:keep]] = True

    node_data = {
        "slide_id": slide_id,
        "patch_index": np.arange(n),
        "x": coords_np[:, 0],
        "y": coords_np[:, 1],
        "attention": attn_np,
        "selected_top_attention": selected.astype(int),
    }
    if clusters_np is not None and len(clusters_np) >= n:
        node_data["cluster"] = clusters_np[:n].astype(int)
    nodes = pd.DataFrame(node_data)
    edges_rows: list[dict[str, float | int | str]] = []
    for source in range(indices_np.shape[0]):
        for rank, target in enumerate(indices_np[source]):
            row: dict[str, float | int | str] = {
                "slide_id": slide_id,
                "source": int(source),
                "target": int(target),
                "rank": int(rank),
            }
            if weights_np is not None:
                row["weight"] = float(weights_np[source, rank])
            edges_rows.append(row)
    edges = pd.DataFrame(edges_rows)

    nodes_path = output_dir / f"{slide_id}_nodes.csv"
    edges_path = output_dir / f"{slide_id}_edges.csv"
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)
    return nodes_path, edges_path
