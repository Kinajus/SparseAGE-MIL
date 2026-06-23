"""Lightweight topology descriptors for WSI patch bags.

The descriptors are intentionally simple and dependency-light: they summarize
pseudo-histology cluster composition, connectedness, boundary/interface ratio,
and spatial dispersion. They are suitable as auxiliary self-supervision targets
for the SparseAGE-MIL topology head.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_TOPOLOGY_DIM = 6


def _as_numpy(array: object | None) -> np.ndarray | None:
    if array is None:
        return None
    try:
        import torch

        if torch.is_tensor(array):
            return array.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(array)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def n_components(self) -> int:
        return len({self.find(i) for i in range(len(self.parent))})


def compute_lightweight_topology_descriptor(
    coords: object | None = None,
    clusters: object | None = None,
    *,
    k_neighbors: int = 6,
    output_dim: int = DEFAULT_TOPOLOGY_DIM,
) -> np.ndarray:
    """Compute a fixed-length spatial-topology descriptor for one slide.

    The first six entries are:

    1. normalized cluster entropy;
    2. dominant cluster fraction;
    3. normalized connected-component count within same-cluster kNN graph;
    4. cross-cluster boundary/interface edge ratio;
    5. mean within-cluster spatial dispersion normalized by slide diagonal;
    6. normalized spatial coverage of the patch cloud.

    Missing coordinates or clusters are allowed; unavailable terms are returned
    as zeros. Extra dimensions are zero-padded to keep the target shape stable.
    """

    coords_np = _as_numpy(coords)
    clusters_np = _as_numpy(clusters)
    descriptor = np.zeros(max(output_dim, DEFAULT_TOPOLOGY_DIM), dtype=np.float32)

    if clusters_np is not None:
        clusters_np = np.asarray(clusters_np).reshape(-1)
        valid_cluster = clusters_np >= 0
        valid_labels = clusters_np[valid_cluster].astype(np.int64)
        if valid_labels.size > 0:
            _, counts = np.unique(valid_labels, return_counts=True)
            proportions = counts.astype(np.float64) / counts.sum()
            if len(proportions) > 1:
                entropy = -np.sum(proportions * np.log(proportions + 1e-12))
                descriptor[0] = float(entropy / math.log(len(proportions)))
            descriptor[1] = float(proportions.max())

    if coords_np is None:
        return descriptor[:output_dim]

    coords_np = np.asarray(coords_np, dtype=np.float64)
    if coords_np.ndim != 2 or coords_np.shape[0] == 0:
        return descriptor[:output_dim]
    coords_np = coords_np[:, :2]
    finite = np.isfinite(coords_np).all(axis=1)
    if clusters_np is not None and len(clusters_np) == len(coords_np):
        finite = finite & (clusters_np >= 0)
    coords_np = coords_np[finite]
    if coords_np.shape[0] < 2:
        return descriptor[:output_dim]
    if clusters_np is not None and len(clusters_np) == len(finite):
        clusters_valid = clusters_np[finite].astype(np.int64)
    else:
        clusters_valid = None

    span = coords_np.max(axis=0) - coords_np.min(axis=0)
    diagonal = float(np.linalg.norm(span) + 1e-8)
    descriptor[5] = float((span[0] * span[1]) / (diagonal * diagonal + 1e-8))

    n = coords_np.shape[0]
    k_eff = max(1, min(int(k_neighbors), n - 1))
    distances = np.linalg.norm(coords_np[:, None, :] - coords_np[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argpartition(distances, kth=k_eff - 1, axis=1)[:, :k_eff]

    if clusters_valid is not None:
        total_edges = 0
        boundary_edges = 0
        uf = _UnionFind(n)
        for i in range(n):
            for j in neighbors[i]:
                total_edges += 1
                if clusters_valid[i] == clusters_valid[j]:
                    uf.union(i, int(j))
                else:
                    boundary_edges += 1
        descriptor[2] = float(uf.n_components() / max(1.0, math.sqrt(n)))
        descriptor[3] = float(boundary_edges / max(1, total_edges))

        dispersions: list[float] = []
        for label in np.unique(clusters_valid):
            idx = clusters_valid == label
            if idx.sum() < 2:
                continue
            centroid = coords_np[idx].mean(axis=0)
            dispersions.append(float(np.linalg.norm(coords_np[idx] - centroid, axis=1).mean()))
        if dispersions:
            descriptor[4] = float(np.mean(dispersions) / diagonal)
    else:
        descriptor[4] = float(
            np.linalg.norm(coords_np - coords_np.mean(axis=0), axis=1).mean() / diagonal
        )

    return descriptor[:output_dim]


def descriptor_column_names(prefix: str = "topo") -> list[str]:
    """Return conventional column names for the six built-in descriptors."""

    return [
        f"{prefix}_cluster_entropy",
        f"{prefix}_dominant_cluster_fraction",
        f"{prefix}_component_count",
        f"{prefix}_boundary_ratio",
        f"{prefix}_spatial_dispersion",
        f"{prefix}_spatial_coverage",
    ]
