"""Train-only pseudo-histology cluster fitting and assignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans

from sparseage_data import load_feature_tensor, resolve_feature_path


def select_training_rows(
    df: pd.DataFrame,
    *,
    split_column: str | None = None,
    train_values: list[str] | tuple[str, ...] = ("train",),
    fold_column: str | None = None,
    fold: int | None = None,
) -> pd.Series:
    """Return a boolean mask identifying rows used to fit cluster prototypes."""

    if split_column and split_column in df.columns:
        values = {str(v).lower() for v in train_values}
        return df[split_column].astype(str).str.lower().isin(values)
    if fold_column and fold_column in df.columns and fold is not None:
        return df[fold_column].astype(int) != int(fold)
    return pd.Series(True, index=df.index)


def sample_patch_features(
    df: pd.DataFrame,
    *,
    root: str | Path | None,
    feature_column: str,
    max_patches_per_slide: int,
    seed: int,
) -> np.ndarray:
    """Sample patch features from training slides for prototype fitting."""

    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    for _, row in df.iterrows():
        path = resolve_feature_path(row[feature_column], root)
        feats = load_feature_tensor(path).detach().cpu().numpy().astype("float32")
        if feats.shape[0] > max_patches_per_slide:
            idx = rng.choice(feats.shape[0], size=max_patches_per_slide, replace=False)
            feats = feats[idx]
        chunks.append(feats)
    if not chunks:
        raise ValueError("No training features were found for pseudo-cluster fitting")
    return np.concatenate(chunks, axis=0)


def fit_pseudo_cluster_model(
    features: np.ndarray,
    *,
    n_clusters: int = 16,
    batch_size: int = 4096,
    seed: int = 2021,
) -> MiniBatchKMeans:
    """Fit a MiniBatchKMeans pseudo-histology prototype model."""

    if features.shape[0] < n_clusters:
        raise ValueError(
            f"Need at least n_clusters samples, got {features.shape[0]} < {n_clusters}"
        )
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=seed,
        n_init="auto",
        reassignment_ratio=0.01,
    )
    model.fit(features)
    return model


def assign_pseudo_clusters(
    df: pd.DataFrame,
    *,
    model: MiniBatchKMeans,
    root: str | Path | None,
    feature_column: str,
    output_dir: str | Path,
    id_column: str = "slide_id",
    cluster_column: str = "cluster_path",
) -> pd.DataFrame:
    """Assign fitted prototypes to all slides and write per-slide cluster tensors."""

    output_dir = Path(output_dir)
    clusters_dir = output_dir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)
    out_df = df.copy()
    paths: list[str] = []
    for idx, row in out_df.iterrows():
        slide_id = str(row[id_column]) if id_column in out_df.columns else f"slide_{idx:05d}"
        features = (
            load_feature_tensor(resolve_feature_path(row[feature_column], root))
            .detach()
            .cpu()
            .numpy()
            .astype("float32")
        )
        labels = model.predict(features).astype("int64")
        path = clusters_dir / f"{slide_id}_pseudo_clusters.pt"
        torch.save(torch.as_tensor(labels, dtype=torch.long), path)
        paths.append(str(path))
    out_df[cluster_column] = paths
    return out_df
