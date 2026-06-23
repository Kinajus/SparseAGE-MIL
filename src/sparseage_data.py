"""Dataset utilities for manifest-based WSI feature bags.

The dataset supports both the single-task SparseAGE-MIL manifests and
joint multi-task manifests with optional patch coordinates, pseudo-histology
clusters, and lightweight topology targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from sparseage_topology import compute_lightweight_topology_descriptor

FEATURE_KEYS = ("features", "feats", "embeddings", "x", "bag")
COORD_KEYS = ("coords", "coordinates", "xy", "patch_coords", "locations")
CLUSTER_KEYS = ("clusters", "cluster", "cluster_labels", "labels", "pseudo_clusters")
TOPOLOGY_KEYS = ("topology", "topology_target", "topo", "topology_descriptor")

_ROMAN_STAGE = {
    "0": 0,
    "I": 0,
    "IA": 0,
    "IA1": 0,
    "IA2": 0,
    "IA3": 0,
    "IB": 0,
    "1": 0,
    "II": 1,
    "IIA": 1,
    "IIB": 1,
    "2": 1,
    "III": 2,
    "IIIA": 2,
    "IIIB": 2,
    "IIIC": 2,
    "3": 2,
    "IV": 3,
    "IVA": 3,
    "IVB": 3,
    "4": 3,
}


def normalize_label_key(value: Any) -> Any:
    """Normalize manifest labels before looking them up in a label map."""

    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return str(value)


def is_missing(value: Any) -> bool:
    """Return True for NaN/None/empty-string manifest values."""

    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return isinstance(value, str) and value.strip() == ""


def read_manifest(manifest: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a CSV/TSV manifest or return a copy of an existing dataframe."""

    if isinstance(manifest, pd.DataFrame):
        return manifest.copy()
    manifest = Path(manifest)
    if manifest.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(manifest, sep="\t")
    return pd.read_csv(manifest)


def resolve_feature_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a feature path relative to a root directory when needed."""

    path = Path(path)
    if path.is_absolute() or root is None:
        return path
    return Path(root) / path


def _first_existing(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _to_tensor(value: Any, *, dtype: torch.dtype | None = None) -> Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().clone()
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _ensure_feature_tensor(obj: Any, path: str | Path) -> Tensor:
    tensor = _to_tensor(obj, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"Feature tensor at {path} must have shape [N, D], got {tuple(tensor.shape)}"
        )
    return tensor


def load_bag_object(path: str | Path, map_location: str = "cpu") -> dict[str, Tensor | None]:
    """Load features plus optional coords/clusters/topology from a `.pt` file."""

    obj = torch.load(path, map_location=map_location)
    coords = clusters = topology = None
    if isinstance(obj, dict):
        features_obj = _first_existing(obj, FEATURE_KEYS)
        if features_obj is None:
            raise KeyError(f"No feature tensor key found in {path}; tried {FEATURE_KEYS}")
        coords = _first_existing(obj, COORD_KEYS)
        clusters = _first_existing(obj, CLUSTER_KEYS)
        topology = _first_existing(obj, TOPOLOGY_KEYS)
        features = _ensure_feature_tensor(features_obj, path)
    else:
        features = _ensure_feature_tensor(obj, path)

    output: dict[str, Tensor | None] = {"features": features}
    if coords is not None:
        coord_tensor = _to_tensor(coords, dtype=torch.float32)
        if coord_tensor.ndim == 1:
            coord_tensor = coord_tensor.view(-1, 2)
        output["coords"] = coord_tensor[:, :2]
    else:
        output["coords"] = None
    if clusters is not None:
        output["clusters"] = _to_tensor(clusters, dtype=torch.long).view(-1)
    else:
        output["clusters"] = None
    if topology is not None:
        output["topology_target"] = _to_tensor(topology, dtype=torch.float32).view(-1)
    else:
        output["topology_target"] = None
    return output


def load_feature_tensor(path: str | Path, map_location: str = "cpu") -> Tensor:
    """Load a feature tensor from a `.pt` file."""

    return load_bag_object(path, map_location=map_location)["features"]  # type: ignore[return-value]


def load_array_file(path: str | Path, *, key_candidates: tuple[str, ...] | None = None) -> Tensor:
    """Load a tensor/array from .pt, .npy, .npz, or CSV/TSV."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and key_candidates is not None:
            value = _first_existing(obj, key_candidates)
            if value is None:
                raise KeyError(f"No matching key found in {path}; tried {key_candidates}")
            obj = value
        return _to_tensor(obj)
    if suffix == ".npy":
        return torch.as_tensor(np.load(path))
    if suffix == ".npz":
        data = np.load(path)
        key = data.files[0]
        return torch.as_tensor(data[key])
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix in {".tsv", ".txt"} else ","
        return torch.as_tensor(pd.read_csv(path, sep=sep).to_numpy())
    raise ValueError(f"Unsupported array file extension: {path}")


def build_label_map(values: pd.Series) -> dict[Any, int]:
    """Build a stable mapping for string or mixed classification labels."""

    cleaned = [normalize_label_key(value) for value in values.dropna().tolist()]
    if all(isinstance(value, int) for value in cleaned):
        unique = sorted(set(cleaned))
        if unique and min(unique) == 0:
            return {value: value for value in unique}
        return {value: idx for idx, value in enumerate(unique)}
    labels = sorted({str(value) for value in cleaned})
    return {label: idx for idx, label in enumerate(labels)}


def build_stage_label_map(values: pd.Series) -> dict[Any, int]:
    """Build an ordinal map for pathological stage labels."""

    cleaned = [value for value in values.dropna().tolist() if not is_missing(value)]
    numeric: list[int] = []
    all_numeric = True
    for value in cleaned:
        try:
            numeric.append(int(float(value)))
        except (TypeError, ValueError):
            all_numeric = False
            break
    if all_numeric and numeric:
        unique = sorted(set(numeric))
        if min(unique) == 0:
            return {value: value for value in unique}
        if min(unique) == 1:
            return {value: value - 1 for value in unique}
        return {value: idx for idx, value in enumerate(unique)}

    mapping: dict[Any, int] = {}
    for value in cleaned:
        key = str(value).upper().replace("STAGE", "").replace(" ", "").strip()
        if key in _ROMAN_STAGE:
            mapping[normalize_label_key(value)] = _ROMAN_STAGE[key]
        else:
            mapping[normalize_label_key(value)] = len(mapping)
    return mapping


def fit_survival_bin_edges(
    df: pd.DataFrame,
    event_time_column: str = "event_time",
    event_observed_column: str = "event_observed",
    n_bins: int = 4,
) -> np.ndarray | None:
    """Fit discrete-time survival bin edges from a training dataframe only.

    The training CLI calls this *after* splitting the manifest so that
    validation/external follow-up times do not influence the discretization used
    by the survival head. Edges are based on uncensored training samples when
    possible, following common discrete-time survival practice, and are expanded
    to ``[-inf, +inf]`` at the boundaries so that external cohorts with longer or
    shorter follow-up can be assigned without refitting.
    """

    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if event_time_column not in df.columns or event_observed_column not in df.columns:
        return None

    valid = df[event_time_column].notna() & df[event_observed_column].notna()
    if valid.sum() == 0:
        return None

    event_times = pd.to_numeric(df.loc[valid, event_time_column], errors="coerce").dropna()
    if event_times.empty:
        return None
    observed = (
        pd.to_numeric(df.loc[event_times.index, event_observed_column], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    basis = event_times[observed == 1]
    if len(basis) < max(2, n_bins):
        basis = event_times

    values = np.asarray(basis, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    if np.unique(values).size <= 1:
        center = float(np.unique(values)[0])
        finite_edges = np.linspace(center - 0.5, center + 0.5, n_bins + 1)
    else:
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        finite_edges = np.quantile(values, quantiles)
        finite_edges = np.asarray(finite_edges, dtype=float)
        # Quantile bins can collapse when follow-up times are tied. Fall back to
        # equal-width bins over the observed training range so that the survival
        # head still has the configured number of discrete intervals.
        if np.unique(finite_edges).size < n_bins + 1:
            finite_edges = np.linspace(float(values.min()), float(values.max()), n_bins + 1)

    # Ensure strict monotonicity for downstream cut/digitize operations.
    finite_edges = np.asarray(finite_edges, dtype=float)
    for idx in range(1, len(finite_edges)):
        if finite_edges[idx] <= finite_edges[idx - 1]:
            finite_edges[idx] = finite_edges[idx - 1] + 1e-6
    finite_edges[0] = -np.inf
    finite_edges[-1] = np.inf
    return finite_edges


def apply_survival_bin_edges(
    df: pd.DataFrame,
    bin_edges: np.ndarray | list[float] | tuple[float, ...] | None,
    *,
    event_time_column: str = "event_time",
    output_column: str = "survival_bin",
) -> pd.DataFrame:
    """Apply pre-fitted survival bin edges to a dataframe without refitting."""

    df = df.copy()
    if bin_edges is None or event_time_column not in df.columns:
        return df
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("bin_edges must be a one-dimensional array with at least two entries")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("bin_edges must be strictly increasing")

    event_times = pd.to_numeric(df[event_time_column], errors="coerce")
    out = pd.Series(np.nan, index=df.index)
    valid = event_times.notna() & np.isfinite(event_times)
    if valid.any():
        # np.digitize returns 1..n_bins; convert to 0..n_bins-1 and clip.
        labels = np.digitize(event_times.loc[valid].to_numpy(dtype=float), edges[1:-1], right=False)
        labels = np.clip(labels, 0, len(edges) - 2)
        out.loc[valid] = labels.astype(int)
    df[output_column] = out
    return df


def add_discrete_survival_bins(
    df: pd.DataFrame,
    event_time_column: str = "event_time",
    event_observed_column: str = "event_observed",
    output_column: str = "survival_bin",
    n_bins: int = 4,
) -> pd.DataFrame:
    """Discretize survival times into bins.

    This helper is retained for backward compatibility. For leakage-free model
    training, prefer ``fit_survival_bin_edges(train_df)`` followed by
    ``apply_survival_bin_edges`` on train/validation/external data.
    """

    df = df.copy()
    if output_column in df.columns and df[output_column].notna().any():
        return df
    edges = fit_survival_bin_edges(
        df,
        event_time_column=event_time_column,
        event_observed_column=event_observed_column,
        n_bins=n_bins,
    )
    return apply_survival_bin_edges(
        df,
        edges,
        event_time_column=event_time_column,
        output_column=output_column,
    )


class FeatureBagDataset(Dataset):
    """WSI bag dataset backed by a CSV manifest."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        *,
        root: str | Path | None = None,
        task: str = "classification",
        tasks: tuple[str, ...] | list[str] = ("subtype", "stage", "survival"),
        feature_column: str = "feature_path",
        coordinate_column: str | None = None,
        cluster_column: str | None = None,
        topology_column: str | None = None,
        topology_columns: list[str] | tuple[str, ...] | None = None,
        compute_topology_target: bool = True,
        topology_target_dim: int = 6,
        id_column: str = "slide_id",
        label_column: str = "label",
        subtype_label_column: str = "subtype",
        stage_label_column: str = "stage",
        event_time_column: str = "event_time",
        event_observed_column: str = "event_observed",
        censorship_column: str = "censorship",
        survival_bin_column: str = "survival_bin",
        label_map: dict[Any, int] | None = None,
        label_maps: dict[str, dict[Any, int]] | None = None,
    ) -> None:
        self.df = read_manifest(manifest).reset_index(drop=True)
        self.root = root
        self.task = task
        self.tasks = tuple(tasks)
        self.feature_column = feature_column
        self.coordinate_column = coordinate_column
        self.cluster_column = cluster_column
        self.topology_column = topology_column
        self.topology_columns = tuple(topology_columns or ())
        self.compute_topology_target = compute_topology_target
        self.topology_target_dim = topology_target_dim
        self.id_column = id_column
        self.label_column = label_column
        self.subtype_label_column = subtype_label_column
        self.stage_label_column = stage_label_column
        self.event_time_column = event_time_column
        self.event_observed_column = event_observed_column
        self.censorship_column = censorship_column
        self.survival_bin_column = survival_bin_column
        self.label_map = label_map
        self.label_maps = label_maps or {}

        if task not in {"classification", "survival", "multitask"}:
            raise ValueError("task must be 'classification', 'survival', or 'multitask'")
        if feature_column not in self.df.columns:
            raise ValueError(f"Manifest is missing feature column: {feature_column}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_optional_path(self, row: pd.Series, column: str | None) -> Path | None:
        if column is None or column not in row.index or is_missing(row[column]):
            return None
        return resolve_feature_path(row[column], self.root)

    def _encode_label(self, value: Any, mapping: dict[Any, int] | None) -> int:
        if is_missing(value):
            return -1
        if mapping is None:
            return int(value)
        key = normalize_label_key(value)
        if key not in mapping and str(key) in mapping:
            key = str(key)
        return int(mapping[key])

    def _load_optional_arrays(
        self,
        row: pd.Series,
        bag: dict[str, Tensor | None],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        coords = bag.get("coords")
        clusters = bag.get("clusters")
        topology_target = bag.get("topology_target")

        coord_path = self._resolve_optional_path(row, self.coordinate_column)
        if coord_path is not None:
            coords = load_array_file(coord_path, key_candidates=COORD_KEYS).float()[:, :2]
        cluster_path = self._resolve_optional_path(row, self.cluster_column)
        if cluster_path is not None:
            clusters = load_array_file(cluster_path, key_candidates=CLUSTER_KEYS).long().view(-1)
        topo_path = self._resolve_optional_path(row, self.topology_column)
        if topo_path is not None:
            topology_target = (
                load_array_file(topo_path, key_candidates=TOPOLOGY_KEYS).float().view(-1)
            )

        if self.topology_columns and all(column in row.index for column in self.topology_columns):
            values = [row[column] for column in self.topology_columns]
            if not any(is_missing(value) for value in values):
                topology_target = torch.tensor(values, dtype=torch.float32).view(-1)

        if topology_target is None and self.compute_topology_target:
            if coords is not None or clusters is not None:
                topology_np = compute_lightweight_topology_descriptor(
                    coords=coords,
                    clusters=clusters,
                    output_dim=self.topology_target_dim,
                )
                topology_target = torch.tensor(topology_np, dtype=torch.float32)

        if coords is not None:
            coords = coords.float()
            if coords.ndim == 1:
                coords = coords.view(-1, 2)
            coords = coords[:, :2]
        if clusters is not None:
            clusters = clusters.long().view(-1)
        if topology_target is not None:
            topology_target = topology_target.float().view(-1)
        return coords, clusters, topology_target

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        feature_path = resolve_feature_path(row[self.feature_column], self.root)
        bag = load_bag_object(feature_path)
        features = bag["features"]
        assert features is not None
        coords, clusters, topology_target = self._load_optional_arrays(row, bag)

        item: dict[str, Any] = {
            "slide_id": str(row[self.id_column]) if self.id_column in row.index else str(idx),
            "features": features,
        }
        if coords is not None:
            if coords.shape[0] != features.shape[0]:
                message = (
                    f"Coordinate count {coords.shape[0]} does not match feature count "
                    f"{features.shape[0]} for {feature_path}"
                )
                raise ValueError(message)
            item["coords"] = coords
        if clusters is not None:
            if clusters.shape[0] != features.shape[0]:
                message = (
                    f"Cluster count {clusters.shape[0]} does not match feature count "
                    f"{features.shape[0]} for {feature_path}"
                )
                raise ValueError(message)
            item["clusters"] = clusters
        if topology_target is not None:
            item["topology_target"] = topology_target

        if self.task == "classification":
            value = row[self.label_column]
            item["label"] = self._encode_label(value, self.label_map)
        elif self.task == "survival":
            item.update(self._read_survival_fields(row))
        else:
            if "subtype" in self.tasks:
                item["subtype_label"] = self._encode_label(
                    row[self.subtype_label_column]
                    if self.subtype_label_column in row.index
                    else None,
                    self.label_maps.get("subtype"),
                )
            if "stage" in self.tasks:
                item["stage_label"] = self._encode_label(
                    row[self.stage_label_column] if self.stage_label_column in row.index else None,
                    self.label_maps.get("stage"),
                )
            if "survival" in self.tasks:
                item.update(self._read_survival_fields(row, allow_missing=True))
        return item

    def _read_survival_fields(self, row: pd.Series, allow_missing: bool = False) -> dict[str, Any]:
        if self.survival_bin_column not in row.index or is_missing(row[self.survival_bin_column]):
            if allow_missing:
                return {"event_time": float("nan"), "censorship": 0, "survival_bin": -1}
            raise ValueError(f"Missing survival bin column/value: {self.survival_bin_column}")
        event_time = (
            float(row[self.event_time_column])
            if self.event_time_column in row.index
            else float("nan")
        )
        if self.censorship_column in row.index and not is_missing(row[self.censorship_column]):
            censorship = int(row[self.censorship_column])
        elif self.event_observed_column in row.index and not is_missing(
            row[self.event_observed_column]
        ):
            censorship = 0 if int(row[self.event_observed_column]) == 1 else 1
        else:
            censorship = 0
        return {
            "event_time": event_time,
            "censorship": censorship,
            "survival_bin": int(row[self.survival_bin_column]),
        }


def collate_bags(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length bags and create masks for labels and topology."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    dims = {item["features"].shape[1] for item in batch}
    if len(dims) != 1:
        raise ValueError(f"All bags in a batch must share feature dim, got {sorted(dims)}")

    batch_size = len(batch)
    max_instances = max(item["features"].shape[0] for item in batch)
    feature_dim = batch[0]["features"].shape[1]
    features = torch.zeros(batch_size, max_instances, feature_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_instances, dtype=torch.bool)

    has_coords = any("coords" in item for item in batch)
    has_clusters = any("clusters" in item for item in batch)
    coords = torch.zeros(batch_size, max_instances, 2, dtype=torch.float32) if has_coords else None
    clusters = (
        torch.full((batch_size, max_instances), -1, dtype=torch.long) if has_clusters else None
    )

    for idx, item in enumerate(batch):
        bag = item["features"]
        n_instances = bag.shape[0]
        features[idx, :n_instances] = bag
        mask[idx, :n_instances] = True
        if coords is not None and "coords" in item:
            coords[idx, :n_instances] = item["coords"].float()[:, :2]
        if clusters is not None and "clusters" in item:
            clusters[idx, :n_instances] = item["clusters"].long().view(-1)

    output: dict[str, Any] = {
        "slide_id": [item["slide_id"] for item in batch],
        "features": features,
        "mask": mask,
    }
    if coords is not None:
        output["coords"] = coords
    if clusters is not None:
        output["clusters"] = clusters

    if "label" in batch[0]:
        output["label"] = torch.tensor([item.get("label", -1) for item in batch], dtype=torch.long)
    if "subtype_label" in batch[0]:
        output["subtype_label"] = torch.tensor(
            [item.get("subtype_label", -1) for item in batch], dtype=torch.long
        )
    if "stage_label" in batch[0]:
        output["stage_label"] = torch.tensor(
            [item.get("stage_label", -1) for item in batch], dtype=torch.long
        )
    if "survival_bin" in batch[0]:
        output["event_time"] = torch.tensor(
            [item.get("event_time", float("nan")) for item in batch],
            dtype=torch.float32,
        )
        output["censorship"] = torch.tensor(
            [item.get("censorship", 0) for item in batch],
            dtype=torch.float32,
        )
        output["survival_bin"] = torch.tensor(
            [item.get("survival_bin", -1) for item in batch],
            dtype=torch.long,
        )

    if any("topology_target" in item for item in batch):
        topo_dim = max(item.get("topology_target", torch.empty(0)).numel() for item in batch)
        topology_target = torch.zeros(batch_size, topo_dim, dtype=torch.float32)
        topology_mask = torch.zeros(batch_size, dtype=torch.bool)
        for idx, item in enumerate(batch):
            if "topology_target" not in item:
                continue
            target = item["topology_target"].float().view(-1)
            topology_target[idx, : target.numel()] = target
            topology_mask[idx] = torch.isfinite(target).all()
        output["topology_target"] = topology_target
        output["topology_mask"] = topology_mask

    return output
