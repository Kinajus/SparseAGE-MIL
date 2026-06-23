"""Prediction export helpers for evaluation tables.

The functions in this module intentionally write one row per slide. This makes
it straightforward to run DeLong tests, paired C-index permutation tests,
multivariable Cox models, calibration summaries, and external-cohort analyses
without re-running neural network inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sparseage_model import SparseAGEMIL
from sparseage_training import move_to_device
from sparseage_visualization import export_sparse_topology


def _forward_kwargs(
    batch: dict[str, Any], *, return_attention: bool, return_topology: bool
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "mask": batch.get("mask"),
        "return_attention": return_attention,
        "return_topology": return_topology,
    }
    if "coords" in batch:
        kwargs["coords"] = batch["coords"]
    if "clusters" in batch:
        kwargs["clusters"] = batch["clusters"]
    return kwargs


def _as_numpy(value: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _event_observed_from_batch(batch: dict[str, Any], index: int) -> float:
    if "censorship" not in batch:
        return float("nan")
    return float(1.0 - float(batch["censorship"][index].detach().cpu()))


def _masked_length(batch: dict[str, Any], index: int) -> int:
    if "mask" not in batch:
        return int(batch["features"].shape[1])
    return int(batch["mask"][index].detach().cpu().sum().item())


def _append_prob_columns(row: dict[str, Any], prefix: str, probs: np.ndarray) -> None:
    for cls_idx, value in enumerate(np.asarray(probs).reshape(-1)):
        row[f"{prefix}_prob_{cls_idx}"] = float(value)
    row[f"{prefix}_pred"] = int(np.asarray(probs).argmax())


@torch.no_grad()
def collect_slide_predictions(
    model: SparseAGEMIL,
    loader: DataLoader,
    *,
    device: torch.device,
    task: str,
    export_topology_dir: str | Path | None = None,
    topology_top_fraction: float = 0.10,
) -> pd.DataFrame:
    """Collect one prediction row per slide from a dataloader.

    Parameters
    ----------
    model:
        Trained SparseAGE model.
    loader:
        Dataloader using ``collate_bags``.
    device:
        Torch device for inference.
    task:
        ``classification``, ``survival`` or ``multitask``.
    export_topology_dir:
        Optional directory for sparse node/edge CSV files. Export requires
        coordinates in the batch and attention/topology outputs from the model.
    topology_top_fraction:
        Fraction of patches flagged as high-attention in node CSV exports.
    """

    model.eval()
    rows: list[dict[str, Any]] = []
    topology_dir = Path(export_topology_dir) if export_topology_dir is not None else None

    for batch in tqdm(loader, desc="predict", leave=False):
        batch = move_to_device(batch, device)
        need_topology = topology_dir is not None
        out = model(
            batch["features"],
            **_forward_kwargs(batch, return_attention=True, return_topology=need_topology),
        )
        slide_ids = batch["slide_id"]
        batch_size = len(slide_ids)

        logits = _as_numpy(out.logits)
        hazards = _as_numpy(out.hazards)
        survival = _as_numpy(out.survival)
        risk = _as_numpy(out.risk)

        task_logits = {key: _as_numpy(value) for key, value in (out.task_logits or {}).items()}
        task_hazards = {key: _as_numpy(value) for key, value in (out.task_hazards or {}).items()}
        task_survival = {key: _as_numpy(value) for key, value in (out.task_survival or {}).items()}
        task_risk = {key: _as_numpy(value) for key, value in (out.task_risk or {}).items()}

        for i in range(batch_size):
            row: dict[str, Any] = {"slide_id": slide_ids[i]}
            if "label" in batch:
                row["label"] = int(batch["label"][i].detach().cpu())
            if "subtype_label" in batch:
                row["subtype_label"] = int(batch["subtype_label"][i].detach().cpu())
            if "stage_label" in batch:
                row["stage_label"] = int(batch["stage_label"][i].detach().cpu())
            if "event_time" in batch:
                row["event_time"] = float(batch["event_time"][i].detach().cpu())
            if "censorship" in batch:
                row["censorship"] = float(batch["censorship"][i].detach().cpu())
                row["event_observed"] = _event_observed_from_batch(batch, i)
            if "survival_bin" in batch:
                row["survival_bin"] = int(batch["survival_bin"][i].detach().cpu())

            if task == "classification" and logits is not None:
                probs = torch.softmax(torch.as_tensor(logits[i]), dim=0).numpy()
                _append_prob_columns(row, "class", probs)
            elif task == "survival":
                if risk is not None:
                    row["risk_score"] = float(risk[i])
                if hazards is not None:
                    for bin_idx, value in enumerate(hazards[i]):
                        row[f"hazard_bin_{bin_idx}"] = float(value)
                if survival is not None:
                    for bin_idx, value in enumerate(survival[i]):
                        row[f"survival_prob_bin_{bin_idx}"] = float(value)
            elif task == "multitask":
                if "subtype" in task_logits and int(row.get("subtype_label", 0)) >= -1:
                    probs = torch.softmax(torch.as_tensor(task_logits["subtype"][i]), dim=0).numpy()
                    _append_prob_columns(row, "subtype", probs)
                if "stage" in task_logits:
                    probs = torch.softmax(torch.as_tensor(task_logits["stage"][i]), dim=0).numpy()
                    _append_prob_columns(row, "stage", probs)
                    row["stage_expected"] = float(np.sum(np.arange(len(probs)) * probs))
                if "survival" in task_risk:
                    row["risk_score"] = float(task_risk["survival"][i])
                    if "survival" in task_hazards:
                        for bin_idx, value in enumerate(task_hazards["survival"][i]):
                            row[f"hazard_bin_{bin_idx}"] = float(value)
                    if "survival" in task_survival:
                        for bin_idx, value in enumerate(task_survival["survival"][i]):
                            row[f"survival_prob_bin_{bin_idx}"] = float(value)

            if topology_dir is not None and "coords" in batch:
                n_instances = _masked_length(batch, i)
                attention = None
                if out.task_attention is not None:
                    # Prefer survival attention for survival-oriented interpretation.
                    attention = out.task_attention.get("survival")
                    if attention is None:
                        attention = out.task_attention.get("subtype")
                if attention is None:
                    attention = out.attention
                if attention is not None and out.topk_indices is not None:
                    coords_np = batch["coords"][i, :n_instances].detach().cpu()
                    attention_np = (
                        attention[i, :, :n_instances].detach().cpu()
                        if attention.ndim == 3
                        else attention[i, :n_instances].detach().cpu()
                    )
                    topk_idx = out.topk_indices[i, :n_instances].detach().cpu()
                    topk_w = (
                        out.topk_weights[i, :n_instances].detach().cpu()
                        if out.topk_weights is not None
                        else None
                    )
                    clusters_np = (
                        batch["clusters"][i, :n_instances].detach().cpu()
                        if "clusters" in batch
                        else None
                    )
                    nodes_path, edges_path = export_sparse_topology(
                        slide_id=str(slide_ids[i]),
                        coords=coords_np,
                        attention=attention_np,
                        topk_indices=topk_idx,
                        topk_weights=topk_w,
                        output_dir=topology_dir,
                        top_fraction=topology_top_fraction,
                        clusters=clusters_np,
                    )
                    row["topology_nodes_path"] = str(nodes_path)
                    row["topology_edges_path"] = str(edges_path)
            rows.append(row)

    return pd.DataFrame(rows)


def export_slide_predictions(
    model: SparseAGEMIL,
    loader: DataLoader,
    *,
    device: torch.device,
    task: str,
    output_path: str | Path,
    export_topology_dir: str | Path | None = None,
    topology_top_fraction: float = 0.10,
) -> Path:
    """Run inference and save per-slide predictions to CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = collect_slide_predictions(
        model,
        loader,
        device=device,
        task=task,
        export_topology_dir=export_topology_dir,
        topology_top_fraction=topology_top_fraction,
    )
    predictions.to_csv(output_path, index=False)
    return output_path
