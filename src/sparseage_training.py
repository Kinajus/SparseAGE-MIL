"""Training utilities used by the SparseAGE-MIL CLI."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from sparseage_losses import (
    NLLSurvivalLoss,
    NoValidLossError,
    compute_multitask_losses,
    reduce_multitask_losses,
    weighted_loss_terms,
)
from sparseage_metrics import classification_metrics, concordance_index
from sparseage_model import SparseAGEMIL


def seed_everything(seed: int = 2021) -> None:
    """Set common random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values from a collated batch to a device."""

    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def _forward_kwargs(batch: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"mask": batch.get("mask")}
    if "coords" in batch:
        kwargs["coords"] = batch["coords"]
    if "clusters" in batch:
        kwargs["clusters"] = batch["clusters"]
    return kwargs


def build_optimizer(
    model: nn.Module,
    name: str = "Adam",
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
) -> torch.optim.Optimizer:
    """Build an optimizer by name."""

    params = [param for param in model.parameters() if param.requires_grad]
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str = "cosine",
    epochs: int = 200,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build a learning-rate scheduler by name."""

    name = str(name).lower()
    if name in {"none", "null", ""}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.1)
    raise ValueError(f"Unsupported scheduler: {name}")


def _flatten_grads(
    parameters: list[nn.Parameter],
) -> tuple[torch.Tensor, list[torch.Size], list[int]]:
    chunks: list[torch.Tensor] = []
    shapes: list[torch.Size] = []
    sizes: list[int] = []
    for param in parameters:
        shapes.append(param.shape)
        sizes.append(param.numel())
        if param.grad is None:
            chunks.append(torch.zeros_like(param, memory_format=torch.preserve_format).reshape(-1))
        else:
            chunks.append(param.grad.detach().clone().reshape(-1))
    return torch.cat(chunks), shapes, sizes


def _assign_flat_grad(
    parameters: list[nn.Parameter],
    flat_grad: torch.Tensor,
    shapes: list[torch.Size],
    sizes: list[int],
) -> None:
    offset = 0
    for param, shape, size in zip(parameters, shapes, sizes, strict=True):
        grad = (
            flat_grad[offset : offset + size].view(shape).to(device=param.device, dtype=param.dtype)
        )
        param.grad = grad.clone()
        offset += size


def _pcgrad_step(
    losses: dict[str, torch.Tensor],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Apply a small PCGrad step for conflicting multi-task gradients."""

    parameters = [param for param in model.parameters() if param.requires_grad]
    flat_grads: list[torch.Tensor] = []
    shapes: list[torch.Size] | None = None
    sizes: list[int] | None = None

    loss_values = list(losses.values())
    for idx, loss in enumerate(loss_values):
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=idx < len(loss_values) - 1)
        flat, current_shapes, current_sizes = _flatten_grads(parameters)
        flat_grads.append(flat)
        if shapes is None:
            shapes, sizes = current_shapes, current_sizes

    projected: list[torch.Tensor] = []
    for i, grad in enumerate(flat_grads):
        g = grad.clone()
        order = list(range(len(flat_grads)))
        random.shuffle(order)
        for j in order:
            if i == j:
                continue
            other = flat_grads[j]
            dot = torch.dot(g, other)
            if dot < 0:
                g = g - dot / (other.norm().pow(2) + 1e-12) * other
        projected.append(g)

    optimizer.zero_grad(set_to_none=True)
    assert shapes is not None and sizes is not None
    _assign_flat_grad(parameters, torch.stack(projected, dim=0).mean(dim=0), shapes, sizes)
    optimizer.step()


def train_one_epoch(
    model: SparseAGEMIL,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    task: str,
    amp: bool = False,
    stage_ordinal_weight: float = 0.15,
    survival_alpha: float = 0.0,
    topology_weight: float = 0.10,
    loss_weights: dict[str, float] | None = None,
    gradient_strategy: str = "standard",
) -> float | dict[str, float]:
    """Train for one epoch.

    Returns a scalar mean loss for single-task training and a dictionary of mean
    total/component losses for multi-task training.
    """

    model.train()
    scaler = GradScaler(
        "cuda", enabled=amp and device.type == "cuda" and gradient_strategy != "pcgrad"
    )
    criterion_cls = nn.CrossEntropyLoss()
    criterion_surv = NLLSurvivalLoss(alpha=survival_alpha)
    total_loss = 0.0
    component_sums: dict[str, float] = {}
    valid_steps = 0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if task == "multitask":
            if gradient_strategy == "pcgrad":
                out = model(batch["features"], **_forward_kwargs(batch))
                try:
                    losses = compute_multitask_losses(
                        out,
                        batch,
                        stage_ordinal_weight=stage_ordinal_weight,
                        survival_alpha=survival_alpha,
                        topology_weight=topology_weight,
                    )
                except NoValidLossError:
                    continue
                weighted_terms = weighted_loss_terms(losses, model=model, weights=loss_weights)
                _pcgrad_step(weighted_terms, model=model, optimizer=optimizer)
                loss = sum(weighted_terms.values())
            else:
                with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                    out = model(batch["features"], **_forward_kwargs(batch))
                    try:
                        losses = compute_multitask_losses(
                            out,
                            batch,
                            stage_ordinal_weight=stage_ordinal_weight,
                            survival_alpha=survival_alpha,
                            topology_weight=topology_weight,
                        )
                    except NoValidLossError:
                        continue
                    loss = reduce_multitask_losses(losses, model=model, weights=loss_weights)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            for name, value in losses.items():
                component_sums[name] = component_sums.get(name, 0.0) + float(value.detach().cpu())
        else:
            with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                out = model(batch["features"], **_forward_kwargs(batch))
                if task == "classification":
                    loss = criterion_cls(out.logits, batch["label"])
                else:
                    loss = criterion_surv(
                        hazards=out.hazards,
                        survival=out.survival,
                        targets=batch["survival_bin"],
                        censorship=batch["censorship"],
                    )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.detach().cpu())
        valid_steps += 1

    mean_loss = total_loss / max(1, valid_steps)
    if task == "multitask":
        result = {"loss": mean_loss}
        result.update(
            {f"loss_{name}": value / max(1, valid_steps) for name, value in component_sums.items()}
        )
        return result
    return mean_loss


@torch.no_grad()
def evaluate_classification(
    model: SparseAGEMIL,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate classification metrics."""

    model.eval()
    labels: list[int] = []
    probabilities: list[np.ndarray] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_to_device(batch, device)
        out = model(batch["features"], **_forward_kwargs(batch))
        prob = torch.softmax(out.logits, dim=1)
        labels.extend(batch["label"].detach().cpu().tolist())
        probabilities.append(prob.detach().cpu().numpy())
    return classification_metrics(np.asarray(labels), np.concatenate(probabilities, axis=0))


@torch.no_grad()
def evaluate_survival(
    model: SparseAGEMIL,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate survival prediction with C-index."""

    model.eval()
    risks: list[float] = []
    event_times: list[float] = []
    event_observed: list[int] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_to_device(batch, device)
        out = model(batch["features"], **_forward_kwargs(batch))
        risks.extend(out.risk.detach().cpu().tolist())
        event_times.extend(batch["event_time"].detach().cpu().tolist())
        event_observed.extend((1.0 - batch["censorship"]).detach().cpu().int().tolist())
    c_index = concordance_index(
        event_times=np.asarray(event_times),
        risks=np.asarray(risks),
        event_observed=np.asarray(event_observed),
    )
    return {"c_index": c_index}


@torch.no_grad()
def evaluate_multitask(
    model: SparseAGEMIL,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate all available multi-task endpoints."""

    model.eval()
    subtype_labels: list[int] = []
    subtype_probs: list[np.ndarray] = []
    stage_labels: list[int] = []
    stage_probs: list[np.ndarray] = []
    survival_risks: list[float] = []
    event_times: list[float] = []
    event_observed: list[int] = []
    topo_abs_errors: list[np.ndarray] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_to_device(batch, device)
        out = model(batch["features"], **_forward_kwargs(batch))
        assert out.task_logits is not None

        if "subtype" in out.task_logits and "subtype_label" in batch:
            valid = batch["subtype_label"] >= 0
            if torch.any(valid):
                subtype_labels.extend(batch["subtype_label"][valid].detach().cpu().tolist())
                subtype_probs.append(
                    torch.softmax(out.task_logits["subtype"][valid], dim=1).cpu().numpy()
                )

        if "stage" in out.task_logits and "stage_label" in batch:
            valid = batch["stage_label"] >= 0
            if torch.any(valid):
                stage_labels.extend(batch["stage_label"][valid].detach().cpu().tolist())
                stage_probs.append(
                    torch.softmax(out.task_logits["stage"][valid], dim=1).cpu().numpy()
                )

        if "survival" in (out.task_risk or {}) and "survival_bin" in batch:
            valid = batch["survival_bin"] >= 0
            if torch.any(valid):
                survival_risks.extend(out.task_risk["survival"][valid].detach().cpu().tolist())
                event_times.extend(batch["event_time"][valid].detach().cpu().tolist())
                event_observed.extend(
                    (1.0 - batch["censorship"][valid]).detach().cpu().int().tolist()
                )

        if out.topology_pred is not None and "topology_target" in batch:
            valid = batch.get(
                "topology_mask", torch.ones_like(batch["topology_target"][:, 0]).bool()
            )
            if torch.any(valid):
                dim = min(out.topology_pred.shape[1], batch["topology_target"].shape[1])
                topo_abs_errors.append(
                    torch.abs(
                        out.topology_pred[valid, :dim] - batch["topology_target"][valid, :dim]
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

    metrics: dict[str, float] = {}
    if subtype_probs:
        metrics.update(
            {
                f"subtype_{key}": value
                for key, value in classification_metrics(
                    np.asarray(subtype_labels), np.concatenate(subtype_probs, axis=0)
                ).items()
            }
        )
    if stage_probs:
        probs = np.concatenate(stage_probs, axis=0)
        labels = np.asarray(stage_labels)
        metrics.update(
            {f"stage_{key}": value for key, value in classification_metrics(labels, probs).items()}
        )
        metrics["stage_mae"] = float(np.mean(np.abs(probs.argmax(axis=1) - labels)))
    if survival_risks:
        metrics["survival_c_index"] = concordance_index(
            event_times=np.asarray(event_times),
            risks=np.asarray(survival_risks),
            event_observed=np.asarray(event_observed),
        )
    if topo_abs_errors:
        metrics["topology_mae"] = float(np.concatenate(topo_abs_errors, axis=0).mean())
    return metrics


def save_checkpoint(
    path: str | Path,
    *,
    model: SparseAGEMIL,
    epoch: int,
    metric_name: str,
    metric_value: float,
    config: dict[str, Any],
) -> None:
    """Save a model checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "model_state": model.state_dict(),
            "model_config": model.config.__dict__,
            "config": config,
        },
        path,
    )


def append_metrics_csv(path: str | Path, row: dict[str, Any]) -> None:
    """Append one metrics row to a CSV file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
