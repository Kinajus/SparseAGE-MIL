"""Loss functions for SparseAGE-MIL and SparseAGE-MIL."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sparseage_model import SparseAGEOutput


class NoValidLossError(RuntimeError):
    """Raised when a batch contains no valid labels for any requested task."""


def nll_survival_loss(
    hazards: Tensor,
    survival: Tensor | None,
    targets: Tensor,
    censorship: Tensor,
    alpha: float = 0.0,
    eps: float = 1e-7,
) -> Tensor:
    """Discrete-time negative log-likelihood survival loss.

    ``censorship`` follows the common computational pathology convention:
    ``0`` means event observed and ``1`` means censored.
    """

    batch_size = targets.shape[0]
    targets = targets.view(batch_size, 1).long()
    censorship = censorship.view(batch_size, 1).float()
    if survival is None:
        survival = torch.cumprod(1.0 - hazards, dim=1)

    survival_padded = torch.cat([torch.ones_like(censorship), survival], dim=1)
    uncensored = -(1.0 - censorship) * (
        torch.log(torch.gather(survival_padded, 1, targets).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, targets).clamp(min=eps))
    )
    censored = -censorship * torch.log(torch.gather(survival_padded, 1, targets + 1).clamp(min=eps))
    loss = censored + uncensored
    loss = (1.0 - alpha) * loss + alpha * uncensored
    return loss.mean()


class NLLSurvivalLoss:
    """Callable wrapper for ``nll_survival_loss``."""

    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha

    def __call__(
        self,
        hazards: Tensor,
        survival: Tensor | None,
        targets: Tensor,
        censorship: Tensor,
    ) -> Tensor:
        return nll_survival_loss(
            hazards=hazards,
            survival=survival,
            targets=targets,
            censorship=censorship,
            alpha=self.alpha,
        )


def masked_cross_entropy(logits: Tensor, targets: Tensor, *, ignore_index: int = -1) -> Tensor:
    """Cross-entropy over samples whose target is not ``ignore_index``."""

    valid = targets != ignore_index
    if not torch.any(valid):
        raise NoValidLossError("no valid classification labels in batch")
    return F.cross_entropy(logits[valid], targets[valid].long())


def ordinal_stage_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    ordinal_weight: float = 0.15,
    ignore_index: int = -1,
) -> Tensor:
    """Stage loss that combines nominal CE with an ordinal-distance penalty."""

    valid = targets != ignore_index
    if not torch.any(valid):
        raise NoValidLossError("no valid stage labels in batch")
    logits_valid = logits[valid]
    targets_valid = targets[valid].long()
    ce = F.cross_entropy(logits_valid, targets_valid)
    if ordinal_weight <= 0:
        return ce
    probs = F.softmax(logits_valid, dim=1)
    stage_axis = torch.arange(logits_valid.shape[1], device=logits.device, dtype=probs.dtype)
    expected_stage = torch.sum(probs * stage_axis.unsqueeze(0), dim=1)
    denom = max(1, logits_valid.shape[1] - 1)
    ordinal = F.smooth_l1_loss(expected_stage / denom, targets_valid.float() / denom)
    return ce + ordinal_weight * ordinal


def topology_descriptor_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    *,
    cumsum_weight: float = 0.25,
) -> Tensor:
    """Huber loss for topology descriptors plus cumulative-shape matching."""

    valid = torch.isfinite(target).all(dim=1)
    if mask is not None:
        valid = valid & mask.bool()
    if not torch.any(valid):
        raise NoValidLossError("no valid topology targets in batch")
    pred_valid = pred[valid]
    target_valid = target[valid]
    dim = min(pred_valid.shape[1], target_valid.shape[1])
    pred_valid = pred_valid[:, :dim]
    target_valid = target_valid[:, :dim]
    base = F.smooth_l1_loss(pred_valid, target_valid)
    if cumsum_weight <= 0 or dim < 2:
        return base
    cumulative = F.smooth_l1_loss(
        torch.cumsum(pred_valid, dim=1), torch.cumsum(target_valid, dim=1)
    )
    return base + cumsum_weight * cumulative


def compute_multitask_losses(
    out: SparseAGEOutput,
    batch: Mapping[str, Tensor],
    *,
    stage_ordinal_weight: float = 0.15,
    survival_alpha: float = 0.0,
    topology_weight: float = 0.10,
) -> dict[str, Tensor]:
    """Compute all valid endpoint and topology losses for one batch."""

    if out.task_logits is None:
        raise ValueError("compute_multitask_losses requires multitask model output")
    losses: dict[str, Tensor] = {}

    if "subtype" in out.task_logits and "subtype_label" in batch:
        try:
            losses["subtype"] = masked_cross_entropy(
                out.task_logits["subtype"], batch["subtype_label"]
            )
        except NoValidLossError:
            pass

    if "stage" in out.task_logits and "stage_label" in batch:
        try:
            losses["stage"] = ordinal_stage_loss(
                out.task_logits["stage"],
                batch["stage_label"],
                ordinal_weight=stage_ordinal_weight,
            )
        except NoValidLossError:
            pass

    if "survival" in out.task_hazards and "survival_bin" in batch:
        valid = batch["survival_bin"] >= 0
        if torch.any(valid):
            losses["survival"] = nll_survival_loss(
                hazards=out.task_hazards["survival"][valid],
                survival=out.task_survival["survival"][valid],
                targets=batch["survival_bin"][valid],
                censorship=batch["censorship"][valid],
                alpha=survival_alpha,
            )

    if topology_weight > 0 and out.topology_pred is not None and "topology_target" in batch:
        try:
            losses["topology"] = topology_weight * topology_descriptor_loss(
                out.topology_pred,
                batch["topology_target"],
                mask=batch.get("topology_mask"),
            )
        except NoValidLossError:
            pass

    if not losses:
        raise NoValidLossError("batch has no valid losses")
    return losses


def reduce_multitask_losses(
    losses: Mapping[str, Tensor],
    *,
    model: nn.Module | None = None,
    weights: Mapping[str, float] | None = None,
) -> Tensor:
    """Reduce task losses with fixed weights or model-owned uncertainty weights."""

    total: Tensor | None = None
    log_vars = getattr(model, "log_vars", None) if model is not None else None
    for name, loss in losses.items():
        if log_vars is not None and name in log_vars:
            s = log_vars[name]
            weighted = torch.exp(-s) * loss + s
        else:
            weight = 1.0 if weights is None else float(weights.get(name, 1.0))
            weighted = weight * loss
        total = weighted if total is None else total + weighted
    if total is None:
        raise NoValidLossError("no losses to reduce")
    return total


def weighted_loss_terms(
    losses: Mapping[str, Tensor],
    *,
    model: nn.Module | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Tensor]:
    """Return individually weighted losses, useful for PCGrad."""

    terms: dict[str, Tensor] = {}
    log_vars = getattr(model, "log_vars", None) if model is not None else None
    for name, loss in losses.items():
        if log_vars is not None and name in log_vars:
            s = log_vars[name]
            terms[name] = torch.exp(-s) * loss + s
        else:
            weight = 1.0 if weights is None else float(weights.get(name, 1.0))
            terms[name] = weight * loss
    return terms
