"""Spatial-topology regularized SparseAGE-MIL model components.

The implementation keeps the SparseAGE-MIL API for single-task
classification/survival with single-task and joint multi-task modes:

    feature projection -> spatial-feature sparse topology aggregation ->
    task-specific MIL attention/adapters -> subtype/stage/survival heads ->
    lightweight topology auxiliary head.

The topology module uses WSI coordinates and pseudo-histology cluster labels
as inductive biases inside supervised MIL.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import sqrt

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _as_tuple_topk(value: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, int):
        values = (value,)
    else:
        values = tuple(int(v) for v in value)
    values = tuple(sorted(set(values)))
    if not values or min(values) < 1:
        raise ValueError("topk must contain positive integers")
    return values


@dataclass
class SparseAGEConfig:
    """Configuration for :class:`SparseAGEMIL`.

    ``task`` may be ``"classification"``, ``"survival"``, or ``"multitask"``.
    In ``"multitask"`` mode, ``tasks`` controls which endpoint heads are built.
    """

    input_dim: int = 1024
    embedding_dim: int = 512
    attention_dim: int = 128
    n_classes: int = 2
    topk: int | tuple[int, ...] | list[int] = 6
    dropout: float = 0.25
    task: str = "classification"
    use_scaled_similarity: bool = True

    # Spatial/topology sparse graph priors.
    use_spatial: bool = True
    spatial_sigma: float = 256.0
    spatial_weight: float = 0.20
    cluster_weight: float = 0.15
    boundary_weight: float = 0.10
    include_self: bool = True

    # Joint multi-task setup.
    tasks: tuple[str, ...] | list[str] = field(
        default_factory=lambda: ("subtype", "stage", "survival")
    )
    subtype_classes: int = 2
    stage_classes: int = 4
    survival_bins: int = 4
    topology_target_dim: int = 6
    use_uncertainty_weighting: bool = True


@dataclass
class SparseAGEOutput:
    """Model output container.

    Single-task fields are kept at top level. Multi-task predictions are returned
    in dictionaries keyed by task name.
    """

    logits: Tensor | None = None
    hazards: Tensor | None = None
    survival: Tensor | None = None
    risk: Tensor | None = None
    attention: Tensor | None = None
    topk_indices: Tensor | None = None
    topk_weights: Tensor | None = None

    task_logits: dict[str, Tensor] | None = None
    task_hazards: dict[str, Tensor] | None = None
    task_survival: dict[str, Tensor] | None = None
    task_risk: dict[str, Tensor] | None = None
    task_attention: dict[str, Tensor] | None = None
    task_embeddings: dict[str, Tensor] | None = None

    topology_pred: Tensor | None = None
    slide_embedding: Tensor | None = None
    scale_weights: Tensor | None = None
    multi_topk_indices: dict[int, Tensor] | None = None
    multi_topk_weights: dict[int, Tensor] | None = None


def initialize_weights(module: nn.Module) -> None:
    """Initialize linear and normalization layers with MIL-friendly defaults."""

    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)


class FeatureProjectionNetwork(nn.Module):
    """Shared nonlinear projection before sparse topology construction."""

    def __init__(self, input_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SpatialFeatureTopologyAggregator(nn.Module):
    """Multi-K sparse aggregator with spatial and pseudo-histology priors.

    For each source patch, neighbors are selected by a score combining feature
    affinity, physical proximity, same-cluster coherence, and cross-cluster
    interface evidence. Multi-K neighborhoods are fused by a learnable scale
    gate, making K sensitivity less brittle while preserving sparse computation.
    """

    def __init__(
        self,
        dim: int = 512,
        topk: int | Sequence[int] = (4, 8, 16),
        *,
        use_scaled_similarity: bool = True,
        use_spatial: bool = True,
        spatial_sigma: float = 256.0,
        spatial_weight: float = 0.20,
        cluster_weight: float = 0.15,
        boundary_weight: float = 0.10,
        include_self: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.topk_values = _as_tuple_topk(topk)
        self.use_scaled_similarity = use_scaled_similarity
        self.use_spatial = use_spatial
        self.spatial_sigma = float(spatial_sigma)
        self.spatial_weight = float(spatial_weight)
        self.cluster_weight = float(cluster_weight)
        self.boundary_weight = float(boundary_weight)
        self.include_self = include_self

        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.message_gate = nn.Linear(dim * 2, dim)
        self.scale_gate = nn.Linear(dim, len(self.topk_values))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        coords: Tensor | None = None,
        clusters: Tensor | None = None,
        return_topology: bool = False,
    ) -> tuple[
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        dict[int, Tensor] | None,
        dict[int, Tensor] | None,
    ]:
        squeeze_batch = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            if mask is not None and mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if coords is not None and coords.ndim == 2:
                coords = coords.unsqueeze(0)
            if clusters is not None and clusters.ndim == 1:
                clusters = clusters.unsqueeze(0)
            squeeze_batch = True
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, N, D] or [N, D], got {tuple(x.shape)}")

        batch_size, n_instances, dim = x.shape
        if n_instances < 1:
            raise ValueError("SparseAGE cannot aggregate an empty bag")

        if mask is None:
            mask = torch.ones(batch_size, n_instances, device=x.device, dtype=torch.bool)
        else:
            mask = mask.to(device=x.device, dtype=torch.bool)
            if mask.shape != (batch_size, n_instances):
                raise ValueError(
                    f"Expected mask shape {(batch_size, n_instances)}, got {tuple(mask.shape)}"
                )

        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        scores = torch.matmul(q, k.transpose(-2, -1))
        if self.use_scaled_similarity:
            scores = scores / sqrt(dim)

        distances: Tensor | None = None
        if coords is not None and self.use_spatial:
            coords = coords.to(device=x.device, dtype=x.dtype)
            if coords.ndim != 3 or coords.shape[:2] != (batch_size, n_instances):
                raise ValueError(
                    "coords must have shape [B, N, C] matching x; "
                    f"got {tuple(coords.shape)} for x {tuple(x.shape)}"
                )
            distances = torch.cdist(coords, coords, p=2)
            sigma = max(self.spatial_sigma, 1e-6)
            scores = scores - self.spatial_weight * torch.log1p(distances / sigma)

        if clusters is not None:
            clusters = clusters.to(device=x.device).long()
            if clusters.shape != (batch_size, n_instances):
                expected_shape = (batch_size, n_instances)
                raise ValueError(
                    f"clusters must have shape {expected_shape}, got {tuple(clusters.shape)}"
                )
            valid_cluster = clusters >= 0
            valid_pair = valid_cluster.unsqueeze(1) & valid_cluster.unsqueeze(2)
            same_cluster = (clusters.unsqueeze(1) == clusters.unsqueeze(2)) & valid_pair
            scores = scores + self.cluster_weight * same_cluster.to(dtype=scores.dtype)
            if distances is not None and self.boundary_weight != 0.0:
                # Nearby cross-cluster edges approximate histologic interfaces.
                interface = (~same_cluster) & valid_pair
                interface_strength = torch.exp(-distances / max(self.spatial_sigma, 1e-6))
                scores = (
                    scores + self.boundary_weight * interface.to(scores.dtype) * interface_strength
                )

        if not self.include_self:
            eye = torch.eye(n_instances, device=x.device, dtype=torch.bool).unsqueeze(0)
            scores = scores.masked_fill(eye, torch.finfo(scores.dtype).min)

        key_mask = mask.unsqueeze(1)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        messages: list[Tensor] = []
        multi_indices: dict[int, Tensor] = {}
        multi_weights: dict[int, Tensor] = {}
        primary_indices: Tensor | None = None
        primary_weights: Tensor | None = None
        primary_k = max(self.topk_values)

        for k_value in self.topk_values:
            k_eff = min(k_value, n_instances)
            topk_scores, topk_indices = torch.topk(scores, k=k_eff, dim=-1)

            expanded_v = v.unsqueeze(1).expand(batch_size, n_instances, n_instances, dim)
            gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, -1, dim)
            neighbor_values = torch.gather(expanded_v, dim=2, index=gather_index)

            source_values = q.unsqueeze(2).expand_as(neighbor_values)
            gate = torch.sigmoid(
                self.message_gate(torch.cat([source_values, neighbor_values], dim=-1))
            )
            mixed = gate * neighbor_values + (1.0 - gate) * source_values

            # First-stage Top-K affinity and second-stage gated compatibility are combined.
            alpha = F.softmax(topk_scores, dim=-1)
            compat = torch.einsum("bnkd,bnkd->bnk", mixed, source_values) / sqrt(dim)
            beta = F.softmax(compat + torch.log(alpha.clamp_min(1e-8)), dim=-1)
            message = torch.sum(beta.unsqueeze(-1) * mixed, dim=2)
            messages.append(message)

            if return_topology:
                multi_indices[k_value] = topk_indices
                multi_weights[k_value] = beta
                if k_value == primary_k:
                    primary_indices = topk_indices
                    primary_weights = beta

        stacked = torch.stack(messages, dim=2)  # [B, N, S, D]
        scale_logits = self.scale_gate(q)  # [B, N, S]
        scale_weights = F.softmax(scale_logits, dim=-1)
        fused = torch.sum(scale_weights.unsqueeze(-1) * stacked, dim=2)
        out = self.norm(q + self.dropout(fused))
        out = out * mask.unsqueeze(-1)

        if squeeze_batch:
            out = out.squeeze(0)
            if primary_indices is not None:
                primary_indices = primary_indices.squeeze(0)
            if primary_weights is not None:
                primary_weights = primary_weights.squeeze(0)
            scale_weights = scale_weights.squeeze(0)
            if return_topology:
                multi_indices = {key: value.squeeze(0) for key, value in multi_indices.items()}
                multi_weights = {key: value.squeeze(0) for key, value in multi_weights.items()}

        if return_topology:
            return (
                out,
                primary_indices,
                primary_weights,
                scale_weights,
                multi_indices,
                multi_weights,
            )
        return out, None, None, None, None, None


class TopologyAwareLocalAggregator(nn.Module):
    """Backward-compatible wrapper around the spatial-feature aggregator."""

    def __init__(
        self,
        dim: int = 512,
        topk: int = 6,
        use_scaled_similarity: bool = True,
    ) -> None:
        super().__init__()
        self.aggregator = SpatialFeatureTopologyAggregator(
            dim=dim,
            topk=topk,
            use_scaled_similarity=use_scaled_similarity,
            use_spatial=False,
            spatial_weight=0.0,
            cluster_weight=0.0,
            boundary_weight=0.0,
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        return_topology: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        out, indices, weights, _, _, _ = self.aggregator(
            x,
            mask=mask,
            return_topology=return_topology,
        )
        return out, indices, weights


class AttentionMILPooling(nn.Module):
    """Attention-based MIL pooling."""

    def __init__(self, dim: int = 512, attention_dim: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        squeeze_batch = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            if mask is not None and mask.ndim == 1:
                mask = mask.unsqueeze(0)
            squeeze_batch = True
        scores = self.attention(x).transpose(1, 2)
        if mask is not None:
            mask = mask.to(device=x.device, dtype=torch.bool)
            scores = scores.masked_fill(~mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        attention = F.softmax(scores, dim=-1)
        pooled = torch.bmm(attention, x).flatten(1)
        if squeeze_batch:
            pooled = pooled.squeeze(0)
            attention = attention.squeeze(0)
        return pooled, attention


class ResidualTaskAdapter(nn.Module):
    """Small task-specific adapter for reducing negative transfer."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class TaskSpecificAttentionPooling(nn.Module):
    """Task-specific attention pooling over a shared sparse WSI representation."""

    def __init__(self, tasks: Iterable[str], dim: int, attention_dim: int) -> None:
        super().__init__()
        self.pools = nn.ModuleDict(
            {task: AttentionMILPooling(dim=dim, attention_dim=attention_dim) for task in tasks}
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        embeddings: dict[str, Tensor] = {}
        attentions: dict[str, Tensor] = {}
        for task, pool in self.pools.items():
            embeddings[task], attentions[task] = pool(x, mask=mask)
        return embeddings, attentions


class SparseAGEMIL(nn.Module):
    """Sparse Adaptive Graph Encoding MIL with optional joint multi-task heads."""

    def __init__(self, config: SparseAGEConfig) -> None:
        super().__init__()
        if config.task not in {"classification", "survival", "multitask"}:
            raise ValueError("task must be 'classification', 'survival', or 'multitask'")
        self.config = config
        self.tasks = tuple(config.tasks) if config.task == "multitask" else ()

        self.project = FeatureProjectionNetwork(
            input_dim=config.input_dim,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
        )
        self.topology = SpatialFeatureTopologyAggregator(
            dim=config.embedding_dim,
            topk=config.topk,
            use_scaled_similarity=config.use_scaled_similarity,
            use_spatial=config.use_spatial,
            spatial_sigma=config.spatial_sigma,
            spatial_weight=config.spatial_weight,
            cluster_weight=config.cluster_weight,
            boundary_weight=config.boundary_weight,
            include_self=config.include_self,
            dropout=config.dropout,
        )

        if config.task == "multitask":
            self.task_pool = TaskSpecificAttentionPooling(
                self.tasks,
                dim=config.embedding_dim,
                attention_dim=config.attention_dim,
            )
            self.task_adapters = nn.ModuleDict(
                {
                    task: ResidualTaskAdapter(config.embedding_dim, config.dropout)
                    for task in self.tasks
                }
            )
            self.heads = nn.ModuleDict()
            if "subtype" in self.tasks:
                self.heads["subtype"] = nn.Linear(config.embedding_dim, config.subtype_classes)
            if "stage" in self.tasks:
                self.heads["stage"] = nn.Linear(config.embedding_dim, config.stage_classes)
            if "survival" in self.tasks:
                self.heads["survival"] = nn.Linear(config.embedding_dim, config.survival_bins)
            extra_tasks = set(self.tasks) - {"subtype", "stage", "survival"}
            for task in sorted(extra_tasks):
                self.heads[task] = nn.Linear(config.embedding_dim, config.n_classes)

            if config.topology_target_dim > 0:
                self.topology_head: nn.Module | None = nn.Sequential(
                    nn.LayerNorm(config.embedding_dim),
                    nn.Linear(config.embedding_dim, max(1, config.embedding_dim // 2)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(config.dropout),
                    nn.Linear(max(1, config.embedding_dim // 2), config.topology_target_dim),
                )
            else:
                self.topology_head = None

            if config.use_uncertainty_weighting:
                log_vars = {task: nn.Parameter(torch.zeros(())) for task in self.tasks}
                if config.topology_target_dim > 0:
                    log_vars["topology"] = nn.Parameter(torch.zeros(()))
                self.log_vars = nn.ParameterDict(log_vars)
            else:
                self.log_vars = nn.ParameterDict()
        else:
            self.pool = AttentionMILPooling(config.embedding_dim, config.attention_dim)
            self.classifier = nn.Linear(config.embedding_dim, config.n_classes)

        self.apply(initialize_weights)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        coords: Tensor | None = None,
        clusters: Tensor | None = None,
        return_attention: bool = False,
        return_topology: bool = False,
    ) -> SparseAGEOutput:
        if x.ndim == 2:
            x = x.unsqueeze(0)
            if mask is not None and mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if coords is not None and coords.ndim == 2:
                coords = coords.unsqueeze(0)
            if clusters is not None and clusters.ndim == 1:
                clusters = clusters.unsqueeze(0)

        features = self.project(x.float())
        features, topk_indices, topk_weights, scale_weights, multi_indices, multi_weights = (
            self.topology(
                features,
                mask=mask,
                coords=coords,
                clusters=clusters,
                return_topology=return_topology,
            )
        )

        if self.config.task != "multitask":
            pooled, attention = self.pool(features, mask=mask)
            logits = self.classifier(pooled)
            hazards = survival = risk = None
            if self.config.task == "survival":
                hazards = torch.sigmoid(logits)
                survival = torch.cumprod(1.0 - hazards, dim=1)
                risk = -torch.sum(survival, dim=1)
            return SparseAGEOutput(
                logits=logits,
                hazards=hazards,
                survival=survival,
                risk=risk,
                attention=attention if return_attention else None,
                topk_indices=topk_indices,
                topk_weights=topk_weights,
                slide_embedding=pooled,
                scale_weights=scale_weights if return_topology else None,
                multi_topk_indices=multi_indices,
                multi_topk_weights=multi_weights,
            )

        task_embeddings, task_attention = self.task_pool(features, mask=mask)
        adapted_embeddings = {
            task: self.task_adapters[task](embedding) for task, embedding in task_embeddings.items()
        }
        task_logits: dict[str, Tensor] = {}
        task_hazards: dict[str, Tensor] = {}
        task_survival: dict[str, Tensor] = {}
        task_risk: dict[str, Tensor] = {}

        for task, embedding in adapted_embeddings.items():
            logits = self.heads[task](embedding)
            task_logits[task] = logits
            if task == "survival":
                hazards = torch.sigmoid(logits)
                survival = torch.cumprod(1.0 - hazards, dim=1)
                risk = -torch.sum(survival, dim=1)
                task_hazards[task] = hazards
                task_survival[task] = survival
                task_risk[task] = risk

        # A stable shared slide embedding for topology regularization.
        slide_embedding = torch.stack(list(adapted_embeddings.values()), dim=0).mean(dim=0)
        topology_pred = (
            self.topology_head(slide_embedding) if self.topology_head is not None else None
        )

        primary_task = "subtype" if "subtype" in task_logits else next(iter(task_logits))
        primary_attention = task_attention.get(primary_task)
        primary_logits = task_logits.get(primary_task)

        return SparseAGEOutput(
            logits=primary_logits,
            hazards=task_hazards.get("survival"),
            survival=task_survival.get("survival"),
            risk=task_risk.get("survival"),
            attention=primary_attention if return_attention else None,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            task_logits=task_logits,
            task_hazards=task_hazards,
            task_survival=task_survival,
            task_risk=task_risk,
            task_attention=task_attention if return_attention else None,
            task_embeddings=adapted_embeddings,
            topology_pred=topology_pred,
            slide_embedding=slide_embedding,
            scale_weights=scale_weights if return_topology else None,
            multi_topk_indices=multi_indices,
            multi_topk_weights=multi_weights,
        )

