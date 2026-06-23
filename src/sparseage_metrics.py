"""Evaluation metrics for SparseAGE-MIL."""

from __future__ import annotations

import numpy as np


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Return accuracy, AUC, precision, recall, and F1."""

    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError as exc:
        raise ImportError("classification_metrics requires scikit-learn") from exc

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predictions = probabilities.argmax(axis=1)
    average = "binary" if probabilities.shape[1] == 2 else "macro"

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, average=average, zero_division=0)),
        "recall": float(recall_score(labels, predictions, average=average, zero_division=0)),
        "f1": float(f1_score(labels, predictions, average=average, zero_division=0)),
    }
    try:
        if probabilities.shape[1] == 2:
            metrics["auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
        else:
            metrics["auc"] = float(roc_auc_score(labels, probabilities, multi_class="ovr"))
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def concordance_index(
    event_times: np.ndarray,
    risks: np.ndarray,
    event_observed: np.ndarray,
) -> float:
    """Compute Harrell's concordance index without external survival packages."""

    event_times = np.asarray(event_times, dtype=float)
    risks = np.asarray(risks, dtype=float)
    event_observed = np.asarray(event_observed, dtype=bool)

    concordant = 0.0
    permissible = 0.0
    n = len(event_times)
    for i in range(n):
        for j in range(i + 1, n):
            if event_times[i] == event_times[j]:
                continue
            if event_times[i] < event_times[j] and event_observed[i]:
                permissible += 1
                concordant += _pair_score(risks[i], risks[j])
            elif event_times[j] < event_times[i] and event_observed[j]:
                permissible += 1
                concordant += _pair_score(risks[j], risks[i])
    if permissible == 0:
        return float("nan")
    return float(concordant / permissible)


def _pair_score(earlier_risk: float, later_risk: float) -> float:
    if earlier_risk > later_risk:
        return 1.0
    if earlier_risk == later_risk:
        return 0.5
    return 0.0
