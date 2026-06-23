"""Statistical comparison helpers for experiments."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score

from sparseage_metrics import concordance_index


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return AUC and percentile bootstrap confidence interval."""

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    auc = float(roc_auc_score(y_true, y_score))
    boot: list[float] = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot.append(float(roc_auc_score(y_true[idx], y_score[idx])))
    if not boot:
        return auc, float("nan"), float("nan")
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return auc, float(lo), float(hi)


def paired_bootstrap_auc_test(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap test for AUC difference between two models."""

    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    observed = float(roc_auc_score(y_true, score_a) - roc_auc_score(y_true, score_b))
    diffs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(
            float(
                roc_auc_score(y_true[idx], score_a[idx]) - roc_auc_score(y_true[idx], score_b[idx])
            )
        )
    if not diffs:
        return {"delta_auc": observed, "p_value": float("nan")}
    diffs_np = np.asarray(diffs)
    p = 2 * min(np.mean(diffs_np <= 0), np.mean(diffs_np >= 0))
    return {"delta_auc": observed, "p_value": float(min(1.0, p))}


def bootstrap_cindex_ci(
    event_times: np.ndarray,
    risks: np.ndarray,
    event_observed: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return Harrell C-index and percentile bootstrap confidence interval."""

    event_times = np.asarray(event_times)
    risks = np.asarray(risks)
    event_observed = np.asarray(event_observed)
    rng = np.random.default_rng(seed)
    c_index = concordance_index(event_times, risks, event_observed)
    boot: list[float] = []
    n = len(event_times)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        value = concordance_index(event_times[idx], risks[idx], event_observed[idx])
        if value == value:
            boot.append(value)
    if not boot:
        return c_index, float("nan"), float("nan")
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return c_index, float(lo), float(hi)


def paired_permutation_cindex_test(
    event_times: np.ndarray,
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    event_observed: np.ndarray,
    *,
    n_perm: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired permutation test for C-index difference."""

    event_times = np.asarray(event_times)
    risk_a = np.asarray(risk_a)
    risk_b = np.asarray(risk_b)
    event_observed = np.asarray(event_observed)
    rng = np.random.default_rng(seed)
    observed = concordance_index(event_times, risk_a, event_observed) - concordance_index(
        event_times, risk_b, event_observed
    )
    diffs: list[float] = []
    for _ in range(n_perm):
        swap = rng.random(len(event_times)) < 0.5
        perm_a = np.where(swap, risk_b, risk_a)
        perm_b = np.where(swap, risk_a, risk_b)
        diff = concordance_index(event_times, perm_a, event_observed) - concordance_index(
            event_times, perm_b, event_observed
        )
        if diff == diff:
            diffs.append(diff)
    if not diffs:
        return {"delta_c_index": float(observed), "p_value": float("nan")}
    p = np.mean(np.abs(diffs) >= abs(observed))
    return {"delta_c_index": float(observed), "p_value": float(p)}


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    midranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return midranks


def _fast_delong(predictions_sorted: np.ndarray, n_positive: int) -> tuple[np.ndarray, np.ndarray]:
    m = n_positive
    n = predictions_sorted.shape[1] - m
    positive = predictions_sorted[:, :m]
    negative = predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _compute_midrank(positive[r])
        ty[r] = _compute_midrank(negative[r])
        tz[r] = _compute_midrank(predictions_sorted[r])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    covariance = sx / m + sy / n
    covariance = np.atleast_2d(covariance)
    return aucs, covariance


def delong_auc_test(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
) -> dict[str, float]:
    """Two-sided DeLong test for paired binary AUCs.

    Returns AUCs, AUC difference, z statistic, and p value. Inputs must be
    binary labels and continuous scores for the positive class.
    """

    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)
    n_positive = int(y_true.sum())
    if n_positive == 0 or n_positive == len(y_true):
        raise ValueError("DeLong test requires both positive and negative samples")
    preds = np.vstack([np.asarray(score_a), np.asarray(score_b)])[:, order]
    aucs, covariance = _fast_delong(preds, n_positive)
    contrast = np.array([[1.0, -1.0]])
    var = float((contrast @ covariance @ contrast.T).item())
    delta = float(aucs[0] - aucs[1])
    if var <= 0:
        z = float("nan")
        p = float("nan")
    else:
        z = delta / math.sqrt(var)
        p = math.erfc(abs(z) / math.sqrt(2.0))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta_auc": delta,
        "z": float(z),
        "p_value": float(p),
    }
