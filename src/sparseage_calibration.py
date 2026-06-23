"""Calibration and stage-diagnostic utilities for exported predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, cohen_kappa_score, confusion_matrix, roc_auc_score


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Compute top-label expected calibration error for classification."""

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    confidence = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (
            (confidence > lo) & (confidence <= hi)
            if hi < 1
            else (confidence > lo) & (confidence <= hi)
        )
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def calibration_curve_table(
    y_true: np.ndarray,
    positive_probability: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return a bin-wise binary calibration curve table."""

    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(positive_probability, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:], strict=True)):
        mask = (prob >= lo) & (prob < hi) if idx < n_bins - 1 else (prob >= lo) & (prob <= hi)
        if not mask.any():
            rows.append(
                {
                    "bin": idx,
                    "prob_low": lo,
                    "prob_high": hi,
                    "n": 0,
                    "mean_predicted": np.nan,
                    "observed_fraction": np.nan,
                }
            )
            continue
        rows.append(
            {
                "bin": idx,
                "prob_low": lo,
                "prob_high": hi,
                "n": int(mask.sum()),
                "mean_predicted": float(prob[mask].mean()),
                "observed_fraction": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def classification_calibration_summary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return ECE and Brier summaries for binary or multiclass classification."""

    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    rows = [
        {
            "metric": "ece_top_label",
            "value": expected_calibration_error(y_true, probs, n_bins=n_bins),
        }
    ]
    if probs.shape[1] == 2:
        rows.append(
            {
                "metric": "brier_positive_class",
                "value": float(brier_score_loss(y_true, probs[:, 1])),
            }
        )
        try:
            rows.append({"metric": "auc", "value": float(roc_auc_score(y_true, probs[:, 1]))})
        except ValueError:
            rows.append({"metric": "auc", "value": np.nan})
    else:
        one_hot = np.eye(probs.shape[1])[y_true]
        rows.append(
            {"metric": "brier_multiclass_mean", "value": float(np.mean((one_hot - probs) ** 2))}
        )
    return pd.DataFrame(rows)


def stage_diagnostic_tables(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return confusion matrix, per-stage sensitivity/specificity, and summary."""

    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    y_pred = probs.argmax(axis=1)
    labels = np.arange(probs.shape[1])
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm, index=[f"true_{i}" for i in labels], columns=[f"pred_{i}" for i in labels]
    )
    rows = []
    total = cm.sum()
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        ppv = tp / (tp + fp) if (tp + fp) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        rows.append(
            {
                "stage_class": int(label),
                "support": int(tp + fn),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "ppv": float(ppv),
                "npv": float(npv),
            }
        )
    per_stage = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "accuracy": float(np.mean(y_pred == y_true)),
                "stage_mae": float(np.mean(np.abs(y_pred - y_true))),
                "quadratic_weighted_kappa": float(
                    cohen_kappa_score(y_true, y_pred, weights="quadratic")
                ),
                "balanced_accuracy_from_sensitivity": float(
                    per_stage["sensitivity"].mean(skipna=True)
                ),
            }
        ]
    )
    return cm_df, per_stage, summary


def probability_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    """Find exported probability columns such as ``stage_prob_0``."""

    columns = [col for col in df.columns if col.startswith(f"{prefix}_prob_")]
    return sorted(columns, key=lambda name: int(name.rsplit("_", 1)[-1]))
