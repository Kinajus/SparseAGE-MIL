"""Analysis survival statistics without heavyweight dependencies.

This module implements compact Cox proportional hazards utilities using a
Breslow partial likelihood. It is not intended to replace a full clinical
statistics package, but it produces the quantities hazard
ratios with confidence intervals, multivariable adjustment, model comparison,
C-index, log-rank tests, and simple horizon-specific AUCs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm
from sklearn.metrics import roc_auc_score

from sparseage_metrics import concordance_index


@dataclass
class CoxPHResult:
    """Container for a fitted Cox proportional hazards model."""

    coefficients: np.ndarray
    covariance: np.ndarray
    feature_names: list[str]
    partial_log_likelihood: float
    n: int
    n_events: int
    linear_predictor: np.ndarray
    c_index: float

    def summary(self) -> pd.DataFrame:
        se = np.sqrt(np.clip(np.diag(self.covariance), 0.0, np.inf))
        z = np.divide(
            self.coefficients, se, out=np.full_like(self.coefficients, np.nan), where=se > 0
        )
        p = 2.0 * norm.sf(np.abs(z))
        hr = np.exp(np.clip(self.coefficients, -50.0, 50.0))
        ci_low = np.exp(np.clip(self.coefficients - 1.96 * se, -50.0, 50.0))
        ci_high = np.exp(np.clip(self.coefficients + 1.96 * se, -50.0, 50.0))
        return pd.DataFrame(
            {
                "term": self.feature_names,
                "coef": self.coefficients,
                "se": se,
                "hazard_ratio": hr,
                "hr_ci_low": ci_low,
                "hr_ci_high": ci_high,
                "z": z,
                "p_value": p,
                "n": self.n,
                "n_events": self.n_events,
                "c_index": self.c_index,
                "partial_log_likelihood": self.partial_log_likelihood,
            }
        )


def _standardize_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()
    values = values.fillna(median if median == median else 0.0)
    std = values.std(ddof=0)
    if not std or std != std:
        return values * 0.0
    return (values - values.mean()) / std


def build_design_matrix(
    df: pd.DataFrame, covariates: Iterable[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Create a numeric design matrix from continuous/categorical covariates."""

    pieces: list[pd.DataFrame] = []
    for covariate in covariates:
        if covariate not in df.columns:
            raise ValueError(f"Missing covariate column: {covariate}")
        series = df[covariate]
        numeric = pd.to_numeric(series, errors="coerce")
        # Treat as numeric if at least 90% of non-missing values parse as numbers.
        non_missing = series.notna().sum()
        numeric_fraction = numeric.notna().sum() / max(1, non_missing)
        if numeric_fraction >= 0.90:
            filled = pd.to_numeric(series, errors="coerce")
            median = filled.median()
            filled = filled.fillna(median if median == median else 0.0).astype(float)
            unique_values = set(np.unique(filled.to_numpy()))
            if unique_values.issubset({0.0, 1.0}):
                pieces.append(pd.DataFrame({covariate: filled}, index=df.index))
            else:
                pieces.append(
                    pd.DataFrame({f"{covariate}_z": _standardize_numeric(series)}, index=df.index)
                )
        else:
            filled = series.astype("object").where(series.notna(), "Missing").astype(str)
            dummies = pd.get_dummies(filled, prefix=covariate, drop_first=True, dtype=float)
            if not dummies.empty:
                pieces.append(dummies)
    if not pieces:
        raise ValueError("No usable covariates were supplied")
    design = pd.concat(pieces, axis=1).astype(float)
    constant_cols = [col for col in design.columns if design[col].nunique(dropna=False) <= 1]
    if constant_cols:
        design = design.drop(columns=constant_cols)
    if design.empty:
        raise ValueError("Design matrix is empty after dropping constant columns")
    return design, list(design.columns)


def _cox_objective(
    beta: np.ndarray, x: np.ndarray, times: np.ndarray, events: np.ndarray
) -> tuple[float, np.ndarray]:
    eta = np.clip(x @ beta, -50.0, 50.0)
    exp_eta = np.exp(eta)
    unique_event_times = np.unique(times[events == 1])
    loglik = 0.0
    grad = np.zeros_like(beta)
    for t in unique_event_times:
        event_mask = (times == t) & (events == 1)
        risk_mask = times >= t
        d = int(event_mask.sum())
        risk_sum = float(exp_eta[risk_mask].sum())
        if risk_sum <= 0:
            continue
        weighted_x = (exp_eta[risk_mask, None] * x[risk_mask]).sum(axis=0)
        loglik += float((x[event_mask] @ beta).sum() - d * np.log(risk_sum))
        grad += x[event_mask].sum(axis=0) - d * weighted_x / risk_sum
    return -loglik, -grad


def _cox_hessian(
    beta: np.ndarray, x: np.ndarray, times: np.ndarray, events: np.ndarray
) -> np.ndarray:
    eta = np.clip(x @ beta, -50.0, 50.0)
    exp_eta = np.exp(eta)
    p = x.shape[1]
    hess = np.zeros((p, p), dtype=float)
    for t in np.unique(times[events == 1]):
        event_mask = (times == t) & (events == 1)
        risk_mask = times >= t
        d = int(event_mask.sum())
        risk_sum = float(exp_eta[risk_mask].sum())
        if risk_sum <= 0:
            continue
        xr = x[risk_mask]
        weights = exp_eta[risk_mask]
        mean = (weights[:, None] * xr).sum(axis=0) / risk_sum
        second = (xr.T @ (weights[:, None] * xr)) / risk_sum
        hess += d * (second - np.outer(mean, mean))
    return hess


def fit_cox_ph(
    df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    covariates: list[str],
    ridge: float = 1e-6,
) -> CoxPHResult:
    """Fit a Cox model and return HR/CI-ready result object."""

    required = [duration_col, event_col, *covariates]
    data = df[required].copy()
    data[duration_col] = pd.to_numeric(data[duration_col], errors="coerce")
    data[event_col] = pd.to_numeric(data[event_col], errors="coerce")
    data = data.dropna(subset=[duration_col, event_col])
    data = data[data[duration_col] > 0].copy()
    if data.empty or data[event_col].sum() == 0:
        raise ValueError("Cox model requires positive durations and at least one observed event")

    design, names = build_design_matrix(data, covariates)
    x = design.to_numpy(dtype=float)
    times = data[duration_col].to_numpy(dtype=float)
    events = data[event_col].to_numpy(dtype=int)
    beta0 = np.zeros(x.shape[1], dtype=float)

    def fun(beta: np.ndarray) -> float:
        nll, _ = _cox_objective(beta, x, times, events)
        return float(nll + 0.5 * ridge * np.dot(beta, beta))

    def jac(beta: np.ndarray) -> np.ndarray:
        _, grad = _cox_objective(beta, x, times, events)
        return grad + ridge * beta

    result = minimize(fun, beta0, jac=jac, method="BFGS", options={"maxiter": 1000, "gtol": 1e-6})
    beta = np.asarray(result.x, dtype=float)
    nll, _ = _cox_objective(beta, x, times, events)
    hess = _cox_hessian(beta, x, times, events) + ridge * np.eye(x.shape[1])
    try:
        covariance = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hess)
    linear_predictor = x @ beta
    return CoxPHResult(
        coefficients=beta,
        covariance=covariance,
        feature_names=names,
        partial_log_likelihood=-float(nll),
        n=len(data),
        n_events=int(events.sum()),
        linear_predictor=linear_predictor,
        c_index=concordance_index(times, linear_predictor, events),
    )


def likelihood_ratio_test(full: CoxPHResult, reduced: CoxPHResult) -> dict[str, float]:
    """Compare nested Cox models with a likelihood-ratio test."""

    df = max(1, len(full.coefficients) - len(reduced.coefficients))
    statistic = max(0.0, 2.0 * (full.partial_log_likelihood - reduced.partial_log_likelihood))
    return {
        "lr_statistic": float(statistic),
        "df": float(df),
        "p_value": float(chi2.sf(statistic, df)),
    }


def logrank_test(
    durations: np.ndarray,
    events: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """Two-group log-rank test for high/low risk groups."""

    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    groups = np.asarray(groups).astype(int)
    if set(np.unique(groups)) - {0, 1}:
        raise ValueError("groups must be binary encoded as 0/1")
    observed_minus_expected = 0.0
    variance = 0.0
    for t in np.unique(durations[events == 1]):
        at_risk = durations >= t
        d_total = int(((durations == t) & (events == 1)).sum())
        n_total = int(at_risk.sum())
        n_high = int((at_risk & (groups == 1)).sum())
        d_high = int(((durations == t) & (events == 1) & (groups == 1)).sum())
        if n_total <= 1:
            continue
        expected = d_total * n_high / n_total
        var = (
            (n_high / n_total)
            * (1.0 - n_high / n_total)
            * d_total
            * (n_total - d_total)
            / (n_total - 1)
        )
        observed_minus_expected += d_high - expected
        variance += var
    statistic = observed_minus_expected**2 / variance if variance > 0 else float("nan")
    p_value = float(chi2.sf(statistic, 1)) if statistic == statistic else float("nan")
    return {"logrank_chi2": float(statistic), "p_value": p_value}


def median_risk_group_stats(
    df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    risk_col: str,
) -> pd.DataFrame:
    """Return median-cut high/low risk HR and log-rank summary."""

    data = df[[duration_col, event_col, risk_col]].copy()
    data = data.dropna()
    cutoff = float(data[risk_col].median())
    data["high_risk"] = (data[risk_col] >= cutoff).astype(int)
    cox = fit_cox_ph(data, duration_col=duration_col, event_col=event_col, covariates=["high_risk"])
    lr = logrank_test(
        data[duration_col].to_numpy(), data[event_col].to_numpy(), data["high_risk"].to_numpy()
    )
    row = cox.summary().iloc[0].to_dict()
    row.update(lr)
    row["risk_cutoff"] = cutoff
    row["n_high"] = int(data["high_risk"].sum())
    row["n_low"] = int((1 - data["high_risk"]).sum())
    return pd.DataFrame([row])


def horizon_auc(
    df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    risk_col: str,
    horizons: Iterable[float],
) -> pd.DataFrame:
    """Compute simple cumulative/dynamic AUC at fixed horizons.

    Cases are patients with an observed event by the horizon. Controls are
    patients known to survive beyond the horizon. Patients censored before or at
    the horizon are excluded for that horizon.
    """

    rows = []
    data = df[[duration_col, event_col, risk_col]].copy().dropna()
    times = pd.to_numeric(data[duration_col], errors="coerce").to_numpy(dtype=float)
    events = pd.to_numeric(data[event_col], errors="coerce").to_numpy(dtype=int)
    risks = pd.to_numeric(data[risk_col], errors="coerce").to_numpy(dtype=float)
    for horizon in horizons:
        h = float(horizon)
        case = (times <= h) & (events == 1)
        control = times > h
        valid = case | control
        y = case[valid].astype(int)
        if len(np.unique(y)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y, risks[valid]))
        rows.append(
            {
                "horizon": h,
                "auc": auc,
                "n_used": int(valid.sum()),
                "n_cases": int(case.sum()),
                "n_controls": int(control.sum()),
            }
        )
    return pd.DataFrame(rows)


def compare_clinical_models(
    df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    risk_col: str,
    clinical_covariates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit WSI-only, clinical-only, and clinical+WSI Cox models."""

    wsi = fit_cox_ph(df, duration_col=duration_col, event_col=event_col, covariates=[risk_col])
    clinical = fit_cox_ph(
        df, duration_col=duration_col, event_col=event_col, covariates=clinical_covariates
    )
    full = fit_cox_ph(
        df,
        duration_col=duration_col,
        event_col=event_col,
        covariates=[risk_col, *clinical_covariates],
    )
    summaries = []
    for name, result in [
        ("wsi_only", wsi),
        ("clinical_only", clinical),
        ("clinical_plus_wsi", full),
    ]:
        frame = result.summary()
        frame.insert(0, "model", name)
        summaries.append(frame)
    comparison = pd.DataFrame(
        [
            {
                "model": "wsi_only",
                "n": wsi.n,
                "events": wsi.n_events,
                "c_index": wsi.c_index,
                "partial_log_likelihood": wsi.partial_log_likelihood,
            },
            {
                "model": "clinical_only",
                "n": clinical.n,
                "events": clinical.n_events,
                "c_index": clinical.c_index,
                "partial_log_likelihood": clinical.partial_log_likelihood,
            },
            {
                "model": "clinical_plus_wsi",
                "n": full.n,
                "events": full.n_events,
                "c_index": full.c_index,
                "partial_log_likelihood": full.partial_log_likelihood,
            },
        ]
    )
    lrt = likelihood_ratio_test(full, clinical)
    lrt_df = pd.DataFrame(
        [
            {
                **lrt,
                "comparison": "clinical_plus_wsi_vs_clinical_only",
                "delta_c_index": full.c_index - clinical.c_index,
            }
        ]
    )
    return pd.concat(summaries, ignore_index=True), comparison, lrt_df
