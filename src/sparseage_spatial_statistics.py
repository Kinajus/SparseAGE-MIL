"""Quantitative attention-vs-spatial phenotype statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment."""

    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return adjusted
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    adjusted[valid] = out
    return adjusted


def attention_region_enrichment(
    nodes: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    region_col: str = "region",
    id_cols: tuple[str, str] = ("slide_id", "patch_index"),
    n_permutations: int = 1000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test whether attention is enriched/depleted in annotated patch regions.

    ``nodes`` should be the exported node table. ``annotations`` can contain
    patch-level assignments from QuPath/napari post-processing with columns
    ``slide_id``, ``patch_index`` and a region label such as tumor_core, stroma,
    or tumor_stroma_interface.
    """

    if "attention" not in nodes.columns:
        raise ValueError("nodes table requires an attention column")
    for col in (*id_cols, region_col):
        if col not in annotations.columns and col != id_cols[0]:
            raise ValueError(f"annotations table is missing required column: {col}")
    merge_cols = [col for col in id_cols if col in nodes.columns and col in annotations.columns]
    if not merge_cols:
        raise ValueError(
            "nodes and annotations must share slide_id/patch_index or patch_index columns"
        )
    merged = nodes.merge(annotations[[*merge_cols, region_col]], on=merge_cols, how="inner")
    rng = np.random.default_rng(seed)
    rows = []
    for region, _group in merged.groupby(region_col):
        in_region = merged[region_col] == region
        a = merged.loc[in_region, "attention"].to_numpy(float)
        b = merged.loc[~in_region, "attention"].to_numpy(float)
        if len(a) == 0 or len(b) == 0:
            p_mwu = np.nan
        else:
            p_mwu = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        observed = float(a.mean() - b.mean()) if len(a) and len(b) else np.nan
        perm_diffs = []
        labels = in_region.to_numpy()
        attention = merged["attention"].to_numpy(float)
        for _ in range(n_permutations):
            perm = rng.permutation(labels)
            if perm.sum() == 0 or (~perm).sum() == 0:
                continue
            perm_diffs.append(float(attention[perm].mean() - attention[~perm].mean()))
        if perm_diffs and observed == observed:
            p_perm = float(np.mean(np.abs(perm_diffs) >= abs(observed)))
        else:
            p_perm = np.nan
        rows.append(
            {
                "region": region,
                "n_region": int(in_region.sum()),
                "n_other": int((~in_region).sum()),
                "mean_attention_region": float(a.mean()) if len(a) else np.nan,
                "mean_attention_other": float(b.mean()) if len(b) else np.nan,
                "delta_mean_attention": observed,
                "mannwhitney_p": p_mwu,
                "permutation_p": p_perm,
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["mannwhitney_fdr"] = benjamini_hochberg(summary["mannwhitney_p"].to_numpy(float))
        summary["permutation_fdr"] = benjamini_hochberg(summary["permutation_p"].to_numpy(float))
    return merged, summary


def attention_phenotype_correlations(
    table: pd.DataFrame,
    *,
    attention_col: str = "attention",
    phenotype_cols: list[str] | None = None,
    group_col: str | None = "slide_id",
) -> pd.DataFrame:
    """Spearman correlations between attention and ST/cell-type variables."""

    if attention_col not in table.columns:
        raise ValueError(f"Missing attention column: {attention_col}")
    if phenotype_cols is None:
        excluded = {attention_col, "slide_id", "spot_id", "patch_index", "x", "y"}
        phenotype_cols = [
            col
            for col in table.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(table[col])
        ]
    rows = []
    groups = (
        [("all", table)]
        if group_col is None or group_col not in table.columns
        else table.groupby(group_col)
    )
    for group_name, group in groups:
        for phenotype in phenotype_cols:
            valid = group[[attention_col, phenotype]].dropna()
            if (
                len(valid) < 3
                or valid[attention_col].nunique() < 2
                or valid[phenotype].nunique() < 2
            ):
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(valid[attention_col], valid[phenotype])
            rows.append(
                {
                    "group": group_name,
                    "phenotype": phenotype,
                    "n": int(len(valid)),
                    "spearman_r": float(rho),
                    "p_value": float(p),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr"] = benjamini_hochberg(result["p_value"].to_numpy(float))
    return result


def risk_group_phenotype_tests(
    table: pd.DataFrame,
    *,
    risk_group_col: str = "risk_group",
    phenotype_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compare phenotype/cell-type scores between high and low risk groups."""

    if risk_group_col not in table.columns:
        raise ValueError(f"Missing risk group column: {risk_group_col}")
    if phenotype_cols is None:
        excluded = {risk_group_col, "slide_id", "patient_id"}
        phenotype_cols = [
            col
            for col in table.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(table[col])
        ]
    rows = []
    groups = table[risk_group_col].astype(str).str.lower()
    high = groups.isin({"high", "1", "true"})
    low = groups.isin({"low", "0", "false"})
    for phenotype in phenotype_cols:
        a = pd.to_numeric(table.loc[high, phenotype], errors="coerce").dropna().to_numpy(float)
        b = pd.to_numeric(table.loc[low, phenotype], errors="coerce").dropna().to_numpy(float)
        if len(a) == 0 or len(b) == 0:
            p = np.nan
        else:
            p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        rows.append(
            {
                "phenotype": phenotype,
                "n_high": int(len(a)),
                "n_low": int(len(b)),
                "mean_high": float(a.mean()) if len(a) else np.nan,
                "mean_low": float(b.mean()) if len(b) else np.nan,
                "delta_mean_high_minus_low": float(a.mean() - b.mean())
                if len(a) and len(b)
                else np.nan,
                "mannwhitney_p": p,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr"] = benjamini_hochberg(result["mannwhitney_p"].to_numpy(float))
    return result
