"""Paired statistical comparison for exported prediction CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sparseage_statistics import (
    bootstrap_auc_ci,
    bootstrap_cindex_ci,
    delong_auc_test,
    paired_bootstrap_auc_test,
    paired_permutation_cindex_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two prediction CSVs with paired tests.")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=["classification", "survival"], required=True)
    parser.add_argument("--id-col", default="slide_id")
    parser.add_argument("--label-col", default="label")
    parser.add_argument(
        "--score-col-a", default=None, help="Positive-class score/risk column for model A."
    )
    parser.add_argument(
        "--score-col-b", default=None, help="Positive-class score/risk column for model B."
    )
    parser.add_argument("--event-time-col", default="event_time")
    parser.add_argument("--event-col", default="event_observed")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-permutation", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    a = pd.read_csv(args.model_a)
    b = pd.read_csv(args.model_b)
    merged = a.merge(b, on=args.id_col, suffixes=("_a", "_b"))
    rows = []
    if args.task == "classification":
        score_a_col = args.score_col_a or "class_prob_1"
        score_b_col = args.score_col_b or "class_prob_1"
        label_col = args.label_col if args.label_col in merged.columns else f"{args.label_col}_a"
        score_a_series = pd.to_numeric(
            merged[score_a_col if score_a_col in merged.columns else f"{score_a_col}_a"],
            errors="coerce",
        )
        score_b_series = pd.to_numeric(
            merged[score_b_col if score_b_col in merged.columns else f"{score_b_col}_b"],
            errors="coerce",
        )
        y_series = pd.to_numeric(merged[label_col], errors="coerce")
        valid = score_a_series.notna() & score_b_series.notna() & y_series.notna() & (y_series >= 0)
        score_a = score_a_series.loc[valid].to_numpy(float)
        score_b = score_b_series.loc[valid].to_numpy(float)
        y = y_series.loc[valid].to_numpy(int)
        rows.append({"test": "delong", **delong_auc_test(y, score_a, score_b)})
        rows.append(
            {
                "test": "paired_bootstrap_auc",
                **paired_bootstrap_auc_test(
                    y, score_a, score_b, n_boot=args.n_bootstrap, seed=args.seed
                ),
            }
        )
        auc_a, lo_a, hi_a = bootstrap_auc_ci(y, score_a, n_boot=args.n_bootstrap, seed=args.seed)
        auc_b, lo_b, hi_b = bootstrap_auc_ci(
            y, score_b, n_boot=args.n_bootstrap, seed=args.seed + 1
        )
        rows.extend(
            [
                {"test": "auc_ci_model_a", "auc": auc_a, "ci_low": lo_a, "ci_high": hi_a},
                {"test": "auc_ci_model_b", "auc": auc_b, "ci_low": lo_b, "ci_high": hi_b},
            ]
        )
    else:
        score_a_col = args.score_col_a or "risk_score"
        score_b_col = args.score_col_b or "risk_score"
        risk_a_series = pd.to_numeric(
            merged[score_a_col if score_a_col in merged.columns else f"{score_a_col}_a"],
            errors="coerce",
        )
        risk_b_series = pd.to_numeric(
            merged[score_b_col if score_b_col in merged.columns else f"{score_b_col}_b"],
            errors="coerce",
        )
        time_col = (
            args.event_time_col
            if args.event_time_col in merged.columns
            else f"{args.event_time_col}_a"
        )
        event_col = args.event_col if args.event_col in merged.columns else f"{args.event_col}_a"
        time_series = pd.to_numeric(merged[time_col], errors="coerce")
        event_series = pd.to_numeric(merged[event_col], errors="coerce")
        valid = (
            risk_a_series.notna()
            & risk_b_series.notna()
            & time_series.notna()
            & event_series.notna()
        )
        risk_a = risk_a_series.loc[valid].to_numpy(float)
        risk_b = risk_b_series.loc[valid].to_numpy(float)
        times = time_series.loc[valid].to_numpy(float)
        events = event_series.loc[valid].to_numpy(int)
        rows.append(
            {
                "test": "paired_permutation_cindex",
                **paired_permutation_cindex_test(
                    times, risk_a, risk_b, events, n_perm=args.n_permutation, seed=args.seed
                ),
            }
        )
        c_a, lo_a, hi_a = bootstrap_cindex_ci(
            times, risk_a, events, n_boot=args.n_bootstrap, seed=args.seed
        )
        c_b, lo_b, hi_b = bootstrap_cindex_ci(
            times, risk_b, events, n_boot=args.n_bootstrap, seed=args.seed + 1
        )
        rows.extend(
            [
                {"test": "cindex_ci_model_a", "c_index": c_a, "ci_low": lo_a, "ci_high": hi_a},
                {"test": "cindex_ci_model_b", "c_index": c_b, "ci_low": lo_b, "ci_high": hi_b},
            ]
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote paired comparison to {output}")


if __name__ == "__main__":
    main()
