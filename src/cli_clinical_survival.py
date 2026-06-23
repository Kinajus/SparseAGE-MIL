"""Run survival analyses from exported prediction CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sparseage_clinical_survival import (
    compare_clinical_models,
    fit_cox_ph,
    horizon_auc,
    median_risk_group_stats,
)
from sparseage_statistics import bootstrap_cindex_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cox/HR/C-index analysis for SparseAGE prediction CSVs."
    )
    parser.add_argument(
        "--predictions", required=True, help="Prediction CSV containing risk score and outcomes."
    )
    parser.add_argument("--output-dir", required=True, help="Directory for CSV summaries.")
    parser.add_argument("--duration-col", default="event_time")
    parser.add_argument("--event-col", default="event_observed")
    parser.add_argument("--risk-col", default="risk_score")
    parser.add_argument(
        "--clinical-covariates", nargs="*", default=[], help="Age sex TNM stage treatment etc."
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        type=float,
        default=[],
        help="Fixed time horizons for AUC, e.g. 365 1095 1825.",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cox = fit_cox_ph(
        df,
        duration_col=args.duration_col,
        event_col=args.event_col,
        covariates=[args.risk_col],
    )
    cox.summary().to_csv(out / "cox_wsi_univariable.csv", index=False)
    cindex_df = df[[args.duration_col, args.event_col, args.risk_col]].copy().dropna()
    cindex, lo, hi = bootstrap_cindex_ci(
        cindex_df[args.duration_col].to_numpy(),
        cindex_df[args.risk_col].to_numpy(),
        cindex_df[args.event_col].to_numpy(),
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    pd.DataFrame(
        [{"c_index": cindex, "c_index_ci_low": lo, "c_index_ci_high": hi, "n_boot": args.bootstrap}]
    ).to_csv(out / "cindex_bootstrap_ci.csv", index=False)
    median_risk_group_stats(
        df,
        duration_col=args.duration_col,
        event_col=args.event_col,
        risk_col=args.risk_col,
    ).to_csv(out / "median_risk_group_logrank_hr.csv", index=False)

    if args.clinical_covariates:
        summary, comparison, lrt = compare_clinical_models(
            df,
            duration_col=args.duration_col,
            event_col=args.event_col,
            risk_col=args.risk_col,
            clinical_covariates=args.clinical_covariates,
        )
        summary.to_csv(out / "cox_multivariable_summary.csv", index=False)
        comparison.to_csv(out / "cox_model_comparison.csv", index=False)
        lrt.to_csv(out / "cox_likelihood_ratio_test.csv", index=False)

    if args.horizons:
        horizon_auc(
            df,
            duration_col=args.duration_col,
            event_col=args.event_col,
            risk_col=args.risk_col,
            horizons=args.horizons,
        ).to_csv(out / "time_horizon_auc.csv", index=False)
    print(f"Wrote survival analysis outputs to {out}")


if __name__ == "__main__":
    main()
