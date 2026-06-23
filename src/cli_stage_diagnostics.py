"""Create stage confusion-matrix and calibration summaries from prediction CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sparseage_calibration import (
    classification_calibration_summary,
    probability_columns,
    stage_diagnostic_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage prediction diagnostics for analysis."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-col", default="stage_label")
    parser.add_argument("--prefix", default="stage")
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prob_cols = probability_columns(df, args.prefix)
    if not prob_cols:
        raise ValueError(f"No probability columns found with prefix {args.prefix!r}")
    valid = df[args.label_col].notna() & (df[args.label_col] >= 0)
    y = df.loc[valid, args.label_col].astype(int).to_numpy()
    p = df.loc[valid, prob_cols].to_numpy(float)
    cm, per_stage, summary = stage_diagnostic_tables(y, p)
    cm.to_csv(out / "stage_confusion_matrix.csv")
    per_stage.to_csv(out / "stage_per_class_sensitivity_specificity.csv", index=False)
    summary.to_csv(out / "stage_summary.csv", index=False)
    classification_calibration_summary(y, p, n_bins=args.calibration_bins).to_csv(
        out / "stage_calibration_summary.csv", index=False
    )
    print(f"Wrote stage diagnostics to {out}")


if __name__ == "__main__":
    main()
