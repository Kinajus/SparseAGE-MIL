"""Attention-ST and risk-group phenotype statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sparseage_spatial_statistics import (
    attention_phenotype_correlations,
    risk_group_phenotype_tests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantitative ST/cell-type statistics for analysis."
    )
    parser.add_argument(
        "--table", required=True, help="CSV with attention and phenotype/cell-type columns."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attention-col", default="attention")
    parser.add_argument("--phenotype-cols", nargs="*", default=None)
    parser.add_argument("--group-col", default="slide_id")
    parser.add_argument("--risk-group-col", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.table)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    attention_phenotype_correlations(
        table,
        attention_col=args.attention_col,
        phenotype_cols=args.phenotype_cols,
        group_col=args.group_col,
    ).to_csv(out / "attention_phenotype_spearman.csv", index=False)
    if args.risk_group_col:
        risk_group_phenotype_tests(
            table,
            risk_group_col=args.risk_group_col,
            phenotype_cols=args.phenotype_cols,
        ).to_csv(out / "risk_group_phenotype_tests.csv", index=False)
    print(f"Wrote spatial statistics to {out}")


if __name__ == "__main__":
    main()
