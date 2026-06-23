"""Summarize per-fold metrics CSV files."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from sparseage_summary import read_last_epoch_metrics, summarize_numeric_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize cross-validation metrics as mean ± SD tables."
    )
    parser.add_argument("--metrics", nargs="*", default=[], help="Explicit metrics.csv files.")
    parser.add_argument(
        "--metrics-glob", default=None, help="Glob pattern, e.g. 'runs/*/*/metrics.csv'."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric-cols", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = list(args.metrics)
    if args.metrics_glob:
        paths.extend(glob.glob(args.metrics_glob))
    if not paths:
        raise ValueError("No metrics files were provided")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = read_last_epoch_metrics(paths)
    metrics.to_csv(out / "cv_last_epoch_metrics.csv", index=False)
    summarize_numeric_metrics(metrics, args.metric_cols).to_csv(
        out / "cv_metric_summary.csv", index=False
    )
    print(f"Wrote CV summary to {out}")


if __name__ == "__main__":
    main()
