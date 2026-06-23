"""Cross-validation and paired-comparison summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_last_epoch_metrics(paths: list[str | Path]) -> pd.DataFrame:
    """Read the last row from each metrics CSV and attach fold/run metadata."""

    rows = []
    for idx, path in enumerate(paths):
        path = Path(path)
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[-1].to_dict()
        row["source_path"] = str(path)
        row["run_index"] = idx
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_numeric_metrics(
    metrics: pd.DataFrame, metric_cols: list[str] | None = None
) -> pd.DataFrame:
    """Return mean/SD/SE/normal-approx CI for numeric metrics."""

    if metric_cols is None:
        metric_cols = [
            col
            for col in metrics.columns
            if pd.api.types.is_numeric_dtype(metrics[col]) and col not in {"epoch", "run_index"}
        ]
    rows = []
    for col in metric_cols:
        values = pd.to_numeric(metrics[col], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
        se = sd / np.sqrt(values.size) if values.size > 1 else float("nan")
        rows.append(
            {
                "metric": col,
                "n_runs": int(values.size),
                "mean": mean,
                "sd": sd,
                "se": se,
                "ci95_low": mean - 1.96 * se if se == se else float("nan"),
                "ci95_high": mean + 1.96 * se if se == se else float("nan"),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)
