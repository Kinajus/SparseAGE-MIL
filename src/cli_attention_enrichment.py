"""Patch-level attention enrichment in pathologist-annotated regions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sparseage_spatial_statistics import attention_region_enrichment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test attention enrichment in annotated regions.")
    parser.add_argument(
        "--nodes", required=True, help="Exported *_nodes.csv or concatenated node table."
    )
    parser.add_argument("--annotations", required=True, help="Patch-level annotation CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--region-col", default="region")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes = pd.read_csv(args.nodes)
    ann = pd.read_csv(args.annotations)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged, summary = attention_region_enrichment(
        nodes,
        ann,
        region_col=args.region_col,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    merged.to_csv(out / "attention_annotation_joined.csv", index=False)
    summary.to_csv(out / "attention_region_enrichment.csv", index=False)
    print(f"Wrote attention enrichment results to {out}")


if __name__ == "__main__":
    main()
