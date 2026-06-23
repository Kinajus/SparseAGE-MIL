"""Summarize exported sparse topology CSVs."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from sparseage_edge_diagnostics import load_and_summarize_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse edge/node diagnostics.")
    parser.add_argument(
        "--topology-dir",
        required=True,
        help="Directory containing *_nodes.csv and *_edges.csv files.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top = Path(args.topology_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    edge_summaries = []
    node_summaries = []
    for nodes_file in sorted(glob.glob(str(top / "*_nodes.csv"))):
        edges_file = nodes_file.replace("_nodes.csv", "_edges.csv")
        if not Path(edges_file).exists():
            continue
        edge_summary, node_summary = load_and_summarize_pair(nodes_file, edges_file)
        edge_summaries.append(edge_summary)
        node_summaries.append(node_summary)
    if edge_summaries:
        pd.concat(edge_summaries, ignore_index=True).to_csv(
            out / "sparse_edge_summary.csv", index=False
        )
    if node_summaries:
        pd.concat(node_summaries, ignore_index=True).to_csv(
            out / "attention_node_summary.csv", index=False
        )
    print(f"Wrote edge diagnostics to {out}")


if __name__ == "__main__":
    main()
