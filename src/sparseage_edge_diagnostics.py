"""Diagnostics for exported sparse topology node/edge CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def summarize_sparse_edges(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Summarize physical distances and cluster/interface composition of sparse edges."""

    required_node_cols = {"patch_index", "x", "y"}
    required_edge_cols = {"source", "target"}
    if not required_node_cols.issubset(nodes.columns):
        raise ValueError(f"nodes table requires columns {sorted(required_node_cols)}")
    if not required_edge_cols.issubset(edges.columns):
        raise ValueError(f"edges table requires columns {sorted(required_edge_cols)}")
    node_lookup = nodes.set_index("patch_index")
    rows = []
    for _, edge in edges.iterrows():
        source = int(edge["source"])
        target = int(edge["target"])
        if source not in node_lookup.index or target not in node_lookup.index:
            continue
        src = node_lookup.loc[source]
        tgt = node_lookup.loc[target]
        distance = float(
            np.hypot(float(src["x"]) - float(tgt["x"]), float(src["y"]) - float(tgt["y"]))
        )
        row = {"distance": distance, "self_edge": int(source == target)}
        if "cluster" in nodes.columns:
            src_cluster = src.get("cluster")
            tgt_cluster = tgt.get("cluster")
            if pd.notna(src_cluster) and pd.notna(tgt_cluster):
                row["same_cluster"] = int(src_cluster == tgt_cluster)
                row["cross_cluster"] = int(src_cluster != tgt_cluster)
        if "selected_top_attention" in nodes.columns:
            row["source_selected"] = int(src.get("selected_top_attention", 0))
            row["target_selected"] = int(tgt.get("selected_top_attention", 0))
        if "weight" in edge.index:
            row["weight"] = float(edge["weight"])
        rows.append(row)
    edge_df = pd.DataFrame(rows)
    if edge_df.empty:
        return pd.DataFrame([{"n_edges": 0}])
    summary = {
        "n_edges": int(len(edge_df)),
        "median_distance": float(edge_df["distance"].median()),
        "mean_distance": float(edge_df["distance"].mean()),
        "self_edge_fraction": float(edge_df["self_edge"].mean()),
    }
    for col in ["same_cluster", "cross_cluster", "source_selected", "target_selected"]:
        if col in edge_df.columns:
            summary[f"{col}_fraction"] = float(edge_df[col].mean())
    if "weight" in edge_df.columns:
        summary["mean_weight"] = float(edge_df["weight"].mean())
        summary["median_weight"] = float(edge_df["weight"].median())
    return pd.DataFrame([summary])


def summarize_node_attention(nodes: pd.DataFrame) -> pd.DataFrame:
    """Summarize attention scores and optional region/cluster enrichment."""

    if "attention" not in nodes.columns:
        raise ValueError("nodes table requires an attention column")
    rows = [
        {
            "group": "all",
            "n": int(len(nodes)),
            "mean_attention": float(nodes["attention"].mean()),
            "median_attention": float(nodes["attention"].median()),
            "selected_fraction": float(
                nodes.get("selected_top_attention", pd.Series(0, index=nodes.index)).mean()
            ),
        }
    ]
    for group_col in ["cluster", "region"]:
        if group_col in nodes.columns:
            for value, group in nodes.groupby(group_col, dropna=True):
                rows.append(
                    {
                        "group": f"{group_col}:{value}",
                        "n": int(len(group)),
                        "mean_attention": float(group["attention"].mean()),
                        "median_attention": float(group["attention"].median()),
                        "selected_fraction": float(
                            group.get(
                                "selected_top_attention", pd.Series(0, index=group.index)
                            ).mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def load_and_summarize_pair(
    nodes_path: str | Path, edges_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    edge_summary = summarize_sparse_edges(nodes, edges)
    node_summary = summarize_node_attention(nodes)
    slide_id = (
        nodes["slide_id"].iloc[0]
        if "slide_id" in nodes.columns and len(nodes)
        else Path(nodes_path).stem
    )
    edge_summary.insert(0, "slide_id", slide_id)
    node_summary.insert(0, "slide_id", slide_id)
    return edge_summary, node_summary
