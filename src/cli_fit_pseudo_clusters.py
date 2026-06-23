"""Fit pseudo-histology clusters using training slides only and assign all slides."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sparseage_data import read_manifest
from sparseage_pseudo_clusters import (
    assign_pseudo_clusters,
    fit_pseudo_cluster_model,
    sample_patch_features,
    select_training_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit train-only pseudo-cluster prototypes.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--feature-column", default="feature_path")
    parser.add_argument("--id-column", default="slide_id")
    parser.add_argument("--cluster-column", default="cluster_path")
    parser.add_argument("--split-column", default=None)
    parser.add_argument("--train-values", nargs="*", default=["train"])
    parser.add_argument("--fold-column", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--n-clusters", type=int, default=16)
    parser.add_argument("--max-patches-per-slide", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--output-manifest", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = read_manifest(args.manifest)
    train_mask = select_training_rows(
        df,
        split_column=args.split_column,
        train_values=args.train_values,
        fold_column=args.fold_column,
        fold=args.fold,
    )
    train_df = df.loc[train_mask].reset_index(drop=True)
    features = sample_patch_features(
        train_df,
        root=args.root,
        feature_column=args.feature_column,
        max_patches_per_slide=args.max_patches_per_slide,
        seed=args.seed,
    )
    model = fit_pseudo_cluster_model(
        features,
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    np.savez(out / "pseudo_cluster_prototypes.npz", centers=model.cluster_centers_)
    assigned = assign_pseudo_clusters(
        df,
        model=model,
        root=args.root,
        feature_column=args.feature_column,
        output_dir=out,
        id_column=args.id_column,
        cluster_column=args.cluster_column,
    )
    output_manifest = (
        Path(args.output_manifest)
        if args.output_manifest
        else out / "manifest_with_pseudo_clusters.csv"
    )
    assigned.to_csv(output_manifest, index=False)
    pd.DataFrame(
        [
            {
                "n_training_slides": int(train_mask.sum()),
                "n_total_slides": len(df),
                "n_clusters": args.n_clusters,
            }
        ]
    ).to_csv(out / "pseudo_cluster_fit_summary.csv", index=False)
    print(f"Wrote cluster-assigned manifest to {output_manifest}")


if __name__ == "__main__":
    main()
