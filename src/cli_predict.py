"""Export per-slide predictions from a trained SparseAGE checkpoint."""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cli_train import _build_model_config
from sparseage_data import (
    FeatureBagDataset,
    apply_survival_bin_edges,
    build_label_map,
    build_stage_label_map,
    collate_bags,
    read_manifest,
)
from sparseage_model import SparseAGEMIL
from sparseage_prediction import export_slide_predictions
from sparseage_training import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-slide SparseAGE predictions."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt checkpoint.")
    parser.add_argument("--manifest", required=True, help="CSV/TSV manifest to score.")
    parser.add_argument("--output", required=True, help="Output prediction CSV.")
    parser.add_argument("--root", default=None, help="Override feature root directory.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--export-topology-dir", default=None)
    parser.add_argument("--topology-top-fraction", type=float, default=0.10)
    return parser.parse_args()


def _build_label_maps(
    task: str, tasks: tuple[str, ...], df: pd.DataFrame, data_cfg: dict[str, Any]
) -> tuple[dict[Any, int] | None, dict[str, dict[Any, int]]]:
    label_map = None
    label_maps: dict[str, dict[Any, int]] = {}
    if task == "classification" and data_cfg.get("label_column", "label") in df.columns:
        label_map = build_label_map(df[data_cfg.get("label_column", "label")])
    elif task == "multitask":
        subtype_col = data_cfg.get("subtype_label_column", "subtype")
        stage_col = data_cfg.get("stage_label_column", "stage")
        if "subtype" in tasks and subtype_col in df.columns:
            label_maps["subtype"] = build_label_map(df[subtype_col])
        if "stage" in tasks and stage_col in df.columns:
            label_maps["stage"] = build_stage_label_map(df[stage_col])
    return label_map, label_maps


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    task = config.get("task", checkpoint.get("model_config", {}).get("task", "classification"))
    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))
    model_cfg.update(checkpoint.get("model_config", {}))
    if args.root is not None:
        data_cfg["root"] = args.root

    seed_everything(int(config.get("train", {}).get("seed", 2021)))
    df = read_manifest(args.manifest)
    tasks = tuple(model_cfg.get("tasks", data_cfg.get("tasks", ["subtype", "stage", "survival"])))
    survival_col = data_cfg.get("survival_bin_column", "survival_bin")
    if task == "survival" or (task == "multitask" and "survival" in tasks):
        if survival_col not in df.columns or df[survival_col].isna().all():
            edges = config.get("derived", {}).get("survival_bin_edges")
            if edges is not None:
                df = apply_survival_bin_edges(
                    df,
                    edges,
                    event_time_column=data_cfg.get("event_time_column", "event_time"),
                    output_column=survival_col,
                )

    label_map, label_maps = _build_label_maps(task, tasks, df, data_cfg)
    derived = config.get("derived", {})
    if derived.get("label_map") is not None:
        label_map = derived["label_map"]
    if derived.get("label_maps") is not None:
        label_maps = derived["label_maps"]
    n_classes = int(
        model_cfg.get("n_classes", checkpoint.get("model_config", {}).get("n_classes", 2))
    )
    sparseage_cfg = _build_model_config(task, model_cfg, n_classes=n_classes, tasks=tasks)
    model = SparseAGEMIL(sparseage_cfg)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    dataset = FeatureBagDataset(
        df,
        root=data_cfg.get("root"),
        task=task,
        tasks=tasks,
        feature_column=data_cfg.get("feature_column", "feature_path"),
        coordinate_column=data_cfg.get("coordinate_column", data_cfg.get("coords_column")),
        cluster_column=data_cfg.get("cluster_column"),
        topology_column=data_cfg.get("topology_column"),
        topology_columns=data_cfg.get("topology_columns"),
        compute_topology_target=bool(data_cfg.get("compute_topology_target", True)),
        topology_target_dim=int(model_cfg.get("topology_target_dim", 6)),
        id_column=data_cfg.get("id_column", "slide_id"),
        label_column=data_cfg.get("label_column", "label"),
        subtype_label_column=data_cfg.get("subtype_label_column", "subtype"),
        stage_label_column=data_cfg.get("stage_label_column", "stage"),
        event_time_column=data_cfg.get("event_time_column", "event_time"),
        event_observed_column=data_cfg.get("event_observed_column", "event_observed"),
        censorship_column=data_cfg.get("censorship_column", "censorship"),
        survival_bin_column=survival_col,
        label_map=label_map,
        label_maps=label_maps,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_bags,
    )
    output = export_slide_predictions(
        model,
        loader,
        device=device,
        task=task,
        output_path=args.output,
        export_topology_dir=args.export_topology_dir,
        topology_top_fraction=args.topology_top_fraction,
    )
    print(f"Exported predictions to {output}")


if __name__ == "__main__":
    main()
