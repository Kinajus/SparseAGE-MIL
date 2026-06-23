"""Command-line training entrypoint for SparseAGE-MIL."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader

from sparseage_data import (
    FeatureBagDataset,
    apply_survival_bin_edges,
    build_label_map,
    build_stage_label_map,
    collate_bags,
    fit_survival_bin_edges,
    read_manifest,
)
from sparseage_model import SparseAGEConfig, SparseAGEMIL
from sparseage_prediction import export_slide_predictions
from sparseage_training import (
    append_metrics_csv,
    build_optimizer,
    build_scheduler,
    evaluate_classification,
    evaluate_multitask,
    evaluate_survival,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SparseAGE-MIL from YAML.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--fold", type=int, default=None, help="Override the validation fold.")
    parser.add_argument("--dry-run", action="store_true", help="Build data/model and exit.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def split_dataframe(
    df: pd.DataFrame,
    *,
    task: str,
    split_cfg: dict[str, Any],
    label_column: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_column = split_cfg.get("split_column", "split")
    fold_column = split_cfg.get("fold_column", "fold")
    fold = int(split_cfg.get("fold", 0))

    if split_column in df.columns:
        train_values = set(split_cfg.get("train_values", ["train"]))
        val_values = set(split_cfg.get("val_values", ["val", "valid", "validation", "test"]))
        train_df = df[df[split_column].astype(str).str.lower().isin(train_values)]
        val_df = df[df[split_column].astype(str).str.lower().isin(val_values)]
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    if fold_column in df.columns:
        train_df = df[df[fold_column].astype(int) != fold]
        val_df = df[df[fold_column].astype(int) == fold]
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    n_splits = int(split_cfg.get("n_splits", 5))
    indices = list(range(len(df)))
    stratify_column = split_cfg.get("stratify_column", label_column)
    can_stratify = task in {"classification", "multitask"} and stratify_column in df.columns
    if can_stratify and df[stratify_column].nunique(dropna=True) > 1:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        y = df[stratify_column].astype(str).to_numpy()
        splits = list(splitter.split(indices, y))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(indices))
    train_idx, val_idx = splits[fold]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def _build_model_config(
    task: str,
    model_cfg: dict[str, Any],
    *,
    n_classes: int,
    tasks: tuple[str, ...],
) -> SparseAGEConfig:
    return SparseAGEConfig(
        input_dim=int(model_cfg.get("input_dim", 1024)),
        embedding_dim=int(model_cfg.get("embedding_dim", 512)),
        attention_dim=int(model_cfg.get("attention_dim", 128)),
        n_classes=n_classes,
        topk=model_cfg.get("topk", 6),
        dropout=float(model_cfg.get("dropout", 0.25)),
        task=task,
        use_scaled_similarity=bool(model_cfg.get("use_scaled_similarity", True)),
        use_spatial=bool(model_cfg.get("use_spatial", True)),
        spatial_sigma=float(model_cfg.get("spatial_sigma", 256.0)),
        spatial_weight=float(model_cfg.get("spatial_weight", 0.20)),
        cluster_weight=float(model_cfg.get("cluster_weight", 0.15)),
        boundary_weight=float(model_cfg.get("boundary_weight", 0.10)),
        include_self=bool(model_cfg.get("include_self", True)),
        tasks=tasks,
        subtype_classes=int(model_cfg.get("subtype_classes", 2)),
        stage_classes=int(model_cfg.get("stage_classes", 4)),
        survival_bins=int(model_cfg.get("survival_bins", model_cfg.get("n_classes", 4))),
        topology_target_dim=int(model_cfg.get("topology_target_dim", 6)),
        use_uncertainty_weighting=bool(model_cfg.get("use_uncertainty_weighting", True)),
    )


def _build_datasets(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    task: str,
    tasks: tuple[str, ...],
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    label_map: dict[Any, int] | None,
    label_maps: dict[str, dict[Any, int]],
) -> tuple[FeatureBagDataset, FeatureBagDataset]:
    dataset_kwargs = {
        "root": data_cfg.get("root"),
        "task": task,
        "tasks": tasks,
        "feature_column": data_cfg.get("feature_column", "feature_path"),
        "coordinate_column": data_cfg.get("coordinate_column", data_cfg.get("coords_column")),
        "cluster_column": data_cfg.get("cluster_column"),
        "topology_column": data_cfg.get("topology_column"),
        "topology_columns": data_cfg.get("topology_columns"),
        "compute_topology_target": bool(data_cfg.get("compute_topology_target", True)),
        "topology_target_dim": int(model_cfg.get("topology_target_dim", 6)),
        "id_column": data_cfg.get("id_column", "slide_id"),
        "label_column": data_cfg.get("label_column", "label"),
        "subtype_label_column": data_cfg.get("subtype_label_column", "subtype"),
        "stage_label_column": data_cfg.get("stage_label_column", "stage"),
        "event_time_column": data_cfg.get("event_time_column", "event_time"),
        "event_observed_column": data_cfg.get("event_observed_column", "event_observed"),
        "censorship_column": data_cfg.get("censorship_column", "censorship"),
        "survival_bin_column": data_cfg.get("survival_bin_column", "survival_bin"),
        "label_map": label_map,
        "label_maps": label_maps,
    }
    return FeatureBagDataset(train_df, **dataset_kwargs), FeatureBagDataset(
        val_df, **dataset_kwargs
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    task = config.get("task", "classification")
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    split_cfg = config.get("split", {})
    if args.fold is not None:
        split_cfg["fold"] = args.fold

    seed = int(train_cfg.get("seed", 2021))
    if "num_threads" in train_cfg:
        torch.set_num_threads(int(train_cfg["num_threads"]))
    seed_everything(seed)

    df = read_manifest(data_cfg["manifest"])
    tasks = tuple(model_cfg.get("tasks", data_cfg.get("tasks", ["subtype", "stage", "survival"])))
    label_column = data_cfg.get("label_column", "label")
    survival_bins = int(model_cfg.get("survival_bins", model_cfg.get("n_classes", 4)))

    if task == "multitask" and "stratify_column" not in split_cfg:
        subtype_col = data_cfg.get("subtype_label_column", "subtype")
        if subtype_col in df.columns:
            split_cfg["stratify_column"] = subtype_col

    train_df, val_df = split_dataframe(
        df,
        task=task,
        split_cfg=split_cfg,
        label_column=label_column,
        seed=seed,
    )
    if train_df.empty or val_df.empty:
        raise ValueError("Train/validation split is empty. Check split or fold settings.")

    survival_bin_edges = None
    survival_bin_column = data_cfg.get("survival_bin_column", "survival_bin")
    has_survival_task = task == "survival" or (task == "multitask" and "survival" in tasks)
    should_fit_survival_bins = bool(data_cfg.get("fit_survival_bins", True)) and has_survival_task
    overwrite_survival_bins = bool(data_cfg.get("overwrite_survival_bins", False))
    survival_bins_missing = (
        survival_bin_column not in train_df.columns or train_df[survival_bin_column].isna().all()
    )
    if should_fit_survival_bins and (overwrite_survival_bins or survival_bins_missing):
        survival_bin_edges = fit_survival_bin_edges(
            train_df,
            event_time_column=data_cfg.get("event_time_column", "event_time"),
            event_observed_column=data_cfg.get("event_observed_column", "event_observed"),
            n_bins=survival_bins,
        )
        train_df = apply_survival_bin_edges(
            train_df,
            survival_bin_edges,
            event_time_column=data_cfg.get("event_time_column", "event_time"),
            output_column=survival_bin_column,
        )
        val_df = apply_survival_bin_edges(
            val_df,
            survival_bin_edges,
            event_time_column=data_cfg.get("event_time_column", "event_time"),
            output_column=survival_bin_column,
        )
        config.setdefault("derived", {})["survival_bin_edges"] = [
            float(edge) for edge in survival_bin_edges
        ]

    label_map: dict[Any, int] | None = None
    label_maps: dict[str, dict[Any, int]] = {}
    n_classes = int(model_cfg.get("n_classes", 2))
    if task == "classification":
        label_map = build_label_map(df[label_column])
        n_classes = int(model_cfg.get("n_classes", len(label_map)))
    elif task == "multitask":
        subtype_col = data_cfg.get("subtype_label_column", "subtype")
        stage_col = data_cfg.get("stage_label_column", "stage")
        if "subtype" in tasks and subtype_col in df.columns:
            label_maps["subtype"] = build_label_map(df[subtype_col])
            model_cfg.setdefault("subtype_classes", len(label_maps["subtype"]))
        if "stage" in tasks and stage_col in df.columns:
            label_maps["stage"] = build_stage_label_map(df[stage_col])
            model_cfg.setdefault("stage_classes", max(label_maps["stage"].values()) + 1)

    if label_map is not None:
        config.setdefault("derived", {})["label_map"] = label_map
    if label_maps:
        config.setdefault("derived", {})["label_maps"] = label_maps

    sparseage_cfg = _build_model_config(task, model_cfg, n_classes=n_classes, tasks=tasks)
    train_set, val_set = _build_datasets(
        df,
        train_df,
        val_df,
        task=task,
        tasks=tasks,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        label_map=label_map,
        label_maps=label_maps,
    )

    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_bags,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_bags,
    )

    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = SparseAGEMIL(sparseage_cfg).to(device)
    optimizer = build_optimizer(
        model,
        name=train_cfg.get("optimizer", "Adam"),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    epochs = int(train_cfg.get("epochs", 200))
    scheduler = build_scheduler(optimizer, train_cfg.get("scheduler", "cosine"), epochs=epochs)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        Path(train_cfg.get("output_dir", "runs")) / config.get("name", "sparseage") / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    best_path = output_dir / "best_model.pt"
    with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    train_df.to_csv(output_dir / "train_manifest_resolved.csv", index=False)
    val_df.to_csv(output_dir / "val_manifest_resolved.csv", index=False)
    if survival_bin_edges is not None:
        pd.DataFrame(
            {"edge_index": range(len(survival_bin_edges)), "edge": survival_bin_edges}
        ).to_csv(output_dir / "survival_bin_edges.csv", index=False)
    if task == "multitask":
        default_metric = "survival_c_index" if "survival" in tasks else "subtype_auc"
    else:
        default_metric = "auc" if task == "classification" else "c_index"
    metric_name = train_cfg.get("best_metric", default_metric)
    amp = bool(train_cfg.get("amp", False))

    print(
        f"[SparseAGE-MIL] task={task} train={len(train_set)} val={len(val_set)} device={device}"
    )
    print(f"[SparseAGE-MIL] tasks={tasks} topk={sparseage_cfg.topk} output_dir={output_dir}")
    if label_map is not None:
        print(f"[SparseAGE-MIL] label_map={label_map}")
    if label_maps:
        print(f"[SparseAGE-MIL] label_maps={label_maps}")
    if args.dry_run:
        return

    best_metric = float("-inf")
    for epoch in range(epochs):
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            task=task,
            amp=amp,
            stage_ordinal_weight=float(train_cfg.get("stage_ordinal_weight", 0.15)),
            survival_alpha=float(train_cfg.get("survival_alpha", 0.0)),
            topology_weight=float(train_cfg.get("topology_weight", 0.10)),
            loss_weights=train_cfg.get("loss_weights"),
            gradient_strategy=train_cfg.get("gradient_strategy", "standard"),
        )
        if scheduler is not None:
            scheduler.step()
        if task == "classification":
            metrics = evaluate_classification(model, val_loader, device=device)
        elif task == "survival":
            metrics = evaluate_survival(model, val_loader, device=device)
        else:
            metrics = evaluate_multitask(model, val_loader, device=device)

        if isinstance(train_result, dict):
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_result.items()}, **metrics}
            train_loss = train_result["loss"]
        else:
            row = {"epoch": epoch, "train_loss": train_result, **metrics}
            train_loss = train_result
        append_metrics_csv(metrics_path, row)

        current = metrics.get(metric_name, float("nan"))
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} "
            + " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
        )
        if current == current and current > best_metric:
            best_metric = current
            save_checkpoint(
                best_path,
                model=model,
                epoch=epoch,
                metric_name=metric_name,
                metric_value=best_metric,
                config=config,
            )
            print(f"[SparseAGE-MIL] saved best checkpoint: {best_path}")

    if bool(train_cfg.get("export_predictions", True)):
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
        topology_dir = (
            output_dir / "topology_exports"
            if bool(train_cfg.get("export_topology", False))
            else None
        )
        pred_path = output_dir / "val_predictions.csv"
        export_slide_predictions(
            model,
            val_loader,
            device=device,
            task=task,
            output_path=pred_path,
            export_topology_dir=topology_dir,
            topology_top_fraction=float(train_cfg.get("topology_top_fraction", 0.10)),
        )
        print(f"[SparseAGE-MIL] exported validation predictions: {pred_path}")
        if bool(train_cfg.get("export_train_predictions", False)):
            train_pred_path = output_dir / "train_predictions.csv"
            export_slide_predictions(
                model,
                train_loader,
                device=device,
                task=task,
                output_path=train_pred_path,
                export_topology_dir=None,
                topology_top_fraction=float(train_cfg.get("topology_top_fraction", 0.10)),
            )
            print(f"[SparseAGE-MIL] exported training predictions: {train_pred_path}")


if __name__ == "__main__":
    main()
