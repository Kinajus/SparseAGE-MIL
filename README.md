# SparseAGE-MIL

Sparse Adaptive Graph Encoding with Spatial-Topology Regularization for Multi-task Multiple Instance Learning.

This repository contains standalone research code for WSI-level multiple instance learning. It runs directly from the source tree and is not structured as a Python package.

## Main features

- Spatial-feature sparse graph encoding using feature affinity, patch coordinates, pseudo-histology cluster consistency, and interface priors.
- Multi-scale Top-K aggregation with `topk` as an integer or a list such as `[4, 8, 16]`.
- Joint multi-task MIL with a shared encoder and task-specific attention/adapters for subtype, stage, and survival.
- Ordinal-aware stage loss, discrete-time survival loss, topology auxiliary loss, uncertainty weighting, and optional PCGrad.
- Utility scripts for prediction export, cross-validation summaries, statistical comparison, Cox analysis, stage diagnostics, sparse-edge diagnostics, attention enrichment, and ablation config generation.

## Setup

```bash
cd SparseAGE-MIL
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install the PyTorch build matching your CUDA driver when using GPU training.

## Data format

Each slide is represented by a `.pt` file. The simplest input is a tensor with shape `[num_patches, feature_dim]`.

For spatial-topology training, save a dictionary:

```python
torch.save(
    {
        "features": features,      # FloatTensor [N, D]
        "coords": coords,          # FloatTensor [N, 2], optional
        "clusters": clusters,      # LongTensor [N], optional
        "topology": topology,      # FloatTensor [6], optional
    },
    "features/slide_001.pt",
)
```

Coordinates and cluster labels can also be stored in separate files and referenced in the manifest.

### Multi-task manifest

```csv
slide_id,feature_path,subtype,stage,event_time,event_observed,fold
S001,features/S001.pt,LUAD,I,812,1,0
S002,features/S002.pt,LUSC,III,455,0,1
```

Missing labels are allowed. A missing endpoint is skipped for that slide while available endpoints still update the shared encoder.

### Single-task classification manifest

```csv
slide_id,feature_path,label,fold
S001,features/S001.pt,LUAD,0
S002,features/S002.pt,LUSC,1
```

### Single-task survival manifest

```csv
slide_id,feature_path,event_time,event_observed,survival_bin,fold
S001,features/S001.pt,812,1,2,0
S002,features/S002.pt,455,0,1,1
```

When `survival_bin` is absent in multi-task mode, `train.py` fits discrete-time survival bins on the training split and applies the same bin edges to validation data.

## Training

Edit the manifest paths and model settings in `configs/multitask.yaml`, then run:

```bash
python train.py --config configs/multitask.yaml
```

Single-task training:

```bash
python train.py --config configs/classification.yaml
python train.py --config configs/survival.yaml
```

Important multi-task options:

```yaml
model:
  input_dim: 1024
  topk: [4, 8, 16]
  use_spatial: true
  spatial_weight: 0.20
  cluster_weight: 0.15
  boundary_weight: 0.10
  use_uncertainty_weighting: true

train:
  stage_ordinal_weight: 0.15
  topology_weight: 0.10
  gradient_strategy: standard  # or pcgrad
```

Training writes `metrics.csv`, `best_model.pt`, resolved manifests, and validation predictions to the run directory.

## Prediction

```bash
python predict.py   --checkpoint runs/<run_name>/best_model.pt   --manifest data/manifests/external.csv   --root .   --output predictions.csv
```

Add `--export-topology-dir topology_exports` to export patch nodes and sparse edges for visualization.

## Script list

```text
python train.py                  train/evaluate models
python predict.py                export slide-level predictions
python summarize_cv.py           summarize cross-validation metrics
python compare_predictions.py    AUC/C-index comparison and confidence intervals
python clinical_survival.py      Cox, log-rank, HR, and fixed-horizon AUC analysis
python stage_diagnostics.py      stage confusion and calibration summaries
python fit_pseudo_clusters.py    fit train-split pseudo-histology prototypes
python edge_diagnostics.py       summarize sparse graph geometry
python attention_enrichment.py   patch-region attention enrichment test
python spatial_stats.py          spatial/cell-type correlation analyses
python ablation_grid.py          generate module and Top-K ablation configs
```

## Repository layout

```text
SparseAGE-MIL/
├── src/                 # model, losses, data loading, training helpers, analysis helpers
├── configs/             # YAML configuration templates
├── train.py             # training entry script
├── predict.py           # inference/export entry script
├── requirements.txt
├── LICENSE
└── README.md
```

## License

MIT License.
