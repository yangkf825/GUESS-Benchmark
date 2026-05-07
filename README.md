<img width="2172" height="724" alt="image" src="https://github.com/user-attachments/assets/21976eac-32ff-4617-aa3e-bdb915c509c4" />


# GUESS-Bench

> **GUESS-Bench: Benchmarking Uncertainty Estimation in GNNs Under Distribution Shifts**
> 
A comprehensive benchmark for GNN Uncertainty EStimation under distribution Shift.


---

## Supported Algorithms

### Standard UQ Methods

| Algorithm | Category | Uncertainty Score | Reference |
|-----------|----------|-------------------|-----------|
| **Vanilla GCN** | Deterministic | Predictive entropy $H[p]$ | Kipf & Welling, ICLR 2017 |
| **GPN** | Bayesian | $1/\sum_c\alpha_c$ (Dirichlet evidence reciprocal) | Stadler et al., NeurIPS 2021 |
| **G-ΔUQ** | Bayesian | Mean std of sigmoid outputs across anchors | Thiagarajan et al., NeurIPS 2022 |
| **S-BGCN-T-K** | Bayesian | Entropy of mean prediction over $K$ stochastic passes | Hasanzadeh et al., ICML 2020 |
| **CaGCN** | Calibration | $1-\max(p)$ after graph-conv temperature scaling | Wang et al., NeurIPS 2021 |
| **GATS** | Calibration | $1-\max(p)$ after attention temperature scaling | Hsu et al., 2022 |
| **RBS**  | Calibration | $1-\max(p)$ after reliability-based bin scaling | Liu et al., 2022 |
### Conformal Prediction Methods

| Algorithm | Score Type | Reference |
|-----------|-----------|-----------|
| **DAPS** | APS on PPR-smoothed logits | Zargarbashi et al., 2023 |
| **NAPS** | Degree-weighted APS quantile | Clarkson, 2023 |
| **CF-GNN** | ConfGNN topology-aware correction + APS | Huang et al., NeurIPS 2023 |

---

## Supported Datasets

| Dataset | OOD Type | Train | Val | OOD Test Splits | Metric |
|---------|----------|-------|-----|-----------------|--------|
| **Elliptic Bitcoin** | Temporal | Steps 7–11 | Steps 12–16 | 9 groups (Steps 17–48) | F1 |
| **OGB-Arxiv** | Temporal | ≤2011 | 2011–2014 | 2014–16 / 2016–18 / 2018–20 | Accuracy |
| **EERM-Cora** | Environmental | env 0 | env 1 | env 2–9 (8 splits) | Accuracy |
| **EERM-Amazon-Photo** | Environmental | env 0 | env 1 | env 2–9 (8 splits) | Accuracy |
| **Twitch-Explicit** | Cross-domain | ES, FR, PTBR, RU | DE | ENGB, TW | Accuracy |
| **Facebook-100** | Cross-domain | 8 universities | Bingham82, Texas80, Yale4 | Caltech36, Duke14, Penn94 | Accuracy |

---

## Evaluation Metrics

| Research Question | Metrics |
|-------------------|---------|
| **RQ1 Calibration** | ECE (M=15), NLL, Brier; OOD: Δ-ECE, Δ-NLL, Δ-Brier |
| **RQ2 Uncertainty Estimation** | UE-AUROC, UE-AUPR, OOD-AUROC |
| **RQ3 Selective Classification** | AURC, Risk@τ (τ∈{0.1,…,1.0}), SRTR@τ, SRTR-AUC |
| **RQ4 Conformal Prediction** | Coverage, Set Size, Singleton Hit Ratio (SHR) |

---

## Installation

```bash
git clone https://anonymous.4open.science/r/GUESS-Benchmark/
cd GUESS-Benchmark
pip install -e .
```

**Dependencies:** 
 
```
Python >= 3.9
torch >= 2.0
torch-geometric >= 2.4
numpy, scipy, scikit-learn, pandas, matplotlib, seaborn
```

**Hardware:** Tested on NVIDIA RTX 4090D (24 GB), CUDA 11.8, Python 3.10 (Ubuntu 22.04)

---

## Quick Start

### Run All Methods (Recommended)
 
```bash
# ── Elliptic / Arxiv / EERM ──────────────────────────────────────────────
# Usage: bash experiments/run_all.sh <dataset> <data_path> <runs> [eerm_ds]
 
bash experiments/run_all.sh elliptic ./data/elliptic 5
bash experiments/run_all.sh arxiv    ./data/arxiv/data.pkl 5
bash experiments/run_all.sh eerm     ./data/eerm/Planetoid/cora 5 cora
bash experiments/run_all.sh eerm     ./data/eerm/Amazon/Photo   5 amazon
 
# ── Facebook100 / Twitch ─────────────────────────────────────────────────
# Usage: bash experiments/run_all_fb_twitch.sh <dataset> <data_root> <runs>
 
bash experiments/run_all_fb_twitch.sh twitch   ./data 5
bash experiments/run_all_fb_twitch.sh facebook ./data 3

# Conformal prediction methods (sweep over alpha)
for alpha in 0.01 0.05 0.10 0.15 0.20 0.25; do
    python experiments/run_cfgnn.py    --dataset elliptic --data_dir ./data/elliptic \
        --alpha $alpha --score aps --runs 5 --save_dir ./results/cfgnn
    python experiments/run_daps.py     --dataset elliptic --data_dir ./data/elliptic \
        --alpha $alpha --score aps --transform daps --n_iters 10 --ppr_alpha 0.85 \
        --runs 5 --save_dir ./results/daps
    python experiments/run_graph_cp.py --dataset elliptic --data_dir ./data/elliptic \
        --alpha $alpha --score aps --mode weighted --runs 5 --save_dir ./results/graph_cp
done

# ConfGNN (CF-GNN core contribution)
python experiments/run_confgnn_elliptic.py --data_dir ./data/elliptic \
    --alpha 0.1 --score aps --runs 5 --use_confgnn --save_dir ./results/confgnn

```
 
---
 
## Per-Method Commands (Elliptic / Arxiv / EERM)

The following commands run the original main experiment scripts.
 
### S-BGCN-T-K（`run_ungnn.py`）
 
```bash
# Elliptic
python experiments/run_ungnn.py --dataset elliptic --data_dir ./data/elliptic --runs 5
 
# OGB-Arxiv
python experiments/run_ungnn.py --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
 
# EERM-Cora
python experiments/run_ungnn.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --runs 5
 
# EERM-Amazon
python experiments/run_ungnn.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --runs 5
```
 
### GATS（`run_gats.py`）
 
```bash
# Elliptic
python experiments/run_gats.py --dataset elliptic --data_dir ./data/elliptic --runs 5
 
# OGB-Arxiv
python experiments/run_gats.py --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
 
# EERM-Cora
python experiments/run_gats.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --runs 5
 
# EERM-Amazon
python experiments/run_gats.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --runs 5
```
 
### CaGCN（`run_cagcn.py`）
 
```bash
# Elliptic
python experiments/run_cagcn.py --dataset elliptic --data_dir ./data/elliptic --runs 5
 
# OGB-Arxiv
python experiments/run_cagcn.py --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
 
# EERM-Cora
python experiments/run_cagcn.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --runs 5
 
# EERM-Amazon
python experiments/run_cagcn.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --runs 5
```
 
### RBS（`run_calgnn.py`）
 
```bash
# Elliptic
python experiments/run_calgnn.py --dataset elliptic --data_dir ./data/elliptic --runs 5
 
# OGB-Arxiv
python experiments/run_calgnn.py --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
 
# EERM-Cora
python experiments/run_calgnn.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --runs 5
 
# EERM-Amazon
python experiments/run_calgnn.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --runs 5
```
 
### G-ΔUQ（`run_gduq.py`）
 
```bash
# Elliptic
python experiments/run_gduq.py --dataset elliptic --data_dir ./data/elliptic --runs 5
 
# OGB-Arxiv
python experiments/run_gduq.py --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
 
# EERM-Cora
python experiments/run_gduq.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --runs 5
 
# EERM-Amazon
python experiments/run_gduq.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --runs 5
```
 
### GPN (`run_gpn.py`)

```bash
python experiments/run_gpn.py --dataset elliptic --data_dir ./data/elliptic --runs 5
python experiments/run_gpn.py --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
python experiments/run_gpn.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --runs 5
python experiments/run_gpn.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --runs 5
```
 
---
 
## Per-Algorithm Commands (Facebook100 / Twitch)

The following commands run the original cross-domain experiment scripts.
 
### S-BGCN-T-K（`run_ungnn_fb_twitch.py`）
 
```bash
python experiments/run_ungnn_fb_twitch.py --dataset twitch   --data_root ./data --runs 5
python experiments/run_ungnn_fb_twitch.py --dataset facebook --data_root ./data --runs 3
```
 
### GATS（`run_gats_fb_twitch.py`）
 
```bash
python experiments/run_gats_fb_twitch.py --dataset twitch   --data_root ./data --runs 5
python experiments/run_gats_fb_twitch.py --dataset facebook --data_root ./data --runs 3
```
 
### CaGCN（`run_cagcn_fb_twitch.py`）
 
```bash
python experiments/run_cagcn_fb_twitch.py --dataset twitch   --data_root ./data --runs 5
python experiments/run_cagcn_fb_twitch.py --dataset facebook --data_root ./data --runs 3
```
 
### RBS（`run_calgnn_fb_twitch.py`）
 
```bash
python experiments/run_calgnn_fb_twitch.py --dataset twitch   --data_root ./data --runs 5
python experiments/run_calgnn_fb_twitch.py --dataset facebook --data_root ./data --runs 3
```
 
### G-ΔUQ（`run_gduq_fb_twitch.py`）
 
```bash
python experiments/run_gduq_fb_twitch.py --dataset twitch   --data_root ./data --runs 5
python experiments/run_gduq_fb_twitch.py --dataset facebook --data_root ./data --runs 3
```
 
### GPN (`run_gpn_fb_twitch.py`)
 
```bash
python experiments/run_gpn_fb_twitch.py --dataset twitch   --data_root ./data --runs 3
python experiments/run_gpn_fb_twitch.py --dataset facebook --data_root ./data --runs 3
```
 
---
 
## Result Visualization
 
```bash
python plot_results.py --data_dir ./results --out_dir ./figures --dataset elliptic
python plot_results.py --data_dir ./results --out_dir ./figures --dataset arxiv
python plot_results.py --data_dir ./results --out_dir ./figures --dataset twitch
python plot_results.py --data_dir ./results --out_dir ./figures --dataset facebook
```
 
---
 
## Project Structure
 
```
graphuq-bench/
├── src/
│   └── gnn_uq_bench/
│       ├── __init__.py
│       ├── metrics.py              # RQ1/RQ2/RQ3 full metrics + RBS summary
│       ├── datasets.py             # Elliptic / OGB-Arxiv / EERM loaders
│       ├── datasets_fb_twitch.py   # Facebook100 / Twitch loaders
│       ├── models.py               # model definitions used by the original scripts
│       └── calibration.py          # TS / HB / Iso / BBQ / MetaCal / RBS
│
├── experiments/
│   ├── run_ungnn.py                ┐
│   ├── run_gats.py                 │
│   ├── run_gpn.py                  │ Elliptic / Arxiv / EERM
│   ├── run_gduq.py                 │
│   ├── run_calgnn.py               │
│   ├── run_cagcn.py                ┘
│   ├── run_ungnn_fb_twitch.py      ┐
│   ├── run_gats_fb_twitch.py       │
│   ├── run_gpn_fb_twitch.py        │ Facebook100 / Twitch
│   ├── run_gduq_fb_twitch.py       │
│   ├── run_calgnn_fb_twitch.py     │
│   ├── run_cagcn_fb_twitch.py      ┘
│   ├── run_all.sh                  
│   └── run_all_fb_twitch.sh        
│
├── data/                           # datasets (not tracked by git)
├── results/                        # experiment outputs (auto-generated)
├── figures/                        # visualization outputs (auto-generated)
├── plot_results.py
├── pyproject.toml
├── .gitignore
└── README.md
```

 
---
 
## Output Format

Each experiment script generates three CSV files under `results/<alg>/`:

| File | Description |
|------|-------------|
| `*_results.csv` | Mean ± standard deviation over each split |
| `*_reliability.csv` | Reliability Diagram raw data (M=15 bins) |
| `*_uncertainty.csv` | Node-level uncertainty scores (first seed only) |

CalGNN additionally outputs results per calibration method (file names include `_Uncal_`, `_RBS_`, etc.).

 
---
 
## Key Hyperparameters

### Common Parameters (All Scripts)

| Parameter | Default | Description |
|----------|--------|-------------|
| `--runs` | 5 | Number of repeated runs |
| `--base_seed` | 42 | Base random seed |
| `--hidden` | 64 | Hidden dimension of GNN |
| `--dropout` | 0.5 | Dropout rate |
| `--lr` | 0.01 | Learning rate |
| `--epochs` | 2000 | Maximum training epochs |
| `--patience` | 100 | Early stopping patience |
| `--save_dir` | `./results/<alg>` | Output directory |

### GATS-Specific Parameters

| Parameter | Default | Description |
|----------|--------|-------------|
| `--gats_heads` | 8 | Number of attention heads in CalibAttentionLayer |
| `--gats_bias` | 1.0 | Initial temperature bias |
| `--gats_epochs` | 200 | Training epochs for GATS temperature layer |
| `--bfs_depth` | 2 | Maximum hop distance for BFS-based computation |
 
### GPN-Specific Parameters

| Parameter | Default | Description |
|----------|--------|-------------|
| `--dim_latent` | 10 | Latent space dimension |
| `--radial_layers` | 10 | Number of normalizing flow layers |
| `--K` | 10 | APPNP propagation steps |
| `--alpha_teleport` | 0.2 | Teleport probability in APPNP |

### G-ΔUQ-Specific Parameters

| Parameter | Default | Description |
|----------|--------|-------------|
| `--n_anchors` | 30 | Number of anchor samples during inference |
| `--anchor_type` | `node` | Anchor type: `node` (per-node) / `graph` (shared) |
| `--num_layers` | 2 | Number of layers in GNN encoder |

### CaGCN-Specific Parameters

| Parameter | Default | Description |
|----------|--------|-------------|
| `--lr_for_cal` | 0.01 | Learning rate for temperature network |
| `--l2_for_cal` | 5e-4 | Weight decay for temperature network |
| `--Lambda` | 0.5 | Intra-class loss weight |
| `--stage` | 1 | Number of self-training stages |
| `--threshold` | 0.8 | Confidence threshold for pseudo-labeling |
 
### RBS-Specific Parameters

| Parameter | Default | Description |
|----------|--------|-------------|
| `--num_bins_rbs` | 10 | Number of bins for RBS temperature calibration |
| `--add_cal_loss` | `False` | Whether to include calibration regularization loss |


---
 
## .gitignore
 
```gitignore
data/
results/
figures/
*.pth
*.pt
__pycache__/
*.pyc
*.egg-info/
dist/
```
 
---

## Data Splits Summary

| Dataset | Train | Val | OOD Test |
|---------|-------|-----|----------|
| Elliptic | Steps 7–11 | Steps 12–16 | 9 groups (steps 17–48) |
| OGB-Arxiv | ≤ 2011 | 2011–2014 | 2014–16 / 2016–18 / 2018–20 |
| EERM-Cora | env 0 | env 1 | env 2–9 |
| EERM-Amazon | env 0 | env 1 | env 2–9 |
| Twitch | ES, FR, PTBR, RU | DE | ENGB, TW |
| Facebook-100 | Amherst41, Brandeis99, Brown11, Carnegie49, Cornell5, Johns Hopkins55, Princeton12, WashU32 | Bingham82, Texas80, Yale4 | Caltech36, Duke14, Penn94 |

---

## Appendix: GAT / GraphSAGE Backbone Extension

This section documents the additional GAT and GraphSAGE backbone variants.
The original scripts above are kept unchanged. The new variants are isolated in newly created `*_gat_sage.py` runners and a dedicated model file, so the original GCN-based workflow remains available.

### Added Model File

```text
src/gnn_uq_bench/model_gat_sage.py
```

The file provides GCN, GAT, and GraphSAGE backbone builders for both execution paths used in this benchmark:

| Path | Datasets | Interface | Backbones |
|------|----------|-----------|-----------|
| Sparse adjacency path | Elliptic / OGB-Arxiv / EERM-Cora / EERM-Amazon | `forward(x, adj)` | GCN / GAT / GraphSAGE |
| PyG `edge_index` path | Facebook100 / Twitch | `forward(x, edge_index)` | GCN / GAT / GraphSAGE |

The new runners use `--model` instead of `--backbone`:

```bash
--model GCN
--model GAT
--model SAGE
--model GraphSAGE
```

`SAGE` and `GraphSAGE` are treated as aliases. The default is kept as `GCN` unless a runner sets a different default internally.

### Added Experiment Scripts

```text
experiments/run_ungnn_gat_sage.py
experiments/run_gats_gat_sage.py
experiments/run_cagcn_gat_sage.py
experiments/run_calgnn_gat_sage.py
experiments/run_gduq_gat_sage.py
experiments/run_gpn_gat_sage.py
experiments/run_graph_cp_gat_sage.py

experiments/run_ungnn_fb_twitch_gat_sage.py
experiments/run_gats_fb_twitch_gat_sage.py
experiments/run_cagcn_fb_twitch_gat_sage.py
experiments/run_calgnn_fb_twitch_gat_sage.py
experiments/run_gduq_fb_twitch_gat_sage.py
experiments/run_gpn_fb_twitch_gat_sage.py
```

### Backbone Extension Commands: Elliptic / Arxiv / EERM

```bash
# S-BGCN-T-K
python experiments/run_ungnn_gat_sage.py --dataset elliptic --data_dir ./data/elliptic --model GAT --runs 5
python experiments/run_ungnn_gat_sage.py --dataset arxiv --data_path ./data/arxiv/data.pkl --model GraphSAGE --runs 5
python experiments/run_ungnn_gat_sage.py --dataset eerm --eerm_dataset cora --eerm_root ./data/eerm/Planetoid/cora --model GAT --runs 5
python experiments/run_ungnn_gat_sage.py --dataset eerm --eerm_dataset amazon --eerm_root ./data/eerm/Amazon/Photo --model GraphSAGE --runs 5

# GATS
python experiments/run_gats_gat_sage.py --dataset elliptic --data_dir ./data/elliptic --model GAT --runs 5
python experiments/run_gats_gat_sage.py --dataset arxiv --data_path ./data/arxiv/data.pkl --model GraphSAGE --runs 5
python experiments/run_gats_gat_sage.py --dataset eerm --eerm_dataset cora --eerm_root ./data/eerm/Planetoid/cora --model GAT --runs 5
python experiments/run_gats_gat_sage.py --dataset eerm --eerm_dataset amazon --eerm_root ./data/eerm/Amazon/Photo --model GraphSAGE --runs 5

# CaGCN
python experiments/run_cagcn_gat_sage.py --dataset elliptic --data_dir ./data/elliptic --model GAT --runs 5
python experiments/run_cagcn_gat_sage.py --dataset arxiv --data_path ./data/arxiv/data.pkl --model GraphSAGE --runs 5
python experiments/run_cagcn_gat_sage.py --dataset eerm --eerm_dataset cora --eerm_root ./data/eerm/Planetoid/cora --model GAT --runs 5
python experiments/run_cagcn_gat_sage.py --dataset eerm --eerm_dataset amazon --eerm_root ./data/eerm/Amazon/Photo --model GraphSAGE --runs 5

# RBS / CalGNN
python experiments/run_calgnn_gat_sage.py --dataset elliptic --data_dir ./data/elliptic --model GAT --runs 5
python experiments/run_calgnn_gat_sage.py --dataset arxiv --data_path ./data/arxiv/data.pkl --model GraphSAGE --runs 5
python experiments/run_calgnn_gat_sage.py --dataset eerm --eerm_dataset cora --eerm_root ./data/eerm/Planetoid/cora --model GAT --runs 5
python experiments/run_calgnn_gat_sage.py --dataset eerm --eerm_dataset amazon --eerm_root ./data/eerm/Amazon/Photo --model GraphSAGE --runs 5

# G-ΔUQ
python experiments/run_gduq_gat_sage.py --dataset elliptic --data_dir ./data/elliptic --model GAT --runs 5
python experiments/run_gduq_gat_sage.py --dataset arxiv --data_path ./data/arxiv/data.pkl --model GraphSAGE --runs 5
python experiments/run_gduq_gat_sage.py --dataset eerm --eerm_dataset cora --eerm_root ./data/eerm/Planetoid/cora --model GAT --runs 5
python experiments/run_gduq_gat_sage.py --dataset eerm --eerm_dataset amazon --eerm_root ./data/eerm/Amazon/Photo --model GraphSAGE --runs 5

# GPN backbone-extension runner
python experiments/run_gpn_gat_sage.py --dataset elliptic --data_dir ./data/elliptic --model GAT --runs 5
python experiments/run_gpn_gat_sage.py --dataset arxiv --data_path ./data/arxiv/data.pkl --model GraphSAGE --runs 5
python experiments/run_gpn_gat_sage.py --dataset eerm --eerm_dataset cora --eerm_root ./data/eerm/Planetoid/cora --model GAT --runs 5
python experiments/run_gpn_gat_sage.py --dataset eerm --eerm_dataset amazon --eerm_root ./data/eerm/Amazon/Photo --model GraphSAGE --runs 5

# Graph-CP backbone-extension runner
python experiments/run_graph_cp_gat_sage.py --dataset elliptic --data_dir ./data/elliptic --model GAT --runs 5
python experiments/run_graph_cp_gat_sage.py --dataset arxiv --data_path ./data/arxiv/data.pkl --model GraphSAGE --runs 5
python experiments/run_graph_cp_gat_sage.py --dataset eerm --eerm_dataset cora --eerm_root ./data/eerm/Planetoid/cora --model GAT --runs 5
python experiments/run_graph_cp_gat_sage.py --dataset eerm --eerm_dataset amazon --eerm_root ./data/eerm/Amazon/Photo --model GraphSAGE --runs 5
```

### Backbone Extension Commands: Facebook100 / Twitch

```bash
# S-BGCN-T-K
python experiments/run_ungnn_fb_twitch_gat_sage.py --dataset twitch --data_root ./data --model GAT --runs 5
python experiments/run_ungnn_fb_twitch_gat_sage.py --dataset facebook --data_root ./data --model GraphSAGE --runs 3

# GATS
python experiments/run_gats_fb_twitch_gat_sage.py --dataset twitch --data_root ./data --model GAT --runs 5
python experiments/run_gats_fb_twitch_gat_sage.py --dataset facebook --data_root ./data --model GraphSAGE --runs 3

# CaGCN
python experiments/run_cagcn_fb_twitch_gat_sage.py --dataset twitch --data_root ./data --model GAT --runs 5
python experiments/run_cagcn_fb_twitch_gat_sage.py --dataset facebook --data_root ./data --model GraphSAGE --runs 3

# RBS / CalGNN
python experiments/run_calgnn_fb_twitch_gat_sage.py --dataset twitch --data_root ./data --model GAT --runs 5
python experiments/run_calgnn_fb_twitch_gat_sage.py --dataset facebook --data_root ./data --model GraphSAGE --runs 3

# G-ΔUQ
python experiments/run_gduq_fb_twitch_gat_sage.py --dataset twitch --data_root ./data --model GAT --runs 5
python experiments/run_gduq_fb_twitch_gat_sage.py --dataset facebook --data_root ./data --model GraphSAGE --runs 3

# GPN backbone-extension runner
python experiments/run_gpn_fb_twitch_gat_sage.py --dataset twitch --data_root ./data --model GAT --runs 3
python experiments/run_gpn_fb_twitch_gat_sage.py --dataset facebook --data_root ./data --model GraphSAGE --runs 3
```

### Extension-Specific Parameters

| Parameter | Default | Description |
|----------|---------|-------------|
| `--model` | `GCN` | Backbone used by the new `*_gat_sage.py` scripts. Options: `GCN`, `GAT`, `SAGE`, `GraphSAGE`. |
| `--backbone_heads` | `8` | Number of attention heads used by the GAT backbone where applicable. |


## License

Apache 2.0

---
