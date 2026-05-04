<img width="2102" height="496" alt="Gemini_Generated_Image_z8ysd9z8ysd9z8ys (2)" src="https://github.com/user-attachments/assets/c59fb4f9-6f8c-47a1-9b6a-aa473ef6ba03" />




# GUESS-Bench

> **GUESS-Bench: Benchmarking Uncertainty Estimation in GNNs Under Distribution Shifts**
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
| **RBS** (CalGNN) | Calibration | $1-\max(p)$ after reliability-based bin scaling | Liu et al., 2022 |

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
git clone https://github.com/YOUR_USERNAME/gnn-uq-bench
cd gnn-uq-bench
pip install -e .
```

**Dependencies:** `torch>=2.1`, `torch-geometric>=2.4`, `scipy`, `scikit-learn`, `numpy`

**Hardware:** Tested on NVIDIA RTX 4090D (24 GB), CUDA 11.8, Python 3.10 (Ubuntu 22.04)

---

## Quick Start

### Elliptic Bitcoin

```bash
# Standard UQ methods
python experiments/run_ungnn.py   --dataset elliptic --data_dir ./data/elliptic --runs 5
python experiments/run_gpn.py     --dataset elliptic --data_dir ./data/elliptic --runs 5
python experiments/run_gduq.py    --dataset elliptic --data_dir ./data/elliptic --runs 5
python experiments/run_gats.py    --dataset elliptic --data_dir ./data/elliptic --runs 5
python experiments/run_cagcn.py   --dataset elliptic --data_dir ./data/elliptic --runs 5
python experiments/run_calgnn.py  --dataset elliptic --data_dir ./data/elliptic --runs 5

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

### OGB-Arxiv

```bash
python experiments/run_ungnn.py  --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
python experiments/run_gpn.py    --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
python experiments/run_gduq.py   --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
python experiments/run_gats.py   --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
python experiments/run_cagcn.py  --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
python experiments/run_calgnn.py --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
```

### EERM (Cora / Amazon-Photo)

```bash
# Cora
python experiments/run_ungnn.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/EERM/Planetoid/cora --runs 5

# Amazon-Photo
python experiments/run_ungnn.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/EERM/Amazon/Photo --runs 5
```

### Twitch-Explicit & Facebook-100

```bash
DATA=./data   # data_root must contain twitch/ and facebook100/ subdirectories

# Twitch
python experiments/run_ungnn_fb_twitch.py  --dataset twitch --data_root $DATA --runs 5
python experiments/run_gpn_fb_twitch.py    --dataset twitch --data_root $DATA --runs 5
python experiments/run_gduq_fb_twitch.py   --dataset twitch --data_root $DATA --runs 5
python experiments/run_gats_fb_twitch.py   --dataset twitch --data_root $DATA --runs 5
python experiments/run_cagcn_fb_twitch.py  --dataset twitch --data_root $DATA --runs 5
python experiments/run_calgnn_fb_twitch.py --dataset twitch --data_root $DATA --runs 5

# Facebook-100
python experiments/run_ungnn_fb_twitch.py  --dataset facebook --data_root $DATA --runs 3
python experiments/run_gpn_fb_twitch.py    --dataset facebook --data_root $DATA --runs 3
python experiments/run_gduq_fb_twitch.py   --dataset facebook --data_root $DATA --runs 3
python experiments/run_gats_fb_twitch.py   --dataset facebook --data_root $DATA --runs 3
python experiments/run_cagcn_fb_twitch.py  --dataset facebook --data_root $DATA --runs 3
python experiments/run_calgnn_fb_twitch.py --dataset facebook --data_root $DATA --runs 3
```

### Run All Algorithms with One Command

```bash
# Elliptic / Arxiv / EERM
bash experiments/run_all.sh elliptic ./data/elliptic 5
bash experiments/run_all.sh arxiv    ./data/arxiv/data.pkl 5
bash experiments/run_all.sh eerm     ./data/EERM/Planetoid/cora 5 cora
bash experiments/run_all.sh eerm     ./data/EERM/Amazon/Photo 5 amazon

# Twitch / Facebook-100
bash experiments/run_all_fb_twitch.sh twitch   ./data 5
bash experiments/run_all_fb_twitch.sh facebook ./data 3
```

---

## Result Visualization

### Standard UQ Methods

```bash
python plot_results.py --data_dir ./results --out_dir ./figures --dataset elliptic
```

### Conformal Prediction Methods

```bash
python plot_conformal.py \
    --cfgnn_dir    ./results/cfgnn \
    --confgnn_dir  ./results/confgnn \
    --graphcp_dir  ./results/graph_cp \
    --daps_dir     ./results/daps \
    --split        all \
    --out          ./figures
```

Generated figures include:
- Calibration comparison (ECE / NLL / Brier bar charts)
- OOD Δ-ECE / Δ-NLL trend curves
- UE-AUROC and OOD-AUROC comparisons
- AURC selective classification curves
- Coverage vs Efficiency (Set Size) — conformal methods
- Coverage vs Singleton Hit Ratio — conformal methods

---

## Project Structure

```
gnn-uq-bench/
├── src/gnn_uq_bench/
│   ├── metrics.py              # RQ1/RQ2/RQ3/RQ4 metrics
│   ├── datasets.py             # Elliptic / Arxiv / EERM data loading
│   ├── datasets_fb_twitch.py   # Facebook-100 / Twitch data loading
│   ├── models.py               # GCNSparse, GPN, G-ΔUQ, CaGCN, GATS...
│   └── calibration.py          # TS, HB, Iso, BBQ, MetaCal, RBS
├── experiments/
│   ├── run_ungnn.py            # Vanilla GCN / S-BGCN-T-K
│   ├── run_gpn.py              # GPN
│   ├── run_gduq.py             # G-ΔUQ
│   ├── run_gats.py             # GATS
│   ├── run_cagcn.py            # CaGCN
│   ├── run_calgnn.py           # CalGNN (Uncal + RBS)
│   ├── run_cfgnn.py            # CF-GNN (split conformal, APS/RAPS/TPS)
│   ├── run_confgnn_elliptic.py # ConfGNN (topology-aware conformal)
│   ├── run_daps.py             # DAPS (PPR-based conformal)
│   ├── run_graph_cp.py         # NAPS (neighbourhood-weighted conformal)
│   ├── run_*_fb_twitch.py      # Facebook-100 / Twitch variants
│   ├── run_all.sh              # Run all standard methods
│   └── run_all_fb_twitch.sh    # Run all methods on FB/Twitch
├── plot_results.py             # Standard UQ result visualisation
├── plot_conformal.py           # Conformal prediction visualisation
└── pyproject.toml
```

---

## Output CSV Format

Each algorithm generates three CSV files under `results/<alg>/`:

| File | Content |
|------|---------|
| `*_results.csv` | Mean ± std for each split across runs (main results) |
| `*_reliability.csv` | Raw data for Reliability Diagram |
| `*_uncertainty.csv` | Per-node uncertainty scores |

Conformal prediction methods additionally output `*_cp.csv` with Coverage, Set Size, and SHR per alpha level.

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

## License

Apache 2.0

## License

Apache 2.0
