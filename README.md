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
| **RBS** (CalGNN) | Calibration | $1-\max(p)$ after reliability-based bin scaling | Liu et al., 2022 |
> **注意：** GPN 使用自有的 normalizing flow encoder + APPNP 传播，结构上不依赖 GCN/GAT/SAGE backbone，因此不支持 `--backbone` 参数。
### Conformal Prediction Methods

| Algorithm | Score Type | Reference |
|-----------|-----------|-----------|
| **DAPS** | APS on PPR-smoothed logits | Zargarbashi et al., 2023 |
| **NAPS** | Degree-weighted APS quantile | Clarkson, 2023 |
| **CF-GNN** | ConfGNN topology-aware correction + APS | Huang et al., NeurIPS 2023 |

---
---
 
## 支持 Backbone
 
| Backbone | 稀疏adj路径<br>（Elliptic / Arxiv / EERM） | PyG edge_index 路径<br>（Facebook / Twitch） |
|----------|------------------------------------------|---------------------------------------------|
| **GCN** | `GCNSparse` | `GCNPyG` |
| **GAT** | `GATSparse` | `GATModel` |
| **GraphSAGE** | `SAGESparse` | `SAGEModel` |
 
所有 backbone 通过统一参数 `--backbone GCN / GAT / GraphSAGE`（默认 `GCN`）切换，**与原有代码完全向后兼容**。
 
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

### 一键运行全部算法（推荐）
 
```bash
# ── Elliptic / Arxiv / EERM ──────────────────────────────────────────────
# 用法: bash experiments/run_all.sh <dataset> <data_path> <runs> <backbone> [eerm_ds]
 
# GCN backbone（默认）
bash experiments/run_all.sh elliptic ./data/elliptic 5 GCN
bash experiments/run_all.sh arxiv    ./data/arxiv/data.pkl 5 GCN
bash experiments/run_all.sh eerm     ./data/eerm/Planetoid/cora 5 GCN cora
bash experiments/run_all.sh eerm     ./data/eerm/Amazon/Photo   5 GCN amazon
 
# GAT backbone
bash experiments/run_all.sh elliptic ./data/elliptic 5 GAT
bash experiments/run_all.sh arxiv    ./data/arxiv/data.pkl 5 GAT
bash experiments/run_all.sh eerm     ./data/eerm/Planetoid/cora 5 GAT cora
bash experiments/run_all.sh eerm     ./data/eerm/Amazon/Photo   5 GAT amazon
 
# GraphSAGE backbone
bash experiments/run_all.sh elliptic ./data/elliptic 5 GraphSAGE
bash experiments/run_all.sh arxiv    ./data/arxiv/data.pkl 5 GraphSAGE
bash experiments/run_all.sh eerm     ./data/eerm/Planetoid/cora 5 GraphSAGE cora
bash experiments/run_all.sh eerm     ./data/eerm/Amazon/Photo   5 GraphSAGE amazon
 
# ── Facebook100 / Twitch ─────────────────────────────────────────────────
# 用法: bash experiments/run_all_fb_twitch.sh <dataset> <data_root> <runs> <backbone>
 
bash experiments/run_all_fb_twitch.sh twitch   ./data 5 GCN
bash experiments/run_all_fb_twitch.sh twitch   ./data 5 GAT
bash experiments/run_all_fb_twitch.sh twitch   ./data 5 GraphSAGE
bash experiments/run_all_fb_twitch.sh facebook ./data 3 GCN
bash experiments/run_all_fb_twitch.sh facebook ./data 3 GAT
bash experiments/run_all_fb_twitch.sh facebook ./data 3 GraphSAGE

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
 
## 逐算法命令（Elliptic / Arxiv / EERM）
 
所有稀疏adj路径脚本均支持 `--backbone GCN / GAT / GraphSAGE`（默认 `GCN`）。
 
### S-BGCN-T-K（`run_ungnn.py`）
 
```bash
# Elliptic
python experiments/run_ungnn.py --dataset elliptic --data_dir ./data/elliptic --runs 5
python experiments/run_ungnn.py --dataset elliptic --data_dir ./data/elliptic --backbone GAT --runs 5
python experiments/run_ungnn.py --dataset elliptic --data_dir ./data/elliptic --backbone GraphSAGE --runs 5
 
# OGB-Arxiv
python experiments/run_ungnn.py --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
python experiments/run_ungnn.py --dataset arxiv --data_path ./data/arxiv/data.pkl --backbone GAT --runs 5
python experiments/run_ungnn.py --dataset arxiv --data_path ./data/arxiv/data.pkl --backbone GraphSAGE --runs 5
 
# EERM-Cora
python experiments/run_ungnn.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --runs 5
python experiments/run_ungnn.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --backbone GAT --runs 5
python experiments/run_ungnn.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --backbone GraphSAGE --runs 5
 
# EERM-Amazon
python experiments/run_ungnn.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --runs 5
python experiments/run_ungnn.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --backbone GAT --runs 5
python experiments/run_ungnn.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --backbone GraphSAGE --runs 5
```
 
### GATS（`run_gats.py`）
 
```bash
# Elliptic
python experiments/run_gats.py --dataset elliptic --data_dir ./data/elliptic --runs 5
python experiments/run_gats.py --dataset elliptic --data_dir ./data/elliptic --backbone GAT --runs 5
python experiments/run_gats.py --dataset elliptic --data_dir ./data/elliptic --backbone GraphSAGE --runs 5
 
# OGB-Arxiv
python experiments/run_gats.py --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
python experiments/run_gats.py --dataset arxiv --data_path ./data/arxiv/data.pkl --backbone GAT --runs 5
python experiments/run_gats.py --dataset arxiv --data_path ./data/arxiv/data.pkl --backbone GraphSAGE --runs 5
 
# EERM-Cora
python experiments/run_gats.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --runs 5
python experiments/run_gats.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --backbone GAT --runs 5
python experiments/run_gats.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --backbone GraphSAGE --runs 5
 
# EERM-Amazon
python experiments/run_gats.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --runs 5 --backbone GAT
python experiments/run_gats.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --runs 5 --backbone GraphSAGE
```
 
### CaGCN（`run_cagcn.py`）
 
```bash
# Elliptic
python experiments/run_cagcn.py --dataset elliptic --data_dir ./data/elliptic --runs 5
python experiments/run_cagcn.py --dataset elliptic --data_dir ./data/elliptic --backbone GAT --runs 5
python experiments/run_cagcn.py --dataset elliptic --data_dir ./data/elliptic --backbone GraphSAGE --runs 5
 
# OGB-Arxiv
python experiments/run_cagcn.py --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
python experiments/run_cagcn.py --dataset arxiv --data_path ./data/arxiv/data.pkl --backbone GAT --runs 5
python experiments/run_cagcn.py --dataset arxiv --data_path ./data/arxiv/data.pkl --backbone GraphSAGE --runs 5
 
# EERM-Cora
python experiments/run_cagcn.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --runs 5
python experiments/run_cagcn.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --backbone GAT --runs 5
python experiments/run_cagcn.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --backbone GraphSAGE --runs 5
 
# EERM-Amazon
python experiments/run_cagcn.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --runs 5 --backbone GAT
python experiments/run_cagcn.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --runs 5 --backbone GraphSAGE
```
 
### CalGNN（`run_calgnn.py`）
 
```bash
# Elliptic
python experiments/run_calgnn.py --dataset elliptic --data_dir ./data/elliptic --runs 5
python experiments/run_calgnn.py --dataset elliptic --data_dir ./data/elliptic --backbone GAT --runs 5
python experiments/run_calgnn.py --dataset elliptic --data_dir ./data/elliptic --backbone GraphSAGE --runs 5
 
# OGB-Arxiv
python experiments/run_calgnn.py --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
python experiments/run_calgnn.py --dataset arxiv --data_path ./data/arxiv/data.pkl --backbone GAT --runs 5
python experiments/run_calgnn.py --dataset arxiv --data_path ./data/arxiv/data.pkl --backbone GraphSAGE --runs 5
 
# EERM-Cora
python experiments/run_calgnn.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --runs 5
python experiments/run_calgnn.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --backbone GAT --runs 5
python experiments/run_calgnn.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --backbone GraphSAGE --runs 5
 
# EERM-Amazon
python experiments/run_calgnn.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --runs 5 --backbone GAT
python experiments/run_calgnn.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --runs 5 --backbone GraphSAGE
```
 
### G-ΔUQ（`run_gduq.py`）
 
```bash
# Elliptic
python experiments/run_gduq.py --dataset elliptic --data_dir ./data/elliptic --runs 5
python experiments/run_gduq.py --dataset elliptic --data_dir ./data/elliptic --backbone GAT --runs 5
python experiments/run_gduq.py --dataset elliptic --data_dir ./data/elliptic --backbone GraphSAGE --runs 5
 
# OGB-Arxiv
python experiments/run_gduq.py --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
python experiments/run_gduq.py --dataset arxiv --data_path ./data/arxiv/data.pkl --backbone GAT --runs 5
python experiments/run_gduq.py --dataset arxiv --data_path ./data/arxiv/data.pkl --backbone GraphSAGE --runs 5
 
# EERM-Cora
python experiments/run_gduq.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --runs 5
python experiments/run_gduq.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --backbone GAT --runs 5
python experiments/run_gduq.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --backbone GraphSAGE --runs 5
 
# EERM-Amazon
python experiments/run_gduq.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --runs 5 --backbone GAT
python experiments/run_gduq.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --runs 5 --backbone GraphSAGE
```
 
### GPN（`run_gpn.py`）—— 无 backbone 参数
 
```bash
# GPN 使用 normalizing flow encoder，不依赖 GCN/GAT/SAGE backbone
python experiments/run_gpn.py --dataset elliptic --data_dir ./data/elliptic --runs 5
python experiments/run_gpn.py --dataset arxiv --data_path ./data/arxiv/data.pkl --runs 5
python experiments/run_gpn.py --dataset eerm --eerm_dataset cora \
    --eerm_root ./data/eerm/Planetoid/cora --runs 5
python experiments/run_gpn.py --dataset eerm --eerm_dataset amazon \
    --eerm_root ./data/eerm/Amazon/Photo --runs 5
```
 
---
 
## 逐算法命令（Facebook100 / Twitch）
 
所有 `_fb_twitch` 脚本均支持 `--backbone GCN / GAT / GraphSAGE`（默认 `GCN`）。
 
### S-BGCN-T-K（`run_ungnn_fb_twitch.py`）
 
```bash
python experiments/run_ungnn_fb_twitch.py --dataset twitch   --data_root ./data --runs 5
python experiments/run_ungnn_fb_twitch.py --dataset twitch   --data_root ./data --backbone GAT --runs 5
python experiments/run_ungnn_fb_twitch.py --dataset twitch   --data_root ./data --backbone GraphSAGE --runs 5
python experiments/run_ungnn_fb_twitch.py --dataset facebook --data_root ./data --runs 3
python experiments/run_ungnn_fb_twitch.py --dataset facebook --data_root ./data --backbone GAT --runs 3
python experiments/run_ungnn_fb_twitch.py --dataset facebook --data_root ./data --backbone GraphSAGE --runs 3
```
 
### GATS（`run_gats_fb_twitch.py`）
 
```bash
python experiments/run_gats_fb_twitch.py --dataset twitch   --data_root ./data --runs 5
python experiments/run_gats_fb_twitch.py --dataset twitch   --data_root ./data --backbone GAT --runs 5
python experiments/run_gats_fb_twitch.py --dataset twitch   --data_root ./data --backbone GraphSAGE --runs 5
python experiments/run_gats_fb_twitch.py --dataset facebook --data_root ./data --runs 3
python experiments/run_gats_fb_twitch.py --dataset facebook --data_root ./data --backbone GAT --runs 3
python experiments/run_gats_fb_twitch.py --dataset facebook --data_root ./data --backbone GraphSAGE --runs 3
```
 
### CaGCN（`run_cagcn_fb_twitch.py`）
 
```bash
python experiments/run_cagcn_fb_twitch.py --dataset twitch   --data_root ./data --runs 5
python experiments/run_cagcn_fb_twitch.py --dataset twitch   --data_root ./data --backbone GAT --runs 5
python experiments/run_cagcn_fb_twitch.py --dataset twitch   --data_root ./data --backbone GraphSAGE --runs 5
python experiments/run_cagcn_fb_twitch.py --dataset facebook --data_root ./data --runs 3
python experiments/run_cagcn_fb_twitch.py --dataset facebook --data_root ./data --backbone GAT --runs 3
python experiments/run_cagcn_fb_twitch.py --dataset facebook --data_root ./data --backbone GraphSAGE --runs 3
```
 
### CalGNN（`run_calgnn_fb_twitch.py`）
 
```bash
python experiments/run_calgnn_fb_twitch.py --dataset twitch   --data_root ./data --runs 5
python experiments/run_calgnn_fb_twitch.py --dataset twitch   --data_root ./data --backbone GAT --runs 5
python experiments/run_calgnn_fb_twitch.py --dataset twitch   --data_root ./data --backbone GraphSAGE --runs 5
python experiments/run_calgnn_fb_twitch.py --dataset facebook --data_root ./data --runs 3
python experiments/run_calgnn_fb_twitch.py --dataset facebook --data_root ./data --backbone GAT --runs 3
python experiments/run_calgnn_fb_twitch.py --dataset facebook --data_root ./data --backbone GraphSAGE --runs 3
```
 
### G-ΔUQ（`run_gduq_fb_twitch.py`）
 
```bash
python experiments/run_gduq_fb_twitch.py --dataset twitch   --data_root ./data --runs 5
python experiments/run_gduq_fb_twitch.py --dataset twitch   --data_root ./data --backbone GAT --runs 5
python experiments/run_gduq_fb_twitch.py --dataset twitch   --data_root ./data --backbone GraphSAGE --runs 5
python experiments/run_gduq_fb_twitch.py --dataset facebook --data_root ./data --runs 3
python experiments/run_gduq_fb_twitch.py --dataset facebook --data_root ./data --backbone GAT --runs 3
python experiments/run_gduq_fb_twitch.py --dataset facebook --data_root ./data --backbone GraphSAGE --runs 3
```
 
### GPN（`run_gpn_fb_twitch.py`）—— 无 backbone 参数
 
```bash
python experiments/run_gpn_fb_twitch.py --dataset twitch   --data_root ./data --runs 3
python experiments/run_gpn_fb_twitch.py --dataset facebook --data_root ./data --runs 3
```
 
---
 
## 结果可视化
 
```bash
python plot_results.py --data_dir ./results --out_dir ./figures --dataset elliptic
python plot_results.py --data_dir ./results --out_dir ./figures --dataset arxiv
python plot_results.py --data_dir ./results --out_dir ./figures --dataset twitch
python plot_results.py --data_dir ./results --out_dir ./figures --dataset facebook
```
 
---
 
## 项目结构
 
```
graphuq-bench/
├── src/
│   └── gnn_uq_bench/
│       ├── __init__.py
│       ├── metrics.py              # RQ1/RQ2/RQ3 全量指标 + CalGNN 专用汇总
│       ├── datasets.py             # Elliptic / OGB-Arxiv / EERM 数据加载
│       ├── datasets_fb_twitch.py   # Facebook100 / Twitch 数据加载
│       ├── models.py               # 六种模型 + GCN/GAT/SAGE backbone 工厂函数
│       └── calibration.py          # TS / HB / Iso / BBQ / MetaCal / RBS
│
├── experiments/
│   ├── run_ungnn.py                ┐
│   ├── run_gats.py                 │
│   ├── run_gpn.py                  │ Elliptic / Arxiv / EERM
│   ├── run_gduq.py                 │ 支持 --backbone（GPN 除外）
│   ├── run_calgnn.py               │
│   ├── run_cagcn.py                ┘
│   ├── run_ungnn_fb_twitch.py      ┐
│   ├── run_gats_fb_twitch.py       │
│   ├── run_gpn_fb_twitch.py        │ Facebook100 / Twitch
│   ├── run_gduq_fb_twitch.py       │ 支持 --backbone（GPN 除外）
│   ├── run_calgnn_fb_twitch.py     │
│   ├── run_cagcn_fb_twitch.py      ┘
│   ├── run_all.sh                  # 新增第4参数 BACKBONE
│   └── run_all_fb_twitch.sh        # 新增第4参数 BACKBONE
│
├── data/                           # 数据集（不纳入 git）
├── results/                        # 实验结果 CSV（自动生成）
├── figures/                        # 可视化图表（自动生成）
├── plot_results.py
├── pyproject.toml
├── .gitignore
└── README.md
```
 
---
 
## 输出格式
 
每个实验脚本在 `results/<alg>/` 下生成三个 CSV，文件名含 backbone 标识（如 `elliptic_gat_ungnn_results.csv`）：
 
| 文件 | 内容 |
|------|------|
| `*_results.csv` | 每个 split 的均值 ± 标准差 |
| `*_reliability.csv` | Reliability Diagram 原始数据（M=15 bins） |
| `*_uncertainty.csv` | 逐节点不确定性分数（第一个 seed） |
 
CalGNN 额外按校准方法分别输出（文件名含 `_Uncal_` / `_RBS_`）。
 
> **多 backbone 结果不互相覆盖**：GCN/GAT/GraphSAGE 的输出文件名分别含 `_gcn_` / `_gat_` / `_graphsage_` 标识。
 
---
 
## 主要参数
 
### 通用参数（所有脚本）
 
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backbone` | `GCN` | GNN backbone：`GCN` / `GAT` / `GraphSAGE`（GPN 除外） |
| `--runs` | 5 | 重复实验次数 |
| `--base_seed` | 42 | 随机种子起点 |
| `--hidden` | 64 | GNN 隐层维度 |
| `--dropout` | 0.5 | Dropout 比例 |
| `--lr` | 0.01 | 学习率 |
| `--epochs` | 2000 | 最大训练轮数 |
| `--patience` | 100 | Early stopping 容忍轮数 |
| `--save_dir` | `./results/<alg>` | 保存路径 |
 
### GATS 专有参数
 
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gats_heads` | 8 | CalibAttentionLayer 注意力头数 |
| `--gats_bias` | 1.0 | 初始温度偏置 |
| `--gats_epochs` | 200 | GATS 温度层训练轮数 |
| `--bfs_depth` | 2 | BFS 距离计算最大跳数 |
 
### GPN 专有参数
 
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dim_latent` | 10 | Latent space 维度 |
| `--radial_layers` | 10 | Normalizing flow 层数 |
| `--K` | 10 | APPNP 传播步数 |
| `--alpha_teleport` | 0.2 | APPNP teleport 概率 |
 
### G-ΔUQ 专有参数
 
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_anchors` | 30 | 推理时锚点采样数 |
| `--anchor_type` | `node` | 锚点类型：`node`（per-node）/ `graph`（shared） |
| `--num_layers` | 2 | GNN encoder 层数 |
 
### CaGCN 专有参数
 
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr_for_cal` | 0.01 | 温度网络学习率 |
| `--l2_for_cal` | 5e-4 | 温度网络 weight decay |
| `--Lambda` | 0.5 | Intra-class loss 权重 |
| `--stage` | 1 | 自训练阶段数 |
| `--threshold` | 0.8 | 伪标签置信度阈值 |
 
### CalGNN 专有参数
 
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_bins_rbs` | 10 | RBS 温度分箱数 |
| `--add_cal_loss` | `False` | 是否加入校准正则项 |
 
---
 
## Backbone 设计说明
 
### 稀疏adj路径（Elliptic / Arxiv / EERM）
 
`GATSparse` 和 `SAGESparse` 内部通过 `_to_edge_index()` 自动将 `torch.sparse_coo_tensor` 转换为 `edge_index`，因此对所有上层调用代码**完全透明**——传入 sparse adj 或 edge_index 均可。
 
```python
# models.py 工厂函数
from gnn_uq_bench.models import build_sparse_backbone, build_pyg_backbone
 
# 稀疏adj路径
model = build_sparse_backbone('GAT', nfeat=128, nhid=64, nclass=2, dropout=0.5)
logits = model(feat, adj)          # adj 可以是 sparse_adj 或 edge_index
 
# PyG edge_index 路径
model = build_pyg_backbone('GraphSAGE', nfeat=128, nhid=64, nclass=2, dropout=0.5)
logits = model(feat, edge_index)
```
 
### CaGCN 的温度网络
 
无论 backbone 是 GCN/GAT/SAGE，CaGCN 的温度缩放网络始终使用两层 `GCNConv`（与原论文一致）。这通过 `CaGCNFlex` 类实现，backbone 与温度网络完全解耦。
 
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

## License

Apache 2.0

## License

Apache 2.0
