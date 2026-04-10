# GNN-UQ-Bench

> **图神经网络不确定性量化基准库**
> Uncertainty Quantification Benchmarking for Graph Neural Networks under Distribution Shift

---

## 支持算法

| 算法 | 不确定性分数 | 原论文 |
|------|------------|--------|
| **S-BGCN-T-K** (Vanilla GCN) | 预测熵 H[p] | — |
| **GATS** | 1 - max(p) + 拓扑温度 | — |
| **GPN** | 1 / α.sum()（Dirichlet evidence） | Stadler et al., NeurIPS 2021 |
| **G-ΔUQ** | 随机锚点 sigmoid std | Thiagarajan et al., NeurIPS 2022 |
| **CalGNN-Uncal** | 1 - max(p) | — |
| **CalGNN-RBS** | 图拓扑感知温度缩放 | — |
| **CaGCN** | 1 - max(p)（图卷积温度缩放后） | Wang et al., NeurIPS 2021 |

## 支持数据集

| 数据集 | OOD 类型 | OOD split 数 |
|--------|---------|-------------|
| **Elliptic Bitcoin** | 时序偏移（time step） | 9 |
| **OGB-Arxiv** | 时序偏移（year） | 3 |
| **EERM-Cora** | 环境偏移（environment） | 8 |
| **EERM-Amazon-Photo** | 环境偏移（environment） | 8 |

## 评估指标

| 研究问题 | 指标 |
|---------|------|
| **RQ1 校准** | ECE (M=15), NLL, Brier; OOD: Δ-ECE, Δ-NLL, Δ-Brier |
| **RQ2 不确定性估计** | UE-AUROC, UE-AUPR, OOD-AUROC |
| **RQ3 选择性分类** | AURC, Risk@τ (τ∈{0.1,…,1.0}), SRTR@τ, SRTR-AUC |

---

## 安装

```bash
git clone https://github.com/YOUR_USERNAME/gnn-uq-bench
cd gnn-uq-bench
pip install -e .
```

依赖：`torch>=2.0`, `torch-geometric>=2.4`, `scipy`, `scikit-learn`, `numpy`

---

## 快速开始

### 运行单个算法

```bash
# S-BGCN-T-K（Vanilla GCN）
python experiments/run_ungnn.py --dataset elliptic --data_dir ./elliptic --runs 5

# GATS
python experiments/run_gats.py --dataset elliptic --data_dir ./elliptic --runs 5

# GPN
python experiments/run_gpn.py --dataset elliptic --data_dir ./elliptic --runs 5

# G-ΔUQ
python experiments/run_gduq.py --dataset elliptic --data_dir ./elliptic --runs 5

# CalGNN（Uncal + RBS 两种校准方法）
python experiments/run_calgnn.py --dataset elliptic --data_dir ./elliptic --runs 5

# CaGCN
python experiments/run_cagcn.py --dataset elliptic --data_dir ./elliptic --runs 5
```

### 一键运行全部算法

```bash
bash experiments/run_all.sh elliptic ./elliptic 5
bash experiments/run_all.sh arxiv    ./data.pkl 5
bash experiments/run_all.sh eerm     ./cora     5 cora
bash experiments/run_all.sh eerm     ./amazon   5 amazon
```

### OGB-Arxiv

```bash
python experiments/run_ungnn.py --dataset arxiv --data_path ./data.pkl --runs 5
```

### EERM（Cora / Amazon-Photo）

```bash
python experiments/run_ungnn.py \
    --dataset eerm --eerm_dataset cora \
    --eerm_root ./EERM/Planetoid/cora --runs 5

python experiments/run_ungnn.py \
    --dataset eerm --eerm_dataset amazon \
    --eerm_root ./EERM/Amazon/Photo --runs 5
```

---

## 结果可视化

```bash
python plot_results.py --data_dir ./results --out_dir ./figures --dataset elliptic
```

生成的图表保存在 `./figures/`，包含：
- 校准对比（ECE / NLL bar chart）
- OOD ECE / Δ-ECE 趋势线
- UE-AUROC 对比
- AURC 选择性分类
- OOD-AUROC 趋势

---

## 项目结构

```
gnn-uq-bench/
├── src/gnn_uq_bench/
│   ├── metrics.py       # 全量 RQ1/RQ2/RQ3 指标
│   ├── datasets.py      # Elliptic / Arxiv / EERM 数据加载
│   ├── models.py        # GCNSparse, GATS, GPN, G-ΔUQ, CaGCN...
│   └── calibration.py   # TS, HB, Iso, BBQ, MetaCal, RBS
├── experiments/
│   ├── run_ungnn.py     # S-BGCN-T-K
│   ├── run_gats.py      # GATS
│   ├── run_gpn.py       # GPN
│   ├── run_gduq.py      # G-ΔUQ
│   ├── run_calgnn.py    # CalGNN (Uncal/RBS)
│   ├── run_cagcn.py     # CaGCN
│   └── run_all.sh       # 一键运行
├── plot_results.py      # 结果可视化
└── pyproject.toml
```

---

## 输出 CSV 格式

每个算法运行完毕后在 `results/<alg>/` 下生成三个 CSV：

| 文件 | 内容 |
|------|------|
| `*_results.csv` | 每个 split 的均值±标准差（主要结果） |
| `*_reliability.csv` | Reliability Diagram 原始数据 |
| `*_uncertainty.csv` | 逐节点不确定性分数 |

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{your2025gnuqbench,
  title   = {GNN-UQ-Bench: Uncertainty Quantification Benchmarking for Graph Neural Networks},
  author  = {},
  year    = {2026},
}
```

## License

Apache 2.0
