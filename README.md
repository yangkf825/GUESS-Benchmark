<img width="1424" height="336" alt="image" src="https://github.com/user-attachments/assets/8b670259-327b-4ffd-ae8d-422a0d4a2fee" />
# GNN-UQ-Bench

> **Graph Neural Network Uncertainty Quantification Benchmark**
> Uncertainty Quantification Benchmarking for Graph Neural Networks under Distribution Shift

---

## Supported Algorithms

| Algorithm                    | Uncertainty Score                                          | Original Paper                   |
| ---------------------------- | ---------------------------------------------------------- | -------------------------------- |
| **S-BGCN-T-K** (Vanilla GCN) | Predictive entropy H[p]                                    | —                                |
| **GATS**                     | 1 - max(p) + topology temperature                          | —                                |
| **GPN**                      | 1 / α.sum() (Dirichlet evidence)                           | Stadler et al., NeurIPS 2021     |
| **G-ΔUQ**                    | Random anchor sigmoid std                                  | Thiagarajan et al., NeurIPS 2022 |
| **CalGNN-Uncal**             | 1 - max(p)                                                 | —                                |
| **CalGNN-RBS**               | Graph topology-aware temperature scaling                   | —                                |
| **CaGCN**                    | 1 - max(p) (after graph convolutional temperature scaling) | Wang et al., NeurIPS 2021        |

## Supported Datasets

| Dataset               | OOD Type                          | Number of OOD Splits |
| --------------------- | --------------------------------- | -------------------- |
| **Elliptic Bitcoin**  | Temporal shift (time step)        | 9                    |
| **OGB-Arxiv**         | Temporal shift (year)             | 3                    |
| **EERM-Cora**         | Environmental shift (environment) | 8                    |
| **EERM-Amazon-Photo** | Environmental shift (environment) | 8                    |

## Evaluation Metrics

| Research Question                | Metrics                                            |
| -------------------------------- | -------------------------------------------------- |
| **RQ1 Calibration**              | ECE (M=15), NLL, Brier; OOD: Δ-ECE, Δ-NLL, Δ-Brier |
| **RQ2 Uncertainty Estimation**   | UE-AUROC, UE-AUPR, OOD-AUROC                       |
| **RQ3 Selective Classification** | AURC, Risk@τ (τ∈{0.1,…,1.0}), SRTR@τ, SRTR-AUC     |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/gnn-uq-bench
cd gnn-uq-bench
pip install -e .
```

Dependencies: `torch>=2.0`, `torch-geometric>=2.4`, `scipy`, `scikit-learn`, `numpy`

---

## Quick Start

### Run a Single Algorithm

```bash
# S-BGCN-T-K (Vanilla GCN)
python experiments/run_ungnn.py --dataset elliptic --data_dir ./elliptic --runs 5

# GATS
python experiments/run_gats.py --dataset elliptic --data_dir ./elliptic --runs 5

# GPN
python experiments/run_gpn.py --dataset elliptic --data_dir ./elliptic --runs 5

# G-ΔUQ
python experiments/run_gduq.py --dataset elliptic --data_dir ./elliptic --runs 5

# CalGNN (Uncal + RBS two calibration methods)
python experiments/run_calgnn.py --dataset elliptic --data_dir ./elliptic --runs 5

# CaGCN
python experiments/run_cagcn.py --dataset elliptic --data_dir ./elliptic --runs 5
```

### Run All Algorithms with One Command

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

### EERM (Cora / Amazon-Photo)

```bash
python experiments/run_ungnn.py \
    --dataset eerm --eerm_dataset cora \
    --eerm_root ./EERM/Planetoid/cora --runs 5

python experiments/run_ungnn.py \
    --dataset eerm --eerm_dataset amazon \
    --eerm_root ./EERM/Amazon/Photo --runs 5
```

---

## Result Visualization

```bash
python plot_results.py --data_dir ./results --out_dir ./figures --dataset elliptic
```

Generated figures are saved in `./figures/`, including:

* Calibration comparison (ECE / NLL bar chart)
* OOD ECE / Δ-ECE trend curves
* UE-AUROC comparison
* AURC selective classification
* OOD-AUROC trends

---

## Project Structure

```
gnn-uq-bench/
├── src/gnn_uq_bench/
│   ├── metrics.py       # Full RQ1/RQ2/RQ3 metrics
│   ├── datasets.py      # Elliptic / Arxiv / EERM data loading
│   ├── models.py        # GCNSparse, GATS, GPN, G-ΔUQ, CaGCN...
│   └── calibration.py   # TS, HB, Iso, BBQ, MetaCal, RBS
├── experiments/
│   ├── run_ungnn.py     # S-BGCN-T-K
│   ├── run_gats.py      # GATS
│   ├── run_gpn.py       # GPN
│   ├── run_gduq.py      # G-ΔUQ
│   ├── run_calgnn.py    # CalGNN (Uncal/RBS)
│   ├── run_cagcn.py     # CaGCN
│   └── run_all.sh       # Run all with one command
├── plot_results.py      # Result visualization
└── pyproject.toml
```

---

## Output CSV Format

After each algorithm finishes, three CSV files are generated under `results/<alg>/`:

| File                | Content                                                 |
| ------------------- | ------------------------------------------------------- |
| `*_results.csv`     | Mean ± standard deviation for each split (main results) |
| `*_reliability.csv` | Raw data for Reliability Diagram                        |
| `*_uncertainty.csv` | Node-level uncertainty scores                           |

---

## Citation

If this project is helpful to your research, please cite:

```bibtex
@article{your2025gnuqbench,
  title   = {GNN-UQ-Bench: Uncertainty Quantification Benchmarking for Graph Neural Networks},
  author  = {},
  year    = {2026},
}
```

## License

Apache 2.0
