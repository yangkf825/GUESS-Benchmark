"""
GNN-UQ-Bench
============
图神经网络不确定性量化基准库。

六种算法：
  S-BGCN-T-K  — 熵作为不确定性的 vanilla GCN
  GATS         — 图注意力温度缩放
  GPN          — 图后验网络（Dirichlet 不确定性）
  G-ΔUQ        — 随机锚点方差不确定性
  CalGNN       — GCN + 六种后处理校准
  CaGCN        — 图卷积温度缩放校准

四个数据集：Elliptic / OGB-Arxiv / EERM-Cora / EERM-Amazon-Photo
"""

__version__ = "0.1.0"
