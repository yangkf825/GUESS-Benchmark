
"""
Dedicated GAT / GraphSAGE backbone module for the GNN-UQ benchmark.

This module is intentionally independent from ``gnn_uq_bench.models`` so the
new runners can coexist with the original GCN-only runners.

Available factories
-------------------
- get_pyg_backbone: forward(x, edge_index), 2-layer PyG backbone.
- get_pyg_backbone_bn: forward(x, edge_index), 3-layer BN/residual backbone
  for Facebook100/Twitch style domain experiments.
- get_sparse_backbone: forward(x, adj), sparse-adjacency backbone for CaGCN
  and other runners that use normalized torch.sparse adjacency.
- GraphANTNodeBackbone: G-DUQ compatible anchor wrapper with GAT/SAGE base.
- GPNBackboneModel: GPN-style evidential output wrapper with GAT/SAGE base.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.parameter import Parameter

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
except Exception:  # pragma: no cover
    GCNConv = None
    GATConv = None
    SAGEConv = None


def canonical_backbone_name(name: str) -> str:
    key = str(name).strip().lower().replace('-', '').replace('_', '')
    if key == 'gcn':
        return 'GCN'
    if key == 'gat':
        return 'GAT'
    if key in {'sage', 'graphsage'}:
        return 'SAGE'
    raise ValueError(f'Unsupported backbone {name!r}; use GCN, GAT, SAGE or GraphSAGE.')


def _need_pyg(cls_name: str) -> None:
    if GCNConv is None or GATConv is None or SAGEConv is None:
        raise ImportError(f'torch_geometric is required for {cls_name}.')


class PyGGCNBackbone(nn.Module):
    def __init__(self, nfeat: int, hidden: int, nclass: int, dropout: float):
        super().__init__(); _need_pyg(type(self).__name__)
        self.conv1 = GCNConv(nfeat, hidden)
        self.conv2 = GCNConv(hidden, nclass)
        self.dp = dropout
    def reset_parameters(self):
        self.conv1.reset_parameters(); self.conv2.reset_parameters()
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dp, self.training)
        return self.conv2(x, edge_index)


class PyGGATBackbone(nn.Module):
    def __init__(self, nfeat: int, hidden: int, nclass: int, dropout: float, heads: int = 8):
        super().__init__(); _need_pyg(type(self).__name__)
        self.dp = dropout; self.heads = heads
        self.conv1 = GATConv(nfeat, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden * heads, nclass, heads=1, dropout=dropout, concat=False)
    def reset_parameters(self):
        self.conv1.reset_parameters(); self.conv2.reset_parameters()
    def forward(self, x, edge_index):
        x = F.dropout(x, self.dp, self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dp, self.training)
        return self.conv2(x, edge_index)


class PyGGraphSAGEBackbone(nn.Module):
    def __init__(self, nfeat: int, hidden: int, nclass: int, dropout: float):
        super().__init__(); _need_pyg(type(self).__name__)
        self.conv1 = SAGEConv(nfeat, hidden)
        self.conv2 = SAGEConv(hidden, nclass)
        self.dp = dropout
    def reset_parameters(self):
        self.conv1.reset_parameters(); self.conv2.reset_parameters()
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dp, self.training)
        return self.conv2(x, edge_index)


class PyGGCNBNBackbone(nn.Module):
    """Three-layer GCN + BN used by Facebook100/Twitch scripts."""
    def __init__(self, nfeat: int, hidden: int, nclass: int, dropout: float):
        super().__init__(); _need_pyg(type(self).__name__)
        self.c1 = GCNConv(nfeat, hidden); self.bn1 = nn.BatchNorm1d(hidden)
        self.c2 = GCNConv(hidden, hidden); self.bn2 = nn.BatchNorm1d(hidden)
        self.c3 = GCNConv(hidden, nclass); self.dp = dropout
    def reset_parameters(self):
        for m in [self.c1, self.c2, self.c3]: m.reset_parameters()
        for b in [self.bn1, self.bn2]: b.reset_parameters()
    def forward(self, x, edge_index):
        x = F.dropout(F.relu(self.bn1(self.c1(x, edge_index))), self.dp, self.training)
        h = F.relu(self.bn2(self.c2(x, edge_index)) + x)
        h = F.dropout(h, self.dp, self.training)
        return self.c3(h, edge_index)


class PyGGATBNBackbone(nn.Module):
    """Three-layer GAT + BN/residual for domain experiments."""
    def __init__(self, nfeat: int, hidden: int, nclass: int, dropout: float, heads: int = 8):
        super().__init__(); _need_pyg(type(self).__name__)
        self.dp = dropout; self.heads = heads
        self.c1 = GATConv(nfeat, hidden, heads=heads, dropout=dropout, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.c2 = GATConv(hidden, hidden, heads=heads, dropout=dropout, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.c3 = GATConv(hidden, nclass, heads=1, dropout=dropout, concat=False)
    def reset_parameters(self):
        for m in [self.c1, self.c2, self.c3]: m.reset_parameters()
        for b in [self.bn1, self.bn2]: b.reset_parameters()
    def forward(self, x, edge_index):
        x = F.dropout(F.elu(self.bn1(self.c1(x, edge_index))), self.dp, self.training)
        h = F.elu(self.bn2(self.c2(x, edge_index)) + x)
        h = F.dropout(h, self.dp, self.training)
        return self.c3(h, edge_index)


class PyGGraphSAGEBNBackbone(nn.Module):
    """Three-layer GraphSAGE + BN/residual for domain experiments."""
    def __init__(self, nfeat: int, hidden: int, nclass: int, dropout: float):
        super().__init__(); _need_pyg(type(self).__name__)
        self.c1 = SAGEConv(nfeat, hidden); self.bn1 = nn.BatchNorm1d(hidden)
        self.c2 = SAGEConv(hidden, hidden); self.bn2 = nn.BatchNorm1d(hidden)
        self.c3 = SAGEConv(hidden, nclass); self.dp = dropout
    def reset_parameters(self):
        for m in [self.c1, self.c2, self.c3]: m.reset_parameters()
        for b in [self.bn1, self.bn2]: b.reset_parameters()
    def forward(self, x, edge_index):
        x = F.dropout(F.relu(self.bn1(self.c1(x, edge_index))), self.dp, self.training)
        h = F.relu(self.bn2(self.c2(x, edge_index)) + x)
        h = F.dropout(h, self.dp, self.training)
        return self.c3(h, edge_index)


def get_pyg_backbone(name: str, nfeat: int, hidden: int, nclass: int,
                     dropout: float, heads: int = 8) -> nn.Module:
    key = canonical_backbone_name(name)
    if key == 'GCN': return PyGGCNBackbone(nfeat, hidden, nclass, dropout)
    if key == 'GAT': return PyGGATBackbone(nfeat, hidden, nclass, dropout, heads=heads)
    return PyGGraphSAGEBackbone(nfeat, hidden, nclass, dropout)


def get_pyg_backbone_bn(name: str, nfeat: int, hidden: int, nclass: int,
                        dropout: float, heads: int = 8) -> nn.Module:
    key = canonical_backbone_name(name)
    if key == 'GCN': return PyGGCNBNBackbone(nfeat, hidden, nclass, dropout)
    if key == 'GAT': return PyGGATBNBackbone(nfeat, hidden, nclass, dropout, heads=heads)
    return PyGGraphSAGEBNBackbone(nfeat, hidden, nclass, dropout)


class SparseGraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = Parameter(torch.empty(in_features, out_features))
        self.b = Parameter(torch.empty(out_features))
        self.reset_parameters()
    def reset_parameters(self):
        s = 1.0 / (self.W.size(1) ** 0.5)
        nn.init.uniform_(self.W, -s, s); nn.init.uniform_(self.b, -s, s)
    def forward(self, x, adj):
        return torch.spmm(adj, x @ self.W) + self.b


class SparseGCNBackbone(nn.Module):
    def __init__(self, nfeat: int, hidden: int, nclass: int, dropout: float):
        super().__init__()
        self.gc1 = SparseGraphConvolution(nfeat, hidden)
        self.gc2 = SparseGraphConvolution(hidden, nclass)
        self.dp = nn.Dropout(dropout)
    def reset_parameters(self):
        self.gc1.reset_parameters(); self.gc2.reset_parameters()
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj)); x = self.dp(x)
        return self.gc2(x, adj)


class SparseGATBackbone(nn.Module):
    """Sparse-adjacency multi-head GAT-style backbone compatible with CaGCN."""
    def __init__(self, nfeat: int, hidden: int, nclass: int, dropout: float,
                 alpha: float = 0.2, nheads: int = 8):
        super().__init__()
        self.dp = dropout; self.alpha = alpha
        self.Ws = nn.ParameterList([Parameter(torch.empty(nfeat, hidden)) for _ in range(nheads)])
        self.out = SparseGraphConvolution(hidden * nheads, nclass)
        self.reset_parameters()
    def reset_parameters(self):
        for w in self.Ws: nn.init.xavier_normal_(w.data, gain=1.414)
        self.out.reset_parameters()
    def forward(self, x, adj):
        x_drop = F.dropout(x, self.dp, self.training)
        heads = [F.elu(torch.spmm(adj, x_drop @ w)) for w in self.Ws]
        h = torch.cat(heads, dim=1)
        h = F.dropout(h, self.dp, self.training)
        return self.out(h, adj)


class SparseSAGELayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin_self = nn.Linear(in_features, out_features, bias=True)
        self.lin_neigh = nn.Linear(in_features, out_features, bias=False)
    def reset_parameters(self):
        self.lin_self.reset_parameters(); self.lin_neigh.reset_parameters()
    def forward(self, x, adj):
        neigh = torch.spmm(adj, x)
        return self.lin_self(x) + self.lin_neigh(neigh)


class SparseGraphSAGEBackbone(nn.Module):
    def __init__(self, nfeat: int, hidden: int, nclass: int, dropout: float):
        super().__init__()
        self.sage1 = SparseSAGELayer(nfeat, hidden)
        self.sage2 = SparseSAGELayer(hidden, nclass)
        self.dp = nn.Dropout(dropout)
    def reset_parameters(self):
        self.sage1.reset_parameters(); self.sage2.reset_parameters()
    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj)); x = self.dp(x)
        return self.sage2(x, adj)


def get_sparse_backbone(name: str, nfeat: int, hidden: int, nclass: int,
                        dropout: float, alpha: float = 0.2, nheads: int = 8) -> nn.Module:
    key = canonical_backbone_name(name)
    if key == 'GCN': return SparseGCNBackbone(nfeat, hidden, nclass, dropout)
    if key == 'GAT': return SparseGATBackbone(nfeat, hidden, nclass, dropout, alpha=alpha, nheads=nheads)
    return SparseGraphSAGEBackbone(nfeat, hidden, nclass, dropout)


class GraphANTNodeBackbone(nn.Module):
    """G-DUQ anchor wrapper with graph backbone.

    It keeps the original runner contract: ``forward(data)`` returns logits for
    one sampled anchor and ``infer(data, n_anchors)`` returns mean probabilities
    and uncertainty. The uncertainty is the mean class-wise std over anchor
    predictions per node.
    """
    def __init__(self, backbone: nn.Module, mean: torch.Tensor, std: torch.Tensor,
                 anchor_type: str = 'node', num_classes: int | None = None):
        super().__init__()
        self.base = backbone
        self.register_buffer('mean', mean.float())
        self.register_buffer('std', std.float())
        self.anchor_type = anchor_type
        self.num_classes = num_classes
    def _anchors(self, x):
        if self.anchor_type == 'graph':
            a = torch.randn(1, self.mean.numel(), device=x.device) * self.std + self.mean
            return a.expand(x.size(0), -1)
        return torch.randn(x.size(0), self.mean.numel(), device=x.device) * self.std + self.mean
    def forward(self, data):
        x = data.x
        anchor = self._anchors(x)
        x_aug = torch.cat([x, anchor], dim=1)
        return self.base(x_aug, data.edge_index)
    @torch.no_grad()
    def infer(self, data, n_anchors: int = 30):
        self.eval()
        probs_list, score_list = [], []
        for _ in range(n_anchors):
            logits = self.forward(data)
            probs_list.append(F.softmax(logits, dim=1))
            score_list.append(torch.sigmoid(logits))
        probs_t = torch.stack(probs_list, dim=0)
        score_t = torch.stack(score_list, dim=0)
        probs = probs_t.mean(dim=0).cpu().numpy()
        u = score_t.std(dim=0).mean(dim=1).cpu().numpy()
        return probs, u


class GPNBackboneModel(nn.Module):
    """GPN-compatible evidential wrapper using GCN/GAT/SAGE backbone.

    The model returns the keys used by existing GPN runners: ``alpha``, ``soft``
    and ``log_soft``. It does not touch the runner's metrics or data protocol.
    """
    def __init__(self, dim_features: int, num_classes: int, dim_hidden: int = 64,
                 dim_latent: int = 16, radial_layers: int = 6, K: int = 10,
                 alpha_teleport: float = 0.1, dropout_prob: float = 0.5,
                 alpha_evidence_scale: str = 'latent-new-plus-classes',
                 backbone: str = 'GAT', heads: int = 8):
        super().__init__()
        self.backbone_name = canonical_backbone_name(backbone)
        self.encoder = get_pyg_backbone(backbone, dim_features, dim_hidden,
                                        num_classes, dropout_prob, heads=heads)
    def get_optimizer(self, lr=0.01, weight_decay=5e-4, flow_lr=0.01, flow_weight_decay=0.0):
        return (Adam(self.parameters(), lr=lr, weight_decay=weight_decay),
                Adam(self.parameters(), lr=flow_lr, weight_decay=flow_weight_decay))
    def forward(self, data, train_mask=None, edge_index=None, edge_weight=None):
        ei = edge_index if edge_index is not None else data.edge_index
        logits = self.encoder(data.x, ei)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        soft = alpha / alpha.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return {
            'alpha': alpha,
            'soft': soft,
            'log_soft': torch.log(soft.clamp(min=1e-8)),
            'logits': logits,
        }
