"""
gnn_uq_bench.models
====================
六种 GNN-UQ 模型：
  GCNSparse   — 稀疏矩阵 GCN（CaGCN / CalGNN base）
  GCNPyG      — PyG GCNConv（GATS base / GDUQ base）
  GATModel    — PyG GATConv
  GATS        — Graph Attention Temperature Scaling
  GPNModel    — Graph Posterior Network
  GraphANTNode— G-ΔUQ 随机锚点模型
  CaGCN       — 图卷积温度缩放校准
"""

import math
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam


# ════════════════════════════════════════════════════════════════════════════
# 1. GCN (稀疏 adj，用于 CaGCN / CalGNN)
# ════════════════════════════════════════════════════════════════════════════

class GraphConvolution(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.W = Parameter(torch.FloatTensor(in_f, out_f))
        self.b = Parameter(torch.FloatTensor(out_f))
        s = 1. / out_f ** 0.5
        self.W.data.uniform_(-s, s); self.b.data.uniform_(-s, s)

    def forward(self, x, adj):
        return torch.spmm(adj, x @ self.W) + self.b


class GCNSparse(nn.Module):
    """双层稀疏 GCN，接受 torch.sparse adj"""
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dp  = nn.Dropout(dropout)

    def reset_parameters(self):
        for gc in [self.gc1, self.gc2]:
            s = 1. / gc.W.shape[1] ** 0.5
            gc.W.data.uniform_(-s, s); gc.b.data.uniform_(-s, s)

    def forward(self, x, adj):
        return self.gc2(self.dp(F.relu(self.gc1(x, adj))), adj)


# ════════════════════════════════════════════════════════════════════════════
# 2. GCN / GAT (PyG edge_index，用于 GATS / GDUQ base)
# ════════════════════════════════════════════════════════════════════════════

class GCNPyG(nn.Module):
    """双层 PyG GCNConv"""
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
        self.dp    = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters(); self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dp, self.training)
        return self.conv2(x, edge_index)


class GATModel(nn.Module):
    """双层 PyG GATConv"""
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, heads=8):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.conv1 = GATConv(nfeat, nhid, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(nhid * heads, nclass, heads=1, dropout=dropout, concat=False)
        self.dp    = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters(); self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dp, self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dp, self.training)
        return self.conv2(x, edge_index)


# ════════════════════════════════════════════════════════════════════════════
# 3. CaGCN（图卷积温度缩放）
# ════════════════════════════════════════════════════════════════════════════

class CaGCN(nn.Module):
    """
    CaGCN: 两层图卷积学习 per-node 温度 T(v)，base 权重冻结。
    输出 logits * T(v)
    """
    def __init__(self, nclass, base_model):
        super().__init__()
        self.base = base_model
        self.s1   = GraphConvolution(nclass, 16)
        self.s2   = GraphConvolution(16, 1)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x, adj):
        logits = self.base(x, adj)
        t = torch.log(torch.exp(self.s2(F.relu(self.s1(logits, adj)), adj)) + 1.1)
        return logits * t


# ════════════════════════════════════════════════════════════════════════════
# 4. GATS（Graph Attention Temperature Scaling）
# ════════════════════════════════════════════════════════════════════════════

class CalibAttentionLayer(torch.nn.Module):
    """拓扑感知的 per-node 温度层（GATS 核心）"""
    def __init__(self, in_channels, edge_index, num_nodes, dist_to_train,
                 heads=8, negative_slope=0.2, bias=1.0):
        super().__init__()
        from torch_geometric.nn.conv import MessagePassing
        from torch_geometric.nn.dense.linear import Linear as PyGLinear
        from torch_geometric.utils import remove_self_loops, add_self_loops, degree

        self._MP = MessagePassing
        self.in_channels    = in_channels
        self.heads          = heads
        self.negative_slope = negative_slope
        self.edge_index     = edge_index
        self.num_nodes      = num_nodes
        self.temp_lin  = PyGLinear(in_channels, heads, bias=False, weight_initializer='glorot')
        self.conf_coef = Parameter(torch.zeros([]))
        self.bias      = Parameter(torch.ones(1) * bias)
        self.train_a   = Parameter(torch.ones(1))
        self.dist1_a   = Parameter(torch.ones(1))
        self.register_buffer('dist_to_train', dist_to_train)

        # 加自环
        ei, _ = remove_self_loops(self.edge_index, None)
        self.edge_index, _ = add_self_loops(ei, None, fill_value='mean',
                                            num_nodes=num_nodes)

        from torch_geometric.utils import softmax as pyg_softmax, degree as pyg_degree
        self._softmax = pyg_softmax
        self._degree  = pyg_degree

    def forward(self, x):
        from torch_geometric.nn.conv import MessagePassing

        N, H   = self.num_nodes, self.heads
        xn     = x - x.min(1, keepdim=True)[0]
        denom  = (x.max(1, keepdim=True)[0] - x.min(1, keepdim=True)[0]).clamp(min=1e-8)
        x_sort = torch.sort(xn / denom, dim=-1)[0]
        temp   = self.temp_lin(x_sort)
        a_clus = torch.ones(N, dtype=torch.float32, device=x.device)
        a_clus[self.dist_to_train == 0] = self.train_a
        a_clus[self.dist_to_train == 1] = self.dist1_a
        conf   = F.softmax(x, dim=1).amax(-1)
        deg    = self._degree(self.edge_index[0], N)
        deg_inv = (1. / deg).clamp(max=1e9)
        deg_inv[deg_inv == float('inf')] = 0.

        # 手动 message passing
        src, dst = self.edge_index
        temp_src = temp.view(N, H) * a_clus.unsqueeze(-1)
        alpha_x  = x / a_clus.unsqueeze(-1).clamp(min=1e-8)

        # alpha = leaky_relu((alpha_src * alpha_dst).sum(-1))
        alpha = (alpha_x[src] * alpha_x[dst]).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = self._softmax(alpha, dst, num_nodes=N)

        msg_temp = temp_src[src] * alpha.unsqueeze(-1).expand(-1, H)
        msg_conf = (conf[dst] - conf[src]).unsqueeze(-1)
        msg      = torch.cat([msg_temp, msg_conf], dim=-1)   # (E, H+1)

        out = torch.zeros(N, H + 1, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, H + 1), msg)

        sim, dconf = out[:, :-1], out[:, -1:]
        out = F.softplus(sim + self.conf_coef * dconf * deg_inv.unsqueeze(-1))
        out = out.mean(dim=1) + self.bias
        return out.unsqueeze(1)


class GATS(nn.Module):
    """
    GATS: base 模型冻结，CalibAttentionLayer 学习 per-node 温度。
    forward 返回校准后 logits。
    """
    def __init__(self, base_model, edge_index, num_nodes, dist_to_train,
                 nclass, heads=8, bias=1.0):
        super().__init__()
        self.base      = base_model
        self.num_nodes = num_nodes
        self.cagat     = CalibAttentionLayer(
            nclass, edge_index, num_nodes, dist_to_train, heads=heads, bias=bias)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x, edge_index):
        logits = self.base(x, edge_index)
        t = self.cagat(logits).view(self.num_nodes, -1)
        t = t.expand(self.num_nodes, logits.size(1))
        return logits / t


def bfs_distance(edge_index, train_mask, max_hop, N, device):
    """BFS 计算每个节点到训练集的最近距离（≤max_hop）"""
    dist = torch.full((N,), max_hop, dtype=torch.long, device=device)
    dist[train_mask] = 0
    src, dst = edge_index[0], edge_index[1]
    for _ in range(max_hop):
        update = dist[src] + 1
        torch.minimum(dist[dst], update, out=dist[dst])
    return dist.clamp(max=max_hop)


# ════════════════════════════════════════════════════════════════════════════
# 5. GPN（Graph Posterior Network）
# ════════════════════════════════════════════════════════════════════════════

class RadialTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim         = dim
        self.x0          = nn.Parameter(torch.empty(dim))
        self.alpha_prime = nn.Parameter(torch.empty(1))
        self.beta_prime  = nn.Parameter(torch.empty(1))
        self._reset()

    def _reset(self):
        s = 1. / math.sqrt(self.dim)
        nn.init.uniform_(self.x0, -s, s)
        nn.init.uniform_(self.alpha_prime, -s, s)
        nn.init.uniform_(self.beta_prime,  -s, s)

    def forward(self, z):
        alpha  = F.softplus(self.alpha_prime)
        beta   = -alpha + F.softplus(self.beta_prime)
        diff   = z - self.x0.unsqueeze(0)
        r      = diff.norm(dim=-1, keepdim=True)
        h      = 1. / (alpha + r)
        bh     = beta * h
        z_new  = z + bh * diff
        hp     = -(h ** 2)
        log_det = ((self.dim - 1) * torch.log1p(bh.squeeze(-1))
                   + torch.log1p(bh.squeeze(-1) + beta * hp.squeeze(-1) * r.squeeze(-1)))
        return z_new, log_det


class PerClassRadialFlow(nn.Module):
    def __init__(self, num_classes, dim, num_layers):
        super().__init__()
        self.num_classes = num_classes
        self.flows = nn.ModuleList([
            nn.ModuleList([RadialTransform(dim) for _ in range(num_layers)])
            for _ in range(num_classes)
        ])
        self.mu = nn.Parameter(torch.zeros(num_classes, dim))

    def forward(self, z):
        N = z.size(0)
        log_q = torch.zeros(N, self.num_classes, device=z.device)
        for c in range(self.num_classes):
            zc = z; ld_sum = torch.zeros(N, device=z.device)
            for radial in self.flows[c]:
                zc, ld = radial(zc); ld_sum += ld
            diff = zc - self.mu[c].unsqueeze(0)
            log_pz = -0.5 * (self.num_classes * math.log(2 * math.pi) + (diff * diff).sum(-1))
            # use dim not num_classes
            log_pz = -0.5 * (self.flows[0][0].dim * math.log(2 * math.pi) + (diff * diff).sum(-1))
            log_q[:, c] = log_pz + ld_sum
        return log_q


class APPNPProp(nn.Module):
    def __init__(self, K, alpha):
        super().__init__()
        self.K = K; self.alpha = alpha

    def forward(self, x, edge_index, edge_weight):
        h0 = x; row, col = edge_index
        for _ in range(self.K):
            out = torch.zeros_like(x)
            out.scatter_add_(0, col.unsqueeze(-1).expand(-1, x.size(1)),
                             edge_weight.unsqueeze(-1) * x[row])
            x = (1 - self.alpha) * out + self.alpha * h0
        return x


def sym_norm_edge(edge_index, N, add_self_loops=True):
    from torch_geometric.utils import add_self_loops as _asl
    if add_self_loops:
        edge_index, _ = _asl(edge_index, num_nodes=N)
    row, col = edge_index
    deg = torch.zeros(N, device=edge_index.device)
    deg.scatter_add_(0, col, torch.ones(edge_index.size(1), device=edge_index.device))
    di  = deg.pow(-0.5); di[di == float('inf')] = 0.
    return edge_index, di[row] * di[col]


class GPNModel(nn.Module):
    def __init__(self, dim_features, num_classes,
                 dim_hidden=64, dim_latent=10, radial_layers=10,
                 K=10, alpha_teleport=0.2, dropout_prob=0.5,
                 alpha_evidence_scale='latent-new-plus-classes'):
        super().__init__()
        self.num_classes = num_classes
        self.dim_latent  = dim_latent
        self.alpha_evidence_scale = alpha_evidence_scale
        self.encoder = nn.Sequential(
            nn.Linear(dim_features, dim_hidden), nn.ReLU(), nn.Dropout(dropout_prob))
        self.latent_encoder = nn.Linear(dim_hidden, dim_latent)
        self.flow        = PerClassRadialFlow(num_classes, dim_latent, radial_layers)
        self.propagation = APPNPProp(K, alpha_teleport)
        scale = 0.5 * dim_latent * math.log(4 * math.pi)
        if 'plus-classes' in alpha_evidence_scale:
            scale += math.log(num_classes)
        self._log_scale = scale

    def forward(self, data, train_mask, edge_index, edge_weight):
        z       = self.latent_encoder(self.encoder(data.x))
        y_tr    = data.y[train_mask]
        counts  = torch.zeros(self.num_classes, device=y_tr.device)
        for c in range(self.num_classes):
            counts[c] = (y_tr == c).float().sum()
        p_c     = counts.clamp(min=1.) / counts.clamp(min=1.).sum()
        log_q   = self.flow(z)
        log_qj  = log_q + p_c.log().unsqueeze(0)
        beta    = (log_qj + self._log_scale).clamp(-30., 30.).exp()
        beta    = self.propagation(beta, edge_index, edge_weight)
        alpha   = 1. + beta
        soft    = alpha / alpha.sum(-1, keepdim=True)
        return dict(alpha=alpha, soft=soft, log_soft=soft.log(), hard=soft.argmax(-1))

    def get_optimizer(self, lr, wd, flow_lr, flow_wd):
        fids   = {id(p) for p in self.flow.parameters()}
        others = [p for p in self.parameters() if id(p) not in fids]
        return (Adam([{'params': others, 'lr': lr, 'weight_decay': wd},
                      {'params': list(self.flow.parameters()),
                       'lr': flow_lr, 'weight_decay': flow_wd}]),
                Adam(self.flow.parameters(), lr=flow_lr, weight_decay=flow_wd))


def gpn_uce_loss(alpha, y):
    return (torch.digamma(alpha.sum(-1))
            - torch.digamma(alpha.gather(-1, y.view(-1, 1)).squeeze(-1))).sum()

def gpn_entropy_reg(alpha, beta_reg):
    ent = torch.distributions.Dirichlet(alpha.clamp(min=1e-6)).entropy()
    return -beta_reg * ent.sum()

def gpn_ce_loss(log_soft, y):
    return F.nll_loss(log_soft, y, reduction='sum')


# ════════════════════════════════════════════════════════════════════════════
# 6. G-ΔUQ (GraphANTNode)
# ════════════════════════════════════════════════════════════════════════════

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, dim_hidden, num_layers, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import GCNConv
        dims = [in_dim] + [dim_hidden] * num_layers
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i+1]) for i in range(num_layers)])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(dims[i+1]) for i in range(num_layers)])
        self.drop  = dropout

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = bn(conv(h, edge_index, edge_weight))
            if i < len(self.convs) - 1: h = F.relu(h)
            h = F.dropout(h, self.drop, training=self.training)
        return h


class BaseModelNode(nn.Module):
    def __init__(self, in_dim, dim_hidden, num_classes, num_layers, dropout=0.5):
        super().__init__()
        self.encoder    = GCNEncoder(in_dim, dim_hidden, num_layers, dropout)
        self.classifier = nn.Linear(dim_hidden, num_classes)

    def forward_graph(self, x, edge_index, edge_weight=None):
        return self.classifier(self.encoder(x, edge_index, edge_weight))


class GraphANTNode(nn.Module):
    """
    G-ΔUQ 随机锚点模型。
    训练：单锚点前向；推理：n_anchors 锚点的 std 作为不确定性。
    """
    def __init__(self, base_net, mu, std, anchor_type='node', num_classes=2):
        super().__init__()
        self.net         = base_net
        self.anchor_type = anchor_type.lower()
        self.num_classes = num_classes
        self.register_buffer('mu',  mu)
        self.register_buffer('std', std)

    def _sample_anchor(self, N, device):
        d = torch.distributions.Normal(self.mu, self.std)
        if self.anchor_type == 'node':
            return d.sample([N]).to(device)
        return d.sample([1]).expand(N, -1).to(device)

    def forward(self, data):
        N      = data.x.shape[0]
        anchor = self._sample_anchor(N, data.x.device)
        new_x  = torch.cat([data.x - anchor, anchor], dim=1)
        return self.net.forward_graph(new_x, data.edge_index)

    @torch.no_grad()
    def infer(self, data, n_anchors=30):
        self.eval()
        N = data.x.shape[0]
        preds = []
        for _ in range(n_anchors):
            anchor = self._sample_anchor(N, data.x.device)
            new_x  = torch.cat([data.x - anchor, anchor], dim=1)
            preds.append(self.net.forward_graph(new_x, data.edge_index))
        stack   = torch.stack(preds, 0)
        mu_log  = stack.mean(0)
        sig_std = stack.sigmoid().std(0)
        c       = sig_std.mean(-1, keepdim=True).expand_as(mu_log)
        mu_cal  = mu_log / (1 + torch.exp(c))
        probs   = F.softmax(mu_cal, dim=1).cpu().numpy()
        std_np  = sig_std.mean(-1).cpu().numpy()   # (N,) scalar uncertainty
        return probs, std_np
