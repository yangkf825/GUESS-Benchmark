"""
GATS — Elliptic & OGB-Arxiv 完整指标版
========================================
【指标体系（实验文档 RQ1/RQ2/RQ3）】

RQ1: acc, ece(M=15), nll, brier; [Elliptic] f1/prec/rec
     [OOD] delta_ece, delta_nll, delta_brier
RQ2: ue_auroc, ue_aupr  (u = 1-max(p)，GATS 无自有不确定性分数)
     [OOD] delta_ue_auroc, ood_auroc
RQ3: aurc, risk@0.1~1.0
     [OOD] aurc_ood, srtr@0.1~1.0, srtr_auc

用法:
    python gats_elliptic_arxiv.py --dataset elliptic --data_dir  ./elliptic --runs 5
    python gats_elliptic_arxiv.py --dataset arxiv    --data_path ./data.pkl  --runs 5
"""
import sys; sys.path.insert(0, 'src')


import os, pickle, argparse, warnings, csv, copy, math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear as PyGLinear
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_geometric.nn import GCNConv, GATConv
from gnn_uq_bench.model_gat_sage import (canonical_backbone_name, get_pyg_backbone, get_pyg_backbone_bn, get_sparse_backbone, GraphANTNodeBackbone, GPNBackboneModel)

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 0. 参数
# ══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',       type=str,   default='elliptic',
                    choices=['elliptic', 'arxiv', 'eerm'])
parser.add_argument('--data_dir',      type=str,   default='./elliptic')
parser.add_argument('--data_path',     type=str,   default='./data.pkl')
parser.add_argument('--runs',          type=int,   default=5)
parser.add_argument('--model',         type=str,   default='GAT',
                    choices=['GCN', 'GAT', 'SAGE', 'GraphSAGE'],
                    help='backbone: GCN, GAT, SAGE/GraphSAGE')
parser.add_argument('--backbone_heads', type=int,   default=8,
                    help='GAT attention heads for the new backbone')
parser.add_argument('--hidden',        type=int,   default=64)
parser.add_argument('--dropout',       type=float, default=0.5)
parser.add_argument('--lr',            type=float, default=0.01)
parser.add_argument('--weight_decay',  type=float, default=5e-4)
parser.add_argument('--epochs',        type=int,   default=2000)
parser.add_argument('--patience',      type=int,   default=100)
parser.add_argument('--gats_heads',    type=int,   default=8)
parser.add_argument('--gats_bias',     type=float, default=1.0)
parser.add_argument('--gats_wdecay',   type=float, default=0.005)
parser.add_argument('--gats_epochs',   type=int,   default=2000)
parser.add_argument('--gats_patience', type=int,   default=100)
parser.add_argument('--bfs_depth',     type=int,   default=2)
parser.add_argument('--id_val_ratio',  type=float, default=0.1)
parser.add_argument('--id_test_ratio', type=float, default=0.1)
parser.add_argument('--class_weight',  type=float, default=10.0)
parser.add_argument('--eerm_dataset', type=str, default='cora',
                    choices=['cora', 'amazon'],
                    help='EERM 数据集: cora 或 amazon')
parser.add_argument('--eerm_root',    type=str, default=None,
                    help='EERM 数据集根目录（含 gen/ raw/）')
parser.add_argument('--save_dir',      type=str,   default='./save_model_gats_gat_sage')
parser.add_argument('--base_seed',     type=int,   default=42)
args = parser.parse_args()

def _backbone_name():
    return canonical_backbone_name(args.model)


def _model_tag():
    return _backbone_name().lower()


def _tagged_prefix(prefix):
    return f'{prefix}_{_model_tag()}'


def _tagged_title(title):
    return f'{title} [{_backbone_name()}]'


os.makedirs(args.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[设备] {device}')

ELLIPTIC_TRAIN = list(range(7,  12))
ELLIPTIC_VAL   = list(range(12, 17))
ELLIPTIC_TESTS = [
    list(range(17, 21)), list(range(21, 25)), list(range(25, 29)),
    list(range(29, 33)), list(range(33, 37)), list(range(37, 41)),
    list(range(41, 44)), list(range(44, 47)), list(range(47, 49)),
]
ARXIV_TRAIN_YEAR   = 2013
ARXIV_OODVAL_YEARS = (2014, 2015)
ARXIV_TESTS        = [(2014, 2016), (2016, 2018), (2018, 2020)]

COV_FULL = [round(0.1 * i, 1) for i in range(1, 11)]


# ══════════════════════════════════════════════════════════════
# 1. 指标计算（RQ1 / RQ2 / RQ3）
# ══════════════════════════════════════════════════════════════
def _reliability_bins(probs, labels, n_bins=15):
    """M=15 bin 的 (avg_conf, accuracy, count) 列表，供 Reliability Diagram 使用"""
    conf  = probs.max(1); pred = probs.argmax(1)
    acc_a = (pred == labels).astype(float)
    edges = np.linspace(0., 1., n_bins + 1)
    bins  = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            bins.append((float(conf[m].mean()), float(acc_a[m].mean()), int(m.sum())))
        else:
            bins.append((float((lo+hi)/2), float('nan'), 0))
    return bins


def _ece(probs, labels, n_bins=15):
    bins = _reliability_bins(probs, labels, n_bins)
    N = len(labels); ece = 0.0
    for avg_c, acc, cnt in bins:
        if cnt > 0 and not math.isnan(acc):
            ece += abs(avg_c - acc) * (cnt / N)
    return float(ece)


def _nll(probs, labels):
    return float(-np.log(probs[np.arange(len(labels)), labels] + 1e-10).mean())


def _brier(probs, labels, nclass):
    oh = np.eye(nclass)[labels]
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def _f1bin(probs, labels):
    pred = probs.argmax(1)
    tp = float(((pred==1)&(labels==1)).sum()); fp = float(((pred==1)&(labels==0)).sum())
    fn = float(((pred==0)&(labels==1)).sum())
    p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8)
    return 2*p*r/(p+r+1e-8), p, r


def _ue_auroc(probs, u, labels):
    z = (probs.argmax(1) != labels).astype(int)
    try:    return float(roc_auc_score(z, u)), float(average_precision_score(z, u))
    except: return float('nan'), float('nan')


def _ood_auroc_fn(u_id, u_ood):
    scores = np.concatenate([u_id, u_ood])
    d = np.concatenate([np.zeros(len(u_id)), np.ones(len(u_ood))])
    try:    return float(roc_auc_score(d, scores))
    except: return float('nan')


def _risk_curve(probs, u, labels):
    N = len(labels); wrong = (probs.argmax(1) != labels).astype(float)
    ws = wrong[np.argsort(u)]
    return {tau: float(ws[:max(1, int(math.ceil(tau*N)))].mean()) for tau in COV_FULL}


def _aurc(rc):
    taus = sorted(rc.keys()); risks = [rc[t] for t in taus]
    return float(sum((risks[j]+risks[j+1])/2*(taus[j+1]-taus[j])
                     for j in range(len(taus)-1)))


def compute_split_metrics(probs, u, labels, nclass, binary=False):
    """
    probs  : (N,C) numpy softmax 概率
    u      : (N,)  numpy 不确定性（越大越不确定）
    labels : (N,)  numpy int
    返回 dict，含 '_probs'/'_u' 内部字段（不写 CSV）
    """
    res = dict(acc=float((probs.argmax(1)==labels).mean()),
               ece=_ece(probs, labels), nll=_nll(probs, labels),
               brier=_brier(probs, labels, nclass))
    if binary:
        f1, pr, re = _f1bin(probs, labels)
        res.update(f1=f1, prec=pr, rec=re)
    res['ue_auroc'], res['ue_aupr'] = _ue_auroc(probs, u, labels)
    rc = _risk_curve(probs, u, labels)
    res['aurc'] = _aurc(rc)
    for tau in COV_FULL:
        res[f'risk@{tau}'] = rc[tau]
    # Coverage @ target risk (图12)
    N = len(labels)
    wrong = (probs.argmax(1) != labels).astype(float)
    ws    = wrong[np.argsort(u)]
    for target, key in [(0.01,'coverage@risk01'),(0.05,'coverage@risk05'),(0.10,'coverage@risk10')]:
        cov = float('nan')
        for tau in COV_FULL:
            if float(ws[:max(1, int(math.ceil(tau*N)))].mean()) <= target:
                cov = tau
        res[key] = cov
    res['_probs']            = probs
    res['_u']                = u
    res['_correct']          = (probs.argmax(1) == labels).astype(int)
    res['_reliability_bins'] = _reliability_bins(probs, labels)
    return res


def add_cross_split_metrics(id_res, ood_res, u_id, u_ood):
    out = dict(ood_res)
    for k in ('ece', 'nll', 'brier'):
        out[f'delta_{k}'] = ood_res.get(k, float('nan')) - id_res.get(k, float('nan'))
    out['delta_ue_auroc'] = (ood_res.get('ue_auroc', float('nan'))
                             - id_res.get('ue_auroc', float('nan')))
    out['ood_auroc'] = _ood_auroc_fn(u_id, u_ood)
    srtr = {}
    for tau in COV_FULL:
        ri = id_res.get(f'risk@{tau}', float('nan'))
        ro = ood_res.get(f'risk@{tau}', float('nan'))
        v  = (ro/ri) if (not math.isnan(ri) and ri > 0) else float('nan')
        srtr[tau] = v; out[f'srtr@{tau}'] = v
    valid = [(t, v) for t, v in sorted(srtr.items()) if not math.isnan(v)]
    out['srtr_auc'] = (float(sum((valid[j][1]+valid[j+1][1])/2*(valid[j+1][0]-valid[j][0])
                       for j in range(len(valid)-1))) if len(valid) >= 2 else float('nan'))
    out['aurc_ood'] = ood_res.get('aurc', float('nan'))
    for k in ('_probs', '_u', '_correct', '_reliability_bins'):
        if k in ood_res:
            out[k] = ood_res[k]
    return out


def build_all_keys(binary):
    base  = (['acc', 'f1', 'prec', 'rec'] if binary else ['acc'])
    base += ['ece', 'nll', 'brier', 'ue_auroc', 'ue_aupr', 'aurc']
    base += [f'risk@{t}' for t in COV_FULL]
    extra  = ['delta_ece', 'delta_nll', 'delta_brier',
              'delta_ue_auroc', 'ood_auroc', 'aurc_ood']
    extra += [f'srtr@{t}' for t in COV_FULL]
    extra += ['srtr_auc']
    extra += ['coverage@risk01', 'coverage@risk05', 'coverage@risk10']
    return base + extra


# ══════════════════════════════════════════════════════════════
# 2. evaluate（模型推理 → 全量指标）
# ══════════════════════════════════════════════════════════════
def evaluate(model, x, edge_index, labels, mask, nclass, binary=False):
    """
    GATS evaluate。
    u = 1 - max(p)（GATS 无自有不确定性分数）。
    """
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    lab_np  = (labels.cpu().numpy() if torch.is_tensor(labels)
               else np.asarray(labels))[mask_np]
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
    probs = torch.softmax(out, dim=1).cpu().numpy()[mask_np]
    u     = 1. - probs.max(1)
    return compute_split_metrics(probs, u, lab_np, nclass, binary=binary)


# ══════════════════════════════════════════════════════════════
# 3. BFS 最短路
# ══════════════════════════════════════════════════════════════
def shortest_path_length(edge_index, train_mask, max_hop):
    N    = train_mask.size(0)
    dist = torch.full((N,), fill_value=max_hop, dtype=torch.long,
                      device=train_mask.device)
    dist[train_mask] = 0
    frontier = train_mask.clone(); seen = train_mask.clone()
    ei = edge_index
    for hop in range(1, max_hop):
        frontier_nodes = frontier.nonzero(as_tuple=True)[0]
        if frontier_nodes.numel() == 0: break
        next_hop = torch.zeros(N, dtype=torch.bool, device=train_mask.device)
        for node in frontier_nodes.tolist():
            nbr_mask = (ei[0] == node)
            next_hop[ei[1][nbr_mask]] = True
        new_nodes = next_hop & ~seen
        dist[new_nodes] = hop; seen[new_nodes] = True; frontier = new_nodes
    return dist


# ══════════════════════════════════════════════════════════════
# 4. 模型
# ══════════════════════════════════════════════════════════════
class BaseGCN(nn.Module):
    def __init__(self, in_ch, hidden, nclass, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, nclass)
        self.dp    = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dp, self.training)
        return self.conv2(x, edge_index)


class BaseGAT(nn.Module):
    def __init__(self, in_ch, hidden, nclass, dropout):
        super().__init__()
        self.conv1 = GATConv(in_ch,      hidden,  heads=8, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden * 8, nclass,  heads=1, dropout=dropout, concat=False)
        self.dp    = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dp, self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dp, self.training)
        return self.conv2(x, edge_index)


def get_base_model(nfeat, nclass):
    return get_pyg_backbone(args.model, nfeat, args.hidden, nclass,
                            args.dropout, heads=getattr(args, 'backbone_heads', 8))


class CalibAttentionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_index, num_nodes,
                 train_mask, dist_to_train,
                 heads=8, negative_slope=0.2, bias=1.0,
                 self_loops=True, fill_value='mean'):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.heads          = heads
        self.negative_slope = negative_slope
        self.fill_value     = fill_value
        self.edge_index     = edge_index
        self.num_nodes      = num_nodes
        self.temp_lin  = PyGLinear(in_channels, heads,
                                   bias=False, weight_initializer='glorot')
        self.conf_coef = Parameter(torch.zeros([]))
        self.bias      = Parameter(torch.ones(1) * bias)
        self.train_a   = Parameter(torch.ones(1))
        self.dist1_a   = Parameter(torch.ones(1))
        self.register_buffer('dist_to_train', dist_to_train)
        self.reset_parameters()
        if self_loops:
            self.edge_index, _ = remove_self_loops(self.edge_index, None)
            self.edge_index, _ = add_self_loops(
                self.edge_index, None,
                fill_value=self.fill_value, num_nodes=num_nodes)

    def reset_parameters(self):
        self.temp_lin.reset_parameters()

    def forward(self, x):
        N, H = self.num_nodes, self.heads
        xn = x - x.min(1, keepdim=True)[0]
        denom = (x.max(1, keepdim=True)[0] - x.min(1, keepdim=True)[0]).clamp(min=1e-8)
        x_sorted = torch.sort(xn / denom, dim=-1)[0]
        temp = self.temp_lin(x_sorted)
        a_cluster = torch.ones(N, dtype=torch.float32, device=x.device)
        a_cluster[self.dist_to_train == 0] = self.train_a
        a_cluster[self.dist_to_train == 1] = self.dist1_a
        conf    = F.softmax(x, dim=1).amax(-1)
        deg     = degree(self.edge_index[0], N)
        deg_inv = (1. / deg).clamp(max=1e9)
        deg_inv[deg_inv == float('inf')] = 0.
        out = self.propagate(
            self.edge_index,
            temp  = temp.view(N, H) * a_cluster.unsqueeze(-1),
            alpha = x / a_cluster.unsqueeze(-1).clamp(min=1e-8),
            conf  = conf)
        sim, dconf = out[:, :-1], out[:, -1:]
        out = F.softplus(sim + self.conf_coef * dconf * deg_inv.unsqueeze(-1))
        out = out.mean(dim=1) + self.bias
        return out.unsqueeze(1)

    def message(self, temp_j, alpha_j, alpha_i, conf_i, conf_j,
                index, ptr, size_i):
        alpha = (alpha_j * alpha_i).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        return torch.cat([
            temp_j * alpha.unsqueeze(-1).expand_as(temp_j),
            (conf_i - conf_j).unsqueeze(-1)
        ], dim=-1)


class GATS(nn.Module):
    def __init__(self, base_model, edge_index, num_nodes,
                 train_mask, dist_to_train, nclass):
        super().__init__()
        self.base      = base_model
        self.num_nodes = num_nodes
        self.cagat     = CalibAttentionLayer(
            in_channels=nclass, out_channels=1,
            edge_index=edge_index, num_nodes=num_nodes,
            train_mask=train_mask, dist_to_train=dist_to_train,
            heads=args.gats_heads, bias=args.gats_bias)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x, edge_index):
        logits = self.base(x, edge_index)
        t      = self.cagat(logits).view(self.num_nodes, -1)
        t      = t.expand(self.num_nodes, logits.size(1))
        return logits / t


# ══════════════════════════════════════════════════════════════
# 5. 图工具
# ══════════════════════════════════════════════════════════════
def stratified_split(labels_t, base_mask, val_ratio, test_ratio, seed):
    rng      = np.random.default_rng(seed)
    lab_np   = labels_t.cpu().numpy()
    base_idx = base_mask.cpu().numpy().nonzero()[0]
    base_lab = lab_np[base_idx]
    N        = len(labels_t)
    val_lst, test_lst, train_lst = [], [], []
    for cls in np.unique(base_lab):
        idx  = base_idx[base_lab == cls]
        n    = len(idx)
        perm = rng.permutation(n); idx = idx[perm]
        nv   = max(1, int(n * val_ratio))
        nt   = max(1, int(n * test_ratio))
        while nv + nt >= n and (nv > 1 or nt > 1):
            if nv >= nt: nv -= 1
            else:        nt -= 1
        if nv + nt >= n: nv = 0; nt = 0
        val_lst  .append(idx[:nv])
        test_lst .append(idx[nv:nv+nt])
        train_lst.append(idx[nv+nt:])

    def to_mask(lists):
        m   = torch.zeros(N, dtype=torch.bool)
        cat = np.concatenate([x for x in lists if len(x) > 0])
        if len(cat): m[cat] = True
        return m
    return to_mask(train_lst), to_mask(val_lst), to_mask(test_lst)


def load_elliptic_step(step, data_dir):
    with open(os.path.join(data_dir, f'{step}.pkl'), 'rb') as f:
        d = pickle.load(f)
    return d[0], d[1], d[2].astype(np.float32)


def merge_elliptic(steps, data_dir):
    all_rows, all_cols, feats, labs = [], [], [], []
    offset = 0
    for s in steps:
        adj_s, lab, feat = load_elliptic_step(s, data_dir)
        coo = adj_s.tocoo()
        all_rows.append(np.concatenate([coo.row, coo.col]) + offset)
        all_cols.append(np.concatenate([coo.col, coo.row]) + offset)
        feats.append(feat); labs.append(lab); offset += feat.shape[0]
    N    = offset
    rows = np.concatenate(all_rows); cols = np.concatenate(all_cols)
    A    = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                         shape=(N, N)).tocsr()
    A.data[:] = 1.
    coo_u    = A.tocoo()
    edge_idx = torch.from_numpy(np.vstack([coo_u.row, coo_u.col]).astype(np.int64))
    features = torch.FloatTensor(np.concatenate(feats, 0))
    labels   = torch.LongTensor(np.concatenate(labs,  0))
    return edge_idx, features, labels, N


def load_arxiv(data_path):
    with open(data_path, 'rb') as f: raw = pickle.load(f)
    graph  = raw['graph']  if isinstance(raw, dict) else raw[0]
    labels = raw['labels'] if isinstance(raw, dict) else raw[1]

    def get(d, *keys):
        for k in keys:
            if k in d: return d[k]
        raise KeyError(keys)

    edge_index = np.array(get(graph, 'edge_index'), dtype=np.int64)
    node_feat  = np.array(get(graph, 'node_feat', 'node_feature', 'x'), dtype=np.float32)
    node_year  = np.array(get(graph, 'node_year', 'year')).flatten().astype(np.int64)
    N          = int(get(graph, 'num_nodes', 'num_node'))
    labels     = np.array(labels).flatten().astype(np.int64)
    nclass     = int(labels.max()) + 1
    print(f'  N={N} nclass={nclass} nfeat={node_feat.shape[1]}')
    src, dst = edge_index[0], edge_index[1]
    su = np.concatenate([src, dst]); du = np.concatenate([dst, src])
    ei_ud    = torch.from_numpy(np.vstack([su, du]).astype(np.int64))
    features = torch.FloatTensor(node_feat)
    labels_t = torch.LongTensor(labels)
    years    = torch.LongTensor(node_year)
    base_tr  = (years <= ARXIV_TRAIN_YEAR)
    oy0, oy1 = ARXIV_OODVAL_YEARS
    ov_mask  = (years >= oy0) & (years <= oy1)
    ood_masks = [(years >= ty0) & (years <= ty1) for ty0, ty1 in ARXIV_TESTS]
    return ei_ud, features, labels_t, years, nclass, base_tr, ov_mask, ood_masks, N


# ══════════════════════════════════════════════════════════════
# EERM Cora / Amazon-Photo 数据加载
# ══════════════════════════════════════════════════════════════

def load_eerm(eerm_root, dataset='cora'):
    """
    加载 EERM 格式的 Cora 或 Amazon-Photo 数据集。
    格式：gen/{i}-gcn.pkl = (x_tensor, y_tensor)
    环境：0=train, 1=OOD-val, 2~9=OOD-test_0~7。

    GATS 使用 PyG 的 edge_index，因此本函数返回无向 edge_index，而不是稀疏邻接矩阵。
    """
    import torch as _torch
    gen_dir = os.path.join(eerm_root, 'gen')

    feat_list, label_ref = [], None
    for i in range(10):
        with open(os.path.join(gen_dir, f'{i}-gcn.pkl'), 'rb') as f:
            d = pickle.load(f)
        x = d[0].detach().cpu().numpy() if isinstance(d[0], _torch.Tensor) else np.array(d[0])
        y = d[1].detach().cpu().numpy() if isinstance(d[1], _torch.Tensor) else np.array(d[1])
        feat_list.append(x.astype(np.float32))
        if label_ref is None:
            label_ref = y.astype(np.int64)

    labels = label_ref
    N      = feat_list[0].shape[0]
    nclass = int(labels.max()) + 1
    nfeat  = feat_list[0].shape[1]
    print(f'  EERM-{dataset}: N={N} nclass={nclass} nfeat={nfeat}')

    if dataset == 'cora':
        raw_dir = os.path.join(eerm_root, 'raw')

        def _load_raw(name):
            with open(os.path.join(raw_dir, f'ind.cora.{name}'), 'rb') as f:
                return pickle.load(f, encoding='latin1')

        allx  = _load_raw('allx')
        y_raw = _load_raw('y')
        graph = _load_raw('graph')
        with open(os.path.join(raw_dir, 'ind.cora.test.index')) as f:
            test_idx_raw = np.array([int(i) for i in f.read().split()])

        rows, cols = [], []
        for src, dsts in graph.items():
            for dst in dsts:
                rows.append(src); cols.append(dst)
        edge_index_np = np.vstack([rows, cols]).astype(np.int64)
        train_idx = np.arange(y_raw.shape[0])
        val_idx   = np.arange(allx.shape[0], allx.shape[0] + 500)
        test_idx  = test_idx_raw

    else:
        raw_dir  = os.path.join(eerm_root, 'raw')
        npz_path = os.path.join(raw_dir, 'amazon_electronics_photo.npz')
        npz = np.load(npz_path, allow_pickle=True)
        adj_npz = sp.csr_matrix(
            (npz['adj_data'], npz['adj_indices'], npz['adj_indptr']),
            shape=tuple(npz['adj_shape']))
        adj_coo = adj_npz.tocoo()
        edge_index_np = np.vstack([adj_coo.row, adj_coo.col]).astype(np.int64)
        rng = np.random.default_rng(42)
        tr  = []
        for c in np.unique(labels):
            idx_c = np.where(labels == c)[0]
            tr.extend(rng.choice(idx_c, min(20, len(idx_c)), replace=False).tolist())
        train_idx = np.array(tr)
        rest = np.setdiff1d(np.arange(N), train_idx)
        rng.shuffle(rest)
        val_idx  = rest[:500]
        test_idx = rest[500:1500]

    src, dst = edge_index_np[0], edge_index_np[1]
    edge_index_np = np.unique(
        np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]),
        axis=1)
    edge_index = _torch.from_numpy(edge_index_np.astype(np.int64))

    labels_t  = _torch.LongTensor(labels)
    feat_tr   = _torch.FloatTensor(feat_list[0])
    feat_val  = _torch.FloatTensor(feat_list[1])
    feat_oods = [_torch.FloatTensor(feat_list[i]) for i in range(2, 10)]

    def idx2mask(idx):
        m = _torch.zeros(N, dtype=_torch.bool)
        m[idx] = True
        return m

    tr_mask   = idx2mask(train_idx)
    val_mask  = idx2mask(val_idx)
    test_mask = idx2mask(test_idx)
    ood_names = [f'OOD-test_{i}(env{i+2})' for i in range(8)]

    return (edge_index, feat_tr, feat_val, feat_oods,
            labels_t, nclass, tr_mask, val_mask, test_mask, ood_names, N)

# ══════════════════════════════════════════════════════════════
# 6. 训练引擎
# ══════════════════════════════════════════════════════════════
def make_criterion(nclass, class_weight=None):
    if class_weight and nclass == 2:
        return nn.CrossEntropyLoss(
            weight=torch.FloatTensor([1.0, class_weight]).to(device))
    return nn.CrossEntropyLoss()


def train_base(x, edge_index, labels, tr_mask, id_val_mask,
               nclass, save_path, crit):
    model = get_base_model(x.shape[1], nclass).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, bad, best_state = 1e9, 0, None
    for _ in range(args.epochs):
        model.train(); opt.zero_grad()
        crit(model(x, edge_index)[tr_mask], labels[tr_mask]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            lv = crit(model(x, edge_index)[id_val_mask], labels[id_val_mask]).item()
        if lv < best:
            best_state = copy.deepcopy(model.state_dict()); best, bad = lv, 0
        else:
            bad += 1
        if bad == args.patience: break
    model.load_state_dict(best_state)
    torch.save(best_state, save_path)
    print(f'    base  | best ID-val loss={best:.4f}')
    return model


def train_gats(base_model, x, edge_index, labels,
               ov_mask, nclass, num_nodes, save_path):
    bfs_mask = ov_mask.cpu()
    print(f'    计算 BFS dist_to_train (max_hop={args.bfs_depth})...')
    dist  = shortest_path_length(edge_index.cpu(), bfs_mask, args.bfs_depth).to(device)
    model = GATS(base_model, edge_index, num_nodes,
                 bfs_mask.to(device), dist, nclass).to(device)
    opt   = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=0.01, weight_decay=args.gats_wdecay)
    best, bad, best_state = 1e9, 0, None
    with torch.no_grad():
        model.base.eval()
        logits_fixed = model.base(x, edge_index)
    for _ in range(args.gats_epochs):
        model.train(); model.base.eval(); opt.zero_grad()
        t   = model.cagat(logits_fixed).view(num_nodes, -1).expand(num_nodes, nclass)
        out = logits_fixed / t
        F.cross_entropy(out[ov_mask], labels[ov_mask]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            t2   = model.cagat(logits_fixed).view(num_nodes, -1).expand(num_nodes, nclass)
            lv   = F.cross_entropy((logits_fixed / t2)[ov_mask], labels[ov_mask]).item()
        if lv < best:
            best_state = copy.deepcopy(
                {k: v for k, v in model.state_dict().items()
                 if not k.startswith('base.') and 'dist_to_train' not in k})
            best, bad = lv, 0
        else: bad += 1
        if bad == args.gats_patience: break
    model.load_state_dict(best_state, strict=False)
    torch.save(best_state, save_path)
    print(f'    GATS  | best OOD-val loss={best:.4f}')
    return model


def run_one_seed(seed, x_tr, ei_tr, lab_tr, tr_base_mask, num_nodes_tr,
                 x_ov, ei_ov, lab_ov, ov_mask, num_nodes_ov,
                 nclass, crit, prefix):
    tr_mask, id_val_mask, id_test_mask = stratified_split(
        lab_tr, tr_base_mask,
        val_ratio=args.id_val_ratio, test_ratio=args.id_test_ratio, seed=seed)
    tr_mask = tr_mask.to(device)
    id_val_mask  = id_val_mask.to(device)
    id_test_mask = id_test_mask.to(device)
    print(f'  train={tr_mask.sum()} | ID-val={id_val_mask.sum()} '
          f'| ID-test={id_test_mask.sum()}')

    base_path = os.path.join(args.save_dir, f'{prefix}_seed{seed}_base.pth')
    gats_path = os.path.join(args.save_dir, f'{prefix}_seed{seed}_gats.pth')
    base_model = train_base(x_tr, ei_tr, lab_tr, tr_mask, id_val_mask,
                            nclass, base_path, crit)
    gats_model = train_gats(base_model, x_ov, ei_ov, lab_ov,
                            ov_mask, nclass, num_nodes_ov, gats_path)
    return gats_model, id_test_mask, base_path, gats_path



def train_base_eerm(x_train, x_val, edge_index, labels, tr_mask, id_val_mask,
                    nclass, save_path, crit):
    """EERM 专用 base 训练：env0 用于监督训练，env1 用于 early stopping。"""
    model = get_base_model(x_train.shape[1], nclass).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, bad, best_state = 1e9, 0, None
    for _ in range(args.epochs):
        model.train(); opt.zero_grad()
        crit(model(x_train, edge_index)[tr_mask], labels[tr_mask]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            lv = crit(model(x_val, edge_index)[id_val_mask], labels[id_val_mask]).item()
        if lv < best:
            best_state = copy.deepcopy(model.state_dict()); best, bad = lv, 0
        else:
            bad += 1
        if bad == args.patience: break
    model.load_state_dict(best_state)
    torch.save(best_state, save_path)
    print(f'    base  | best env1-val loss={best:.4f}')
    return model


def run_one_seed_eerm(seed, x_tr, x_val, edge_index, labels,
                      tr_mask, val_mask, num_nodes, nclass, crit, prefix):
    print(f'  train={tr_mask.sum()} | OOD-val={val_mask.sum()}')
    base_path = os.path.join(args.save_dir, f'{prefix}_seed{seed}_base.pth')
    gats_path = os.path.join(args.save_dir, f'{prefix}_seed{seed}_gats.pth')
    base_model = train_base_eerm(x_tr, x_val, edge_index, labels,
                                 tr_mask, val_mask, nclass, base_path, crit)
    _ = train_gats(base_model, x_val, edge_index, labels,
                   val_mask, nclass, num_nodes, gats_path)
    return base_path, gats_path

def load_gats_for_test(base_path, gats_path, nfeat, nclass,
                       x_te, ei_te, lab_te, num_nodes_te):
    base = get_base_model(nfeat, nclass)
    base.load_state_dict(torch.load(base_path, map_location=device))
    base.to(device).eval()
    bfs_mask = (lab_te >= 0).cpu()
    dist     = shortest_path_length(ei_te.cpu(), bfs_mask, args.bfs_depth).to(device)
    model    = GATS(base, ei_te, num_nodes_te,
                    bfs_mask.to(device), dist, nclass).to(device)
    model.load_state_dict(torch.load(gats_path, map_location=device), strict=False)
    return model


# ══════════════════════════════════════════════════════════════
# 7. 汇总输出
# ══════════════════════════════════════════════════════════════
def summarize(all_runs, split_names, all_keys, csv_path, title,
             reliability_path=None, uncertainty_path=None):
    col_w  = 18
    show_k = ['acc', 'ece', 'nll', 'brier', 'ue_auroc', 'aurc']
    sep    = '═' * (26 + col_w * len(show_k))
    print(f'\n{sep}'); print(f'  {title}  ({len(all_runs)} runs)'); print(sep)
    print(f'  {"split":<24}' + ''.join(f'{k:>{col_w}}' for k in show_k))
    print('  ' + '─' * (24 + col_w * len(show_k)))

    mean_rows = [['split']
                 + [f'{k}_mean' for k in all_keys]
                 + [f'{k}_std'  for k in all_keys]]
    for sname in split_names:
        vals = defaultdict(list)
        for r in all_runs:
            for k in all_keys:
                v = r.get(sname, {}).get(k)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    vals[k].append(v)
        mu = {k: np.mean(vals[k]) if vals[k] else float('nan') for k in all_keys}
        sd = {k: np.std(vals[k])  if vals[k] else float('nan') for k in all_keys}
        print(f'  {sname[:24]:<24}' +
              ''.join(f'{mu[k]:.4f}±{sd[k]:.4f}'.rjust(col_w) for k in show_k))
        mean_rows.append([sname]
                         + [f'{mu[k]:.6f}' for k in all_keys]
                         + [f'{sd[k]:.6f}'  for k in all_keys])

    ood_names = [n for n in split_names if n.startswith('OOD')]
    if ood_names:
        vals = defaultdict(list)
        for r in all_runs:
            for n in ood_names:
                for k in all_keys:
                    v = r.get(n, {}).get(k)
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        vals[k].append(v)
        mu = {k: np.mean(vals[k]) if vals[k] else float('nan') for k in all_keys}
        sd = {k: np.std(vals[k])  if vals[k] else float('nan') for k in all_keys}
        print(f'  {"OOD-avg":<24}' +
              ''.join(f'{mu[k]:.4f}±{sd[k]:.4f}'.rjust(col_w) for k in show_k))
        mean_rows.append(['OOD-avg']
                         + [f'{mu[k]:.6f}' for k in all_keys]
                         + [f'{sd[k]:.6f}'  for k in all_keys])

    print(f'\n{sep}')
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.',
                exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerows(mean_rows)
    print(f'  结果已保存 → {csv_path}')

    if reliability_path:
        rel_rows = [['run', 'split', 'bin', 'avg_confidence', 'accuracy', 'count']]
        for run_idx, r in enumerate(all_runs):
            for sname in split_names:
                for b_idx, (avg_c, acc, cnt) in enumerate(
                        r.get(sname, {}).get('_reliability_bins', [])):
                    rel_rows.append([run_idx, sname, b_idx+1,
                                     f'{avg_c:.6f}',
                                     f'{acc:.6f}' if not math.isnan(acc) else 'nan',
                                     cnt])
        with open(reliability_path, 'w', newline='') as f:
            csv.writer(f).writerows(rel_rows)
        print(f'  Reliability  → {reliability_path}')

    if uncertainty_path:
        unc_rows = [['run', 'split', 'u', 'correct']]
        for run_idx, r in enumerate(all_runs):
            for sname in split_names:
                m = r.get(sname, {})
                u_arr, cor_arr = m.get('_u'), m.get('_correct')
                if u_arr is not None and cor_arr is not None:
                    for u_val, c_val in zip(u_arr.tolist(), cor_arr.tolist()):
                        unc_rows.append([run_idx, sname, f'{u_val:.6f}', int(c_val)])
        with open(uncertainty_path, 'w', newline='') as f:
            csv.writer(f).writerows(unc_rows)
        print(f'  不确定性样本 → {uncertainty_path}')


# ══════════════════════════════════════════════════════════════
# 8. 主函数
# ══════════════════════════════════════════════════════════════
def main():
    print(f'[配置] dataset={args.dataset} backbone={_backbone_name()} runs={args.runs}')
    print(f'[配置] hidden={args.hidden} epochs={args.epochs} patience={args.patience}')
    print(f'[配置] gats_heads={args.gats_heads} bfs_depth={args.bfs_depth}')

    # ── Elliptic ──────────────────────────────────────────────
    if args.dataset == 'elliptic':
        crit = make_criterion(2, args.class_weight)

        print('\n[Elliptic] 加载数据...')
        ei_tr, feat_tr, lab_tr, N_tr = merge_elliptic(ELLIPTIC_TRAIN, args.data_dir)
        ei_tr   = ei_tr.to(device);   feat_tr = feat_tr.to(device)
        lab_tr  = lab_tr.to(device);  tr_base_mask = (lab_tr >= 0)

        ei_ov, feat_ov, lab_ov, N_ov = merge_elliptic(ELLIPTIC_VAL, args.data_dir)
        ei_ov   = ei_ov.to(device);   feat_ov = feat_ov.to(device)
        lab_ov  = lab_ov.to(device);  ov_mask = (lab_ov >= 0).to(device)

        print('[Elliptic] 加载 OOD-test 图...')
        test_graphs = []
        for i, steps in enumerate(ELLIPTIC_TESTS):
            ei_te, feat_te, lab_te, N_te = merge_elliptic(steps, args.data_dir)
            te_mask = (lab_te >= 0)
            print(f'  OOD-test_{i} steps={steps}: labeled={te_mask.sum()} '
                  f'illicit={(lab_te[te_mask]==1).sum()}')
            test_graphs.append((ei_te.to(device), feat_te.to(device),
                                lab_te.to(device), te_mask.to(device), N_te))

        split_names = ['ID-test'] + [
            f'OOD-test_{i}(s{ELLIPTIC_TESTS[i][0]}-{ELLIPTIC_TESTS[i][-1]})'
            for i in range(len(ELLIPTIC_TESTS))]
        all_keys = build_all_keys(binary=True)

        all_runs = []
        for r in range(args.runs):
            seed = args.base_seed + r
            print(f'\n{"="*65}')
            print(f'  GATS  Run {r+1}/{args.runs}  seed={seed}')
            print(f'{"="*65}')
            torch.manual_seed(seed); np.random.seed(seed)

            gats_model, id_test_mask, base_path, gats_path = run_one_seed(
                seed,
                feat_tr, ei_tr, lab_tr, tr_base_mask, N_tr,
                feat_ov, ei_ov, lab_ov, ov_mask, N_ov,
                nclass=2, crit=crit, prefix=_tagged_prefix('elliptic'))

            run_res = {}

            # ID-test（在训练图上重建 GATS）
            model_id = load_gats_for_test(
                base_path, gats_path, feat_tr.shape[1], 2,
                feat_tr, ei_tr, lab_tr, N_tr)
            r_id = evaluate(model_id, feat_tr, ei_tr, lab_tr,
                            id_test_mask, 2, binary=True)
            print(f'  ID-test | acc={r_id["acc"]:.4f} f1={r_id["f1"]:.4f} '
                  f'ece={r_id["ece"]:.4f} nll={r_id["nll"]:.4f} '
                  f'ue_auroc={r_id["ue_auroc"]:.4f} aurc={r_id["aurc"]:.4f}')
            run_res['ID-test'] = r_id

            # OOD-test × 9
            for i, (ei_te, feat_te, lab_te, te_mask, N_te) in enumerate(test_graphs):
                model_te  = load_gats_for_test(
                    base_path, gats_path, feat_te.shape[1], 2,
                    feat_te, ei_te, lab_te, N_te)
                r_ood_raw = evaluate(model_te, feat_te, ei_te, lab_te,
                                     te_mask, 2, binary=True)
                r_ood     = add_cross_split_metrics(
                    r_id, r_ood_raw, r_id['_u'], r_ood_raw['_u'])
                name = split_names[1 + i]
                print(f'  {name[:32]} | acc={r_ood["acc"]:.4f} '
                      f'delta_ece={r_ood["delta_ece"]:.4f} '
                      f'ood_auroc={r_ood["ood_auroc"]:.4f} '
                      f'srtr_auc={r_ood["srtr_auc"]:.4f}')
                run_res[name] = r_ood

            all_runs.append(run_res)

        summarize(all_runs, split_names, all_keys,
                  csv_path=os.path.join(args.save_dir, 'elliptic_gats_results.csv'),
                  title=_tagged_title('Elliptic — GATS'),
                  reliability_path=os.path.join(args.save_dir, 'elliptic_gats_reliability.csv'),
                  uncertainty_path=os.path.join(args.save_dir, 'elliptic_gats_uncertainty_samples.csv'))

    # ── OGB-Arxiv ─────────────────────────────────────────────
    elif args.dataset == 'arxiv':
        print('\n[Arxiv] 加载数据...')
        (ei, features, labels, years, nclass,
         base_tr, ov_mask, ood_masks, N) = load_arxiv(args.data_path)

        ei        = ei.to(device);       features  = features.to(device)
        labels    = labels.to(device);   ov_mask   = ov_mask.to(device)
        ood_masks = [m.to(device) for m in ood_masks]
        crit      = make_criterion(nclass)

        split_names = ['ID-test'] + [
            f'OOD-test_{i}({ty0}-{ty1})'
            for i, (ty0, ty1) in enumerate(ARXIV_TESTS)]
        all_keys = build_all_keys(binary=False)

        all_runs = []
        for r in range(args.runs):
            seed = args.base_seed + r
            print(f'\n{"="*65}')
            print(f'  GATS  Run {r+1}/{args.runs}  seed={seed}')
            print(f'{"="*65}')
            torch.manual_seed(seed); np.random.seed(seed)

            gats_model, id_test_mask, base_path, gats_path = run_one_seed(
                seed,
                features, ei, labels, base_tr, N,
                features, ei, labels, ov_mask, N,
                nclass=nclass, crit=crit, prefix=_tagged_prefix('arxiv'))

            run_res = {}

            r_id = evaluate(gats_model, features, ei, labels,
                            id_test_mask, nclass, binary=False)
            print(f'  ID-test | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f} '
                  f'nll={r_id["nll"]:.4f} ue_auroc={r_id["ue_auroc"]:.4f} '
                  f'aurc={r_id["aurc"]:.4f}')
            run_res['ID-test'] = r_id

            for i, (te_mask, (ty0, ty1)) in enumerate(zip(ood_masks, ARXIV_TESTS)):
                r_ood_raw = evaluate(gats_model, features, ei, labels,
                                     te_mask, nclass, binary=False)
                r_ood     = add_cross_split_metrics(
                    r_id, r_ood_raw, r_id['_u'], r_ood_raw['_u'])
                name = split_names[1 + i]
                print(f'  {name} | acc={r_ood["acc"]:.4f} '
                      f'delta_ece={r_ood["delta_ece"]:.4f} '
                      f'ood_auroc={r_ood["ood_auroc"]:.4f} '
                      f'srtr_auc={r_ood["srtr_auc"]:.4f}')
                run_res[name] = r_ood

            all_runs.append(run_res)

        summarize(all_runs, split_names, all_keys,
                  csv_path=os.path.join(args.save_dir, 'arxiv_gats_results.csv'),
                  title=_tagged_title('OGB-Arxiv — GATS'),
                  reliability_path=os.path.join(args.save_dir, 'arxiv_gats_reliability.csv'),
                  uncertainty_path=os.path.join(args.save_dir, 'arxiv_gats_uncertainty_samples.csv'))



    elif args.dataset == 'eerm':
        assert args.eerm_root, '--eerm_root 必须指定'
        print(f'\n[EERM-{args.eerm_dataset}] 加载数据...')
        (ei, feat_tr, feat_val, feat_oods,
         labels_t, nclass,
         tr_mask, val_mask, test_mask,
         ood_names, N) = load_eerm(args.eerm_root, args.eerm_dataset)

        ei       = ei.to(device)
        feat_tr  = feat_tr.to(device)
        feat_val = feat_val.to(device)
        feat_oods= [f.to(device) for f in feat_oods]
        labels_t = labels_t.to(device)
        tr_mask  = tr_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask= test_mask.to(device)
        crit     = make_criterion(nclass)

        split_names = ['ID-test'] + ood_names
        all_keys    = build_all_keys(binary=False)
        all_runs    = []
        ds_tag      = args.eerm_dataset

        for r in range(args.runs):
            seed = args.base_seed + r
            print(f'\n{"="*65}')
            print(f'  GATS  EERM-{ds_tag}  Run {r+1}/{args.runs}  seed={seed}')
            print(f'{"="*65}')
            torch.manual_seed(seed); np.random.seed(seed)

            base_path, gats_path = run_one_seed_eerm(
                seed, feat_tr, feat_val, ei, labels_t,
                tr_mask, val_mask, N, nclass, crit,
                prefix=_tagged_prefix(f'eerm_{ds_tag}'))

            run_res = {}

            # ID-test：env0 特征，test 节点；加载同一组 base/GATS 参数。
            model_id = load_gats_for_test(
                base_path, gats_path, feat_tr.shape[1], nclass,
                feat_tr, ei, labels_t, N)
            r_id = evaluate(model_id, feat_tr, ei, labels_t,
                            test_mask, nclass, binary=False)
            print(f'  ID-test | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f} '
                  f'ue_auroc={r_id["ue_auroc"]:.4f} aurc={r_id["aurc"]:.4f}')
            run_res['ID-test'] = r_id

            # OOD-test：env2~9 特征，图结构和 test 节点不变。
            for feat_ood, name in zip(feat_oods, ood_names):
                model_ood = load_gats_for_test(
                    base_path, gats_path, feat_ood.shape[1], nclass,
                    feat_ood, ei, labels_t, N)
                r_ood_raw = evaluate(model_ood, feat_ood, ei, labels_t,
                                     test_mask, nclass, binary=False)
                r_ood = add_cross_split_metrics(
                    r_id, r_ood_raw, r_id['_u'], r_ood_raw['_u'])
                print(f'  {name} | acc={r_ood["acc"]:.4f} '
                      f'delta_ece={r_ood["delta_ece"]:.4f} '
                      f'ood_auroc={r_ood["ood_auroc"]:.4f}')
                run_res[name] = r_ood

            all_runs.append(run_res)

        summarize(all_runs, split_names, all_keys,
                  csv_path=os.path.join(args.save_dir,
                      f'{ds_tag}_gats_results.csv'),
                  title=_tagged_title(f'EERM-{ds_tag} — GATS'),
                  reliability_path=os.path.join(args.save_dir,
                      f'{ds_tag}_gats_reliability.csv'),
                  uncertainty_path=os.path.join(args.save_dir,
                      f'{ds_tag}_gats_uncertainty_samples.csv'))

if __name__ == '__main__':

    main()
