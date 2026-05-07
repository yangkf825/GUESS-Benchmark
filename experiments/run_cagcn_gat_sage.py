"""
CaGCN — Elliptic & OGB-Arxiv 完整指标版
=========================================
【指标体系（实验文档 RQ1/RQ2/RQ3）】

RQ1（每个 split）:
  acc, ece(M=15), nll, brier
  [Elliptic额外] f1, prec, rec
  [OOD额外] delta_ece, delta_nll, delta_brier

RQ2（每个 split）:
  ue_auroc, ue_aupr          — 错误检测（u = 1-max(p)，CaGCN 无自有不确定性）
  [OOD额外] delta_ue_auroc, ood_auroc

RQ3（每个 split）:
  aurc                       — τ 从 0.1~1.0 梯形积分
  risk@0.1 ~ risk@1.0        — 10 个 coverage 点
  [OOD额外] aurc_ood, srtr@0.1~srtr@1.0, srtr_auc

【额外输出】
  *_boxplot.csv — 每 run × 每 split 的 acc/ece/nll/brier 原始值，供箱线图使用

用法:
    python elliptic_arxiv_cagcn.py --dataset elliptic --data_dir  ./elliptic --runs 5
    python elliptic_arxiv_cagcn.py --dataset arxiv    --data_path ./data.pkl  --runs 5
"""
import sys; sys.path.insert(0, 'src')


import os, pickle, argparse, warnings, csv, math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
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
parser.add_argument('--stage',         type=int,   default=2)
parser.add_argument('--epochs',        type=int,   default=1000)
parser.add_argument('--epoch_for_st',  type=int,   default=200)
parser.add_argument('--lr',            type=float, default=0.01)
parser.add_argument('--lr_for_cal',    type=float, default=0.01)
parser.add_argument('--weight_decay',  type=float, default=5e-4)
parser.add_argument('--l2_for_cal',    type=float, default=5e-3)
parser.add_argument('--backbone_heads', type=int,   default=8,
                    help='GAT attention heads for the new backbone')
parser.add_argument('--hidden',        type=int,   default=64)
parser.add_argument('--dropout',       type=float, default=0.5)
parser.add_argument('--Lambda',        type=float, default=0.5)
parser.add_argument('--patience',      type=int,   default=100)
parser.add_argument('--threshold',     type=float, default=0.85)
parser.add_argument('--class_weight',  type=float, default=10.0)
parser.add_argument('--id_val_ratio',  type=float, default=0.1)
parser.add_argument('--id_test_ratio', type=float, default=0.1)
parser.add_argument('--alpha',         type=float, default=0.2)
parser.add_argument('--nb_heads',      type=int,   default=8)
parser.add_argument('--eerm_dataset', type=str, default='cora',
                    choices=['cora', 'amazon'],
                    help='EERM 数据集: cora 或 amazon')
parser.add_argument('--eerm_root',    type=str, default=None,
                    help='EERM 数据集根目录（含 gen/ raw/）')
parser.add_argument('--save_dir',      type=str,   default='./save_model_cagcn_gat_sage')
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

# Coverage 点：0.1~1.0 共 10 点
COV_FULL = [round(0.1 * i, 1) for i in range(1, 11)]


# ══════════════════════════════════════════════════════════════
# 1. 模型
# ══════════════════════════════════════════════════════════════
class GraphConvolution(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.W = Parameter(torch.FloatTensor(in_f, out_f))
        self.b = Parameter(torch.FloatTensor(out_f))
        s = 1. / out_f ** 0.5
        self.W.data.uniform_(-s, s); self.b.data.uniform_(-s, s)

    def forward(self, x, adj):
        return torch.spmm(adj, x @ self.W) + self.b


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dp  = nn.Dropout(dropout)

    def forward(self, x, adj):
        return self.gc2(self.dp(F.relu(self.gc1(x, adj))), adj)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super().__init__()
        self.dp = dropout
        self.Ws = nn.ParameterList([
            Parameter(torch.FloatTensor(nfeat, nhid)) for _ in range(nheads)])
        for w in self.Ws: nn.init.xavier_normal_(w.data, gain=1.414)
        self.out = GraphConvolution(nhid * nheads, nclass)

    def forward(self, x, adj):
        h = torch.cat([F.elu(torch.spmm(
                adj, F.dropout(x, self.dp, self.training) @ w))
            for w in self.Ws], dim=1)
        return self.out(F.dropout(h, self.dp, self.training), adj)


class CaGCN(nn.Module):
    def __init__(self, nclass, base_model):
        super().__init__()
        self.base = base_model
        self.s1   = GraphConvolution(nclass, 16)
        self.s2   = GraphConvolution(16, 1)
        for p in self.base.parameters(): p.requires_grad = False

    def forward(self, x, adj):
        logits = self.base(x, adj)
        t = torch.log(torch.exp(self.s2(F.relu(self.s1(logits, adj)), adj)) + 1.1)
        return logits * t


def get_model(nfeat, nclass):
    return get_sparse_backbone(args.model, nfeat, args.hidden, nclass,
                               args.dropout, alpha=getattr(args, 'alpha', 0.2),
                               nheads=getattr(args, 'backbone_heads', getattr(args, 'nb_heads', 8)))


# ══════════════════════════════════════════════════════════════
# 2. 指标计算（RQ1 / RQ2 / RQ3）
# ══════════════════════════════════════════════════════════════

def _reliability_bins(probs, labels, n_bins=15):
    """返回 M=15 bin 的 (avg_conf, accuracy, count) 列表，供 Reliability Diagram 使用"""
    conf  = probs.max(1)
    pred  = probs.argmax(1)
    acc_a = (pred == labels).astype(float)
    edges = np.linspace(0., 1., n_bins + 1)
    bins  = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            bins.append((float(conf[m].mean()), float(acc_a[m].mean()), int(m.sum())))
        else:
            mid = float((lo + hi) / 2)
            bins.append((mid, float('nan'), 0))
    return bins  # list of (avg_conf, accuracy, count), length=n_bins


def _ece(probs, labels, n_bins=15):
    """ECE，M=15 等宽 bin（实验文档规定）"""
    bins = _reliability_bins(probs, labels, n_bins)
    N    = len(labels)
    ece  = 0.0
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
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    p  = tp / (tp + fp + 1e-8)
    r  = tp / (tp + fn + 1e-8)
    return 2 * p * r / (p + r + 1e-8), p, r


def _ue_auroc(probs, u, labels):
    """错误检测 AUROC/AUPR（z=1 表示预测错，u 越大越不确定）"""
    z = (probs.argmax(1) != labels).astype(int)
    try:
        return float(roc_auc_score(z, u)), float(average_precision_score(z, u))
    except Exception:
        return float('nan'), float('nan')


def _ood_auroc_fn(u_id, u_ood):
    """OOD 检测 AUROC（ID=0, OOD=1，u 作 score）"""
    scores = np.concatenate([u_id, u_ood])
    d      = np.concatenate([np.zeros(len(u_id)), np.ones(len(u_ood))])
    try:
        return float(roc_auc_score(d, scores))
    except Exception:
        return float('nan')


def _risk_curve(probs, u, labels):
    """Selective risk at each coverage τ（按 u 升序，最确定在前被选中）"""
    N     = len(labels)
    wrong = (probs.argmax(1) != labels).astype(float)
    ws    = wrong[np.argsort(u)]   # 升序：最确定在前
    return {tau: float(ws[:max(1, int(math.ceil(tau * N)))].mean())
            for tau in COV_FULL}


def _aurc(rc):
    taus  = sorted(rc.keys())
    risks = [rc[t] for t in taus]
    return float(sum((risks[j] + risks[j+1]) / 2 * (taus[j+1] - taus[j])
                     for j in range(len(taus) - 1)))


def compute_split_metrics(probs, u, labels, nclass, binary=False):
    """
    单个 split 全量指标。
    probs  : (N, C) numpy float，softmax 概率
    u      : (N,)   numpy float，不确定性分数（越大越不确定）
    labels : (N,)   numpy int，真实标签
    返回 dict，含 '_probs'/'_u' 内部字段供跨 split 计算，不写入 CSV。
    """
    res = dict(
        acc   = float((probs.argmax(1) == labels).mean()),
        ece   = _ece(probs, labels),
        nll   = _nll(probs, labels),
        brier = _brier(probs, labels, nclass),
    )
    if binary:
        f1, pr, re = _f1bin(probs, labels)
        res.update(f1=f1, prec=pr, rec=re)
    res['ue_auroc'], res['ue_aupr'] = _ue_auroc(probs, u, labels)
    rc = _risk_curve(probs, u, labels)
    res['aurc'] = _aurc(rc)
    for tau in COV_FULL:
        res[f'risk@{tau}'] = rc[tau]
    # 内部传递（不进 CSV）
    # Coverage @ target risk (图12)
    N = len(labels)
    wrong = (probs.argmax(1) != labels).astype(float)
    ws    = wrong[np.argsort(u)]
    for target in (0.01, 0.05, 0.10):
        cov = float('nan')
        for tau in COV_FULL:
            r_tau = float(ws[:max(1, int(math.ceil(tau * N)))].mean())
            if r_tau <= target:
                cov = tau
        key = f'coverage@risk{int(target*100):02d}'
        res[key] = cov
    # 内部传递（不进 CSV）
    res['_probs']            = probs
    res['_u']                = u
    res['_correct']          = (probs.argmax(1) == labels).astype(int)
    res['_reliability_bins'] = _reliability_bins(probs, labels)
    return res


def add_cross_split_metrics(id_res, ood_res, u_id, u_ood):
    """追加需要两个 split 才能计算的指标（delta / ood_auroc / srtr）"""
    out = dict(ood_res)
    for k in ('ece', 'nll', 'brier'):
        out[f'delta_{k}'] = (ood_res.get(k, float('nan'))
                             - id_res.get(k, float('nan')))
    out['delta_ue_auroc'] = (ood_res.get('ue_auroc', float('nan'))
                             - id_res.get('ue_auroc', float('nan')))
    out['ood_auroc'] = _ood_auroc_fn(u_id, u_ood)

    srtr = {}
    for tau in COV_FULL:
        ri = id_res.get(f'risk@{tau}', float('nan'))
        ro = ood_res.get(f'risk@{tau}', float('nan'))
        v  = (ro / ri) if (not math.isnan(ri) and ri > 0) else float('nan')
        srtr[tau] = v
        out[f'srtr@{tau}'] = v

    valid = [(t, v) for t, v in sorted(srtr.items()) if not math.isnan(v)]
    out['srtr_auc'] = (
        float(sum((valid[j][1] + valid[j+1][1]) / 2 * (valid[j+1][0] - valid[j][0])
                  for j in range(len(valid) - 1)))
        if len(valid) >= 2 else float('nan'))
    out['aurc_ood'] = ood_res.get('aurc', float('nan'))
    # 传递逐样本内部字段（供 summarize 使用）
    for k in ('_probs', '_u', '_correct', '_reliability_bins'):
        if k in ood_res:
            out[k] = ood_res[k]
    return out


def build_all_keys(binary):
    """返回写入 CSV 的全量指标列名（不含内部字段 _probs/_u）"""
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
# 3. evaluate（模型推理 → 全量指标）
# ══════════════════════════════════════════════════════════════
def evaluate(model, feat, adj, labels, mask, nclass, binary=False):
    """
    CaGCN evaluate。
    不确定性 u = 1 - max(p)（CaGCN 无自有不确定性分数）。
    """
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    lab_np  = (labels.cpu().numpy() if torch.is_tensor(labels)
               else np.asarray(labels))[mask_np]
    model.eval()
    with torch.no_grad():
        out = model(feat, adj)
    probs = torch.softmax(out, dim=1).cpu().numpy()[mask_np]
    u     = 1. - probs.max(1)   # 不确定性 = 1 - 置信度
    return compute_split_metrics(probs, u, lab_np, nclass, binary=binary)


# ══════════════════════════════════════════════════════════════
# 4. 图工具
# ══════════════════════════════════════════════════════════════
def normalize_adj(A_sp, N):
    A_sp = A_sp + sp.eye(N)
    deg  = np.array(A_sp.sum(1)).flatten()
    d    = np.where(deg > 0, deg ** -0.5, 0.)
    D    = sp.diags(d)
    A    = (D @ A_sp @ D).tocoo().astype(np.float32)
    idx  = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
    val  = torch.from_numpy(A.data)
    return torch.sparse.FloatTensor(idx, val, torch.Size([N, N]))


def edges_to_adj(edge_index, N):
    s, d = edge_index[0], edge_index[1]
    su   = np.concatenate([s, d]); du = np.concatenate([d, s])
    A    = sp.coo_matrix((np.ones(len(su), np.float32), (su, du)),
                         shape=(N, N)).tocsr()
    A.data[:] = 1.
    return normalize_adj(A, N)


# ══════════════════════════════════════════════════════════════
# 5. 分层划分 train / ID-val / ID-test
# ══════════════════════════════════════════════════════════════
def stratified_split(labels_tensor, base_mask, val_ratio, test_ratio, seed):
    rng      = np.random.default_rng(seed)
    lab_np   = labels_tensor.cpu().numpy()
    base_idx = base_mask.cpu().numpy().nonzero()[0]
    base_lab = lab_np[base_idx]
    N        = len(labels_tensor)

    val_idx, test_idx, train_idx = [], [], []
    for cls in np.unique(base_lab):
        idx  = base_idx[base_lab == cls]
        n    = len(idx)
        perm = rng.permutation(n); idx = idx[perm]
        n_val  = max(1, int(n * val_ratio))
        n_test = max(1, int(n * test_ratio))
        while n_val + n_test >= n and (n_val > 1 or n_test > 1):
            if n_val >= n_test: n_val  -= 1
            else:               n_test -= 1
        if n_val + n_test >= n: n_val = 0; n_test = 0
        val_idx  .append(idx[:n_val])
        test_idx .append(idx[n_val : n_val + n_test])
        train_idx.append(idx[n_val + n_test:])

    def to_mask(lists):
        m   = torch.zeros(N, dtype=torch.bool)
        cat = np.concatenate([x for x in lists if len(x) > 0])
        if len(cat): m[cat] = True
        return m

    return to_mask(train_idx), to_mask(val_idx), to_mask(test_idx)


# ══════════════════════════════════════════════════════════════
# 6. 数据加载
# ══════════════════════════════════════════════════════════════
def load_elliptic_step(step, data_dir):
    with open(os.path.join(data_dir, f'{step}.pkl'), 'rb') as f:
        d = pickle.load(f)
    return d[0], d[1], d[2].astype(np.float32)


def merge_elliptic(steps, data_dir):
    rows_all, cols_all, feats, labs = [], [], [], []
    offset = 0
    for s in steps:
        adj_s, lab, feat = load_elliptic_step(s, data_dir)
        coo = adj_s.tocoo()
        rows_all.append(coo.row + offset); cols_all.append(coo.col + offset)
        feats.append(feat); labs.append(lab); offset += feat.shape[0]
    rows = np.concatenate(rows_all); cols = np.concatenate(cols_all)
    su   = np.concatenate([rows, cols]); du = np.concatenate([cols, rows])
    A    = sp.coo_matrix((np.ones(len(su), np.float32), (su, du)),
                         shape=(offset, offset)).tocsr()
    A.data[:] = 1.
    return (normalize_adj(A, offset),
            torch.FloatTensor(np.concatenate(feats, 0)),
            torch.LongTensor(np.concatenate(labs,  0)))


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

    adj       = edges_to_adj(edge_index, N)
    features  = torch.FloatTensor(node_feat)
    labels_t  = torch.LongTensor(labels)
    years     = torch.LongTensor(node_year)
    oy0, oy1  = ARXIV_OODVAL_YEARS
    ov_mask   = (years >= oy0) & (years <= oy1)
    ood_masks = [(years >= ty0) & (years <= ty1) for ty0, ty1 in ARXIV_TESTS]
    base_tr   = (years <= ARXIV_TRAIN_YEAR)
    return adj, features, labels_t, years, nclass, base_tr, ov_mask, ood_masks


# ══════════════════════════════════════════════════════════════
# EERM Cora / Amazon-Photo 数据加载
# ══════════════════════════════════════════════════════════════

def load_eerm(eerm_root, dataset='cora'):
    """
    加载 EERM 格式的 Cora 或 Amazon-Photo 数据集。
    格式：gen/{i}-gcn.pkl = (x_tensor, y_tensor)
    环境：0=train, 1=val, 2~9=OOD-test_0~7
    """
    import torch as _torch
    gen_dir = os.path.join(eerm_root, 'gen')

    # 读全部10个环境的特征
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

    # 读边
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
        edge_index = np.vstack([rows, cols]).astype(np.int64)
        train_idx = np.arange(y_raw.shape[0])
        val_idx   = np.arange(allx.shape[0], allx.shape[0] + 500)
        test_idx  = test_idx_raw

    else:  # amazon
        raw_dir  = os.path.join(eerm_root, 'raw')
        npz_path = os.path.join(raw_dir, 'amazon_electronics_photo.npz')
        npz = np.load(npz_path, allow_pickle=True)
        adj_npz = sp.csr_matrix(
            (npz['adj_data'], npz['adj_indices'], npz['adj_indptr']),
            shape=tuple(npz['adj_shape']))
        adj_coo = adj_npz.tocoo()
        edge_index = np.vstack([adj_coo.row, adj_coo.col]).astype(np.int64)
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

    # 构建邻接矩阵
    adj = edges_to_adj(edge_index, N)

    # 转 torch
    labels_t  = _torch.LongTensor(labels)
    feat_tr   = _torch.FloatTensor(feat_list[0])   # env 0
    feat_val  = _torch.FloatTensor(feat_list[1])   # env 1
    feat_oods = [_torch.FloatTensor(feat_list[i]) for i in range(2, 10)]

    def idx2mask(idx):
        m = _torch.zeros(N, dtype=_torch.bool)
        m[idx] = True
        return m

    tr_mask   = idx2mask(train_idx)
    val_mask  = idx2mask(val_idx)
    test_mask = idx2mask(test_idx)
    ood_names = [f'OOD-test_{i}(env{i+2})' for i in range(8)]

    return (adj, feat_tr, feat_val, feat_oods,
            labels_t, nclass,
            tr_mask, val_mask, test_mask,
            ood_names)

# ══════════════════════════════════════════════════════════════
# 7. 训练引擎
# ══════════════════════════════════════════════════════════════
def make_criterion(nclass, class_weight=None):
    if class_weight and nclass == 2:
        return nn.CrossEntropyLoss(
            weight=torch.FloatTensor([1.0, class_weight]).to(device))
    return nn.CrossEntropyLoss()


def intra_loss(output, labels):
    p = F.softmax(output, dim=1); pred = p.max(1)[1]
    ok  = (pred == labels).nonzero(as_tuple=True)[0]
    bad = (pred != labels).nonzero(as_tuple=True)[0]
    s   = p.sort(1, descending=True)[0]; t1, t2 = s[:, 0], s[:, 1]
    return ((1-t1[ok]+t2[ok]).sum() + (t1[bad]-t2[bad]).sum()) / labels.size(0)


def train_base_model(feat, adj, lab, tr_mask, id_val_mask,
                     nclass, path, crit, dyn_tr, pseudo_lab):
    model = get_model(feat.shape[1], nclass).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, bad = 1e9, 0
    for _ in range(args.epochs):
        model.train(); opt.zero_grad()
        crit(model(feat, adj)[dyn_tr], pseudo_lab[dyn_tr]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            lv = crit(model(feat, adj)[id_val_mask], lab[id_val_mask]).item()
        if lv < best: torch.save(model.state_dict(), path); best, bad = lv, 0
        else: bad += 1
        if bad == args.patience: break
    print(f'    base  | ID-val loss={best:.4f}')


def train_cagcn_model(feat_ov, adj_ov, lab_ov, ov_mask,
                      nfeat, nclass, base_path, save_path, crit, epochs):
    base  = get_model(nfeat, nclass)
    base.load_state_dict(torch.load(base_path, map_location=device))
    model = CaGCN(nclass, base).to(device)
    opt   = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=args.lr_for_cal, weight_decay=args.l2_for_cal)
    best, bad = 1e9, 0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(feat_ov, adj_ov)
        (crit(out[ov_mask], lab_ov[ov_mask]) +
         args.Lambda * intra_loss(out[ov_mask], lab_ov[ov_mask])).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            lv = crit(model(feat_ov, adj_ov)[ov_mask], lab_ov[ov_mask]).item()
        if lv < best: torch.save(model.state_dict(), save_path); best, bad = lv, 0
        else: bad += 1
        if bad == args.patience: break
    print(f'    CaGCN | OOD-val loss={best:.4f}')


def gen_pseudo(feat_tr, adj_tr, lab_tr, nfeat, nclass,
               base_path, cagcn_path, dyn_tr, pseudo_lab):
    base = get_model(nfeat, nclass)
    base.load_state_dict(torch.load(base_path, map_location=device))
    m = CaGCN(nclass, base).to(device)
    m.load_state_dict(torch.load(cagcn_path, map_location=device))
    m.eval()
    with torch.no_grad(): conf, pred = F.softmax(m(feat_tr, adj_tr), 1).max(1)
    in_tr = set(dyn_tr.nonzero(as_tuple=True)[0].tolist())
    added = 0
    for i in (conf > args.threshold).nonzero(as_tuple=True)[0].tolist():
        if i not in in_tr:
            pseudo_lab[i] = pred[i]; dyn_tr[i] = True; added += 1
    print(f'    伪标签新增={added}，train={dyn_tr.sum().item()}')
    return dyn_tr, pseudo_lab


def run_one_seed(seed, feat_tr, adj_tr, lab_tr, tr_mask_base,
                 feat_ov, adj_ov, lab_ov, ov_mask,
                 nclass, crit, prefix):
    tr_mask, id_val_mask, id_test_mask = stratified_split(
        lab_tr, tr_mask_base,
        val_ratio=args.id_val_ratio, test_ratio=args.id_test_ratio, seed=seed)
    tr_mask      = tr_mask.to(device)
    id_val_mask  = id_val_mask.to(device)
    id_test_mask = id_test_mask.to(device)

    base_path  = os.path.join(args.save_dir, f'{prefix}_seed{seed}_base.pth')
    cagcn_path = os.path.join(args.save_dir, f'{prefix}_seed{seed}_cagcn.pth')
    nfeat      = feat_tr.shape[1]
    dyn_tr     = tr_mask.clone()
    pseudo_lab = lab_tr.clone()

    for stage in range(1, args.stage + 1):
        is_last = (stage == args.stage)
        print(f'  [Stage {stage}/{args.stage}]')
        train_base_model(feat_tr, adj_tr, lab_tr,
                         tr_mask, id_val_mask, nclass, base_path,
                         crit, dyn_tr, pseudo_lab)
        cal_ep = args.epochs if is_last else args.epoch_for_st
        train_cagcn_model(feat_ov, adj_ov, lab_ov, ov_mask,
                          nfeat, nclass, base_path, cagcn_path, crit, cal_ep)
        if not is_last:
            dyn_tr, pseudo_lab = gen_pseudo(
                feat_tr, adj_tr, lab_tr, nfeat, nclass,
                base_path, cagcn_path, dyn_tr, pseudo_lab)

    def load_final():
        base = get_model(nfeat, nclass)
        base.load_state_dict(torch.load(base_path, map_location=device))
        m = CaGCN(nclass, base)
        m.load_state_dict(torch.load(cagcn_path, map_location=device))
        return m.to(device)

    return load_final, id_test_mask



def train_base_model_eerm(feat_train, feat_val, adj, lab, tr_mask, id_val_mask,
                          nclass, path, crit, dyn_tr, pseudo_lab):
    """EERM 专用 base 训练：env0 训练，env1 validation early stopping。"""
    model = get_model(feat_train.shape[1], nclass).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, bad = 1e9, 0
    for _ in range(args.epochs):
        model.train(); opt.zero_grad()
        crit(model(feat_train, adj)[dyn_tr], pseudo_lab[dyn_tr]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            lv = crit(model(feat_val, adj)[id_val_mask], lab[id_val_mask]).item()
        if lv < best:
            torch.save(model.state_dict(), path); best, bad = lv, 0
        else:
            bad += 1
        if bad == args.patience: break
    print(f'    base  | env1-val loss={best:.4f}')


def run_one_seed_eerm(seed, feat_tr, feat_val, adj, lab,
                      tr_mask, val_mask, nclass, crit, prefix):
    tr_mask = tr_mask.to(device); val_mask = val_mask.to(device)
    base_path  = os.path.join(args.save_dir, f'{prefix}_seed{seed}_base.pth')
    cagcn_path = os.path.join(args.save_dir, f'{prefix}_seed{seed}_cagcn.pth')
    nfeat      = feat_tr.shape[1]
    dyn_tr     = tr_mask.clone()
    pseudo_lab = lab.clone()

    print(f'  train={tr_mask.sum()} | OOD-val={val_mask.sum()}')
    for stage in range(1, args.stage + 1):
        is_last = (stage == args.stage)
        print(f'  [Stage {stage}/{args.stage}]')
        train_base_model_eerm(feat_tr, feat_val, adj, lab,
                              tr_mask, val_mask, nclass, base_path,
                              crit, dyn_tr, pseudo_lab)
        cal_ep = args.epochs if is_last else args.epoch_for_st
        train_cagcn_model(feat_val, adj, lab, val_mask,
                          nfeat, nclass, base_path, cagcn_path, crit, cal_ep)
        if not is_last:
            dyn_tr, pseudo_lab = gen_pseudo(
                feat_tr, adj, lab, nfeat, nclass,
                base_path, cagcn_path, dyn_tr, pseudo_lab)

    def load_final():
        base = get_model(nfeat, nclass)
        base.load_state_dict(torch.load(base_path, map_location=device))
        m = CaGCN(nclass, base)
        m.load_state_dict(torch.load(cagcn_path, map_location=device))
        return m.to(device)

    return load_final

# ══════════════════════════════════════════════════════════════
# 8. 汇总输出（均值±标准差 + 箱线图原始数据）
# ══════════════════════════════════════════════════════════════
def summarize(all_runs, split_names, all_keys, csv_path, boxplot_path, title,
             reliability_path=None, uncertainty_path=None):
    """
    all_runs          : list of {split_name: metric_dict}，长度 = runs
    all_keys          : 全量指标列名（不含 _probs/_u）
    csv_path          : 均值 ± 标准差 CSV
    boxplot_path      : 每 run × 每 split 的 acc/ece/nll/brier 原始值（箱线图用）
    reliability_path  : Reliability Diagram 数据 CSV（可选）
    uncertainty_path  : 逐样本不确定性数据 CSV（可选）
    """
    col_w  = 18
    show_k = ['acc', 'ece', 'nll', 'brier', 'ue_auroc', 'aurc']
    sep    = '═' * (26 + col_w * len(show_k))

    print(f'\n{sep}')
    print(f'  {title}  ({len(all_runs)} runs)')
    print(sep)
    print(f'  {"split":<24}' + ''.join(f'{k:>{col_w}}' for k in show_k))
    print('  ' + '─' * (24 + col_w * len(show_k)))

    # ── 均值±标准差 CSV ──────────────────────────────────────
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
        cells = ''.join(f'{mu[k]:.4f}±{sd[k]:.4f}'.rjust(col_w) for k in show_k)
        print(f'  {sname[:24]:<24}' + cells)
        mean_rows.append([sname]
                         + [f'{mu[k]:.6f}' for k in all_keys]
                         + [f'{sd[k]:.6f}'  for k in all_keys])

    # OOD 平均行
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
        cells = ''.join(f'{mu[k]:.4f}±{sd[k]:.4f}'.rjust(col_w) for k in show_k)
        print(f'  {"OOD-avg":<24}' + cells)
        mean_rows.append(['OOD-avg']
                         + [f'{mu[k]:.6f}' for k in all_keys]
                         + [f'{sd[k]:.6f}'  for k in all_keys])

    print(f'\n{sep}')
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.',
                exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerows(mean_rows)
    print(f'  均值±标准差 → {csv_path}')

    # ── 箱线图原始数据 CSV ───────────────────────────────────
    # 格式：run, split, acc, ece, nll, brier
    box_keys = ['acc', 'ece', 'nll', 'brier']
    box_rows = [['run', 'split'] + box_keys]
    for run_idx, r in enumerate(all_runs):
        for sname in split_names:
            m   = r.get(sname, {})
            row = [run_idx, sname] + [
                f'{m.get(k, float("nan")):.6f}' for k in box_keys]
            box_rows.append(row)
    with open(boxplot_path, 'w', newline='') as f:
        csv.writer(f).writerows(box_rows)
    print(f'  箱线图数据   → {boxplot_path}')

    # ── Reliability Diagram CSV ──────────────────────────────
    if reliability_path:
        rel_rows = [['run', 'split', 'bin', 'avg_confidence', 'accuracy', 'count']]
        for run_idx, r in enumerate(all_runs):
            for sname in split_names:
                m = r.get(sname, {})
                probs  = m.get('_probs')
                labels = m.get('_correct')  # reuse correct for label reconstruction
                # _probs and true labels needed — store reliability in metric dict
                bins_data = m.get('_reliability_bins', [])
                for b_idx, (avg_c, acc, cnt) in enumerate(bins_data):
                    rel_rows.append([run_idx, sname, b_idx+1,
                                     f'{avg_c:.6f}',
                                     f'{acc:.6f}' if not math.isnan(acc) else 'nan',
                                     cnt])
        with open(reliability_path, 'w', newline='') as f:
            csv.writer(f).writerows(rel_rows)
        print(f'  Reliability  → {reliability_path}')

    # ── 逐样本不确定性 CSV ───────────────────────────────────
    if uncertainty_path:
        unc_rows = [['run', 'split', 'u', 'correct']]
        for run_idx, r in enumerate(all_runs):
            for sname in split_names:
                m       = r.get(sname, {})
                u_arr   = m.get('_u')
                cor_arr = m.get('_correct')
                if u_arr is not None and cor_arr is not None:
                    for u_val, c_val in zip(u_arr.tolist(), cor_arr.tolist()):
                        unc_rows.append([run_idx, sname,
                                         f'{u_val:.6f}', int(c_val)])
        with open(uncertainty_path, 'w', newline='') as f:
            csv.writer(f).writerows(unc_rows)
        print(f'  不确定性样本 → {uncertainty_path}')


# ══════════════════════════════════════════════════════════════
# 9. 主函数
# ══════════════════════════════════════════════════════════════
def main():
    print(f'[配置] dataset={args.dataset} backbone={_backbone_name()} runs={args.runs}')
    print(f'[配置] stage={args.stage} epochs={args.epochs} patience={args.patience}')
    print(f'[配置] id_val_ratio={args.id_val_ratio} id_test_ratio={args.id_test_ratio}')

    # ── Elliptic ──────────────────────────────────────────────
    if args.dataset == 'elliptic':
        crit = make_criterion(2, args.class_weight)

        print('\n[Elliptic] 加载数据...')
        adj_tr, feat_tr, lab_tr = merge_elliptic(ELLIPTIC_TRAIN, args.data_dir)
        adj_tr = adj_tr.to(device); feat_tr = feat_tr.to(device)
        lab_tr = lab_tr.to(device)
        tr_mask_base = (lab_tr >= 0)

        adj_ov, feat_ov, lab_ov = merge_elliptic(ELLIPTIC_VAL, args.data_dir)
        adj_ov = adj_ov.to(device); feat_ov = feat_ov.to(device)
        lab_ov = lab_ov.to(device)
        ov_mask = (lab_ov >= 0).to(device)

        print('[Elliptic] 加载 OOD-test 图...')
        test_graphs = []
        for i, steps in enumerate(ELLIPTIC_TESTS):
            adj_te, feat_te, lab_te = merge_elliptic(steps, args.data_dir)
            te_mask = (lab_te >= 0)
            print(f'  OOD-test_{i} steps={steps}: '
                  f'labeled={te_mask.sum()} illicit={(lab_te[te_mask]==1).sum()}')
            test_graphs.append((adj_te.to(device), feat_te.to(device),
                                lab_te.to(device), te_mask.to(device)))

        split_names = ['ID-test'] + [
            f'OOD-test_{i}(s{ELLIPTIC_TESTS[i][0]}-{ELLIPTIC_TESTS[i][-1]})'
            for i in range(len(ELLIPTIC_TESTS))]
        all_keys = build_all_keys(binary=True)

        all_runs = []
        for r in range(args.runs):
            seed = args.base_seed + r
            print(f'\n{"="*65}')
            print(f'  CaGCN  Run {r+1}/{args.runs}  seed={seed}')
            print(f'{"="*65}')

            load_final, id_test_mask = run_one_seed(
                seed, feat_tr, adj_tr, lab_tr, tr_mask_base,
                feat_ov, adj_ov, lab_ov, ov_mask,
                nclass=2, crit=crit, prefix=_tagged_prefix('elliptic'))

            run_res = {}

            # ID-test
            model = load_final()
            r_id  = evaluate(model, feat_tr, adj_tr, lab_tr,
                             id_test_mask, 2, binary=True)
            print(f'  ID-test | acc={r_id["acc"]:.4f} f1={r_id["f1"]:.4f} '
                  f'ece={r_id["ece"]:.4f} nll={r_id["nll"]:.4f} '
                  f'ue_auroc={r_id["ue_auroc"]:.4f} aurc={r_id["aurc"]:.4f}')
            run_res['ID-test'] = r_id

            # OOD-test × 9
            for i, (adj_te, feat_te, lab_te, te_mask) in enumerate(test_graphs):
                model     = load_final()
                r_ood_raw = evaluate(model, feat_te, adj_te, lab_te,
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
                  csv_path=os.path.join(args.save_dir, 'elliptic_cagcn_results.csv'),
                  boxplot_path=os.path.join(args.save_dir, 'elliptic_cagcn_boxplot.csv'),
                  title=_tagged_title('Elliptic — CaGCN'),
                  reliability_path=os.path.join(args.save_dir, 'elliptic_cagcn_reliability.csv'),
                  uncertainty_path=os.path.join(args.save_dir, 'elliptic_cagcn_uncertainty_samples.csv'))

    # ── OGB-Arxiv ─────────────────────────────────────────────
    elif args.dataset == 'arxiv':
        print('\n[Arxiv] 加载数据...')
        (adj, features, labels, years, nclass,
         base_tr, ov_mask, ood_masks) = load_arxiv(args.data_path)

        adj       = adj.to(device);     features = features.to(device)
        labels    = labels.to(device);  ov_mask  = ov_mask.to(device)
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
            print(f'  CaGCN  Run {r+1}/{args.runs}  seed={seed}')
            print(f'{"="*65}')

            load_final, id_test_mask = run_one_seed(
                seed, features, adj, labels, base_tr,
                features, adj, labels, ov_mask,
                nclass=nclass, crit=crit, prefix=_tagged_prefix('arxiv'))

            run_res = {}

            model = load_final()
            r_id  = evaluate(model, features, adj, labels,
                             id_test_mask, nclass, binary=False)
            print(f'  ID-test | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f} '
                  f'nll={r_id["nll"]:.4f} ue_auroc={r_id["ue_auroc"]:.4f} '
                  f'aurc={r_id["aurc"]:.4f}')
            run_res['ID-test'] = r_id

            for i, (te_mask, (ty0, ty1)) in enumerate(zip(ood_masks, ARXIV_TESTS)):
                model     = load_final()
                r_ood_raw = evaluate(model, features, adj, labels,
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
                  csv_path=os.path.join(args.save_dir, 'arxiv_cagcn_results.csv'),
                  boxplot_path=os.path.join(args.save_dir, 'arxiv_cagcn_boxplot.csv'),
                  title=_tagged_title('OGB-Arxiv — CaGCN'),
                  reliability_path=os.path.join(args.save_dir, 'arxiv_cagcn_reliability.csv'),
                  uncertainty_path=os.path.join(args.save_dir, 'arxiv_cagcn_uncertainty_samples.csv'))



    elif args.dataset == 'eerm':
        assert args.eerm_root, '--eerm_root 必须指定'
        print(f'\n[EERM-{args.eerm_dataset}] 加载数据...')
        (adj, feat_tr, feat_val, feat_oods,
         labels_t, nclass,
         tr_mask, val_mask, test_mask,
         ood_names) = load_eerm(args.eerm_root, args.eerm_dataset)

        adj      = adj.to(device)
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
            print(f'  CaGCN  EERM-{ds_tag}  Run {r+1}/{args.runs}  seed={seed}')
            print(f'{"="*65}')
            torch.manual_seed(seed); np.random.seed(seed)

            load_final = run_one_seed_eerm(
                seed, feat_tr, feat_val, adj, labels_t,
                tr_mask, val_mask,
                nclass=nclass, crit=crit,
                prefix=_tagged_prefix(f'eerm_{ds_tag}'))

            run_res = {}
            model = load_final()

            r_id = evaluate(model, feat_tr, adj, labels_t,
                            test_mask, nclass, binary=False)
            print(f'  ID-test | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f} '
                  f'ue_auroc={r_id["ue_auroc"]:.4f} aurc={r_id["aurc"]:.4f}')
            run_res['ID-test'] = r_id

            for feat_ood, name in zip(feat_oods, ood_names):
                model = load_final()
                r_ood_raw = evaluate(model, feat_ood, adj, labels_t,
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
                      f'{ds_tag}_cagcn_results.csv'),
                  boxplot_path=os.path.join(args.save_dir,
                      f'{ds_tag}_cagcn_boxplot.csv'),
                  title=_tagged_title(f'EERM-{ds_tag} — CaGCN'),
                  reliability_path=os.path.join(args.save_dir,
                      f'{ds_tag}_cagcn_reliability.csv'),
                  uncertainty_path=os.path.join(args.save_dir,
                      f'{ds_tag}_cagcn_uncertainty_samples.csv'))

if __name__ == '__main__':

    main()
