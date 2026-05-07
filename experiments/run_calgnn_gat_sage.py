"""
calGNN — Elliptic & OGB-Arxiv 完整指标版
==========================================
【指标体系（实验文档 RQ1/RQ2/RQ3）】

RQ1: acc, ece(M=15), nll, brier; [Elliptic] f1/prec/rec
     [OOD] delta_ece, delta_nll, delta_brier
RQ2: ue_auroc, ue_aupr  (u = 1-max(p)，每种校准方法独立计算)
     [OOD] delta_ue_auroc, ood_auroc
RQ3: aurc, risk@0.1~1.0
     [OOD] aurc_ood, srtr@0.1~1.0, srtr_auc

calGNN 特殊性：每种校准方法（Uncal/TS/HB/Iso/BBQ/MetaCal/RBS）
各自独立输出一套指标，CSV 按 (split, cal_method) 为行。

用法:
    python calgnn_elliptic_arxiv.py --dataset elliptic --data_dir  ./elliptic --runs 5
    python calgnn_elliptic_arxiv.py --dataset arxiv    --data_path ./data.pkl  --runs 5
"""
import sys; sys.path.insert(0, 'src')


import os, pickle, argparse, warnings, csv, copy, math
import numpy as np
import scipy.sparse as sp
import scipy.special
import scipy.optimize
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression as SklearnIso
from sklearn.model_selection import train_test_split
from scipy.stats import entropy as scipy_entropy
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
parser.add_argument('--add_cal_loss',  action='store_true', default=False)
parser.add_argument('--alpha',         type=float, default=0.5)
parser.add_argument('--lmbda',         type=float, default=0.1)
parser.add_argument('--num_bins_rbs',  type=int,   default=10)
parser.add_argument('--id_val_ratio',  type=float, default=0.1)
parser.add_argument('--id_test_ratio', type=float, default=0.1)
parser.add_argument('--class_weight',  type=float, default=10.0)
parser.add_argument('--eerm_dataset', type=str, default='cora',
                    choices=['cora', 'amazon'],
                    help='EERM 数据集: cora 或 amazon')
parser.add_argument('--eerm_root',    type=str, default=None,
                    help='EERM 数据集根目录（含 gen/ raw/）')
parser.add_argument('--save_dir',      type=str,   default='./save_model_calgnn_gat_sage')
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

CAL_METHODS     = ['Uncal', 'TS', 'HB', 'Iso', 'BBQ', 'MetaCal', 'RBS']  # 内部计算全保留
CAL_METHODS_OUT = ['Uncal', 'RBS']  # 只输出这两种方法
COV_FULL    = [round(0.1 * i, 1) for i in range(1, 11)]


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


def compute_split_metrics(probs, labels, nclass, binary=False):
    """
    probs  : (N,C) numpy softmax 概率
    labels : (N,)  numpy int
    u = 1 - max(p)，calGNN 无自有不确定性
    返回 dict，含 '_probs'/'_u' 内部字段（不写 CSV）
    """
    u = 1. - probs.max(1)
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
# 2. GNN 模型
# ══════════════════════════════════════════════════════════════
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
        self.dp = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters(); self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dp, self.training)
        return self.conv2(x, edge_index)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.conv1 = GATConv(nfeat, nhid, heads=8, dropout=dropout, concat=True)
        self.conv2 = GATConv(nhid*8, nclass, heads=1, dropout=dropout, concat=False)
        self.dp = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters(); self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dp, self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dp, self.training)
        return self.conv2(x, edge_index)


def get_model(nfeat, nclass):
    return get_pyg_backbone(args.model, nfeat, args.hidden, nclass,
                            args.dropout, heads=getattr(args, 'backbone_heads', 8))


# ══════════════════════════════════════════════════════════════
# 3. calGNN 校准 loss
# ══════════════════════════════════════════════════════════════
def cal_loss(y_true, logits, lmbda, epoch, epochs, bin_num=15):
    probs      = F.softmax(logits, dim=1)
    y_pred     = probs.max(1)[1]
    confidence = probs.max(1)[0]
    bin_size   = torch.tensor(1.0 / bin_num)
    upper_bounds = torch.arange(bin_size, 1 + bin_size, bin_size)
    acc_per_sample = []
    for conf_thresh in upper_bounds:
        lo = conf_thresh - bin_size
        mask = (confidence > lo) & (confidence <= conf_thresh)
        acc_bin = (y_pred[mask] == y_true[mask]).float().mean() if mask.sum() > 0 \
                  else torch.tensor(0.0, device=logits.device)
        acc_per_sample.append((mask, acc_bin))
    acc_vector = torch.zeros(len(y_true), device=logits.device)
    for mask, acc_bin in acc_per_sample:
        acc_vector[mask] = acc_bin
    cal_term = -(acc_vector * torch.log(confidence.clamp(min=1e-8))).sum()
    anneal   = torch.min(torch.tensor(lmbda),
                         torch.tensor(lmbda * (epoch + 1) / epochs))
    return cal_term * anneal


# ══════════════════════════════════════════════════════════════
# 4. Post-hoc 校准方法
# ══════════════════════════════════════════════════════════════
class TemperatureScaling:
    def __init__(self): self.T = 1.0

    def fit(self, logits_np, labels_np):
        def obj(T):
            P  = scipy.special.softmax(logits_np / T, axis=1)
            return -np.sum(np.log(P[np.arange(len(labels_np)), labels_np] + 1e-30))
        def grad(T):
            E  = np.exp(logits_np / T)
            li = logits_np[np.arange(len(labels_np)), labels_np]
            dT = (np.sum(E*(logits_np - li.reshape(-1,1)), 1)/np.sum(E,1)).sum()
            return -dT / T**2
        T_opt = scipy.optimize.fmin_bfgs(obj, x0=1.0, fprime=grad, gtol=1e-6, disp=False)[0]
        self.T = max(T_opt, 1e-3); return self

    def predict_proba(self, logits_np):
        return scipy.special.softmax(logits_np / self.T, axis=1)


class HistogramBinning:
    def __init__(self, n_bins=20): self.n_bins = n_bins

    def fit(self, probs_np, labels_np):
        nclass = probs_np.shape[1]; self.calibrators_ = []
        for c in range(nclass):
            p1 = probs_np[:, c]; y1 = (labels_np == c).astype(float)
            bins = np.quantile(p1, np.linspace(0, 1, self.n_bins + 1))
            bins[0] = 0.; bins[-1] = 1.; bins = np.unique(bins)
            dig  = np.digitize(p1, bins).clip(1, len(bins)-1)
            freq = np.array([y1[dig==i].mean() if (dig==i).sum()>0 else 0.
                             for i in range(1, len(bins))])
            self.calibrators_.append((bins, freq))
        return self

    def predict_proba(self, probs_np):
        nclass = probs_np.shape[1]; out = np.zeros_like(probs_np)
        for c, (bins, freq) in enumerate(self.calibrators_):
            dig = np.digitize(probs_np[:, c], bins).clip(1, len(bins)-1)
            out[:, c] = np.array([freq[i-1] for i in dig])
        return (out / out.sum(1, keepdims=True).clip(min=1e-8))


class IsotonicCalib:
    def fit(self, probs_np, labels_np):
        self.calibrators_ = []
        for c in range(probs_np.shape[1]):
            ir = SklearnIso(out_of_bounds='clip')
            ir.fit(probs_np[:, c], (labels_np == c).astype(float))
            self.calibrators_.append(ir)
        return self

    def predict_proba(self, probs_np):
        out = np.stack([ir.predict(probs_np[:,c])
                        for c, ir in enumerate(self.calibrators_)], axis=1)
        return (out / out.sum(1, keepdims=True).clip(min=1e-8)).clip(0, 1)


class BBQ:
    def __init__(self, C=10): self.C = C

    def _fit_binary(self, p1, y):
        N = len(y)
        n_min = max(1, int(np.floor(N**(1/3)/self.C)))
        n_max = max(max(int(np.ceil(N/5)), int(np.ceil(self.C*N**(1/3)))), n_min)
        n_max = min(n_max, 50)  # 大图(N~2.5万,C=40)下原始n_max~1168→4.7万次binning，限制为50
        n_min = min(n_min, n_max)
        T = n_max - n_min + 1
        binnings, scores, freqs = [], [], []
        for nb in range(n_min, n_max+1):
            bins = np.quantile(p1, np.linspace(0, 1, nb+1))
            bins[0] = 0.; bins[-1] = 1.
            bins = np.maximum.accumulate(bins)
            dig  = np.digitize(p1, bins).clip(1, len(bins)-1)
            N_b  = np.bincount(dig, minlength=len(bins))[1:]
            m_b  = np.array([y[dig==i].sum() for i in range(1, len(bins))])
            n_b  = N_b - m_b
            p_b  = np.clip((bins[1:]+bins[:-1])/2, 1e-9, 1-1e-9)
            al   = np.clip(N/T*p_b, 1e-9, None)
            bt   = np.clip(N/T*(1-p_b), 1e-9, None)
            ll   = (scipy.special.gammaln(N/T)
                    + scipy.special.gammaln(m_b+al)
                    + scipy.special.gammaln(n_b+bt)
                    - scipy.special.gammaln(N_b+N/T)
                    - scipy.special.gammaln(al)
                    - scipy.special.gammaln(bt)).sum()
            scores.append(-np.log(T) + ll); binnings.append(bins)
            freqs.append([y[dig==i].mean() if (dig==i).sum()>0 else p_b[i-1]
                          for i in range(1, len(bins))])
        self._binnings = binnings; self._scores = scores; self._freqs = freqs

    def fit(self, probs_np, labels_np):
        self._models = []
        for c in range(probs_np.shape[1]):
            self._fit_binary(probs_np[:,c], (labels_np==c).astype(float))
            self._models.append((self._binnings, self._scores, self._freqs))
            self._binnings = self._scores = self._freqs = []
        return self

    def predict_proba(self, probs_np):
        out = np.zeros_like(probs_np)
        for c, (binnings, scores, freqs) in enumerate(self._models):
            w   = np.exp(np.array(scores) - scipy.special.logsumexp(scores))
            col = np.zeros(len(probs_np))
            for bins, freq, wi in zip(binnings, freqs, w):
                dig = np.searchsorted(bins, probs_np[:,c]).clip(0, len(bins)-2)
                col += wi * np.array([freq[d] for d in dig])
            out[:, c] = col
        return (out / out.sum(1, keepdims=True).clip(min=1e-8)).clip(0, 1)


class MetaCalTS:
    def fit(self, logits_np, labels_np):
        ts = TemperatureScaling(); ts.fit(logits_np, labels_np); self.T = ts.T; return self
    def predict(self, logits_np):
        return scipy.special.softmax(logits_np / self.T, axis=1)


class MetaCalMisCoverage:
    def __init__(self, alpha=0.05): self.alpha = alpha

    def fit(self, logits_np, labels_np):
        neg_ind = logits_np.argmax(1) == labels_np
        xs_neg, ys_neg = logits_np[neg_ind], labels_np[neg_ind]
        xs_pos, ys_pos = logits_np[~neg_ind], labels_np[~neg_ind]
        if len(xs_neg) <= 1:
            self.threshold = float('inf')
            self.base_model = MetaCalTS().fit(logits_np, labels_np)
            return self
        n1 = min(max(1, len(xs_neg)//10), 500)
        x1, x2, _, y2 = train_test_split(
            xs_neg, ys_neg, train_size=min(n1, len(xs_neg)-1), random_state=0)
        x2 = np.r_[x2, xs_pos]; y2 = np.r_[y2, ys_pos]
        scores_x1 = scipy_entropy(scipy.special.softmax(x1, axis=1), axis=1)
        self.threshold = np.quantile(scores_x1, 1 - self.alpha)
        scores_x2 = scipy_entropy(scipy.special.softmax(x2, axis=1), axis=1)
        cond = scores_x2 < self.threshold
        self.base_model = MetaCalTS()
        self.base_model.fit(x2[cond] if cond.sum() > 1 else logits_np,
                            y2[cond] if cond.sum() > 1 else labels_np)
        return self

    def predict(self, logits_np):
        scores  = scipy_entropy(scipy.special.softmax(logits_np, axis=1), axis=1)
        neg_ind = scores < self.threshold
        out     = np.full_like(logits_np, 1.0/logits_np.shape[1], dtype=float)
        if neg_ind.sum() > 0:
            out[neg_ind] = self.base_model.predict(logits_np[neg_ind])
        return out


def compute_sm_conf(edge_index_t, N, probs_all_t):
    """
    GPU sparse 版本的 smoothed confidence。
    edge_index_t : torch.LongTensor (2, E) on device
    probs_all_t  : torch.FloatTensor (N, C) on device
    返回         : numpy (N,)  —  每节点邻域平均预测中，预测类别对应的概率
    """
    src, dst = edge_index_t[0], edge_index_t[1]
    vals = torch.ones(src.size(0), dtype=torch.float32, device=device)
    # torch.sparse_coo_tensor: (N, N) sparse
    A_sp = torch.sparse_coo_tensor(
        torch.stack([dst, src], 0), vals, (N, N)   # dst←src，行=目标节点
    ).coalesce()
    # sparse @ dense: (N,N) × (N,C) → (N,C)，全在 GPU
    AP  = torch.sparse.mm(A_sp, probs_all_t)        # (N, C)
    deg = torch.sparse.sum(A_sp, dim=1).to_dense()  # (N,)
    no_nbr  = (deg == 0)
    deg_safe = deg.clone(); deg_safe[no_nbr] = 1.
    AP = AP / deg_safe.unsqueeze(1)
    AP[no_nbr] = probs_all_t[no_nbr]
    pred_cls = probs_all_t.argmax(1)                # (N,)
    sm = AP[torch.arange(N, device=device), pred_cls]
    return sm.cpu().numpy()                          # (N,)


def rbs_fit(sm_conf_sub, logits_np, labels_np, num_bins):
    bins = [1./num_bins*(i+1) for i in range(num_bins)]
    T_list = []
    for i, b in enumerate(bins):
        lo   = 0. if i == 0 else bins[i-1]
        mask = (sm_conf_sub > lo) & (sm_conf_sub <= b)
        if mask.sum() > 1:
            ts = TemperatureScaling(); ts.fit(logits_np[mask], labels_np[mask])
            T_list.append(ts.T)
        else:
            T_list.append(1.0)
    return T_list, bins


def apply_rbs(T_list, bins, sm_conf_sub, logits_np):
    T_vec = np.ones(len(logits_np), dtype=np.float32)
    for i, b in enumerate(bins):
        lo = 0. if i == 0 else bins[i-1]
        m  = (sm_conf_sub > lo) & (sm_conf_sub <= b)
        T_vec[m] = T_list[i]
    with torch.no_grad():
        t = torch.tensor(logits_np, dtype=torch.float32, device=device)
        v = torch.tensor(T_vec, dtype=torch.float32, device=device).unsqueeze(1)
        return torch.softmax(t / v, dim=1).cpu().numpy()


# ══════════════════════════════════════════════════════════════
# 5. 图工具 / 数据加载
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
        m = torch.zeros(N, dtype=torch.bool)
        c = np.concatenate([x for x in lists if len(x) > 0])
        if len(c): m[c] = True
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
    ei   = np.unique(np.vstack([rows, cols]), axis=1)
    ei_t = torch.from_numpy(ei.astype(np.int64))
    return ei_t, torch.FloatTensor(np.concatenate(feats, 0)), \
           torch.LongTensor(np.concatenate(labs, 0)), N, ei


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
    ei_np = np.unique(np.vstack([np.concatenate([src,dst]),
                                  np.concatenate([dst,src])]), axis=1)
    ei_t  = torch.from_numpy(ei_np.astype(np.int64))
    years = torch.LongTensor(node_year)
    base_tr  = (years <= ARXIV_TRAIN_YEAR)
    oy0, oy1 = ARXIV_OODVAL_YEARS
    ov_mask  = (years >= oy0) & (years <= oy1)
    ood_masks = [(years >= ty0) & (years <= ty1) for ty0, ty1 in ARXIV_TESTS]
    return (ei_t, torch.FloatTensor(node_feat), torch.LongTensor(labels),
            years, nclass, base_tr, ov_mask, ood_masks, N, ei_np)


# ══════════════════════════════════════════════════════════════
# EERM Cora / Amazon-Photo 数据加载
# ══════════════════════════════════════════════════════════════

def load_eerm(eerm_root, dataset='cora'):
    """
    加载 EERM 格式的 Cora 或 Amazon-Photo 数据集。
    格式：gen/{i}-gcn.pkl = (x_tensor, y_tensor)
    环境：0=train, 1=OOD-val, 2~9=OOD-test_0~7。

    calGNN 使用 PyG 的 edge_index，因此本函数返回无向 edge_index，而不是稀疏邻接矩阵。
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
# 6. 训练 base model
# ══════════════════════════════════════════════════════════════
def make_criterion(nclass, class_weight=None):
    if class_weight and nclass == 2:
        return nn.CrossEntropyLoss(
            weight=torch.FloatTensor([1., class_weight]).to(device))
    return nn.CrossEntropyLoss()


def train_base(x, ei, labels, tr_mask, id_val_mask, nclass, save_path, crit):
    model = get_model(x.shape[1], nclass).to(device)
    model.reset_parameters()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, bad, best_state = 1e9, 0, None
    for epoch in range(args.epochs):
        model.train(); opt.zero_grad()
        logits   = model(x, ei)
        nll_loss = crit(logits[tr_mask], labels[tr_mask])
        if args.add_cal_loss:
            loss = (args.alpha * nll_loss
                    + (1-args.alpha) * cal_loss(labels[tr_mask], logits[tr_mask],
                                                args.lmbda, epoch, args.epochs))
        else:
            loss = nll_loss
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lv = F.cross_entropy(model(x, ei)[id_val_mask],
                                 labels[id_val_mask]).item()
        if lv < best:
            best_state = copy.deepcopy(model.state_dict()); best, bad = lv, 0
        else:
            bad += 1
        if bad == args.patience: break
    model.load_state_dict(best_state)
    torch.save(best_state, save_path)
    print(f'    base  | best ID-val loss={best:.4f}')
    return model


@torch.no_grad()
@torch.no_grad()
def get_logits_probs(model, x, ei, mask):
    model.eval()
    out    = model(x, ei)
    logits_t = out[mask]
    probs_t  = torch.softmax(logits_t, dim=1)
    return logits_t.cpu().numpy(), probs_t.cpu().numpy()


@torch.no_grad()
def get_all_probs(model, x, ei):
    """返回全图 softmax 概率，GPU 计算后转 numpy"""
    model.eval()
    return torch.softmax(model(x, ei), dim=1).cpu().numpy()


@torch.no_grad()
def get_all_probs_tensor(model, x, ei):
    """返回全图 softmax 概率，保留在 GPU 上（供 compute_sm_conf 使用）"""
    model.eval()
    return torch.softmax(model(x, ei), dim=1)  # (N, C) on device


# ══════════════════════════════════════════════════════════════
# 7. 单次实验：为每种校准方法计算全量指标
# ══════════════════════════════════════════════════════════════
def eval_all_methods(logits, probs, labels, nclass, binary,
                     ts_model, hb_model, iso_model, bbq_model,
                     meta_model, rbs_probs):
    """
    对一个 split 的七种校准方法分别调用 compute_split_metrics。
    返回 { cal_method: metric_dict }，含 _probs/_u 内部字段。
    """
    results = {}
    cal_probs_map = {
        'Uncal'  : probs,
        'TS'     : ts_model.predict_proba(logits),
        'HB'     : hb_model.predict_proba(probs),
        'Iso'    : iso_model.predict_proba(probs),
        'BBQ'    : bbq_model.predict_proba(probs),
        'MetaCal': meta_model.predict(logits),
        'RBS'    : rbs_probs,
    }
    for cm, p in cal_probs_map.items():
        results[cm] = compute_split_metrics(p, labels, nclass, binary=binary)
    return results


def run_one_seed_elliptic(seed,
        ei_tr, x_tr, lab_tr, tr_base_mask, N_tr,
        ei_ov, x_ov, lab_ov, ov_mask, N_ov,
        test_graphs, nclass, crit):

    tr_mask, id_val_mask, id_test_mask = stratified_split(
        lab_tr, tr_base_mask,
        val_ratio=args.id_val_ratio, test_ratio=args.id_test_ratio, seed=seed)
    tr_mask = tr_mask.to(device); id_val_mask = id_val_mask.to(device)
    id_test_mask = id_test_mask.to(device)
    print(f'  train={tr_mask.sum()} (illicit={(lab_tr[tr_mask]==1).sum()}) '
          f'| ID-val={id_val_mask.sum()} | ID-test={id_test_mask.sum()}')

    base_path = os.path.join(args.save_dir, f'elliptic_seed{seed}_base.pth')
    model     = train_base(x_tr, ei_tr, lab_tr, tr_mask, id_val_mask,
                           nclass, base_path, crit)

    # 拟合校准方法（OOD-val上）
    ov_logits, ov_probs = get_logits_probs(model, x_ov, ei_ov, ov_mask)
    ov_labels    = lab_ov[ov_mask].cpu().numpy()
    probs_ov_t   = get_all_probs_tensor(model, x_ov, ei_ov)
    probs_all_ov = probs_ov_t.cpu().numpy()
    ts_model   = TemperatureScaling().fit(ov_logits, ov_labels)
    hb_model   = HistogramBinning().fit(ov_probs, ov_labels)
    iso_model  = IsotonicCalib().fit(ov_probs, ov_labels)
    bbq_model  = BBQ().fit(ov_probs, ov_labels)
    meta_model = MetaCalMisCoverage().fit(ov_logits, ov_labels)
    sm_conf_ov  = compute_sm_conf(ei_ov, N_ov, probs_ov_t)
    ov_idx      = ov_mask.cpu().numpy().nonzero()[0]
    T_list_rbs, bins_rbs = rbs_fit(sm_conf_ov[ov_idx], ov_logits, ov_labels,
                                    args.num_bins_rbs)

    run_res = {}

    # ── ID-test ──
    id_logits, id_probs = get_logits_probs(model, x_tr, ei_tr, id_test_mask)
    id_labels   = lab_tr[id_test_mask].cpu().numpy()
    sm_conf_tr  = compute_sm_conf(ei_tr, N_tr, get_all_probs_tensor(model, x_tr, ei_tr))
    id_idx      = id_test_mask.cpu().numpy().nonzero()[0]
    rbs_id      = apply_rbs(T_list_rbs, bins_rbs, sm_conf_tr[id_idx], id_logits)
    run_res['ID-test'] = eval_all_methods(
        id_logits, id_probs, id_labels, nclass, binary=True,
        ts_model=ts_model, hb_model=hb_model, iso_model=iso_model,
        bbq_model=bbq_model, meta_model=meta_model, rbs_probs=rbs_id)

    # ── OOD-test × 9 ──
    for i, (ei_te, x_te, lab_te, te_mask, N_te) in enumerate(test_graphs):
        te_logits, te_probs = get_logits_probs(model, x_te, ei_te, te_mask)
        te_labels  = lab_te[te_mask].cpu().numpy()
        sm_conf_te = compute_sm_conf(ei_te, N_te, get_all_probs_tensor(model, x_te, ei_te))
        te_idx     = te_mask.cpu().numpy().nonzero()[0]
        rbs_te     = apply_rbs(T_list_rbs, bins_rbs, sm_conf_te[te_idx], te_logits)

        ood_raw = eval_all_methods(
            te_logits, te_probs, te_labels, nclass, binary=True,
            ts_model=ts_model, hb_model=hb_model, iso_model=iso_model,
            bbq_model=bbq_model, meta_model=meta_model, rbs_probs=rbs_te)

        # 追加跨 split 指标（每种校准方法独立计算）
        name = f'OOD-test_{i}(s{ELLIPTIC_TESTS[i][0]}-{ELLIPTIC_TESTS[i][-1]})'
        run_res[name] = {}
        for cm in CAL_METHODS_OUT:
            id_m  = run_res['ID-test'][cm]
            ood_m = ood_raw[cm]
            run_res[name][cm] = add_cross_split_metrics(
                id_m, ood_m, id_m['_u'], ood_m['_u'])

    return run_res


def run_one_seed_arxiv(seed,
        ei, x, labels, base_tr, N, ei_np,
        ov_mask, ood_masks, nclass, crit):

    tr_mask, id_val_mask, id_test_mask = stratified_split(
        labels, base_tr,
        val_ratio=args.id_val_ratio, test_ratio=args.id_test_ratio, seed=seed)
    tr_mask = tr_mask.to(device); id_val_mask = id_val_mask.to(device)
    id_test_mask = id_test_mask.to(device); ov_mask = ov_mask.to(device)
    print(f'  train={tr_mask.sum()} | ID-val={id_val_mask.sum()} '
          f'| ID-test={id_test_mask.sum()}')

    base_path = os.path.join(args.save_dir, f'arxiv_seed{seed}_base.pth')
    model     = train_base(x, ei, labels, tr_mask, id_val_mask,
                           nclass, base_path, crit)

    ov_logits, ov_probs = get_logits_probs(model, x, ei, ov_mask)
    ov_labels   = labels[ov_mask].cpu().numpy()
    probs_all_t = get_all_probs_tensor(model, x, ei)   # (N,C) on GPU
    probs_all   = probs_all_t.cpu().numpy()
    ts_model   = TemperatureScaling().fit(ov_logits, ov_labels)
    hb_model   = HistogramBinning().fit(ov_probs, ov_labels)
    iso_model  = IsotonicCalib().fit(ov_probs, ov_labels)
    bbq_model  = BBQ().fit(ov_probs, ov_labels)
    meta_model = MetaCalMisCoverage().fit(ov_logits, ov_labels)
    sm_conf_all = compute_sm_conf(ei, N, probs_all_t)  # GPU sparse mm
    ov_idx      = ov_mask.cpu().numpy().nonzero()[0]
    T_list_rbs, bins_rbs = rbs_fit(sm_conf_all[ov_idx], ov_logits, ov_labels,
                                    args.num_bins_rbs)

    run_res = {}

    # ── ID-test ──
    id_logits, id_probs = get_logits_probs(model, x, ei, id_test_mask)
    id_labels = labels[id_test_mask].cpu().numpy()
    id_idx    = id_test_mask.cpu().numpy().nonzero()[0]
    rbs_id    = apply_rbs(T_list_rbs, bins_rbs, sm_conf_all[id_idx], id_logits)
    run_res['ID-test'] = eval_all_methods(
        id_logits, id_probs, id_labels, nclass, binary=False,
        ts_model=ts_model, hb_model=hb_model, iso_model=iso_model,
        bbq_model=bbq_model, meta_model=meta_model, rbs_probs=rbs_id)

    # ── OOD-test × 3 ──
    for i, (te_mask_cpu, (ty0, ty1)) in enumerate(zip(ood_masks, ARXIV_TESTS)):
        te_mask = te_mask_cpu.to(device)
        te_logits, te_probs = get_logits_probs(model, x, ei, te_mask)
        te_labels = labels[te_mask].cpu().numpy()
        te_idx    = te_mask.cpu().numpy().nonzero()[0]
        rbs_te    = apply_rbs(T_list_rbs, bins_rbs, sm_conf_all[te_idx], te_logits)

        ood_raw = eval_all_methods(
            te_logits, te_probs, te_labels, nclass, binary=False,
            ts_model=ts_model, hb_model=hb_model, iso_model=iso_model,
            bbq_model=bbq_model, meta_model=meta_model, rbs_probs=rbs_te)

        name = f'OOD-test_{i}({ty0}-{ty1})'
        run_res[name] = {}
        for cm in CAL_METHODS_OUT:
            id_m  = run_res['ID-test'][cm]
            ood_m = ood_raw[cm]
            run_res[name][cm] = add_cross_split_metrics(
                id_m, ood_m, id_m['_u'], ood_m['_u'])

    return run_res



def train_base_eerm(x_train, x_val, ei, labels, tr_mask, id_val_mask,
                    nclass, save_path, crit):
    """EERM 专用 base 训练：env0 训练，env1 validation early stopping。"""
    model = get_model(x_train.shape[1], nclass).to(device)
    model.reset_parameters()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, bad, best_state = 1e9, 0, None
    for epoch in range(args.epochs):
        model.train(); opt.zero_grad()
        logits = model(x_train, ei)
        nll_loss = crit(logits[tr_mask], labels[tr_mask])
        if args.add_cal_loss:
            loss = (args.alpha * nll_loss
                    + (1-args.alpha) * cal_loss(labels[tr_mask], logits[tr_mask],
                                                args.lmbda, epoch, args.epochs))
        else:
            loss = nll_loss
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lv = F.cross_entropy(model(x_val, ei)[id_val_mask],
                                 labels[id_val_mask]).item()
        if lv < best:
            best_state = copy.deepcopy(model.state_dict()); best, bad = lv, 0
        else:
            bad += 1
        if bad == args.patience: break
    model.load_state_dict(best_state)
    torch.save(best_state, save_path)
    print(f'    base  | best env1-val loss={best:.4f}')
    return model


def run_one_seed_eerm(seed, ei, x_tr, x_val, feat_oods, labels,
                      tr_mask, val_mask, test_mask, ood_names, N,
                      nclass, crit, prefix):
    tr_mask = tr_mask.to(device); val_mask = val_mask.to(device); test_mask = test_mask.to(device)
    print(f'  train={tr_mask.sum()} | OOD-val={val_mask.sum()} | ID-test={test_mask.sum()}')

    base_path = os.path.join(args.save_dir, f'{prefix}_seed{seed}_base.pth')
    model = train_base_eerm(x_tr, x_val, ei, labels,
                            tr_mask, val_mask, nclass, base_path, crit)

    # 拟合校准方法：env1 / OOD-val。
    ov_logits, ov_probs = get_logits_probs(model, x_val, ei, val_mask)
    ov_labels = labels[val_mask].cpu().numpy()
    probs_val_t = get_all_probs_tensor(model, x_val, ei)
    ts_model   = TemperatureScaling().fit(ov_logits, ov_labels)
    hb_model   = HistogramBinning().fit(ov_probs, ov_labels)
    iso_model  = IsotonicCalib().fit(ov_probs, ov_labels)
    bbq_model  = BBQ().fit(ov_probs, ov_labels)
    meta_model = MetaCalMisCoverage().fit(ov_logits, ov_labels)
    sm_conf_val = compute_sm_conf(ei, N, probs_val_t)
    ov_idx      = val_mask.cpu().numpy().nonzero()[0]
    T_list_rbs, bins_rbs = rbs_fit(sm_conf_val[ov_idx], ov_logits, ov_labels,
                                    args.num_bins_rbs)

    run_res = {}

    # ID-test：env0 特征，test 节点。
    id_logits, id_probs = get_logits_probs(model, x_tr, ei, test_mask)
    id_labels = labels[test_mask].cpu().numpy()
    sm_conf_tr = compute_sm_conf(ei, N, get_all_probs_tensor(model, x_tr, ei))
    id_idx = test_mask.cpu().numpy().nonzero()[0]
    rbs_id = apply_rbs(T_list_rbs, bins_rbs, sm_conf_tr[id_idx], id_logits)
    run_res['ID-test'] = eval_all_methods(
        id_logits, id_probs, id_labels, nclass, binary=False,
        ts_model=ts_model, hb_model=hb_model, iso_model=iso_model,
        bbq_model=bbq_model, meta_model=meta_model, rbs_probs=rbs_id)

    for feat_ood, name in zip(feat_oods, ood_names):
        te_logits, te_probs = get_logits_probs(model, feat_ood, ei, test_mask)
        te_labels = labels[test_mask].cpu().numpy()
        sm_conf_te = compute_sm_conf(ei, N, get_all_probs_tensor(model, feat_ood, ei))
        te_idx = test_mask.cpu().numpy().nonzero()[0]
        rbs_te = apply_rbs(T_list_rbs, bins_rbs, sm_conf_te[te_idx], te_logits)

        ood_raw = eval_all_methods(
            te_logits, te_probs, te_labels, nclass, binary=False,
            ts_model=ts_model, hb_model=hb_model, iso_model=iso_model,
            bbq_model=bbq_model, meta_model=meta_model, rbs_probs=rbs_te)
        run_res[name] = {}
        for cm in CAL_METHODS_OUT:
            id_m  = run_res['ID-test'][cm]
            ood_m = ood_raw[cm]
            run_res[name][cm] = add_cross_split_metrics(
                id_m, ood_m, id_m['_u'], ood_m['_u'])

    return run_res

# ══════════════════════════════════════════════════════════════
# 8. 汇总输出
# ══════════════════════════════════════════════════════════════
def summarize(all_runs, split_names, all_keys, csv_path, title,
             reliability_path=None, uncertainty_path=None):
    """
    all_runs: list of { split_name: { cal_method: metric_dict } }
    CSV 行 = (split, cal_method)，列 = 均值/标准差
    只输出 CAL_METHODS_OUT 中的方法（Uncal / RBS）
    """
    col_w  = 18
    show_k = ['acc', 'ece', 'nll', 'brier', 'ue_auroc', 'aurc']
    sep    = '═' * (30 + col_w * len(show_k))

    print(f'\n{sep}'); print(f'  {title}  ({len(all_runs)} runs)'); print(sep)

    mean_rows = [['split', 'cal_method']
                 + [f'{k}_mean' for k in all_keys]
                 + [f'{k}_std'  for k in all_keys]]

    for sname in split_names:
        print(f'\n  [{sname}]')
        print(f'  {"cal_method":<14}' + ''.join(f'{k:>{col_w}}' for k in show_k))
        print('  ' + '─' * (14 + col_w * len(show_k)))
        for cm in CAL_METHODS_OUT:
            vals = defaultdict(list)
            for r in all_runs:
                m = r.get(sname, {}).get(cm, {})
                for k in all_keys:
                    v = m.get(k)
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        vals[k].append(v)
            mu = {k: np.mean(vals[k]) if vals[k] else float('nan') for k in all_keys}
            sd = {k: np.std(vals[k])  if vals[k] else float('nan') for k in all_keys}
            print(f'  {cm:<14}' +
                  ''.join(f'{mu[k]:.4f}±{sd[k]:.4f}'.rjust(col_w) for k in show_k))
            mean_rows.append([sname, cm]
                             + [f'{mu[k]:.6f}' for k in all_keys]
                             + [f'{sd[k]:.6f}'  for k in all_keys])

    # OOD 平均行
    ood_names = [n for n in split_names if n.startswith('OOD')]
    if ood_names:
        print(f'\n  [OOD-avg]')
        print(f'  {"cal_method":<14}' + ''.join(f'{k:>{col_w}}' for k in show_k))
        print('  ' + '─' * (14 + col_w * len(show_k)))
        for cm in CAL_METHODS_OUT:
            vals = defaultdict(list)
            for r in all_runs:
                for n in ood_names:
                    m = r.get(n, {}).get(cm, {})
                    for k in all_keys:
                        v = m.get(k)
                        if v is not None and not (isinstance(v, float) and math.isnan(v)):
                            vals[k].append(v)
            mu = {k: np.mean(vals[k]) if vals[k] else float('nan') for k in all_keys}
            sd = {k: np.std(vals[k])  if vals[k] else float('nan') for k in all_keys}
            print(f'  {cm:<14}' +
                  ''.join(f'{mu[k]:.4f}±{sd[k]:.4f}'.rjust(col_w) for k in show_k))
            mean_rows.append(['OOD-avg', cm]
                             + [f'{mu[k]:.6f}' for k in all_keys]
                             + [f'{sd[k]:.6f}'  for k in all_keys])

    print(f'\n{sep}')
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.',
                exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerows(mean_rows)
    print(f'  结果已保存 → {csv_path}')

    if reliability_path:
        rel_rows = [['run', 'split', 'cal_method', 'bin', 'avg_confidence', 'accuracy', 'count']]
        for run_idx, r in enumerate(all_runs):
            for sname in split_names:
                for cm in CAL_METHODS_OUT:
                    for b_idx, (avg_c, acc, cnt) in enumerate(
                            r.get(sname, {}).get(cm, {}).get('_reliability_bins', [])):
                        rel_rows.append([run_idx, sname, cm, b_idx+1,
                                         f'{avg_c:.6f}',
                                         f'{acc:.6f}' if not math.isnan(acc) else 'nan',
                                         cnt])
        with open(reliability_path, 'w', newline='') as f:
            csv.writer(f).writerows(rel_rows)
        print(f'  Reliability  → {reliability_path}')

    if uncertainty_path:
        unc_rows = [['run', 'split', 'cal_method', 'u', 'correct']]
        for run_idx, r in enumerate(all_runs):
            for sname in split_names:
                for cm in CAL_METHODS_OUT:
                    m = r.get(sname, {}).get(cm, {})
                    u_arr, cor_arr = m.get('_u'), m.get('_correct')
                    if u_arr is not None and cor_arr is not None:
                        for u_val, c_val in zip(u_arr.tolist(), cor_arr.tolist()):
                            unc_rows.append([run_idx, sname, cm,
                                             f'{u_val:.6f}', int(c_val)])
        with open(uncertainty_path, 'w', newline='') as f:
            csv.writer(f).writerows(unc_rows)
        print(f'  不确定性样本 → {uncertainty_path}')


# ══════════════════════════════════════════════════════════════
# 9. 主函数
# ══════════════════════════════════════════════════════════════
def main():
    print(f'[配置] dataset={args.dataset} backbone={_backbone_name()} runs={args.runs}')
    print(f'[配置] hidden={args.hidden} epochs={args.epochs} patience={args.patience}')
    print(f'[配置] add_cal_loss={args.add_cal_loss} num_bins_rbs={args.num_bins_rbs}')

    # ── Elliptic ──────────────────────────────────────────────
    if args.dataset == 'elliptic':
        crit = make_criterion(2, args.class_weight)

        print('\n[Elliptic] 加载数据...')
        ei_tr, x_tr, lab_tr, N_tr, _ = merge_elliptic(ELLIPTIC_TRAIN, args.data_dir)
        ei_tr = ei_tr.to(device); x_tr = x_tr.to(device); lab_tr = lab_tr.to(device)
        tr_base_mask = (lab_tr >= 0)

        ei_ov, x_ov, lab_ov, N_ov, _ = merge_elliptic(ELLIPTIC_VAL, args.data_dir)
        ei_ov = ei_ov.to(device); x_ov = x_ov.to(device); lab_ov = lab_ov.to(device)
        ov_mask = (lab_ov >= 0)

        print('[Elliptic] 加载 OOD-test 图...')
        test_graphs = []
        for i, steps in enumerate(ELLIPTIC_TESTS):
            ei_te, x_te, lab_te, N_te, _ = merge_elliptic(steps, args.data_dir)
            te_mask = (lab_te >= 0)
            print(f'  OOD-test_{i} steps={steps}: labeled={te_mask.sum()} '
                  f'illicit={(lab_te[te_mask]==1).sum()}')
            test_graphs.append((ei_te.to(device), x_te.to(device),
                                lab_te.to(device), te_mask, N_te))

        split_names = ['ID-test'] + [
            f'OOD-test_{i}(s{ELLIPTIC_TESTS[i][0]}-{ELLIPTIC_TESTS[i][-1]})'
            for i in range(len(ELLIPTIC_TESTS))]
        all_keys = build_all_keys(binary=True)

        all_runs = []
        for r in range(args.runs):
            seed = args.base_seed + r
            print(f'\n{"="*65}')
            print(f'  calGNN  Run {r+1}/{args.runs}  seed={seed}')
            print(f'{"="*65}')
            torch.manual_seed(seed); np.random.seed(seed)

            run_res = run_one_seed_elliptic(
                seed, ei_tr, x_tr, lab_tr, tr_base_mask, N_tr,
                ei_ov, x_ov, lab_ov, ov_mask, N_ov,
                test_graphs, nclass=2, crit=crit)
            all_runs.append(run_res)

            for cm in ['Uncal', 'TS', 'RBS']:
                m = run_res.get('ID-test', {}).get(cm, {})
                if m: print(f'  ID-test {cm:<8} | acc={m["acc"]:.4f} '
                             f'f1={m.get("f1",0):.4f} ece={m["ece"]:.4f} '
                             f'nll={m["nll"]:.4f} ue_auroc={m["ue_auroc"]:.4f}')

        summarize(all_runs, split_names, all_keys,
                  csv_path=os.path.join(args.save_dir, 'elliptic_calgnn_results.csv'),
                  title=_tagged_title('Elliptic — calGNN'),
                  reliability_path=os.path.join(args.save_dir, 'elliptic_calgnn_reliability.csv'),
                  uncertainty_path=os.path.join(args.save_dir, 'elliptic_calgnn_uncertainty_samples.csv'))

    # ── OGB-Arxiv ─────────────────────────────────────────────
    elif args.dataset == 'arxiv':
        print('\n[Arxiv] 加载数据...')
        (ei, features, labels, years, nclass,
         base_tr, ov_mask, ood_masks, N, ei_np) = load_arxiv(args.data_path)

        ei       = ei.to(device); features = features.to(device)
        labels   = labels.to(device)
        crit     = make_criterion(nclass)

        split_names = ['ID-test'] + [
            f'OOD-test_{i}({ty0}-{ty1})'
            for i, (ty0, ty1) in enumerate(ARXIV_TESTS)]
        all_keys = build_all_keys(binary=False)

        all_runs = []
        for r in range(args.runs):
            seed = args.base_seed + r
            print(f'\n{"="*65}')
            print(f'  calGNN  Run {r+1}/{args.runs}  seed={seed}')
            print(f'{"="*65}')
            torch.manual_seed(seed); np.random.seed(seed)

            run_res = run_one_seed_arxiv(
                seed, ei, features, labels, base_tr, N, ei_np,
                ov_mask, ood_masks, nclass, crit)
            all_runs.append(run_res)

            for cm in ['Uncal', 'TS', 'RBS']:
                m = run_res.get('ID-test', {}).get(cm, {})
                if m: print(f'  ID-test {cm:<8} | acc={m["acc"]:.4f} '
                             f'ece={m["ece"]:.4f} nll={m["nll"]:.4f} '
                             f'ue_auroc={m["ue_auroc"]:.4f}')

        summarize(all_runs, split_names, all_keys,
                  csv_path=os.path.join(args.save_dir, 'arxiv_calgnn_results.csv'),
                  title=_tagged_title('OGB-Arxiv — calGNN'),
                  reliability_path=os.path.join(args.save_dir, 'arxiv_calgnn_reliability.csv'),
                  uncertainty_path=os.path.join(args.save_dir, 'arxiv_calgnn_uncertainty_samples.csv'))



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
            print(f'  calGNN  EERM-{ds_tag}  Run {r+1}/{args.runs}  seed={seed}')
            print(f'{"="*65}')
            torch.manual_seed(seed); np.random.seed(seed)

            run_res = run_one_seed_eerm(
                seed, ei, feat_tr, feat_val, feat_oods, labels_t,
                tr_mask, val_mask, test_mask, ood_names, N,
                nclass=nclass, crit=crit,
                prefix=f'eerm_{ds_tag}')

            for cm in CAL_METHODS_OUT:
                m = run_res.get('ID-test', {}).get(cm, {})
                if m:
                    print(f'  ID-test {cm:<8} | acc={m["acc"]:.4f} '
                          f'ece={m["ece"]:.4f} nll={m["nll"]:.4f} '
                          f'ue_auroc={m["ue_auroc"]:.4f}')
            for name in ood_names:
                m = run_res.get(name, {}).get('RBS', {})
                if m:
                    print(f'  {name} RBS | acc={m["acc"]:.4f} '
                          f'delta_ece={m["delta_ece"]:.4f} '
                          f'ood_auroc={m["ood_auroc"]:.4f}')
            all_runs.append(run_res)

        summarize(all_runs, split_names, all_keys,
                  csv_path=os.path.join(args.save_dir,
                      f'{ds_tag}_calgnn_results.csv'),
                  title=_tagged_title(f'EERM-{ds_tag} — calGNN'),
                  reliability_path=os.path.join(args.save_dir,
                      f'{ds_tag}_calgnn_reliability.csv'),
                  uncertainty_path=os.path.join(args.save_dir,
                      f'{ds_tag}_calgnn_uncertainty_samples.csv'))

if __name__ == '__main__':

    main()
