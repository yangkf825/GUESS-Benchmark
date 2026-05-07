"""
Graph-CP — Distribution-Free Prediction Sets for Node Classification
=====================================================================
Clarkson, 2023. https://github.com/jase-clarkson/graph_cp

核心思路：
  Split Conformal Prediction + 可选邻域加权，基于 ProbabilityAccumulator
  （APS 累积概率 score），额外评估 WSC（worst-slab coverage）衡量条件覆盖。

两种模式：
  standard  — 标准 split CP，不使用图结构权重
  weighted  — 邻域加权 CP，用节点度数分布做权重，对稀疏节点更友好

三种额外指标（相比标准 UQ 指标）：
  coverage_mean / std  (实际覆盖率，目标 >= 1-alpha)
  set_size_mean / std  (预测集平均大小，越小越好)
  wsc                  (worst-slab coverage，条件覆盖质量，越接近 1-alpha 越好)

用法:
    python experiments/run_graph_cp.py --dataset elliptic --data_dir ./data/elliptic --runs 5
    python experiments/run_graph_cp.py --dataset arxiv    --data_path ./data/arxiv/data.pkl --runs 5
    python experiments/run_graph_cp.py --dataset eerm --eerm_dataset cora \
        --eerm_root ./data/eerm/Planetoid/cora --runs 5
    python experiments/run_graph_cp.py --dataset twitch   --data_root ./data --runs 5
    python experiments/run_graph_cp.py --dataset facebook --data_root ./data --runs 3

    # 使用邻域加权模式
    python experiments/run_graph_cp.py --dataset elliptic --data_dir ./data/elliptic \
        --mode weighted --score aps --runs 5
"""
import sys; sys.path.insert(0, 'src')
from gnn_uq_bench.model_gat_sage import (canonical_backbone_name, get_pyg_backbone, get_pyg_backbone_bn, get_sparse_backbone, GraphANTNodeBackbone, GPNBackboneModel)

import os, time, argparse, copy, warnings
import numpy as np
import pandas as pd
import scipy.stats
from scipy.optimize import brentq
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv

from gnn_uq_bench.datasets import (
    load_elliptic, load_arxiv, load_eerm, stratified_split,
    ELLIPTIC_TRAIN, ELLIPTIC_TESTS, ARXIV_TRAIN_YEAR, ARXIV_TESTS,
)
from gnn_uq_bench.datasets_fb_twitch import load_facebook_twitch
from gnn_uq_bench.models import GCNSparse
from gnn_uq_bench.metrics import (
    compute_split_metrics, add_cross_split_metrics, build_all_keys, summarize,
)

warnings.filterwarnings('ignore')

# ── 参数 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',       type=str, default='elliptic',
                    choices=['elliptic','arxiv','eerm','twitch','facebook'])
parser.add_argument('--data_dir',      type=str, default='./data/elliptic')
parser.add_argument('--data_path',     type=str, default='./data/arxiv/data.pkl')
parser.add_argument('--eerm_dataset',  type=str, default='cora', choices=['cora','amazon'])
parser.add_argument('--eerm_root',     type=str, default=None)
parser.add_argument('--data_root',     type=str, default='./data')
parser.add_argument('--runs',          type=int, default=5)
parser.add_argument('--model',         type=str,   default='GAT',
                    choices=['GCN', 'GAT', 'SAGE', 'GraphSAGE'],
                    help='backbone: GCN, GAT, SAGE/GraphSAGE')
parser.add_argument('--alpha',         type=float, default=0.1)
parser.add_argument('--score',         type=str, default='aps', choices=['aps','tps'])
parser.add_argument('--mode',          type=str, default='standard',
                    choices=['standard','weighted'])
parser.add_argument('--n_calib',       type=int, default=None)
parser.add_argument('--n_repeats',     type=int, default=100)
parser.add_argument('--wsc_M',         type=int, default=200)
parser.add_argument('--backbone_heads', type=int,   default=8,
                    help='GAT attention heads for the new backbone')
parser.add_argument('--hidden',        type=int, default=64)
parser.add_argument('--dropout',       type=float, default=0.5)
parser.add_argument('--lr',            type=float, default=0.01)
parser.add_argument('--weight_decay',  type=float, default=5e-4)
parser.add_argument('--epochs',        type=int, default=2000)
parser.add_argument('--patience',      type=int, default=100)
parser.add_argument('--id_val_ratio',  type=float, default=0.1)
parser.add_argument('--id_test_ratio', type=float, default=0.1)
parser.add_argument('--class_weight',  type=float, default=10.0)
parser.add_argument('--base_seed',     type=int, default=42)
parser.add_argument('--save_dir',      type=str, default='./results/graph_cp_gat_sage')
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
print(f'[设备] {device}  alpha={args.alpha}  score={args.score}  mode={args.mode}')


# ════════════════════════════════════════════════════════════════════════
# 1. ProbabilityAccumulator (来自 graph_cp/prob_accum.py)
# ════════════════════════════════════════════════════════════════════════

class ProbabilityAccumulator:
    def __init__(self, prob):
        self.n, self.K = prob.shape
        self.order = np.argsort(-prob, axis=1)
        self.ranks  = np.empty_like(self.order)
        for i in range(self.n):
            self.ranks[i, self.order[i]] = np.arange(self.K)
        self.prob_sort = -np.sort(-prob, axis=1)
        self.Z = np.round(self.prob_sort.cumsum(axis=1), 9)

    def calibrate_scores(self, Y, epsilon=None):
        Y = np.atleast_1d(Y); n2 = len(Y)
        ranks    = np.array([self.ranks[i, Y[i]] for i in range(n2)])
        prob_cum = np.array([self.Z[i, ranks[i]]    for i in range(n2)])
        prob     = np.array([self.prob_sort[i, ranks[i]] for i in range(n2)])
        alpha_max = 1.0 - prob_cum
        alpha_max += np.multiply(prob, epsilon) if epsilon is not None else prob
        return np.minimum(alpha_max, 1.0)

    def predict_sets(self, alpha, epsilon=None, allow_empty=True):
        L = np.argmax(self.Z >= 1.0 - alpha, axis=1).flatten()
        if epsilon is not None:
            Z_excess = np.array([self.Z[i, L[i]] for i in range(self.n)]) - (1.0 - float(alpha))
            p_rm = Z_excess / np.array([self.prob_sort[i, L[i]] for i in range(self.n)])
            rm   = epsilon <= p_rm
            for i in np.where(rm)[0]:
                L[i] = max(0, L[i] - 1) if not allow_empty else L[i] - 1
        return [self.order[i, :L[i]+1] for i in range(self.n)]


# ════════════════════════════════════════════════════════════════════════
# 2. Calibration / Prediction helpers (来自 graph_cp/utils.py)
# ════════════════════════════════════════════════════════════════════════

# ── numpy 版本兼容（1.21 用 interpolation=，1.22+ 用 method=）────────
def _quantile_higher(a, q):
    try:
        return float(np.quantile(a, q, method='higher'))
    except TypeError:
        return float(np.quantile(a, q, interpolation='higher'))


def _aps_calibrate(probs, labels, alpha):
    n   = len(labels)
    cal = ProbabilityAccumulator(probs)
    eps = np.random.uniform(0, 1, n)
    alpha_max  = cal.calibrate_scores(labels, eps)
    raw_scores = alpha - alpha_max
    level      = (1. - alpha) * (1. + 1. / n)
    correction = scipy.stats.mstats.mquantiles(raw_scores, prob=level)[0]
    return alpha - correction   # corrected alpha

def _tps_calibrate(probs, labels, alpha):
    n       = len(labels)
    scores  = 1. - probs[np.arange(n), labels]
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    return _quantile_higher(scores, q_level)   # qhat

def _weighted_quantile(scores, weights, alpha):
    wtildes = weights / (weights.sum() + 1)
    def f(q): return (wtildes * (scores <= q)).sum() - (1 - alpha)
    try:
        return brentq(f, scores.min() - 1., scores.max() + 1.)
    except ValueError:
        return 0.

def _aps_calibrate_weighted(probs, labels, weights, alpha):
    n   = len(labels)
    cal = ProbabilityAccumulator(probs)
    eps = np.random.uniform(0, 1, n)
    alpha_max  = cal.calibrate_scores(labels, eps)
    raw_scores = alpha - alpha_max
    return alpha - _weighted_quantile(raw_scores, weights, alpha)

def _tps_calibrate_weighted(probs, labels, weights, alpha):
    n       = len(labels)
    scores  = 1. - probs[np.arange(n), labels]
    return _weighted_quantile(alpha - scores, weights, alpha)

def _aps_predict(probs, corr_alpha):
    n   = len(probs)
    cal = ProbabilityAccumulator(probs)
    eps = np.random.uniform(0, 1, n)
    return cal.predict_sets(corr_alpha, eps)

def _tps_predict(probs, qhat):
    return [np.where(probs[i] >= (1. - qhat))[0] for i in range(len(probs))]

def _build_weights(edge_index_np, N, val_node_count):
    """节点度数归一化权重（简单近似邻域权重）"""
    deg = np.zeros(N, dtype=np.float32)
    if edge_index_np is not None and edge_index_np.shape[1] > 0:
        for d in edge_index_np[0]: deg[d] += 1
    weights = np.ones(val_node_count, dtype=np.float32)
    return weights / weights.sum()


# ════════════════════════════════════════════════════════════════════════
# 3. WSC (Worst-Slab Coverage, 来自 graph_cp/coverage.py 简化版)
# ════════════════════════════════════════════════════════════════════════

def _wsc(X, covered, delta=0.1, M=200, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    n, p = X.shape
    if n < 20 or p < 1: return float('nan')
    V = rng.standard_normal((M, p))
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    wsc_best = 1.0
    for v in V:
        z     = X @ v
        order = np.argsort(z)
        cov_s = covered[order].astype(float)
        ai_max = int(round((1.0 - delta) * n))
        for ai in range(0, ai_max):
            bi_min = min(ai + int(round(delta * n)), n)
            coverage = np.cumsum(cov_s[ai:]) / np.arange(1, n - ai + 1)
            coverage[:bi_min - ai] = 1.0
            wsc_best = min(wsc_best, coverage.min())
    return float(wsc_best)


# ════════════════════════════════════════════════════════════════════════
# 4. 核心评估入口
# ════════════════════════════════════════════════════════════════════════

def evaluate_cp(smx, labels, ei_np, N, alpha, score, mode,
                n_calib, n_repeats, seed_base, feat_np=None):
    M = len(labels)
    if n_calib is None: n_calib = min(1000, M // 2)
    n_calib = min(n_calib, M - 1)

    covs, sizes, wscs, shrs = [], [], [], []
    for k in range(n_repeats):
        rng     = np.random.default_rng(seed_base + k)
        cal_idx = rng.choice(M, n_calib, replace=False)
        mask    = np.zeros(M, bool); mask[cal_idx] = True

        cal_smx, val_smx = smx[mask], smx[~mask]
        cal_lab, val_lab = labels[mask], labels[~mask]
        n_val = len(val_lab)

        if mode == 'standard':
            if score == 'aps':
                thresh = _aps_calibrate(cal_smx, cal_lab, alpha)
                psets  = _aps_predict(val_smx, thresh)
            else:
                thresh = _tps_calibrate(cal_smx, cal_lab, alpha)
                psets  = _tps_predict(val_smx, thresh)
        else:
            w = _build_weights(ei_np, N, n_val)
            if score == 'aps':
                thresh = _aps_calibrate_weighted(cal_smx, cal_lab, w, alpha)
                psets  = _aps_predict(val_smx, thresh)
            else:
                thresh = _tps_calibrate_weighted(cal_smx, cal_lab, w, alpha)
                psets  = _tps_predict(val_smx, thresh)

        covered  = np.array([val_lab[i] in psets[i] for i in range(n_val)])
        set_size = np.array([len(s) for s in psets])
        covs.append(float(covered.mean()))
        sizes.append(float(set_size.mean()))
        # Singleton Hit Ratio
        singleton_hit = float(((set_size == 1) & covered).sum() / max(n_val, 1))
        shrs.append(singleton_hit)

        if k == 0 and feat_np is not None:
            try:
                wscs.append(_wsc(feat_np[~mask], covered, M=args.wsc_M, rng=rng))
            except Exception:
                wscs.append(float('nan'))

    return (float(np.mean(covs)), float(np.std(covs)),
            float(np.mean(sizes)), float(np.std(sizes)),
            float(np.nanmean(wscs)) if wscs else float('nan'),
            float(np.nanstd(wscs))  if wscs else float('nan'),
            float(np.mean(shrs)), float(np.std(shrs)))


# ════════════════════════════════════════════════════════════════════════
# 5. 模型 / 训练工具
# ════════════════════════════════════════════════════════════════════════

class GCN3BN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dp):
        super().__init__()
        self.c1  = GCNConv(nfeat, nhid); self.bn1 = nn.BatchNorm1d(nhid)
        self.c2  = GCNConv(nhid,  nhid); self.bn2 = nn.BatchNorm1d(nhid)
        self.c3  = GCNConv(nhid, nclass); self.dp = dp
    def forward(self, x, ei):
        x = F.dropout(F.relu(self.bn1(self.c1(x, ei))), self.dp, self.training)
        x = F.dropout(F.relu(self.bn2(self.c2(x, ei)) + x), self.dp, self.training)
        return self.c3(x, ei)


def _train_sparse(adj, feat, lab_np, tr_mask, val_mask, nclass, seed, path, cw=None):
    torch.manual_seed(seed)
    model = get_sparse_backbone(args.model, feat.shape[1], args.hidden, nclass, args.dropout, nheads=getattr(args, 'backbone_heads', 8)).to(device)
    model.reset_parameters()
    crit  = (nn.CrossEntropyLoss(weight=torch.FloatTensor([1., cw]).to(device))
             if cw and nclass == 2 else nn.CrossEntropyLoss())
    opt   = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lab_t = torch.tensor(lab_np, dtype=torch.long).to(device)
    tr_t  = torch.tensor(tr_mask, dtype=torch.bool).to(device)
    val_t = torch.tensor(val_mask, dtype=torch.bool).to(device)
    best, bad, bs = 1e9, 0, None
    for _ in range(args.epochs):
        model.train(); opt.zero_grad()
        crit(model(feat, adj)[tr_t], lab_t[tr_t]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lv = crit(model(feat, adj)[val_t], lab_t[val_t]).item()
        if lv < best: bs = copy.deepcopy(model.state_dict()); best, bad = lv, 0
        else: bad += 1
        if bad >= args.patience: break
    model.load_state_dict(bs); torch.save(bs, path)
    print(f'    val={best:.4f}'); return model


def _train_pyg(train_data, val_data, nfeat, nclass, seed, path):
    torch.manual_seed(seed)
    model = get_pyg_backbone_bn(args.model, nfeat, args.hidden, nclass, args.dropout, heads=getattr(args, 'backbone_heads', 8)).to(device)
    opt   = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, bad, bs = 1e9, 0, None
    for _ in range(min(args.epochs, 200)):
        model.train()
        for d in train_data:
            d = d.to(device); opt.zero_grad()
            F.cross_entropy(model(d.x, d.edge_index), d.y).backward(); opt.step()
        model.eval(); v = 0.
        with torch.no_grad():
            for d in val_data:
                d = d.to(device)
                v += F.cross_entropy(model(d.x, d.edge_index), d.y).item()
        if v < best: bs = copy.deepcopy(model.state_dict()); best, bad = v, 0
        else: bad += 1
        if bad >= args.patience: break
    model.load_state_dict(bs); torch.save(bs, path)
    print(f'    val={best:.4f}'); return model


@torch.no_grad()
def _infer(model, feat, adj):
    model.eval()
    logits = model(feat, adj).cpu().numpy()
    ex = np.exp(logits - logits.max(1, keepdims=True))
    return ex / ex.sum(1, keepdims=True)

@torch.no_grad()
def _infer_pyg(model, data):
    model.eval(); data = data.to(device)
    logits = model(data.x, data.edge_index).cpu().numpy()
    ex = np.exp(logits - logits.max(1, keepdims=True))
    return ex / ex.sum(1, keepdims=True)


# ════════════════════════════════════════════════════════════════════════
# 6. 保存辅助
# ════════════════════════════════════════════════════════════════════════

def _pr(name, cm, cs, sm, ss, wsc, shrm, shrs, alpha):
    print(f'  {name[:30]:<30} | cov={cm:.3f}±{cs:.3f}'
          f' (tgt={1-alpha:.2f})  size={sm:.2f}±{ss:.2f}  wsc={wsc:.3f}  shr={shrm:.3f}±{shrs:.3f}')

def _row(seed, split, cm, cs, sm, ss, wsc, shrm, shrs, alpha, score, mode):
    return {'seed': seed, 'split': split, 'alpha': alpha, 'score': score, 'mode': mode,
            'coverage_mean': cm, 'coverage_std': cs,
            'set_size_mean': sm, 'set_size_std': ss, 'wsc': wsc,
            'shr_mean': shrm, 'shr_std': shrs,
            'target_coverage': 1.-alpha, 'coverage_gap': cm-(1.-alpha)}

def _save(cp_rows, uq_runs, split_names, all_keys, prefix):
    if cp_rows:
        df  = pd.DataFrame(cp_rows)
        agg = (df.groupby('split')[
            ['coverage_mean','coverage_std','set_size_mean','set_size_std','wsc','shr_mean','shr_std']]
               .mean().reset_index())
        agg.to_csv(prefix + '_cp.csv', index=False)
        print(f'\n  Graph-CP CSV → {prefix}_cp.csv')
        print(agg[['split','coverage_mean','set_size_mean','wsc','shr_mean']].to_string(index=False))
    if uq_runs:
        summarize(uq_runs, split_names, all_keys,
                  prefix + '_uq_results.csv', prefix.split('/')[-1],
                  reliability_path=prefix + '_reliability.csv',
                  uncertainty_path=prefix + '_uncertainty.csv')


# ════════════════════════════════════════════════════════════════════════
# 7. 各数据集流程
# ════════════════════════════════════════════════════════════════════════

def run_elliptic():
    print('\n[Elliptic] Loading...')
    adj_tr, ei_tr, feat_tr, _, lab_tr_np, N_tr = load_elliptic(
        ELLIPTIC_TRAIN, args.data_dir, device)
    tr_base   = (lab_tr_np >= 0)
    ei_tr_np  = ei_tr.cpu().numpy()
    feat_np   = feat_tr.cpu().numpy()

    test_graphs = []
    for i, steps in enumerate(ELLIPTIC_TESTS):
        a, ei, f, _, lnp, N = load_elliptic(steps, args.data_dir, device)
        tm = (lnp >= 0)
        test_graphs.append((a, ei.cpu().numpy(), f, f.cpu().numpy(), lnp, tm, N))

    all_keys    = build_all_keys(binary=True)
    split_names = ['ID-test'] + [
        f'OOD-test_{i}(s{ELLIPTIC_TESTS[i][0]}-{ELLIPTIC_TESTS[i][-1]})'
        for i in range(len(ELLIPTIC_TESTS))]
    uq_runs, cp_rows = [], []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  Graph-CP Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(
            lab_tr_np, tr_base, args.id_val_ratio, args.id_test_ratio, seed)
        model = _train_sparse(adj_tr, feat_tr, lab_tr_np, tr_m, val_m, 2, seed,
                               os.path.join(args.save_dir, f'elliptic_seed{seed}.pth'),
                               args.class_weight)
        probs = _infer(model, feat_tr, adj_tr)
        run_res = {}

        u_id = 1. - probs[id_m].max(1)
        r_id = compute_split_metrics(probs[id_m], u_id, lab_tr_np[id_m], 2, binary=True)
        run_res['ID-test'] = r_id
        cm,cs,sm,ss,wsc,_,shrm,shrs = evaluate_cp(probs[id_m], lab_tr_np[id_m], ei_tr_np, N_tr,
                                          args.alpha, args.score, args.mode,
                                          args.n_calib, args.n_repeats, seed,
                                          feat_np=feat_np[id_m])
        _pr('ID-test', cm,cs,sm,ss,wsc,shrm,shrs,args.alpha)
        cp_rows.append(_row(seed,'ID-test',cm,cs,sm,ss,wsc,shrm,shrs,args.alpha,args.score,args.mode))

        # OOD splits — 用同一模型直接推理，不重新训练
        for i, (a_te,ei_te_np,f_te,fnp,lnp,tm,N_te) in enumerate(test_graphs):
            p_te  = _infer(model, f_te, a_te)
            u_ood = 1. - p_te[tm].max(1)
            r_ood = compute_split_metrics(p_te[tm], u_ood, lnp[tm], 2, binary=True)
            r_ood = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            name  = split_names[1+i]; run_res[name] = r_ood
            cm,cs,sm,ss,wsc,_,shrm,shrs = evaluate_cp(p_te[tm], lnp[tm], ei_te_np, N_te,
                                              args.alpha, args.score, args.mode,
                                              args.n_calib, args.n_repeats, seed,
                                              feat_np=fnp[tm])
            _pr(name, cm,cs,sm,ss,wsc,shrm,shrs,args.alpha)
            cp_rows.append(_row(seed,name,cm,cs,sm,ss,wsc,shrm,shrs,args.alpha,args.score,args.mode))

        uq_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    _save(cp_rows, uq_runs, split_names, all_keys,
          os.path.join(args.save_dir, f'elliptic_{_model_tag()}_graphcp_{args.score}_{args.mode}_a{args.alpha}'))


def run_arxiv():
    print('\n[Arxiv] Loading...')
    adj, ei, feat, _, lab_np, node_year, nclass, N = load_arxiv(args.data_path, device)
    base_mask   = (node_year <= ARXIV_TRAIN_YEAR)
    ei_np       = ei.cpu().numpy()
    feat_np     = feat.cpu().numpy()
    split_names = ['ID-test'] + [f'OOD-test_{i}({ty0}-{ty1})'
                                  for i,(ty0,ty1) in enumerate(ARXIV_TESTS)]
    all_keys = build_all_keys(binary=False)
    uq_runs, cp_rows = [], []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  Graph-CP Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(
            lab_np, base_mask, args.id_val_ratio, args.id_test_ratio, seed)
        model = _train_sparse(adj, feat, lab_np, tr_m, val_m, nclass, seed,
                               os.path.join(args.save_dir, f'arxiv_seed{seed}.pth'))
        probs = _infer(model, feat, adj)
        run_res = {}

        u_id = 1. - probs[id_m].max(1)
        r_id = compute_split_metrics(probs[id_m], u_id, lab_np[id_m], nclass)
        run_res['ID-test'] = r_id
        cm,cs,sm,ss,wsc,_,shrm,shrs = evaluate_cp(probs[id_m], lab_np[id_m], ei_np, N,
                                          args.alpha, args.score, args.mode,
                                          args.n_calib, args.n_repeats, seed,
                                          feat_np=feat_np[id_m])
        _pr('ID-test', cm,cs,sm,ss,wsc,shrm,shrs,args.alpha)
        cp_rows.append(_row(seed,'ID-test',cm,cs,sm,ss,wsc,shrm,shrs,args.alpha,args.score,args.mode))

        for i,(ty0,ty1) in enumerate(ARXIV_TESTS):
            te = (node_year >= ty0) & (node_year <= ty1)
            u_ood = 1. - probs[te].max(1)
            r_ood = compute_split_metrics(probs[te], u_ood, lab_np[te], nclass)
            r_ood = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            name  = split_names[1+i]; run_res[name] = r_ood
            cm,cs,sm,ss,wsc,_,shrm,shrs = evaluate_cp(probs[te], lab_np[te], ei_np, N,
                                              args.alpha, args.score, args.mode,
                                              args.n_calib, args.n_repeats, seed,
                                              feat_np=feat_np[te])
            _pr(name, cm,cs,sm,ss,wsc,shrm,shrs,args.alpha)
            cp_rows.append(_row(seed,name,cm,cs,sm,ss,wsc,shrm,shrs,args.alpha,args.score,args.mode))

        uq_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    _save(cp_rows, uq_runs, split_names, all_keys,
          os.path.join(args.save_dir, f'arxiv_{_model_tag()}_graphcp_{args.score}_{args.mode}_a{args.alpha}'))


def run_eerm():
    print(f'\n[EERM-{args.eerm_dataset}] Loading...')
    (adj, ei, feat_envs, lab_t, lab_np, nclass,
     tr_mask, val_mask, test_mask, ood_names, N) = load_eerm(
        args.eerm_root, args.eerm_dataset, device)
    feat_tr  = feat_envs[0]; feat_oods = feat_envs[2:]
    tr_np    = tr_mask.cpu().numpy(); val_np = val_mask.cpu().numpy()
    te_np    = test_mask.cpu().numpy()
    ei_np    = ei.cpu().numpy()
    feat_np  = feat_tr.cpu().numpy()
    split_names = ['ID-test'] + ood_names
    all_keys = build_all_keys(binary=False)
    uq_runs, cp_rows = [], []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  Graph-CP Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        model = _train_sparse(adj, feat_tr, lab_np, tr_np, val_np, nclass, seed,
                               os.path.join(args.save_dir,
                                             f'eerm_{args.eerm_dataset}_seed{seed}.pth'))
        probs_id = _infer(model, feat_tr, adj)
        run_res  = {}

        u_id = 1. - probs_id[te_np].max(1)
        r_id = compute_split_metrics(probs_id[te_np], u_id, lab_np[te_np], nclass)
        run_res['ID-test'] = r_id
        cm,cs,sm,ss,wsc,_,shrm,shrs = evaluate_cp(probs_id[te_np], lab_np[te_np], ei_np, N,
                                          args.alpha, args.score, args.mode,
                                          args.n_calib, args.n_repeats, seed,
                                          feat_np=feat_np[te_np])
        _pr('ID-test', cm,cs,sm,ss,wsc,shrm,shrs,args.alpha)
        cp_rows.append(_row(seed,'ID-test',cm,cs,sm,ss,wsc,shrm,shrs,args.alpha,args.score,args.mode))

        for feat_ood, name in zip(feat_oods, ood_names):
            p_ood = _infer(model, feat_ood, adj)
            u_ood = 1. - p_ood[te_np].max(1)
            r_ood = compute_split_metrics(p_ood[te_np], u_ood, lab_np[te_np], nclass)
            r_ood = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            run_res[name] = r_ood
            cm,cs,sm,ss,wsc,_,shrm,shrs = evaluate_cp(p_ood[te_np], lab_np[te_np], ei_np, N,
                                              args.alpha, args.score, args.mode,
                                              args.n_calib, args.n_repeats, seed,
                                              feat_np=feat_ood.cpu().numpy()[te_np])
            _pr(name, cm,cs,sm,ss,wsc,shrm,shrs,args.alpha)
            cp_rows.append(_row(seed,name,cm,cs,sm,ss,wsc,shrm,shrs,args.alpha,args.score,args.mode))

        uq_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    ds = args.eerm_dataset
    _save(cp_rows, uq_runs, split_names, all_keys,
          os.path.join(args.save_dir, f'{ds}_{_model_tag()}_graphcp_{args.score}_{args.mode}_a{args.alpha}'))


def run_fb_twitch():
    ds = args.dataset
    print(f'\n[{ds}] Loading...')
    (label_map, nclass, nfeat,
     train_data, val_data, test_data,
     domain_names, scaler) = load_facebook_twitch(ds, args.data_root, device=None)
    id_dom    = domain_names['train'][-1]
    id_data   = train_data[-1]
    ood_doms  = domain_names['val'] + domain_names['test']
    ood_datas = val_data + test_data
    split_names = ['ID-test'] + [f'OOD-{d}' for d in ood_doms]
    all_keys    = build_all_keys(binary=False)
    uq_runs, cp_rows = [], []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  Graph-CP Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        model = _train_pyg(train_data, val_data, nfeat, nclass, seed,
                            os.path.join(args.save_dir, f'{ds}_seed{seed}.pth'))
        run_res = {}

        probs_id  = _infer_pyg(model, id_data)
        labels_id = id_data.y.numpy()
        ei_id_np  = id_data.edge_index.numpy()
        feat_id   = id_data.x.numpy()

        u_id = 1. - probs_id.max(1)
        r_id = compute_split_metrics(probs_id, u_id, labels_id, nclass)
        run_res['ID-test'] = r_id
        cm,cs,sm,ss,wsc,_,shrm,shrs = evaluate_cp(probs_id, labels_id, ei_id_np, len(labels_id),
                                          args.alpha, args.score, args.mode,
                                          args.n_calib, args.n_repeats, seed,
                                          feat_np=feat_id)
        _pr(f'ID-test ({id_dom})', cm,cs,sm,ss,wsc,shrm,shrs,args.alpha)
        cp_rows.append(_row(seed,'ID-test',cm,cs,sm,ss,wsc,shrm,shrs,args.alpha,args.score,args.mode))

        for dom, data_ood in zip(ood_doms, ood_datas):
            p_ood   = _infer_pyg(model, data_ood)
            lab_ood = data_ood.y.numpy()
            ei_ood  = data_ood.edge_index.numpy()
            f_ood   = data_ood.x.numpy()
            u_ood   = 1. - p_ood.max(1)
            r_ood   = compute_split_metrics(p_ood, u_ood, lab_ood, nclass)
            r_ood   = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            name    = f'OOD-{dom}'; run_res[name] = r_ood
            cm,cs,sm,ss,wsc,_,shrm,shrs = evaluate_cp(p_ood, lab_ood, ei_ood, len(lab_ood),
                                              args.alpha, args.score, args.mode,
                                              args.n_calib, args.n_repeats, seed,
                                              feat_np=f_ood)
            _pr(name, cm,cs,sm,ss,wsc,shrm,shrs,args.alpha)
            cp_rows.append(_row(seed,name,cm,cs,sm,ss,wsc,shrm,shrs,args.alpha,args.score,args.mode))

        uq_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    _save(cp_rows, uq_runs, split_names, all_keys,
          os.path.join(args.save_dir, f'{ds}_{_model_tag()}_graphcp_{args.score}_{args.mode}_a{args.alpha}'))


# ════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    {
        'elliptic': run_elliptic,
        'arxiv':    run_arxiv,
        'eerm':     run_eerm,
        'twitch':   run_fb_twitch,
        'facebook': run_fb_twitch,
    }[args.dataset]()