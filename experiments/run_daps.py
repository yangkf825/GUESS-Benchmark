"""
DAPS — Distribution-Aware Prediction Sets via Graph Conformal Scores
=====================================================================
Zargar, Clarkson et al. (2024). https://github.com/soroushzargar/DAPS
(torch-conformal/gnn_cp 子模块)

核心思路（来自源码 graph_transformations.py + transformations.py）：
  标准 split conformal prediction 的改进版，关键在于：
  **在计算 non-conformity score 之前**，先用图结构对 logits 做变换（传播平滑），
  使每个节点的分数融入邻域信息，从而得到更适合图结构的 prediction set。

  三种 non-conformity score（来自 transformations.py）：
    TPS    — 1 - softmax(logit_y)（threshold prediction set）
    APS    — 累积 softmax（随机化版本）
    Margin — softmax_y - max_{y'≠y} softmax_y'

  三种图变换（来自 graph_transformations.py）：
    none      — 不变换（标准 split CP 基线）
    vertex_mp — 1跳邻域平均：score = α·Alogits/deg + (1-α)·logits
    daps      — k跳 PPR 传播（Personalized PageRank 近似）

  变换顺序：logits → [图变换] → [score变换] → calibration/prediction

支持所有 6 个数据集。

指标（除标准 UQ 指标外）：
  coverage_mean / std  (实际覆盖率，目标 >= 1-alpha)
  set_size_mean / std  (预测集平均大小，越小越好)
  coverage_gap         (coverage - (1-alpha))

用法:
    python experiments/run_daps.py --dataset elliptic --data_dir ./data/elliptic \\
        --alpha 0.1 --score aps --transform none --runs 5

    python experiments/run_daps.py --dataset elliptic --data_dir ./data/elliptic \\
        --alpha 0.1 --score aps --transform vertex_mp --neigh_coef 0.5 --runs 5

    python experiments/run_daps.py --dataset elliptic --data_dir ./data/elliptic \\
        --alpha 0.1 --score aps --transform daps --n_iters 10 --ppr_alpha 0.85 --runs 5

    python experiments/run_daps.py --dataset arxiv --data_path ./data/arxiv/data.pkl \\
        --alpha 0.1 --score aps --transform daps --runs 5

    python experiments/run_daps.py --dataset eerm --eerm_dataset cora \\
        --eerm_root ./data/eerm/Planetoid/cora --alpha 0.1 --score aps --transform daps

    python experiments/run_daps.py --dataset twitch --data_root ./data \\
        --alpha 0.1 --score aps --transform vertex_mp --neigh_coef 0.5 --runs 5

    python experiments/run_daps.py --dataset facebook --data_root ./data \\
        --alpha 0.1 --score aps --transform daps --runs 3
"""
import sys; sys.path.insert(0, 'src')

import os, time, argparse, copy
import numpy as np
import pandas as pd
import scipy.sparse as sp
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

# ── 参数 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',       type=str,   default='elliptic',
                    choices=['elliptic', 'arxiv', 'eerm', 'twitch', 'facebook'])
parser.add_argument('--data_dir',      type=str,   default='./data/elliptic')
parser.add_argument('--data_path',     type=str,   default='./data/arxiv/data.pkl')
parser.add_argument('--eerm_dataset',  type=str,   default='cora', choices=['cora', 'amazon'])
parser.add_argument('--eerm_root',     type=str,   default=None)
parser.add_argument('--data_root',     type=str,   default='./data')
parser.add_argument('--runs',          type=int,   default=5)
# DAPS 专有参数
parser.add_argument('--alpha',         type=float, default=0.1,
                    help='目标错误覆盖率，预测集覆盖率目标 >= 1-alpha')
parser.add_argument('--score',         type=str,   default='aps',
                    choices=['aps', 'tps', 'margin'],
                    help='non-conformity score 类型')
parser.add_argument('--transform',     type=str,   default='daps',
                    choices=['none', 'vertex_mp', 'daps'],
                    help='图变换类型：none=标准CP / vertex_mp=1跳邻域 / daps=PPR传播')
parser.add_argument('--neigh_coef',    type=float, default=0.5,
                    help='[vertex_mp] 邻域平均权重，logit = α·Alogit/deg + (1-α)·logit')
parser.add_argument('--n_iters',       type=int,   default=10,
                    help='[daps] PPR 幂次迭代次数')
parser.add_argument('--ppr_alpha',     type=float, default=0.85,
                    help='[daps] PPR 重启概率')
parser.add_argument('--n_calib',       type=int,   default=None,
                    help='校准集大小，None=自动取 min(1000, N//2)')
parser.add_argument('--n_repeats',     type=int,   default=100,
                    help='随机重复次数（减小随机性影响）')
# 基础模型参数
parser.add_argument('--hidden',        type=int,   default=64)
parser.add_argument('--dropout',       type=float, default=0.5)
parser.add_argument('--lr',            type=float, default=0.01)
parser.add_argument('--weight_decay',  type=float, default=5e-4)
parser.add_argument('--epochs',        type=int,   default=2000)
parser.add_argument('--patience',      type=int,   default=100)
parser.add_argument('--id_val_ratio',  type=float, default=0.1)
parser.add_argument('--id_test_ratio', type=float, default=0.1)
parser.add_argument('--class_weight',  type=float, default=10.0)
parser.add_argument('--base_seed',     type=int,   default=42)
parser.add_argument('--save_dir',      type=str,   default='./results/daps')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[设备] {device}  alpha={args.alpha}  score={args.score}  transform={args.transform}')


# ════════════════════════════════════════════════════════════════════════
# 1. 三种 Score 变换（来自 transformations.py）
# ════════════════════════════════════════════════════════════════════════

def _tps_transform(logits_t):
    """TPS: score = softmax(logit)，越高越可能在预测集"""
    return F.softmax(logits_t, dim=1)


def _aps_transform(logits_t):
    """
    APS: score_y = softmax_y * U + sum_{y': rank(y') > rank(y)} softmax_y'
    （随机化版本，参考 APSTransformation.transform）
    返回 (N, K)，值越大（负向）越不确定 → 我们取 score = -aps_raw
    """
    smx   = F.softmax(logits_t, dim=1)
    ranks = torch.argsort(torch.argsort(smx, dim=1), dim=1)
    K     = smx.shape[1]
    cls_scores = []
    for c in range(K):
        y_rank          = ranks[:, c].reshape(-1, 1)
        larger_smx_sum  = (smx * (ranks > y_rank).float()).sum(dim=1)
        u_vec           = torch.rand_like(smx[:, c])
        cls_result      = smx[:, c] * u_vec + larger_smx_sum
        cls_scores.append(cls_result.unsqueeze(1))
    # 原始代码返回 * -1（越大 = score 越小 = 更不确定）
    # 为与 calibration 逻辑统一，返回负值（score 越高 → 预测集内）
    return -torch.hstack(cls_scores)


def _margin_transform(logits_t):
    """Margin: score_y = softmax_y - max_{y'≠y} softmax_y'"""
    smx  = F.softmax(logits_t, dim=1)
    K    = smx.shape[1]
    all_classes = torch.arange(K, device=logits_t.device)
    cls_scores  = []
    for c in range(K):
        others    = smx[:, all_classes[all_classes != c]]
        max_other = others.max(dim=1)[0]
        cls_scores.append((smx[:, c] - max_other).unsqueeze(1))
    return torch.hstack(cls_scores)


SCORE_FNS = {'tps': _tps_transform, 'aps': _aps_transform, 'margin': _margin_transform}


# ════════════════════════════════════════════════════════════════════════
# 2. 三种图变换（来自 graph_transformations.py）
# ════════════════════════════════════════════════════════════════════════

def _vertex_mp(logits_t, edge_index_t, N, neigh_coef):
    """
    1跳邻域平均变换（VertexMPTransformation）：
        agg = A · logits / deg
        result = neigh_coef · agg + (1 - neigh_coef) · logits
    """
    if edge_index_t is None or neigh_coef == 0:
        return logits_t
    A    = torch.sparse_coo_tensor(
        edge_index_t,
        torch.ones(edge_index_t.shape[1], device=device),
        (N, N)).coalesce()
    degs = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1e-10)
    agg  = torch.sparse.mm(A, logits_t) / degs.unsqueeze(1)
    return neigh_coef * agg + (1 - neigh_coef) * logits_t


def _approx_ppr(adj_scipy, N, alpha=0.85, n_iter=10):
    """
    近似 PPR 矩阵（PPRCVertexMPTransformation.approx_ppr_product）：
        H = α · D^{-1}A · H + (1-α) · I
    返回 scipy sparse matrix。
    """
    # 行归一化
    deg      = np.array(adj_scipy.sum(1)).flatten()
    deg_inv  = np.where(deg > 0, 1. / deg, 0.)
    pi       = sp.diags(deg_inv) @ adj_scipy

    H  = sp.eye(N, format='csr')
    H0 = H.copy()
    for _ in range(n_iter):
        H = alpha * (pi @ H) + (1 - alpha) * H0
    return H


def _daps_transform(logits_t, edge_index_np, N, n_iters, ppr_alpha):
    """
    DAPS / PPR 传播变换（PPRCVertexMPTransformation）：
        scores = PPR · logits
    """
    if edge_index_np is None:
        return logits_t
    adj = sp.coo_matrix(
        (np.ones(edge_index_np.shape[1]),
         (edge_index_np[0], edge_index_np[1])),
        shape=(N, N)).tocsr()
    ppr = _approx_ppr(adj, N, alpha=ppr_alpha, n_iter=n_iters)
    # 转成 sparse tensor
    ppr_coo = ppr.tocoo()
    idx  = torch.from_numpy(np.vstack([ppr_coo.row, ppr_coo.col]).astype(np.int64)).to(device)
    vals = torch.from_numpy(ppr_coo.data.astype(np.float32)).to(device)
    ppr_t = torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce()
    result = torch.sparse.mm(ppr_t, logits_t)
    return result


def apply_graph_transform(logits_t, edge_index_t, edge_index_np, N, transform, args):
    """统一图变换入口，返回变换后 logits tensor"""
    if transform == 'none':
        return logits_t
    elif transform == 'vertex_mp':
        return _vertex_mp(logits_t, edge_index_t, N, args.neigh_coef)
    elif transform == 'daps':
        return _daps_transform(logits_t, edge_index_np, N, args.n_iters, args.ppr_alpha)
    else:
        raise ValueError(f'Unknown transform: {transform}')


# ════════════════════════════════════════════════════════════════════════
# 3. GraphCP 核心逻辑（来自 graph_cp.py）
# ════════════════════════════════════════════════════════════════════════

def _get_quantile_idx(n_points, alpha):
    """calibrate_from_scores 中的分位数索引（低端分位）"""
    return int((n_points - 1) * alpha)


def _calibrate(scores_t, y_one_hot_t, alpha):
    """
    从分数张量中，取真实类别对应的分数，求 alpha 分位数。
    scores_t   : (N, K) tensor — 越高越可能在预测集（与 score_quantile 比较）
    y_one_hot_t: (N, K) bool tensor
    返回 score_quantile（标量）
    """
    score_points = scores_t[y_one_hot_t]   # (N,) 仅取真实类别的 score
    sorted_scores, _ = torch.sort(score_points)
    q_idx = _get_quantile_idx(score_points.shape[0], alpha)
    return sorted_scores[q_idx].item()


def _predict(scores_t, quantile):
    """score > quantile 的类别进入预测集"""
    return scores_t > quantile   # (N, K) bool


def run_daps_cp(logits_full_np, labels_np, node_mask, edge_index_t, edge_index_np, N,
                alpha, score_type, transform, args, n_calib, n_repeats, seed_base):
    """
    DAPS 完整流程：图变换（全图） → score变换 → 取子集节点 → calibration → prediction → 指标
    logits_full_np: (N, K) 全图 logits
    node_mask     : bool (N,) 或 int array，指定参与评估的节点
    labels_np     : (M,) 对应 node_mask 节点的标签
    """
    M = len(labels_np)
    if n_calib is None:
        n_calib = min(1000, M // 2)
    n_calib = min(n_calib, M - 1)

    # 图变换在全图 logits 上做（保证 PPR/VertexMP 维度正确），然后取目标节点子集
    with torch.no_grad():
        logits_t   = torch.tensor(logits_full_np, dtype=torch.float32, device=device)
        logits_tr  = apply_graph_transform(logits_t, edge_index_t, edge_index_np,
                                            N, transform, args)
        scores_all = SCORE_FNS[score_type](logits_tr)   # (N, K)
        scores_t   = scores_all[node_mask]              # (M, K) 只取目标节点

    scores_np = scores_t.cpu().numpy()

    covs, sizes, shrs = [], [], []
    for k in range(n_repeats):
        rng     = np.random.default_rng(seed_base + k)
        cal_idx = rng.choice(M, n_calib, replace=False)
        mask    = np.zeros(M, dtype=bool); mask[cal_idx] = True

        cal_scores = torch.tensor(scores_np[mask],  dtype=torch.float32, device=device)
        val_scores = torch.tensor(scores_np[~mask], dtype=torch.float32, device=device)
        cal_lab    = labels_np[mask]
        val_lab    = labels_np[~mask]
        n_val      = len(val_lab)

        # one-hot for calibration
        K          = cal_scores.shape[1]
        y_one_hot  = torch.zeros_like(cal_scores, dtype=torch.bool)
        y_one_hot[torch.arange(len(cal_lab)), cal_lab] = True

        quantile   = _calibrate(cal_scores, y_one_hot, alpha)
        pred_sets  = _predict(val_scores, quantile).cpu().numpy()

        covered    = pred_sets[np.arange(n_val), val_lab]
        set_size   = pred_sets.sum(axis=1)
        covs.append(float(covered.mean()))
        sizes.append(float(set_size.mean()))
        # Singleton Hit Ratio
        shrs.append(float(((set_size == 1) & covered).sum() / max(n_val, 1)))

    return (float(np.mean(covs)), float(np.std(covs)),
            float(np.mean(sizes)), float(np.std(sizes)),
            float(np.mean(shrs)), float(np.std(shrs)))


# ════════════════════════════════════════════════════════════════════════
# 4. 模型 / 训练工具
# ════════════════════════════════════════════════════════════════════════

class GCN3BN(nn.Module):
    """三层 GCN + BatchNorm（FB/Twitch 用）"""
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
    model = GCNSparse(feat.shape[1], args.hidden, nclass, args.dropout).to(device)
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
    model = GCN3BN(nfeat, args.hidden, nclass, args.dropout).to(device)
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
def _get_logits(model, feat, adj):
    model.eval()
    return model(feat, adj).cpu().numpy()

@torch.no_grad()
def _get_logits_pyg(model, data):
    model.eval(); data = data.to(device)
    return model(data.x, data.edge_index).cpu().numpy()


def _logits_to_probs(logits_np):
    ex = np.exp(logits_np - logits_np.max(1, keepdims=True))
    return ex / ex.sum(1, keepdims=True)


# ════════════════════════════════════════════════════════════════════════
# 5. 结果保存辅助
# ════════════════════════════════════════════════════════════════════════

def _pr(name, cm, cs, sm, ss, shrm, shrs, alpha):
    print(f'  {name[:30]:<30} | cov={cm:.3f}±{cs:.3f} '
          f'(tgt={1-alpha:.2f})  size={sm:.2f}±{ss:.2f}  shr={shrm:.3f}±{shrs:.3f}')


def _row(seed, split, cm, cs, sm, ss, shrm, shrs, alpha, score, transform):
    return {'seed': seed, 'split': split, 'alpha': alpha,
            'score': score, 'transform': transform,
            'coverage_mean': cm, 'coverage_std': cs,
            'set_size_mean': sm, 'set_size_std': ss,
            'shr_mean': shrm, 'shr_std': shrs,
            'target_coverage': 1. - alpha,
            'coverage_gap': cm - (1. - alpha)}


def _save(cp_rows, uq_runs, split_names, all_keys, prefix):
    if cp_rows:
        df  = pd.DataFrame(cp_rows)
        agg = (df.groupby('split')[
            ['coverage_mean', 'coverage_std', 'set_size_mean', 'set_size_std', 'shr_mean', 'shr_std']]
               .mean().reset_index())
        agg.to_csv(prefix + '_daps.csv', index=False)
        print(f'\n  DAPS CSV → {prefix}_daps.csv')
        print(agg[['split', 'coverage_mean', 'set_size_mean', 'shr_mean']].to_string(index=False))
    if uq_runs:
        summarize(uq_runs, split_names, all_keys,
                  prefix + '_uq_results.csv',
                  prefix.split('/')[-1],
                  reliability_path=prefix + '_reliability.csv',
                  uncertainty_path=prefix + '_uncertainty.csv')


# ════════════════════════════════════════════════════════════════════════
# 6. 各数据集流程
# ════════════════════════════════════════════════════════════════════════

def run_elliptic():
    print('\n[Elliptic] Loading...')
    adj_tr, ei_tr, feat_tr, _, lab_tr_np, N_tr = load_elliptic(
        ELLIPTIC_TRAIN, args.data_dir, device)
    tr_base  = (lab_tr_np >= 0)
    ei_tr_t  = ei_tr                            # (2,E) LongTensor on device
    ei_tr_np = ei_tr.cpu().numpy()

    test_graphs = []
    for i, steps in enumerate(ELLIPTIC_TESTS):
        a, ei_te, f_te, _, lnp, N_te = load_elliptic(steps, args.data_dir, device)
        tm = (lnp >= 0)
        test_graphs.append((a, ei_te, ei_te.cpu().numpy(), f_te, lnp, tm, N_te))

    all_keys    = build_all_keys(binary=True)
    split_names = ['ID-test'] + [
        f'OOD-test_{i}(s{ELLIPTIC_TESTS[i][0]}-{ELLIPTIC_TESTS[i][-1]})'
        for i in range(len(ELLIPTIC_TESTS))]
    uq_runs, cp_rows = [], []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  DAPS Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(
            lab_tr_np, tr_base, args.id_val_ratio, args.id_test_ratio, seed)
        model = _train_sparse(adj_tr, feat_tr, lab_tr_np, tr_m, val_m, 2, seed,
                               os.path.join(args.save_dir, f'elliptic_seed{seed}.pth'),
                               args.class_weight)
        logits_all = _get_logits(model, feat_tr, adj_tr)
        probs_all  = _logits_to_probs(logits_all)
        run_res    = {}

        # UQ 指标（使用 1-max(p) 作为 uncertainty）
        u_id = 1. - probs_all[id_m].max(1)
        r_id = compute_split_metrics(probs_all[id_m], u_id, lab_tr_np[id_m], 2, binary=True)
        run_res['ID-test'] = r_id

        # DAPS 共形预测 on ID-test pool
        cm, cs, sm, ss, shrm, shrs = run_daps_cp(
            logits_all, lab_tr_np[id_m], id_m,
            ei_tr_t, ei_tr_np, N_tr,
            args.alpha, args.score, args.transform, args,
            args.n_calib, args.n_repeats, seed)
        _pr('ID-test', cm, cs, sm, ss, shrm, shrs, args.alpha)
        cp_rows.append(_row(seed, 'ID-test', cm, cs, sm, ss, shrm, shrs,
                             args.alpha, args.score, args.transform))

        # OOD splits — 用同一模型直接推理，不重新训练
        for i, (a_te, ei_te_t, ei_te_np, f_te, lnp, tm, N_te) in enumerate(test_graphs):
            logits_te = _get_logits(model, f_te, a_te)
            probs_te  = _logits_to_probs(logits_te)
            u_ood     = 1. - probs_te[tm].max(1)
            r_ood     = compute_split_metrics(probs_te[tm], u_ood, lnp[tm], 2, binary=True)
            r_ood     = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            name      = split_names[1 + i]; run_res[name] = r_ood

            cm, cs, sm, ss, shrm, shrs = run_daps_cp(
                logits_te, lnp[tm], tm,
                ei_te_t, ei_te_np, N_te,
                args.alpha, args.score, args.transform, args,
                args.n_calib, args.n_repeats, seed)
            _pr(name, cm, cs, sm, ss, shrm, shrs, args.alpha)
            cp_rows.append(_row(seed, name, cm, cs, sm, ss, shrm, shrs,
                                 args.alpha, args.score, args.transform))

        uq_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    tag  = f'{args.score}_{args.transform}'
    _save(cp_rows, uq_runs, split_names, all_keys,
          os.path.join(args.save_dir, f'elliptic_daps_{tag}_a{args.alpha}'))


def run_arxiv():
    print('\n[Arxiv] Loading...')
    adj, ei, feat, _, lab_np, node_year, nclass, N = load_arxiv(args.data_path, device)
    base_mask   = (node_year <= ARXIV_TRAIN_YEAR)
    ei_t        = ei
    ei_np       = ei.cpu().numpy()
    split_names = ['ID-test'] + [f'OOD-test_{i}({ty0}-{ty1})'
                                  for i, (ty0, ty1) in enumerate(ARXIV_TESTS)]
    all_keys = build_all_keys(binary=False)
    uq_runs, cp_rows = [], []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  DAPS Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(
            lab_np, base_mask, args.id_val_ratio, args.id_test_ratio, seed)
        model = _train_sparse(adj, feat, lab_np, tr_m, val_m, nclass, seed,
                               os.path.join(args.save_dir, f'arxiv_seed{seed}.pth'))
        logits_all = _get_logits(model, feat, adj)
        probs_all  = _logits_to_probs(logits_all)
        run_res    = {}

        u_id = 1. - probs_all[id_m].max(1)
        r_id = compute_split_metrics(probs_all[id_m], u_id, lab_np[id_m], nclass)
        run_res['ID-test'] = r_id

        cm, cs, sm, ss, shrm, shrs = run_daps_cp(
            logits_all, lab_np[id_m], id_m, ei_t, ei_np, N,
            args.alpha, args.score, args.transform, args,
            args.n_calib, args.n_repeats, seed)
        _pr('ID-test', cm, cs, sm, ss, shrm, shrs, args.alpha)
        cp_rows.append(_row(seed, 'ID-test', cm, cs, sm, ss, shrm, shrs,
                             args.alpha, args.score, args.transform))

        for i, (ty0, ty1) in enumerate(ARXIV_TESTS):
            te    = (node_year >= ty0) & (node_year <= ty1)
            u_ood = 1. - probs_all[te].max(1)
            r_ood = compute_split_metrics(probs_all[te], u_ood, lab_np[te], nclass)
            r_ood = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            name  = split_names[1 + i]; run_res[name] = r_ood

            cm, cs, sm, ss, shrm, shrs = run_daps_cp(
                logits_all, lab_np[te], te, ei_t, ei_np, N,
                args.alpha, args.score, args.transform, args,
                args.n_calib, args.n_repeats, seed)
            _pr(name, cm, cs, sm, ss, shrm, shrs, args.alpha)
            cp_rows.append(_row(seed, name, cm, cs, sm, ss, shrm, shrs,
                                 args.alpha, args.score, args.transform))

        uq_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    tag = f'{args.score}_{args.transform}'
    _save(cp_rows, uq_runs, split_names, all_keys,
          os.path.join(args.save_dir, f'arxiv_daps_{tag}_a{args.alpha}'))


def run_eerm():
    print(f'\n[EERM-{args.eerm_dataset}] Loading...')
    (adj, ei, feat_envs, lab_t, lab_np, nclass,
     tr_mask, val_mask, test_mask, ood_names, N) = load_eerm(
        args.eerm_root, args.eerm_dataset, device)
    feat_tr  = feat_envs[0]; feat_oods = feat_envs[2:]
    tr_np    = tr_mask.cpu().numpy(); val_np = val_mask.cpu().numpy()
    te_np    = test_mask.cpu().numpy()
    ei_t     = ei; ei_np = ei.cpu().numpy()

    split_names = ['ID-test'] + ood_names
    all_keys    = build_all_keys(binary=False)
    uq_runs, cp_rows = [], []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  DAPS Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        model = _train_sparse(adj, feat_tr, lab_np, tr_np, val_np, nclass, seed,
                               os.path.join(args.save_dir,
                                             f'eerm_{args.eerm_dataset}_seed{seed}.pth'))
        logits_id = _get_logits(model, feat_tr, adj)
        probs_id  = _logits_to_probs(logits_id)
        run_res   = {}

        u_id = 1. - probs_id[te_np].max(1)
        r_id = compute_split_metrics(probs_id[te_np], u_id, lab_np[te_np], nclass)
        run_res['ID-test'] = r_id

        cm, cs, sm, ss, shrm, shrs = run_daps_cp(
            logits_id, lab_np[te_np], te_np, ei_t, ei_np, N,
            args.alpha, args.score, args.transform, args,
            args.n_calib, args.n_repeats, seed)
        _pr('ID-test', cm, cs, sm, ss, shrm, shrs, args.alpha)
        cp_rows.append(_row(seed, 'ID-test', cm, cs, sm, ss, shrm, shrs,
                             args.alpha, args.score, args.transform))

        for feat_ood, name in zip(feat_oods, ood_names):
            logits_ood = _get_logits(model, feat_ood, adj)
            probs_ood  = _logits_to_probs(logits_ood)
            u_ood      = 1. - probs_ood[te_np].max(1)
            r_ood      = compute_split_metrics(probs_ood[te_np], u_ood, lab_np[te_np], nclass)
            r_ood      = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            run_res[name] = r_ood

            cm, cs, sm, ss, shrm, shrs = run_daps_cp(
                logits_ood, lab_np[te_np], te_np, ei_t, ei_np, N,
                args.alpha, args.score, args.transform, args,
                args.n_calib, args.n_repeats, seed)
            _pr(name, cm, cs, sm, ss, shrm, shrs, args.alpha)
            cp_rows.append(_row(seed, name, cm, cs, sm, ss, shrm, shrs,
                                 args.alpha, args.score, args.transform))

        uq_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    ds  = args.eerm_dataset
    tag = f'{args.score}_{args.transform}'
    _save(cp_rows, uq_runs, split_names, all_keys,
          os.path.join(args.save_dir, f'{ds}_daps_{tag}_a{args.alpha}'))


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
        print(f'\n{"="*60}\n  DAPS Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        model = _train_pyg(train_data, val_data, nfeat, nclass, seed,
                            os.path.join(args.save_dir, f'{ds}_seed{seed}.pth'))
        run_res = {}

        logits_id = _get_logits_pyg(model, id_data)
        probs_id  = _logits_to_probs(logits_id)
        labels_id = id_data.y.numpy()
        ei_id_t   = id_data.edge_index.to(device)
        ei_id_np  = id_data.edge_index.numpy()
        N_id      = len(labels_id)

        u_id = 1. - probs_id.max(1)
        r_id = compute_split_metrics(probs_id, u_id, labels_id, nclass)
        run_res['ID-test'] = r_id

        # fb/twitch: 全图即目标节点，mask = 全True
        id_mask_all = np.ones(len(labels_id), dtype=bool)
        cm, cs, sm, ss, shrm, shrs = run_daps_cp(
            logits_id, labels_id, id_mask_all, ei_id_t, ei_id_np, N_id,
            args.alpha, args.score, args.transform, args,
            args.n_calib, args.n_repeats, seed)
        _pr(f'ID-test ({id_dom})', cm, cs, sm, ss, shrm, shrs, args.alpha)
        cp_rows.append(_row(seed, 'ID-test', cm, cs, sm, ss, shrm, shrs,
                             args.alpha, args.score, args.transform))

        for dom, data_ood in zip(ood_doms, ood_datas):
            logits_ood = _get_logits_pyg(model, data_ood)
            probs_ood  = _logits_to_probs(logits_ood)
            lab_ood    = data_ood.y.numpy()
            ei_ood_t   = data_ood.edge_index.to(device)
            ei_ood_np  = data_ood.edge_index.numpy()
            N_ood      = len(lab_ood)
            u_ood      = 1. - probs_ood.max(1)
            r_ood      = compute_split_metrics(probs_ood, u_ood, lab_ood, nclass)
            r_ood      = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            name       = f'OOD-{dom}'; run_res[name] = r_ood

            ood_mask_all = np.ones(len(lab_ood), dtype=bool)
            cm, cs, sm, ss, shrm, shrs = run_daps_cp(
                logits_ood, lab_ood, ood_mask_all, ei_ood_t, ei_ood_np, N_ood,
                args.alpha, args.score, args.transform, args,
                args.n_calib, args.n_repeats, seed)
            _pr(name, cm, cs, sm, ss, shrm, shrs, args.alpha)
            cp_rows.append(_row(seed, name, cm, cs, sm, ss, shrm, shrs,
                                 args.alpha, args.score, args.transform))

        uq_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    tag = f'{args.score}_{args.transform}'
    _save(cp_rows, uq_runs, split_names, all_keys,
          os.path.join(args.save_dir, f'{ds}_daps_{tag}_a{args.alpha}'))


# ════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    {
        'elliptic': run_elliptic,
        'arxiv':    run_arxiv,
        'eerm':     run_eerm,
        'twitch':   run_fb_twitch,
        'facebook': run_fb_twitch,
    }[args.dataset]()
