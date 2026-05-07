"""
CF-GNN — Conformalized Graph Neural Networks (NeurIPS 2023)
============================================================
Huang, Jin, Candès, Leskovec. NeurIPS 2023.

核心思路：
  Split Conformal Prediction — 将测试节点分成 calibration / test 两半，
  用 calibration 半的 non-conformity scores 计算分位数 q_hat，
  生成预测集 S(x) = {y: score(x,y) ≤ q_hat}，
  理论保证 P(y* ∈ S(x)) ≥ 1-alpha。

额外输出指标：
  - coverage_mean / std  (实际覆盖率，目标 ≥ 1-alpha)
  - set_size_mean / std  (预测集平均大小，越小越好)
  - coverage_gap         (coverage - (1-alpha)，越接近 0 越好)
  同时输出标准 UQ 指标 (ECE/NLL/Brier/UE-AUROC/AURC)

用法:
    python experiments/run_cfgnn.py --dataset elliptic --data_dir ./data/elliptic --runs 5 --alpha 0.1
    python experiments/run_cfgnn.py --dataset arxiv    --data_path ./data/arxiv/data.pkl --runs 5
    python experiments/run_cfgnn.py --dataset eerm --eerm_dataset cora \
        --eerm_root ./data/eerm/Planetoid/cora --runs 5
    python experiments/run_cfgnn.py --dataset eerm --eerm_dataset amazon \
        --eerm_root ./data/eerm/Amazon/Photo --runs 5
    python experiments/run_cfgnn.py --dataset twitch   --data_root ./data --runs 5
    python experiments/run_cfgnn.py --dataset facebook --data_root ./data --runs 3
"""
import sys; sys.path.insert(0, 'src')

import os, time, argparse, copy
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv

from gnn_uq_bench.datasets import (
    load_elliptic, load_arxiv, load_eerm, stratified_split,
    ELLIPTIC_TRAIN, ELLIPTIC_TESTS, ARXIV_TRAIN_YEAR, ARXIV_TESTS,
)
from gnn_uq_bench.datasets_fb_twitch import load_facebook_twitch, DOMAIN_SETTINGS
from gnn_uq_bench.models import GCNSparse
from gnn_uq_bench.metrics import (
    compute_split_metrics, add_cross_split_metrics, build_all_keys, summarize,
)

# ── 参数 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',       type=str,   default='elliptic',
                    choices=['elliptic','arxiv','eerm','twitch','facebook'])
parser.add_argument('--data_dir',      type=str,   default='./data/elliptic')
parser.add_argument('--data_path',     type=str,   default='./data/arxiv/data.pkl')
parser.add_argument('--eerm_dataset',  type=str,   default='cora', choices=['cora','amazon'])
parser.add_argument('--eerm_root',     type=str,   default=None)
parser.add_argument('--data_root',     type=str,   default='./data')
parser.add_argument('--runs',          type=int,   default=5)
# 共形预测参数
parser.add_argument('--alpha',         type=float, default=0.1,
                    help='目标错误覆盖率，预测集覆盖率 >= 1-alpha')
parser.add_argument('--score',         type=str,   default='aps',
                    choices=['aps','raps','tps'],
                    help='non-conformity score 类型')
parser.add_argument('--n_calib',       type=int,   default=None,
                    help='校准集大小，None=自动取 min(1000, N//2)')
parser.add_argument('--n_repeats',     type=int,   default=100,
                    help='随机重复共形估计次数（用于减小随机性）')
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
parser.add_argument('--save_dir',      type=str,   default='./results/cfgnn')
# ConfGNN 参数
parser.add_argument('--use_confgnn',   action='store_true', default=False,
                    help='是否使用 ConfGNN 拓扑校正（仅 Elliptic 支持）')
parser.add_argument('--confgnn_hidden', type=int,  default=64)
parser.add_argument('--confgnn_layers', type=int,  default=2)
parser.add_argument('--confgnn_lr',    type=float, default=1e-3)
parser.add_argument('--confgnn_epochs',type=int,   default=500)
parser.add_argument('--confgnn_patience',type=int, default=50)
parser.add_argument('--tau',           type=float, default=1.0,
                    help='ConfGNN size loss 的 sigmoid 温度')
parser.add_argument('--size_loss_weight', type=float, default=0.001,
                    help='ConfGNN size loss 权重')
parser.add_argument('--target_size',   type=int,   default=1,
                    help='ConfGNN 目标预测集大小')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[设备] {device}  alpha={args.alpha}  score={args.score}')


# ── numpy 版本兼容（1.21 用 interpolation=，1.22+ 用 method=）────────
def _quantile_higher(a, q):
    try:
        return float(np.quantile(a, q, method='higher'))
    except TypeError:
        return float(np.quantile(a, q, interpolation='higher'))


# ════════════════════════════════════════════════════════════════════════
# 1. Non-conformity scores (来自 conformalized-gnn 论文)
# ════════════════════════════════════════════════════════════════════════

def _aps_calib(smx, labels):
    """APS 校准分数：累积 softmax 到真实类别为止"""
    pi  = smx.argsort(1)[:, ::-1]
    srt = np.take_along_axis(smx, pi, axis=1).cumsum(axis=1)
    by_class = np.take_along_axis(srt, pi.argsort(axis=1), axis=1)
    return by_class[np.arange(len(labels)), labels]

def _aps_predict(smx, qhat):
    """APS 预测集：累积概率 <= qhat 的类别"""
    pi  = smx.argsort(1)[:, ::-1]
    srt = np.take_along_axis(smx, pi, axis=1).cumsum(axis=1)
    by_class = np.take_along_axis(srt, pi.argsort(axis=1), axis=1)
    return by_class <= qhat

def _raps_calib(smx, labels, lam_reg=0.01, k_reg=5):
    """RAPS 校准分数（带正则化）"""
    K = smx.shape[1]; k_reg = min(k_reg, K)
    reg = np.array(k_reg * [0.] + (K - k_reg) * [lam_reg])[None, :]
    pi  = smx.argsort(1)[:, ::-1]
    srt = np.take_along_axis(smx, pi, axis=1)
    srt_reg = srt + reg
    u = np.random.uniform(0, 1, len(smx))
    srt_cum = srt_reg.cumsum(1) - u[:, None] * srt_reg
    by_class = np.take_along_axis(srt_cum, pi.argsort(1), axis=1)
    return by_class[np.arange(len(labels)), labels]

def _raps_predict(smx, qhat, lam_reg=0.01, k_reg=5):
    K = smx.shape[1]; k_reg = min(k_reg, K)
    reg = np.array(k_reg * [0.] + (K - k_reg) * [lam_reg])[None, :]
    pi  = smx.argsort(1)[:, ::-1]
    srt = np.take_along_axis(smx, pi, axis=1)
    srt_reg = srt + reg
    u = np.random.uniform(0, 1, len(smx))
    srt_cum = srt_reg.cumsum(1) - u[:, None] * srt_reg
    by_class = np.take_along_axis(srt_cum, pi.argsort(1), axis=1)
    return by_class <= qhat

def _tps_calib(smx, labels):
    return 1. - smx[np.arange(len(labels)), labels]

def _tps_predict(smx, qhat):
    return smx >= (1. - qhat)


def _conformal_split(cal_smx, cal_labels, val_smx, alpha, score):
    n       = len(cal_labels)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    if score == 'aps':
        scores = _aps_calib(cal_smx, cal_labels)
        qhat   = _quantile_higher(scores, q_level)
        psets  = _aps_predict(val_smx, qhat)
    elif score == 'raps':
        scores = _raps_calib(cal_smx, cal_labels)
        qhat   = _quantile_higher(scores, q_level)
        psets  = _raps_predict(val_smx, qhat)
    else:  # tps
        scores = _tps_calib(cal_smx, cal_labels)
        qhat   = _quantile_higher(scores, q_level)
        psets  = _tps_predict(val_smx, qhat)
    return psets


def run_conformal(smx, labels, alpha, score, n_calib, n_repeats, seed_base):
    """随机重复 n_repeats 次 split conformal，返回 (cov_mean, cov_std, size_mean, size_std, shr_mean, shr_std)"""
    N = len(labels)
    if n_calib is None:
        n_calib = min(1000, N // 2)
    n_calib = min(n_calib, N - 1)
    covs, sizes, shrs = [], [], []
    for k in range(n_repeats):
        rng = np.random.default_rng(seed_base + k)
        cal_idx = rng.choice(N, n_calib, replace=False)
        mask = np.zeros(N, dtype=bool); mask[cal_idx] = True
        psets = _conformal_split(smx[mask], labels[mask], smx[~mask], alpha, score)
        val_lab = labels[~mask]
        covered  = psets[np.arange(len(val_lab)), val_lab]
        set_size = psets.sum(1)
        covs.append(float(covered.mean()))
        sizes.append(float(set_size.mean()))
        # Singleton Hit Ratio: 预测集大小=1 且 覆盖真实标签 的节点比例
        singleton_hit = ((set_size == 1) & covered).sum() / max(len(val_lab), 1)
        shrs.append(float(singleton_hit))
    return (float(np.mean(covs)), float(np.std(covs)),
            float(np.mean(sizes)), float(np.std(sizes)),
            float(np.mean(shrs)), float(np.std(shrs)))


# ════════════════════════════════════════════════════════════════════════
# 2. 模型 / 训练工具
# ════════════════════════════════════════════════════════════════════════

class GCN3BN(nn.Module):
    """三层 GCN + BatchNorm（FB/Twitch 用）"""
    def __init__(self, nfeat, nhid, nclass, dp):
        super().__init__()
        self.c1  = GCNConv(nfeat, nhid); self.bn1 = nn.BatchNorm1d(nhid)
        self.c2  = GCNConv(nhid,  nhid); self.bn2 = nn.BatchNorm1d(nhid)
        self.c3  = GCNConv(nhid,  nclass); self.dp = dp
    def forward(self, x, ei):
        x = F.dropout(F.relu(self.bn1(self.c1(x, ei))), self.dp, self.training)
        x = F.dropout(F.relu(self.bn2(self.c2(x, ei)) + x), self.dp, self.training)
        return self.c3(x, ei)


def _train_sparse(adj, feat, lab_np, tr_mask, val_mask, nclass, seed, path, class_weight=None):
    torch.manual_seed(seed)
    model = GCNSparse(feat.shape[1], args.hidden, nclass, args.dropout).to(device)
    model.reset_parameters()
    crit  = (nn.CrossEntropyLoss(weight=torch.FloatTensor([1., class_weight]).to(device))
             if class_weight and nclass == 2 else nn.CrossEntropyLoss())
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
def _infer_np(model, feat, adj):
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
# 2.5 ConfGNN — 拓扑感知输出校正模型（来自 CF-GNN 论文核心贡献）
# ════════════════════════════════════════════════════════════════════════

class ConfGNN(nn.Module):
    """
    用一个小 GNN 调整 softmax 输出，使预测集更紧凑。
    输入：原始 softmax 概率 (N, K)
    输出：调整后的 logits (N, K)，用于重新计算 conformal score
    """
    def __init__(self, nclass, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        dims = [nclass] + [nhid] * (nlayers - 1) + [nclass]
        for i in range(nlayers):
            self.convs.append(GCNConv(dims[i], dims[i+1]))
        self.dp = dropout

    def forward(self, prob, edge_index):
        x = prob
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dp, self.training)
        return x


def _train_confgnn(confgnn, base_smx_t, edge_index, labels_t,
                   train_mask, calib_mask, alpha, seed):
    """
    训练 ConfGNN：
      - train_mask: 用于 prediction loss（CE）
      - calib_mask: 用于 size loss（conformal training）
    base_smx_t: 原始模型 softmax 概率 tensor (N, K)，已 detach
    """
    torch.manual_seed(seed)
    opt  = Adam(confgnn.parameters(), lr=args.confgnn_lr, weight_decay=5e-4)
    best, bad, bs = 1e9, 0, None

    for epoch in range(args.confgnn_epochs):
        confgnn.train(); opt.zero_grad()
        out = confgnn(base_smx_t, edge_index)   # (N, K) adjusted logits

        # prediction loss on train nodes
        pred_loss = F.cross_entropy(out[train_mask], labels_t[train_mask])

        # conformal size loss on calib nodes (来自原论文 train.py)
        out_smx = F.softmax(out, dim=1)
        n_calib = int(calib_mask.sum())
        if n_calib > 1 and epoch > 100:
            q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
            # 用当前 calib 前半段估计 q_hat
            calib_idx = calib_mask.nonzero(as_tuple=True)[0]
            half = len(calib_idx) // 2
            cal_idx  = calib_idx[:half]
            test_idx = calib_idx[half:]

            tps_scores = out_smx[cal_idx][
                torch.arange(half), labels_t[cal_idx]]
            q_hat = torch.quantile(tps_scores, 1 - q_level)

            # size loss: sigmoid 近似集合大小
            c = torch.sigmoid((out_smx[test_idx] - q_hat) / args.tau)
            size_loss = torch.mean(
                torch.relu(c.sum(dim=1) - args.target_size))
            loss = pred_loss + args.size_loss_weight * size_loss
        else:
            loss = pred_loss

        loss.backward(); opt.step()

        # early stopping on calib val loss
        confgnn.eval()
        with torch.no_grad():
            out_val = confgnn(base_smx_t, edge_index)
            val_loss = F.cross_entropy(out_val[calib_mask], labels_t[calib_mask]).item()
        if val_loss < best:
            bs = copy.deepcopy(confgnn.state_dict()); best, bad = val_loss, 0
        else:
            bad += 1
        if bad >= args.confgnn_patience: break

    confgnn.load_state_dict(bs)
    print(f'    ConfGNN val={best:.4f}')
    return confgnn


@torch.no_grad()
def _apply_confgnn(confgnn, base_smx_t, edge_index):
    """用训练好的 ConfGNN 调整 softmax，返回新 softmax numpy"""
    confgnn.eval()
    out = confgnn(base_smx_t, edge_index)
    return F.softmax(out, dim=1).cpu().numpy()


# ════════════════════════════════════════════════════════════════════════
# 3. 保存辅助
# ════════════════════════════════════════════════════════════════════════

def _print_conf(name, cm, cs, sm, ss, shrm, shrs, alpha):
    print(f'  {name[:32]:<32} | cov={cm:.3f}±{cs:.3f} '
          f'(tgt={1-alpha:.2f})  size={sm:.2f}±{ss:.2f}  shr={shrm:.3f}±{shrs:.3f}')


def _cf_row(seed, split, cm, cs, sm, ss, shrm, shrs, alpha, score):
    return {'seed': seed, 'split': split, 'alpha': alpha, 'score': score,
            'coverage_mean': cm, 'coverage_std': cs,
            'set_size_mean': sm, 'set_size_std': ss,
            'shr_mean': shrm, 'shr_std': shrs,
            'target_coverage': 1.-alpha, 'coverage_gap': cm-(1.-alpha)}


def _save(cf_rows, uq_runs, split_names, all_keys, prefix):
    if cf_rows:
        df = pd.DataFrame(cf_rows)
        agg = (df.groupby('split')[
            ['coverage_mean','coverage_std','set_size_mean','set_size_std','shr_mean','shr_std']]
               .mean().reset_index())
        agg.to_csv(prefix + '_conformal.csv', index=False)
        print(f'\n  Conformal CSV → {prefix}_conformal.csv')
        print(agg[['split','coverage_mean','set_size_mean','shr_mean']].to_string(index=False))
    if uq_runs:
        summarize(uq_runs, split_names, all_keys,
                  prefix + '_uq_results.csv',
                  prefix.split('/')[-1],
                  reliability_path=prefix + '_reliability.csv',
                  uncertainty_path=prefix + '_uncertainty.csv')


# ════════════════════════════════════════════════════════════════════════
# 4. 各数据集流程
# ════════════════════════════════════════════════════════════════════════

def run_elliptic():
    print('\n[Elliptic] Loading...')
    adj_tr, _, feat_tr, _, lab_tr_np, _ = load_elliptic(
        ELLIPTIC_TRAIN, args.data_dir, device)
    tr_base = (lab_tr_np >= 0)
    test_graphs = []
    for i, steps in enumerate(ELLIPTIC_TESTS):
        a, _, f, _, lnp, _ = load_elliptic(steps, args.data_dir, device)
        tm = (lnp >= 0)
        test_graphs.append((a, f, lnp, tm))

    all_keys    = build_all_keys(binary=True)
    split_names = ['ID-test'] + [
        f'OOD-test_{i}(s{ELLIPTIC_TESTS[i][0]}-{ELLIPTIC_TESTS[i][-1]})'
        for i in range(len(ELLIPTIC_TESTS))]
    uq_runs, cf_rows = [], []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        tag = 'ConfGNN' if args.use_confgnn else 'CF-GNN'
        print(f'\n{"="*60}\n  {tag} Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(
            lab_tr_np, tr_base, args.id_val_ratio, args.id_test_ratio, seed)
        model = _train_sparse(adj_tr, feat_tr, lab_tr_np, tr_m, val_m, 2, seed,
                               os.path.join(args.save_dir, f'elliptic_seed{seed}.pth'),
                               args.class_weight)

        # ── ConfGNN 校正（可选）────────────────────────────────────────
        # 用 OOD-val 图（步骤12-16）的节点训练 ConfGNN，用训练域节点做 pred loss
        if args.use_confgnn:
            print('  [ConfGNN] Training correction model on OOD-val graph...')
            # 加载 OOD-val 图
            from gnn_uq_bench.datasets import ELLIPTIC_VAL
            adj_ov, ei_ov, feat_ov, _, lab_ov_np, N_ov = load_elliptic(
                ELLIPTIC_VAL, args.data_dir, device)
            ov_mask = torch.tensor(lab_ov_np >= 0, dtype=torch.bool).to(device)
            lab_ov_t = torch.tensor(lab_ov_np, dtype=torch.long).to(device)

            # 在 OOD-val 图上推理原始模型，得到 softmax
            with torch.no_grad():
                base_smx_ov = torch.softmax(
                    model(feat_ov, adj_ov), dim=1).detach()

            # 用 OOD-val 图上一半节点做 train，另一半做 calib
            ov_idx = ov_mask.nonzero(as_tuple=True)[0]
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(ov_idx))
            half = max(1, len(ov_idx) // 2)
            tr_ov  = torch.zeros(N_ov, dtype=torch.bool, device=device)
            cal_ov = torch.zeros(N_ov, dtype=torch.bool, device=device)
            tr_ov[ov_idx[perm[:half]]]  = True
            cal_ov[ov_idx[perm[half:]]] = True

            confgnn = ConfGNN(2, args.confgnn_hidden,
                               args.confgnn_layers, args.dropout).to(device)
            confgnn = _train_confgnn(confgnn, base_smx_ov, ei_ov, lab_ov_t,
                                      tr_ov, cal_ov, args.alpha, seed)

            # 用 ConfGNN 校正训练域的 softmax（用于 ID-test CP）
            with torch.no_grad():
                base_smx_tr = torch.softmax(
                    model(feat_tr, adj_tr), dim=1).detach()
            probs = _apply_confgnn(confgnn, base_smx_tr, _)
            # 注意：训练域 edge_index 需要用 ei_tr
            adj_tr2, ei_tr, feat_tr2, _, _, _ = load_elliptic(
                ELLIPTIC_TRAIN, args.data_dir, device)
            with torch.no_grad():
                base_smx_tr = torch.softmax(
                    model(feat_tr, adj_tr), dim=1).detach()
            probs = _apply_confgnn(confgnn, base_smx_tr, ei_tr)
        else:
            probs = _infer_np(model, feat_tr, adj_tr)
            confgnn = None

        run_res = {}

        # ID-test
        u_id = 1. - probs[id_m].max(1)
        r_id = compute_split_metrics(probs[id_m], u_id, lab_tr_np[id_m], 2, binary=True)
        run_res['ID-test'] = r_id
        cm, cs, sm, ss, shrm, shrs = run_conformal(probs[id_m], lab_tr_np[id_m],
                                        args.alpha, args.score, args.n_calib, args.n_repeats, seed)
        _print_conf('ID-test', cm, cs, sm, ss, shrm, shrs, args.alpha)
        cf_rows.append(_cf_row(seed, 'ID-test', cm, cs, sm, ss, shrm, shrs, args.alpha, args.score))

        # OOD splits
        for i, (a_te, f_te, lnp, tm) in enumerate(test_graphs):
            if args.use_confgnn and confgnn is not None:
                # 用 ConfGNN 校正 OOD 图的 softmax
                _, ei_te, _, _, _, _ = load_elliptic(
                    ELLIPTIC_TESTS[i], args.data_dir, device)
                with torch.no_grad():
                    base_smx_te = torch.softmax(
                        model(f_te, a_te), dim=1).detach()
                p_te = _apply_confgnn(confgnn, base_smx_te, ei_te)
            else:
                p_te = _infer_np(model, f_te, a_te)

            u_ood = 1. - p_te[tm].max(1)
            r_ood = compute_split_metrics(p_te[tm], u_ood, lnp[tm], 2, binary=True)
            r_ood = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            name  = split_names[1 + i]; run_res[name] = r_ood
            cm, cs, sm, ss, shrm, shrs = run_conformal(p_te[tm], lnp[tm],
                                            args.alpha, args.score, args.n_calib, args.n_repeats, seed)
            _print_conf(name, cm, cs, sm, ss, shrm, shrs, args.alpha)
            cf_rows.append(_cf_row(seed, name, cm, cs, sm, ss, shrm, shrs, args.alpha, args.score))

        uq_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    score_tag = f'confgnn_{args.score}' if args.use_confgnn else f'cfgnn_{args.score}'
    _save(cf_rows, uq_runs, split_names, all_keys,
          os.path.join(args.save_dir, f'elliptic_{score_tag}_a{args.alpha}'))


def run_arxiv():
    print('\n[Arxiv] Loading...')
    adj, _, feat, _, lab_np, node_year, nclass, _ = load_arxiv(args.data_path, device)
    base_mask   = (node_year <= ARXIV_TRAIN_YEAR)
    split_names = ['ID-test'] + [f'OOD-test_{i}({ty0}-{ty1})'
                                  for i, (ty0, ty1) in enumerate(ARXIV_TESTS)]
    all_keys = build_all_keys(binary=False)
    uq_runs, cf_rows = [], []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  CF-GNN Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(
            lab_np, base_mask, args.id_val_ratio, args.id_test_ratio, seed)
        model = _train_sparse(adj, feat, lab_np, tr_m, val_m, nclass, seed,
                               os.path.join(args.save_dir, f'arxiv_seed{seed}.pth'))
        probs = _infer_np(model, feat, adj)
        run_res = {}

        u_id = 1. - probs[id_m].max(1)
        r_id = compute_split_metrics(probs[id_m], u_id, lab_np[id_m], nclass)
        run_res['ID-test'] = r_id
        cm, cs, sm, ss, shrm, shrs = run_conformal(probs[id_m], lab_np[id_m],
                                        args.alpha, args.score, args.n_calib, args.n_repeats, seed)
        _print_conf('ID-test', cm, cs, sm, ss, shrm, shrs, args.alpha)
        cf_rows.append(_cf_row(seed, 'ID-test', cm, cs, sm, ss, shrm, shrs, args.alpha, args.score))

        for i, (ty0, ty1) in enumerate(ARXIV_TESTS):
            te = (node_year >= ty0) & (node_year <= ty1)
            u_ood = 1. - probs[te].max(1)
            r_ood = compute_split_metrics(probs[te], u_ood, lab_np[te], nclass)
            r_ood = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            name  = split_names[1 + i]; run_res[name] = r_ood
            cm, cs, sm, ss, shrm, shrs = run_conformal(probs[te], lab_np[te],
                                            args.alpha, args.score, args.n_calib, args.n_repeats, seed)
            _print_conf(name, cm, cs, sm, ss, shrm, shrs, args.alpha)
            cf_rows.append(_cf_row(seed, name, cm, cs, sm, ss, shrm, shrs, args.alpha, args.score))

        uq_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    _save(cf_rows, uq_runs, split_names, all_keys,
          os.path.join(args.save_dir, f'arxiv_cfgnn_{args.score}_a{args.alpha}'))


def run_eerm():
    print(f'\n[EERM-{args.eerm_dataset}] Loading...')
    (adj, _, feat_envs, lab_t, lab_np, nclass,
     tr_mask, val_mask, test_mask, ood_names, _) = load_eerm(
        args.eerm_root, args.eerm_dataset, device)
    feat_tr  = feat_envs[0]; feat_oods = feat_envs[2:]
    tr_np    = tr_mask.cpu().numpy(); val_np = val_mask.cpu().numpy()
    te_np    = test_mask.cpu().numpy()
    split_names = ['ID-test'] + ood_names
    all_keys = build_all_keys(binary=False)
    uq_runs, cf_rows = [], []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  CF-GNN Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        model = _train_sparse(adj, feat_tr, lab_np, tr_np, val_np, nclass, seed,
                               os.path.join(args.save_dir,
                                             f'eerm_{args.eerm_dataset}_seed{seed}.pth'))
        probs_id = _infer_np(model, feat_tr, adj)
        run_res = {}

        u_id = 1. - probs_id[te_np].max(1)
        r_id = compute_split_metrics(probs_id[te_np], u_id, lab_np[te_np], nclass)
        run_res['ID-test'] = r_id
        cm, cs, sm, ss, shrm, shrs = run_conformal(probs_id[te_np], lab_np[te_np],
                                        args.alpha, args.score, args.n_calib, args.n_repeats, seed)
        _print_conf('ID-test', cm, cs, sm, ss, shrm, shrs, args.alpha)
        cf_rows.append(_cf_row(seed, 'ID-test', cm, cs, sm, ss, shrm, shrs, args.alpha, args.score))

        for feat_ood, name in zip(feat_oods, ood_names):
            p_ood = _infer_np(model, feat_ood, adj)
            u_ood = 1. - p_ood[te_np].max(1)
            r_ood = compute_split_metrics(p_ood[te_np], u_ood, lab_np[te_np], nclass)
            r_ood = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            run_res[name] = r_ood
            cm, cs, sm, ss, shrm, shrs = run_conformal(p_ood[te_np], lab_np[te_np],
                                            args.alpha, args.score, args.n_calib, args.n_repeats, seed)
            _print_conf(name, cm, cs, sm, ss, shrm, shrs, args.alpha)
            cf_rows.append(_cf_row(seed, name, cm, cs, sm, ss, shrm, shrs, args.alpha, args.score))

        uq_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    ds = args.eerm_dataset
    _save(cf_rows, uq_runs, split_names, all_keys,
          os.path.join(args.save_dir, f'{ds}_cfgnn_{args.score}_a{args.alpha}'))


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
    all_keys = build_all_keys(binary=False)
    uq_runs, cf_rows = [], []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  CF-GNN Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        model = _train_pyg(train_data, val_data, nfeat, nclass, seed,
                            os.path.join(args.save_dir, f'{ds}_seed{seed}.pth'))
        run_res = {}

        probs_id = _infer_pyg(model, id_data)
        labels_id = id_data.y.numpy()
        u_id = 1. - probs_id.max(1)
        r_id = compute_split_metrics(probs_id, u_id, labels_id, nclass)
        run_res['ID-test'] = r_id
        cm, cs, sm, ss, shrm, shrs = run_conformal(probs_id, labels_id,
                                        args.alpha, args.score, args.n_calib, args.n_repeats, seed)
        _print_conf(f'ID-test ({id_dom})', cm, cs, sm, ss, args.alpha)
        cf_rows.append(_cf_row(seed, 'ID-test', cm, cs, sm, ss, shrm, shrs, args.alpha, args.score))

        for dom, data_ood in zip(ood_doms, ood_datas):
            p_ood = _infer_pyg(model, data_ood)
            lab_ood = data_ood.y.numpy()
            u_ood = 1. - p_ood.max(1)
            r_ood = compute_split_metrics(p_ood, u_ood, lab_ood, nclass)
            r_ood = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            name  = f'OOD-{dom}'; run_res[name] = r_ood
            cm, cs, sm, ss, shrm, shrs = run_conformal(p_ood, lab_ood,
                                            args.alpha, args.score, args.n_calib, args.n_repeats, seed)
            _print_conf(name, cm, cs, sm, ss, shrm, shrs, args.alpha)
            cf_rows.append(_cf_row(seed, name, cm, cs, sm, ss, shrm, shrs, args.alpha, args.score))

        uq_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    _save(cf_rows, uq_runs, split_names, all_keys,
          os.path.join(args.save_dir, f'{ds}_cfgnn_{args.score}_a{args.alpha}'))


# ════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    {
        'elliptic': run_elliptic,
        'arxiv':    run_arxiv,
        'eerm':     run_eerm,
        'twitch':   run_fb_twitch,
        'facebook': run_fb_twitch,
    }[args.dataset]()
