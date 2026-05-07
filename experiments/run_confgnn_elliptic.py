"""
ConfGNN — Topology-Aware Conformal Correction for Elliptic Dataset
===================================================================
Huang, Jin, Candès, Leskovec. NeurIPS 2023.

ConfGNN 是 CF-GNN 论文的核心贡献：
  在做 conformal prediction 之前，用一个小 GNN 调整原模型的 softmax 输出，
  使预测集更紧凑（Set Size 更小），同时保持覆盖率理论保证。

Elliptic 上的实现方案（方案一）：
  训练域（步骤7-11）  → 训练 GCN backbone
  OOD-val（步骤12-16）→ 训练 ConfGNN（有标签，可用）
  OOD-test（步骤17-48）→ 用 backbone + ConfGNN 联合推理，做 split CP

对比实验：
  --use_confgnn False  → 标准 CF-GNN（APS/RAPS/TPS）
  --use_confgnn True   → ConfGNN 校正后再做 CP

用法:
    # 标准 CF-GNN 基线
    python experiments/run_confgnn_elliptic.py \\
        --data_dir ./data/elliptic --alpha 0.1 --score aps --runs 5

    # ConfGNN 校正版
    python experiments/run_confgnn_elliptic.py \\
        --data_dir ./data/elliptic --alpha 0.1 --score aps --runs 5 \\
        --use_confgnn --confgnn_hidden 64 --confgnn_layers 2 \\
        --tau 1.0 --size_loss_weight 0.001 --target_size 1
"""
import sys; sys.path.insert(0, 'src')

import os, time, argparse, copy
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv

from gnn_uq_bench.datasets import (
    load_elliptic, stratified_split,
    ELLIPTIC_TRAIN, ELLIPTIC_VAL, ELLIPTIC_TESTS,
)
from gnn_uq_bench.models import GCNSparse
from gnn_uq_bench.metrics import (
    compute_split_metrics, add_cross_split_metrics, build_all_keys, summarize,
)

# ── 参数 ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',        type=str,   default='./data/elliptic')
parser.add_argument('--runs',            type=int,   default=5)
parser.add_argument('--alpha',           type=float, default=0.1)
parser.add_argument('--score',           type=str,   default='aps',
                    choices=['aps', 'raps', 'tps'])
parser.add_argument('--n_calib',         type=int,   default=None)
parser.add_argument('--n_repeats',       type=int,   default=100)
# GCN backbone 参数
parser.add_argument('--hidden',          type=int,   default=64)
parser.add_argument('--dropout',         type=float, default=0.5)
parser.add_argument('--lr',             type=float, default=0.01)
parser.add_argument('--weight_decay',    type=float, default=5e-4)
parser.add_argument('--epochs',          type=int,   default=2000)
parser.add_argument('--patience',        type=int,   default=100)
parser.add_argument('--id_val_ratio',    type=float, default=0.1)
parser.add_argument('--id_test_ratio',   type=float, default=0.1)
parser.add_argument('--class_weight',    type=float, default=10.0)
# ConfGNN 参数
parser.add_argument('--use_confgnn',     action='store_true', default=False)
parser.add_argument('--confgnn_hidden',  type=int,   default=64)
parser.add_argument('--confgnn_layers',  type=int,   default=2)
parser.add_argument('--confgnn_lr',      type=float, default=1e-3)
parser.add_argument('--confgnn_epochs',  type=int,   default=500)
parser.add_argument('--confgnn_patience',type=int,   default=50)
parser.add_argument('--tau',             type=float, default=1.0,
                    help='size loss 中 sigmoid 的温度，越小越近似 step function')
parser.add_argument('--size_loss_weight',type=float, default=0.001)
parser.add_argument('--target_size',     type=int,   default=1,
                    help='ConfGNN 期望的目标预测集大小')
parser.add_argument('--base_seed',       type=int,   default=42)
parser.add_argument('--save_dir',        type=str,   default='./results/confgnn')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
method = 'ConfGNN' if args.use_confgnn else 'CF-GNN'
print(f'[设备] {device}  方法={method}  alpha={args.alpha}  score={args.score}')


# ════════════════════════════════════════════════════════════════════════
# 1. Non-conformity scores
# ════════════════════════════════════════════════════════════════════════

def _quantile_higher(a, q):
    try:
        return float(np.quantile(a, q, method='higher'))
    except TypeError:
        return float(np.quantile(a, q, interpolation='higher'))


def _aps_calib(smx, labels):
    pi  = smx.argsort(1)[:, ::-1]
    srt = np.take_along_axis(smx, pi, axis=1).cumsum(axis=1)
    by_class = np.take_along_axis(srt, pi.argsort(axis=1), axis=1)
    return by_class[np.arange(len(labels)), labels]

def _aps_predict(smx, qhat):
    pi  = smx.argsort(1)[:, ::-1]
    srt = np.take_along_axis(smx, pi, axis=1).cumsum(axis=1)
    by_class = np.take_along_axis(srt, pi.argsort(axis=1), axis=1)
    return by_class <= qhat

def _raps_calib(smx, labels, lam=0.01, k=5):
    K = smx.shape[1]; k = min(k, K)
    reg = np.array(k*[0.] + (K-k)*[lam])[None, :]
    pi  = smx.argsort(1)[:, ::-1]
    srt = np.take_along_axis(smx, pi, axis=1)
    srt_reg = srt + reg
    u   = np.random.uniform(0, 1, len(smx))
    cum = srt_reg.cumsum(1) - u[:, None] * srt_reg
    by_class = np.take_along_axis(cum, pi.argsort(1), axis=1)
    return by_class[np.arange(len(labels)), labels]

def _raps_predict(smx, qhat, lam=0.01, k=5):
    K = smx.shape[1]; k = min(k, K)
    reg = np.array(k*[0.] + (K-k)*[lam])[None, :]
    pi  = smx.argsort(1)[:, ::-1]
    srt = np.take_along_axis(smx, pi, axis=1)
    srt_reg = srt + reg
    u   = np.random.uniform(0, 1, len(smx))
    cum = srt_reg.cumsum(1) - u[:, None] * srt_reg
    by_class = np.take_along_axis(cum, pi.argsort(1), axis=1)
    return by_class <= qhat

def _tps_calib(smx, labels):
    return 1. - smx[np.arange(len(labels)), labels]

def _tps_predict(smx, qhat):
    return smx >= (1. - qhat)


def _split_conformal(cal_smx, cal_lab, val_smx, alpha, score):
    n       = len(cal_lab)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    if score == 'aps':
        qhat  = _quantile_higher(_aps_calib(cal_smx, cal_lab), q_level)
        psets = _aps_predict(val_smx, qhat)
    elif score == 'raps':
        qhat  = _quantile_higher(_raps_calib(cal_smx, cal_lab), q_level)
        psets = _raps_predict(val_smx, qhat)
    else:
        qhat  = _quantile_higher(_tps_calib(cal_smx, cal_lab), q_level)
        psets = _tps_predict(val_smx, qhat)
    return psets


def run_conformal(smx, labels, alpha, score, n_calib, n_repeats, seed_base):
    N = len(labels)
    if n_calib is None: n_calib = min(1000, N // 2)
    n_calib = min(n_calib, N - 1)
    covs, sizes, shrs = [], [], []
    for k in range(n_repeats):
        rng     = np.random.default_rng(seed_base + k)
        cal_idx = rng.choice(N, n_calib, replace=False)
        mask    = np.zeros(N, bool); mask[cal_idx] = True
        psets   = _split_conformal(smx[mask], labels[mask], smx[~mask], alpha, score)
        val_lab = labels[~mask]
        covered  = psets[np.arange(len(val_lab)), val_lab]
        set_size = psets.sum(1)
        covs.append(float(covered.mean()))
        sizes.append(float(set_size.mean()))
        shrs.append(float(((set_size == 1) & covered).sum() / max(len(val_lab), 1)))
    return (float(np.mean(covs)), float(np.std(covs)),
            float(np.mean(sizes)), float(np.std(sizes)),
            float(np.mean(shrs)), float(np.std(shrs)))


# ════════════════════════════════════════════════════════════════════════
# 2. GCN Backbone
# ════════════════════════════════════════════════════════════════════════

def _train_backbone(adj, feat, lab_np, tr_mask, val_mask, seed, path):
    torch.manual_seed(seed)
    model = GCNSparse(feat.shape[1], args.hidden, 2, args.dropout).to(device)
    model.reset_parameters()
    crit  = nn.CrossEntropyLoss(
        weight=torch.FloatTensor([1., args.class_weight]).to(device))
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
    print(f'    Backbone val={best:.4f}')
    return model


@torch.no_grad()
def _infer(model, feat, adj):
    model.eval()
    logits = model(feat, adj).cpu().numpy()
    ex = np.exp(logits - logits.max(1, keepdims=True))
    return ex / ex.sum(1, keepdims=True)


# ════════════════════════════════════════════════════════════════════════
# 3. ConfGNN — 核心贡献
# ════════════════════════════════════════════════════════════════════════

class ConfGNNModel(nn.Module):
    """
    小型 GNN，输入原始 softmax 概率，输出校正后的 logits。
    论文中的 GNN_Multi_Layer，用 GCN 作为 backbone。
    """
    def __init__(self, nclass, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        dims = [nclass] + [nhid] * (nlayers - 1) + [nclass]
        for i in range(nlayers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
        self.dp = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dp, self.training)
        return x


def _train_confgnn(backbone, adj_ov, ei_ov, feat_ov, lab_ov_np, seed):
    """
    在 OOD-val 图（步骤12-16）上训练 ConfGNN。

    训练目标（来自原论文 train.py）：
      1. Prediction loss（CE）：保持预测准确性
      2. Size loss（conformal training）：用 sigmoid 近似缩小预测集大小

    训练/校准分割：
      OOD-val 有标签节点的前一半 → prediction loss
      OOD-val 有标签节点的后一半 → size loss（conformal calibration）
    """
    torch.manual_seed(seed)
    ov_valid = (lab_ov_np >= 0)
    ov_idx   = np.where(ov_valid)[0]

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(ov_idx))
    half = max(2, len(ov_idx) // 2)
    tr_idx  = ov_idx[perm[:half]]
    cal_idx = ov_idx[perm[half:]]

    # 用训练好的 backbone 在 OOD-val 图上推理，得到 base softmax（固定不变）
    with torch.no_grad():
        backbone.eval()
        logits_ov  = backbone(feat_ov, adj_ov)
        base_smx_t = F.softmax(logits_ov, dim=1).detach()  # (N_ov, 2)

    lab_t = torch.tensor(lab_ov_np, dtype=torch.long).to(device)
    tr_t  = torch.zeros(len(lab_ov_np), dtype=torch.bool, device=device)
    tr_t[tr_idx]  = True
    cal_t = torch.zeros(len(lab_ov_np), dtype=torch.bool, device=device)
    cal_t[cal_idx] = True

    confgnn = ConfGNNModel(2, args.confgnn_hidden,
                            args.confgnn_layers, args.dropout).to(device)
    opt  = Adam(confgnn.parameters(), lr=args.confgnn_lr, weight_decay=5e-4)
    best, bad, bs = 1e9, 0, None

    n_cal   = int(cal_t.sum().item())
    q_level = np.ceil((n_cal + 1) * (1 - args.alpha)) / max(n_cal, 1)

    for epoch in range(args.confgnn_epochs):
        confgnn.train(); opt.zero_grad()
        out = confgnn(base_smx_t, ei_ov)          # (N_ov, 2) 调整后 logits

        # ── Prediction loss（CE）────────────────────────────────────
        pred_loss = F.cross_entropy(out[tr_t], lab_t[tr_t])

        # ── Size loss（conformal training，epoch > 100 后开始）───────
        if epoch > 100 and n_cal > 4:
            out_smx = F.softmax(out, dim=1)

            # 用 cal 节点的前半段估计 q_hat
            cal_nodes  = cal_t.nonzero(as_tuple=True)[0]
            half_cal   = len(cal_nodes) // 2
            cal_half1  = cal_nodes[:half_cal]   # 用于估计 q_hat
            cal_half2  = cal_nodes[half_cal:]   # 用于计算 size loss

            # TPS score（简单稳定，原论文用此估计 q_hat）
            tps_scores = out_smx[cal_half1][
                torch.arange(len(cal_half1)), lab_t[cal_half1]]
            q_hat = torch.quantile(tps_scores, max(0., 1. - q_level))

            # sigmoid 近似预测集大小
            c = torch.sigmoid((out_smx[cal_half2] - q_hat) / args.tau)
            size_loss = torch.mean(
                torch.relu(c.sum(dim=1) - args.target_size))

            loss = pred_loss + args.size_loss_weight * size_loss
        else:
            loss = pred_loss

        loss.backward(); opt.step()

        # early stopping
        confgnn.eval()
        with torch.no_grad():
            val_out  = confgnn(base_smx_t, ei_ov)
            val_loss = F.cross_entropy(val_out[cal_t], lab_t[cal_t]).item()
        if val_loss < best:
            bs = copy.deepcopy(confgnn.state_dict()); best, bad = val_loss, 0
        else:
            bad += 1
        if bad >= args.confgnn_patience: break

    confgnn.load_state_dict(bs)
    print(f'    ConfGNN val={best:.4f}  (trained on OOD-val, '
          f'{len(tr_idx)} tr / {len(cal_idx)} cal nodes)')
    return confgnn


@torch.no_grad()
def _infer_confgnn(backbone, confgnn, feat, adj, ei):
    """backbone → base softmax → ConfGNN → 校正 softmax"""
    backbone.eval(); confgnn.eval()
    logits   = backbone(feat, adj)
    base_smx = F.softmax(logits, dim=1).detach()
    out      = confgnn(base_smx, ei)
    probs    = F.softmax(out, dim=1).cpu().numpy()
    return probs


# ════════════════════════════════════════════════════════════════════════
# 4. 输出辅助
# ════════════════════════════════════════════════════════════════════════

def _pr(name, cm, cs, sm, ss, shrm, shrs, alpha):
    print(f'  {name[:32]:<32} | cov={cm:.3f}±{cs:.3f} '
          f'(tgt={1-alpha:.2f})  size={sm:.2f}±{ss:.2f}  shr={shrm:.3f}±{shrs:.3f}')

def _row(seed, split, cm, cs, sm, ss, shrm, shrs):
    return {'seed': seed, 'split': split, 'alpha': args.alpha, 'score': args.score,
            'method': method,
            'coverage_mean': cm, 'coverage_std': cs,
            'set_size_mean': sm, 'set_size_std': ss,
            'shr_mean': shrm,   'shr_std': shrs,
            'target_coverage': 1. - args.alpha,
            'coverage_gap': cm - (1. - args.alpha)}

def _save(cp_rows, uq_runs, split_names, all_keys, prefix):
    if cp_rows:
        df  = pd.DataFrame(cp_rows)
        agg = (df.groupby('split')[
            ['coverage_mean','coverage_std','set_size_mean','set_size_std',
             'shr_mean','shr_std']]
               .mean().reset_index())
        agg.to_csv(prefix + '_cp.csv', index=False)
        print(f'\n  CSV → {prefix}_cp.csv')
        print(agg[['split','coverage_mean','set_size_mean','shr_mean']].to_string(index=False))
    if uq_runs:
        summarize(uq_runs, split_names, all_keys,
                  prefix + '_uq.csv', prefix.split('/')[-1],
                  reliability_path=prefix + '_reliability.csv',
                  uncertainty_path=prefix + '_uncertainty.csv')


# ════════════════════════════════════════════════════════════════════════
# 5. 主流程
# ════════════════════════════════════════════════════════════════════════

def main():
    print('\n[Elliptic] Loading data...')
    # 训练域
    adj_tr, ei_tr, feat_tr, _, lab_tr_np, _ = load_elliptic(
        ELLIPTIC_TRAIN, args.data_dir, device)
    tr_base = (lab_tr_np >= 0)

    # OOD-val 域（步骤12-16）：用于训练 ConfGNN
    adj_ov, ei_ov, feat_ov, _, lab_ov_np, _ = load_elliptic(
        ELLIPTIC_VAL, args.data_dir, device)

    # OOD-test 域
    test_graphs = []
    for i, steps in enumerate(ELLIPTIC_TESTS):
        a, ei, f, _, lnp, _ = load_elliptic(steps, args.data_dir, device)
        tm = (lnp >= 0)
        test_graphs.append((a, ei, f, lnp, tm))
        print(f'  OOD-test_{i} (steps {steps[0]}-{steps[-1]}): '
              f'{tm.sum()} labeled nodes')

    all_keys    = build_all_keys(binary=True)
    split_names = ['ID-test'] + [
        f'OOD-test_{i}(s{ELLIPTIC_TESTS[i][0]}-{ELLIPTIC_TESTS[i][-1]})'
        for i in range(len(ELLIPTIC_TESTS))]
    uq_runs, cp_rows = [], []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  {method} Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')

        # ── Step 1: 训练 GCN backbone（仅用训练域）────────────────────
        tr_m, val_m, id_m = stratified_split(
            lab_tr_np, tr_base, args.id_val_ratio, args.id_test_ratio, seed)
        backbone = _train_backbone(
            adj_tr, feat_tr, lab_tr_np, tr_m, val_m, seed,
            os.path.join(args.save_dir, f'backbone_seed{seed}.pth'))

        # ── Step 2: 训练 ConfGNN（用 OOD-val 图）─────────────────────
        if args.use_confgnn:
            print('  Training ConfGNN on OOD-val graph (steps 12-16)...')
            confgnn = _train_confgnn(backbone, adj_ov, ei_ov,
                                      feat_ov, lab_ov_np, seed)
            # 推理函数：backbone → base_smx → ConfGNN → 校正 softmax
            def get_probs(feat, adj, ei):
                return _infer_confgnn(backbone, confgnn, feat, adj, ei)
        else:
            confgnn = None
            def get_probs(feat, adj, ei):
                return _infer(backbone, feat, adj)

        # ── Step 3: ID-test 推理 ─────────────────────────────────────
        probs_tr = get_probs(feat_tr, adj_tr, ei_tr)
        run_res  = {}

        u_id = 1. - probs_tr[id_m].max(1)
        r_id = compute_split_metrics(probs_tr[id_m], u_id,
                                      lab_tr_np[id_m], 2, binary=True)
        run_res['ID-test'] = r_id

        cm, cs, sm, ss, shrm, shrs = run_conformal(
            probs_tr[id_m], lab_tr_np[id_m],
            args.alpha, args.score, args.n_calib, args.n_repeats, seed)
        _pr('ID-test', cm, cs, sm, ss, shrm, shrs, args.alpha)
        cp_rows.append(_row(seed, 'ID-test', cm, cs, sm, ss, shrm, shrs))

        # ── Step 4: OOD-test 推理 ────────────────────────────────────
        for i, (a_te, ei_te, f_te, lnp, tm) in enumerate(test_graphs):
            p_te  = get_probs(f_te, a_te, ei_te)
            u_ood = 1. - p_te[tm].max(1)
            r_ood = compute_split_metrics(p_te[tm], u_ood, lnp[tm], 2, binary=True)
            r_ood = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            name  = split_names[1 + i]; run_res[name] = r_ood

            cm, cs, sm, ss, shrm, shrs = run_conformal(
                p_te[tm], lnp[tm],
                args.alpha, args.score, args.n_calib, args.n_repeats, seed)
            _pr(name, cm, cs, sm, ss, shrm, shrs, args.alpha)
            cp_rows.append(_row(seed, name, cm, cs, sm, ss, shrm, shrs))

        uq_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    tag  = f'confgnn_{args.score}' if args.use_confgnn else f'cfgnn_{args.score}'
    _save(cp_rows, uq_runs, split_names, all_keys,
          os.path.join(args.save_dir, f'elliptic_{tag}_a{args.alpha}'))


if __name__ == '__main__':
    main()
