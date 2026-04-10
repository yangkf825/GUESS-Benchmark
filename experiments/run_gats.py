"""
GATS — Graph Attention Temperature Scaling
==========================================
不确定性 u = 1 - max(p)（最大 softmax 置信度倒数）

用法:
    python experiments/run_gats.py --dataset elliptic --data_dir ./elliptic --runs 5
    python experiments/run_gats.py --dataset arxiv --data_path ./data.pkl --runs 5
    python experiments/run_gats.py --dataset eerm --eerm_dataset cora --eerm_root ./cora --runs 5
"""
import sys; sys.path.insert(0, 'src')

import os, time, argparse, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
import scipy.special

from gnn_uq_bench.datasets import (
    load_elliptic, load_arxiv, load_eerm, stratified_split,
    ELLIPTIC_TRAIN, ELLIPTIC_VAL, ELLIPTIC_TESTS,
    ARXIV_TRAIN_YEAR, ARXIV_TESTS, ARXIV_OODVAL_YEARS,
)
from gnn_uq_bench.models import GCNPyG, GATModel, GATS, bfs_distance
from gnn_uq_bench.metrics import (
    compute_split_metrics, add_cross_split_metrics, build_all_keys, summarize,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',        type=str,   default='elliptic',
                    choices=['elliptic', 'arxiv', 'eerm'])
parser.add_argument('--data_dir',       type=str,   default='./elliptic')
parser.add_argument('--data_path',      type=str,   default='./data.pkl')
parser.add_argument('--eerm_dataset',   type=str,   default='cora', choices=['cora', 'amazon'])
parser.add_argument('--eerm_root',      type=str,   default=None)
parser.add_argument('--model',          type=str,   default='GCN', choices=['GCN', 'GAT'])
parser.add_argument('--runs',           type=int,   default=5)
parser.add_argument('--hidden',         type=int,   default=64)
parser.add_argument('--dropout',        type=float, default=0.5)
parser.add_argument('--lr',             type=float, default=0.01)
parser.add_argument('--weight_decay',   type=float, default=5e-4)
parser.add_argument('--epochs',         type=int,   default=2000)
parser.add_argument('--patience',       type=int,   default=100)
# GATS 专有参数
parser.add_argument('--gats_heads',     type=int,   default=8)
parser.add_argument('--gats_bias',      type=float, default=1.0)
parser.add_argument('--gats_wdecay',    type=float, default=0.005)
parser.add_argument('--gats_epochs',    type=int,   default=2000)
parser.add_argument('--gats_patience',  type=int,   default=100)
parser.add_argument('--bfs_depth',      type=int,   default=2)
parser.add_argument('--id_val_ratio',   type=float, default=0.1)
parser.add_argument('--id_test_ratio',  type=float, default=0.1)
parser.add_argument('--class_weight',   type=float, default=10.0)
parser.add_argument('--base_seed',      type=int,   default=42)
parser.add_argument('--save_dir',       type=str,   default='./results/gats')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[设备] {device}')


# ── 工具函数 ──────────────────────────────────────────────────

def _get_base(nfeat, nclass):
    if args.model == 'GCN':
        m = GCNPyG(nfeat, args.hidden, nclass, args.dropout)
    else:
        m = GATModel(nfeat, args.hidden, nclass, args.dropout)
    return m.to(device)


def _train_base(base, feat, ei, lab_np, tr_mask, val_mask, save_path, nclass, seed,
                class_weight=None):
    """训练 base GCN/GAT"""
    torch.manual_seed(seed)
    base.reset_parameters()
    crit = (nn.CrossEntropyLoss(weight=torch.FloatTensor([1., class_weight]).to(device))
            if class_weight and nclass == 2 else nn.CrossEntropyLoss())
    opt   = Adam(base.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lab_t = torch.tensor(lab_np, dtype=torch.long).to(device)
    tr_t  = torch.tensor(tr_mask, dtype=torch.bool).to(device)
    val_t = torch.tensor(val_mask, dtype=torch.bool).to(device)
    best, bad, bs = 1e9, 0, None
    for _ in range(args.epochs):
        base.train(); opt.zero_grad()
        crit(base(feat, ei)[tr_t], lab_t[tr_t]).backward(); opt.step()
        base.eval()
        with torch.no_grad():
            lv = crit(base(feat, ei)[val_t], lab_t[val_t]).item()
        if lv < best: bs = copy.deepcopy(base.state_dict()); best, bad = lv, 0
        else: bad += 1
        if bad >= args.patience: break
    base.load_state_dict(bs); torch.save(bs, save_path)
    print(f'    base val={best:.4f}')
    return base


def _train_gats(base, feat, ei, lab_np, tr_mask, ov_mask, N, save_path, nclass, seed):
    """在 ov_mask 上训练 GATS 温度层（base 冻结）"""
    torch.manual_seed(seed)
    dist = bfs_distance(ei, torch.tensor(tr_mask, dtype=torch.bool).to(device),
                        args.bfs_depth, N, device)
    gats = GATS(base, ei, N, dist, nclass,
                heads=args.gats_heads, bias=args.gats_bias).to(device)

    crit  = nn.CrossEntropyLoss()
    opt   = Adam(gats.cagat.parameters(), lr=args.lr, weight_decay=args.gats_wdecay)
    lab_t = torch.tensor(lab_np, dtype=torch.long).to(device)
    ov_t  = torch.tensor(ov_mask, dtype=torch.bool).to(device)
    best, bad, bs = 1e9, 0, None
    for _ in range(args.gats_epochs):
        gats.train(); opt.zero_grad()
        crit(gats(feat, ei)[ov_t], lab_t[ov_t]).backward(); opt.step()
        gats.eval()
        with torch.no_grad():
            lv = crit(gats(feat, ei)[ov_t], lab_t[ov_t]).item()
        if lv < best: bs = copy.deepcopy(gats.state_dict()); best, bad = lv, 0
        else: bad += 1
        if bad >= args.gats_patience: break
    gats.load_state_dict(bs); torch.save(bs, save_path)
    print(f'    GATS val={best:.4f}')
    return gats


@torch.no_grad()
def _infer(model, feat, ei):
    model.eval()
    logits = model(feat, ei).cpu().numpy()
    ex = np.exp(logits - logits.max(1, keepdims=True))
    return ex / ex.sum(1, keepdims=True)


def _metrics(probs, labels, nclass, binary):
    u = 1. - probs.max(1)
    return compute_split_metrics(probs, u, labels, nclass, binary)


# ── Elliptic ──────────────────────────────────────────────────

def run_elliptic():
    all_keys = build_all_keys(binary=True)
    print('\n[Elliptic] Loading...')
    _, ei_tr, feat_tr, lab_tr, lab_tr_np, N_tr = load_elliptic(ELLIPTIC_TRAIN, args.data_dir, device)
    _, ei_ov, feat_ov, lab_ov, lab_ov_np, N_ov = load_elliptic(ELLIPTIC_VAL,   args.data_dir, device)
    tr_base  = (lab_tr_np >= 0)
    ov_mask  = (lab_ov_np >= 0)

    test_graphs = []
    for i, steps in enumerate(ELLIPTIC_TESTS):
        _, ei_te, f_te, _, lnp, N_te = load_elliptic(steps, args.data_dir, device)
        tm = (lnp >= 0)
        test_graphs.append((ei_te, f_te, lnp, tm, N_te))
        print(f'  OOD-test_{i}: labeled={tm.sum()}')

    split_names = ['ID-test'] + [
        f'OOD-test_{i}(s{ELLIPTIC_TESTS[i][0]}-{ELLIPTIC_TESTS[i][-1]})'
        for i in range(len(ELLIPTIC_TESTS))]
    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  GATS Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(lab_tr_np, tr_base,
                                              args.id_val_ratio, args.id_test_ratio, seed)

        # 1) train base on training graph
        base = _get_base(feat_tr.shape[1], 2)
        _train_base(base, feat_tr, ei_tr, lab_tr_np, tr_m, val_m,
                    os.path.join(args.save_dir, f'elliptic_base_seed{seed}.pth'),
                    2, seed, args.class_weight)

        # 2) train GATS on OOD-val graph using val split as overlap
        gats = _train_gats(base, feat_ov, ei_ov, lab_ov_np,
                           tr_m,   # dist computed from training mask
                           ov_mask, N_ov,
                           os.path.join(args.save_dir, f'elliptic_gats_seed{seed}.pth'),
                           2, seed)

        run_res = {}
        # ID-test: infer on train graph with GATS trained on OOD-val
        # (consistent with original: base trained on tr, GATS on ov)
        probs_tr = _infer(gats, feat_ov, ei_ov)   # GATS is anchored to ov graph
        # For ID-test use base model on training graph
        probs_id = _infer(base, feat_tr, ei_tr)
        r_id = _metrics(probs_id[id_m], lab_tr_np[id_m], 2, binary=True)
        print(f'  ID-test | acc={r_id["acc"]:.4f} f1={r_id["f1"]:.4f} ece={r_id["ece"]:.4f}')
        run_res['ID-test'] = r_id

        # OOD-test: retrain base+GATS on each OOD graph
        for i, (ei_te, f_te, lnp, tm, N_te) in enumerate(test_graphs):
            rng = np.random.default_rng(seed + i + 100)
            te_idx = np.where(tm)[0]; perm = rng.permutation(len(te_idx))
            nh = max(1, len(te_idx) // 2)
            tr_te = np.zeros(N_te, bool); tr_te[te_idx[perm[:nh]]] = True
            vl_te = np.zeros(N_te, bool); vl_te[te_idx[perm[nh:]]] = True
            if vl_te.sum() == 0: vl_te = tr_te.copy()

            base_te = _get_base(f_te.shape[1], 2)
            _train_base(base_te, f_te, ei_te, lnp, tr_te, vl_te,
                        os.path.join(args.save_dir, f'elliptic_base_seed{seed}_te{i}.pth'),
                        2, seed + i + 100, args.class_weight)
            gats_te = _train_gats(base_te, f_te, ei_te, lnp,
                                   tr_te, vl_te, N_te,
                                   os.path.join(args.save_dir, f'elliptic_gats_seed{seed}_te{i}.pth'),
                                   2, seed + i + 100)

            probs_te = _infer(gats_te, f_te, ei_te)
            r_ood    = _metrics(probs_te[tm], lnp[tm], 2, binary=True)
            r_ood    = add_cross_split_metrics(r_id, r_ood, r_id['_u'], r_ood['_u'])
            name     = split_names[1 + i]
            print(f'  {name[:30]} | ood_auroc={r_ood["ood_auroc"]:.4f}')
            run_res[name] = r_ood

        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, 'elliptic_gats')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv',
              'Elliptic — GATS',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


# ── OGB-Arxiv ─────────────────────────────────────────────────

def run_arxiv():
    all_keys = build_all_keys(binary=False)
    print('\n[Arxiv] Loading...')
    _, ei, feat, lab_t, lab_np, node_year, nclass, N = load_arxiv(args.data_path, device)
    base_mask = (node_year <= ARXIV_TRAIN_YEAR)
    oy0, oy1  = ARXIV_OODVAL_YEARS
    ov_mask   = (node_year >= oy0) & (node_year <= oy1)

    split_names = ['ID-test'] + [f'OOD-test_{i}({ty0}-{ty1})'
                                  for i, (ty0, ty1) in enumerate(ARXIV_TESTS)]
    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  GATS Arxiv Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(lab_np, base_mask,
                                              args.id_val_ratio, args.id_test_ratio, seed)

        base = _get_base(feat.shape[1], nclass)
        _train_base(base, feat, ei, lab_np, tr_m, val_m,
                    os.path.join(args.save_dir, f'arxiv_base_seed{seed}.pth'),
                    nclass, seed)

        gats = _train_gats(base, feat, ei, lab_np, tr_m, ov_mask, N,
                           os.path.join(args.save_dir, f'arxiv_gats_seed{seed}.pth'),
                           nclass, seed)

        probs_all = _infer(gats, feat, ei)
        run_res   = {}
        r_id = _metrics(probs_all[id_m], lab_np[id_m], nclass, binary=False)
        print(f'  ID-test | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f}')
        run_res['ID-test'] = r_id

        for i, (ty0, ty1) in enumerate(ARXIV_TESTS):
            te_mask = (node_year >= ty0) & (node_year <= ty1)
            r_ood   = _metrics(probs_all[te_mask], lab_np[te_mask], nclass, binary=False)
            r_ood   = add_cross_split_metrics(r_id, r_ood, r_id['_u'], r_ood['_u'])
            run_res[split_names[1 + i]] = r_ood
            print(f'  {split_names[1+i]} | ood_auroc={r_ood["ood_auroc"]:.4f}')

        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, 'arxiv_gats')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv',
              'OGB-Arxiv — GATS',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


# ── EERM ──────────────────────────────────────────────────────

def run_eerm():
    all_keys = build_all_keys(binary=False)
    print(f'\n[EERM-{args.eerm_dataset}] Loading...')
    (_, ei, feat_envs, lab_t, lab_np, nclass,
     tr_mask, val_mask, test_mask, ood_names, N) = load_eerm(args.eerm_root, args.eerm_dataset, device)

    feat_tr   = feat_envs[0]
    feat_oods = feat_envs[2:]
    tr_np     = tr_mask.cpu().numpy()
    val_np    = val_mask.cpu().numpy()
    te_np     = test_mask.cpu().numpy()

    split_names = ['ID-test'] + ood_names
    all_runs    = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  GATS EERM Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')

        # Train base on env-0 features
        base = _get_base(feat_tr.shape[1], nclass)
        _train_base(base, feat_tr, ei, lab_np, tr_np, val_np,
                    os.path.join(args.save_dir, f'eerm_{args.eerm_dataset}_base_seed{seed}.pth'),
                    nclass, seed)

        # Train GATS on env-1 (val) features using val_mask as overlap
        gats = _train_gats(base, feat_envs[1], ei, lab_np,
                           tr_np, val_np, N,
                           os.path.join(args.save_dir, f'eerm_{args.eerm_dataset}_gats_seed{seed}.pth'),
                           nclass, seed)

        run_res  = {}
        probs_id = _infer(gats, feat_tr, ei)
        r_id     = _metrics(probs_id[te_np], lab_np[te_np], nclass, binary=False)
        print(f'  ID-test | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f}')
        run_res['ID-test'] = r_id

        for feat_ood, name in zip(feat_oods, ood_names):
            probs_ood = _infer(gats, feat_ood, ei)
            r_ood     = _metrics(probs_ood[te_np], lab_np[te_np], nclass, binary=False)
            r_ood     = add_cross_split_metrics(r_id, r_ood, r_id['_u'], r_ood['_u'])
            run_res[name] = r_ood
            print(f'  {name[:30]} | ood_auroc={r_ood["ood_auroc"]:.4f}')

        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    ds   = args.eerm_dataset
    pref = os.path.join(args.save_dir, f'{ds}_gats')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv',
              f'EERM-{ds} — GATS',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


if __name__ == '__main__':
    {'elliptic': run_elliptic, 'arxiv': run_arxiv, 'eerm': run_eerm}[args.dataset]()
