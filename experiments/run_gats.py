"""
GATS — Graph Attention Temperature Scaling
==========================================
支持 --backbone GCN / GAT / GraphSAGE

用法:
    python experiments/run_gats.py --dataset elliptic --data_dir ./elliptic --runs 5
    python experiments/run_gats.py --dataset elliptic --data_dir ./elliptic --backbone GAT --runs 5
    python experiments/run_gats.py --dataset elliptic --data_dir ./elliptic --backbone GraphSAGE --runs 5
    python experiments/run_gats.py --dataset arxiv --data_path ./data.pkl --runs 5
    python experiments/run_gats.py --dataset eerm --eerm_dataset cora --eerm_root ./cora --runs 5
"""
import sys; sys.path.insert(0, 'src')

import os, time, argparse, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data

from gnn_uq_bench.datasets import (
    load_elliptic, load_arxiv, load_eerm, stratified_split,
    ELLIPTIC_TRAIN, ELLIPTIC_VAL, ELLIPTIC_TESTS,
    ARXIV_TRAIN_YEAR, ARXIV_TESTS, ARXIV_OODVAL_YEARS,
)
from gnn_uq_bench.models import build_pyg_backbone, GATS, bfs_distance
from gnn_uq_bench.metrics import (
    compute_split_metrics, add_cross_split_metrics, build_all_keys, summarize,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',       type=str,   default='elliptic',
                    choices=['elliptic', 'arxiv', 'eerm'])
parser.add_argument('--data_dir',      type=str,   default='./elliptic')
parser.add_argument('--data_path',     type=str,   default='./data.pkl')
parser.add_argument('--eerm_dataset',  type=str,   default='cora', choices=['cora', 'amazon'])
parser.add_argument('--eerm_root',     type=str,   default=None)
parser.add_argument('--backbone',      type=str,   default='GCN',
                    choices=['GCN', 'GAT', 'GraphSAGE'],
                    help='GNN backbone: GCN (default) / GAT / GraphSAGE')
parser.add_argument('--runs',          type=int,   default=5)
parser.add_argument('--hidden',        type=int,   default=64)
parser.add_argument('--dropout',       type=float, default=0.5)
parser.add_argument('--lr',            type=float, default=0.01)
parser.add_argument('--weight_decay',  type=float, default=5e-4)
parser.add_argument('--epochs',        type=int,   default=2000)
parser.add_argument('--patience',      type=int,   default=100)
parser.add_argument('--gats_heads',    type=int,   default=8)
parser.add_argument('--gats_bias',     type=float, default=1.0)
parser.add_argument('--gats_wdecay',   type=float, default=0.005)
parser.add_argument('--gats_epochs',   type=int,   default=200)
parser.add_argument('--gats_patience', type=int,   default=100)
parser.add_argument('--bfs_depth',     type=int,   default=2)
parser.add_argument('--id_val_ratio',  type=float, default=0.1)
parser.add_argument('--id_test_ratio', type=float, default=0.1)
parser.add_argument('--class_weight',  type=float, default=10.0)
parser.add_argument('--base_seed',     type=int,   default=42)
parser.add_argument('--save_dir',      type=str,   default='./results/gats')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BTAG = args.backbone.lower()
print(f'[设备] {device}  [backbone] {args.backbone}')


def _make_base(nfeat, nclass):
    return build_pyg_backbone(
        args.backbone, nfeat, args.hidden, nclass, args.dropout).to(device)


def _train_base(base, ei, feat, lab_np, tr_mask, val_mask, save_path, nclass, seed, class_weight=None):
    torch.manual_seed(seed); base.reset_parameters()
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


def _train_gats(base, ei, feat, lab_np, tr_mask, N, nclass, seed, save_path):
    torch.manual_seed(seed)
    tr_t  = torch.tensor(tr_mask, dtype=torch.bool).to(device)
    lab_t = torch.tensor(lab_np, dtype=torch.long).to(device)
    dist  = bfs_distance(ei, tr_t, args.bfs_depth, N, device)
    gats  = GATS(base, ei, N, dist, nclass,
                 heads=args.gats_heads, bias=args.gats_bias).to(device)
    opt   = Adam(gats.cagat.parameters(), lr=args.lr, weight_decay=args.gats_wdecay)
    best, bad, bs = 1e9, 0, None
    for _ in range(args.gats_epochs):
        gats.train(); opt.zero_grad()
        F.cross_entropy(gats(feat, ei)[tr_t], lab_t[tr_t]).backward(); opt.step()
        gats.eval()
        with torch.no_grad():
            lv = F.cross_entropy(gats(feat, ei)[tr_t], lab_t[tr_t]).item()
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


def run_elliptic():
    all_keys = build_all_keys(binary=True)
    print('\n[Elliptic] Loading...')
    _, ei_tr, feat_tr, _, lab_tr_np, N_tr = load_elliptic(ELLIPTIC_TRAIN, args.data_dir, device)
    _, ei_ov, feat_ov, _, lab_ov_np, N_ov = load_elliptic(ELLIPTIC_VAL,   args.data_dir, device)
    tr_base = (lab_tr_np >= 0); ov_mask = (lab_ov_np >= 0)

    test_graphs = []
    for i, steps in enumerate(ELLIPTIC_TESTS):
        _, ei_te, f_te, _, lnp, N_te = load_elliptic(steps, args.data_dir, device)
        tm = (lnp >= 0)
        test_graphs.append((ei_te, f_te, lnp, tm, N_te))

    split_names = ['ID-test'] + [
        f'OOD-test_{i}(s{ELLIPTIC_TESTS[i][0]}-{ELLIPTIC_TESTS[i][-1]})'
        for i in range(len(ELLIPTIC_TESTS))]
    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  [{args.backbone}] GATS Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(lab_tr_np, tr_base,
                                              args.id_val_ratio, args.id_test_ratio, seed)
        base = _make_base(feat_tr.shape[1], 2)
        _train_base(base, ei_tr, feat_tr, lab_tr_np, tr_m, val_m,
                    os.path.join(args.save_dir, f'elliptic_{BTAG}_base_seed{seed}.pth'),
                    2, seed, args.class_weight)
        gats = _train_gats(base, ei_ov, feat_ov, lab_ov_np, ov_mask, N_ov, 2, seed,
                            os.path.join(args.save_dir, f'elliptic_{BTAG}_gats_seed{seed}.pth'))

        run_res = {}
        probs_id = _infer(base, feat_tr, ei_tr)
        u_id = 1. - probs_id[id_m].max(1)
        r_id = compute_split_metrics(probs_id[id_m], u_id, lab_tr_np[id_m], 2, binary=True)
        print(f'  ID-test | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f}')
        run_res['ID-test'] = r_id

        for i, (ei_te, f_te, lnp, tm, N_te) in enumerate(test_graphs):
            rng = np.random.default_rng(seed + i + 100)
            te_idx = np.where(tm)[0]; perm = rng.permutation(len(te_idx))
            nh = max(1, len(te_idx)//2)
            tr_te = np.zeros(N_te, bool); tr_te[te_idx[perm[:nh]]] = True
            vl_te = np.zeros(N_te, bool); vl_te[te_idx[perm[nh:]]] = True
            if vl_te.sum() == 0: vl_te = tr_te.copy()

            base_te = _make_base(f_te.shape[1], 2)
            _train_base(base_te, ei_te, f_te, lnp, tr_te, vl_te,
                        os.path.join(args.save_dir, f'elliptic_{BTAG}_base_seed{seed}_te{i}.pth'),
                        2, seed + i + 100, args.class_weight)
            gats_te = _train_gats(base_te, ei_te, f_te, lnp, tr_te, N_te, 2,
                                   seed + i + 100,
                                   os.path.join(args.save_dir, f'elliptic_{BTAG}_gats_seed{seed}_te{i}.pth'))
            probs_te = _infer(base_te, f_te, ei_te)
            u_ood    = 1. - probs_te[tm].max(1)
            r_ood    = compute_split_metrics(probs_te[tm], u_ood, lnp[tm], 2, binary=True)
            r_ood    = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            run_res[split_names[1+i]] = r_ood
            print(f'  {split_names[1+i][:30]} | ood_auroc={r_ood["ood_auroc"]:.4f}')

        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, f'elliptic_{BTAG}_gats')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv',
              f'Elliptic — GATS [{args.backbone}]',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


def run_arxiv():
    all_keys = build_all_keys(binary=False)
    print('\n[Arxiv] Loading...')
    _, ei, feat, _, lab_np, node_year, nclass, N = load_arxiv(args.data_path, device)
    base_mask = (node_year <= ARXIV_TRAIN_YEAR)
    oy0, oy1  = ARXIV_OODVAL_YEARS
    ov_mask   = (node_year >= oy0) & (node_year <= oy1)
    split_names = ['ID-test'] + [f'OOD-test_{i}({ty0}-{ty1})'
                                  for i, (ty0, ty1) in enumerate(ARXIV_TESTS)]
    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  [{args.backbone}] GATS Arxiv Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(lab_np, base_mask,
                                              args.id_val_ratio, args.id_test_ratio, seed)
        base = _make_base(feat.shape[1], nclass)
        _train_base(base, ei, feat, lab_np, tr_m, val_m,
                    os.path.join(args.save_dir, f'arxiv_{BTAG}_base_seed{seed}.pth'), nclass, seed)
        gats = _train_gats(base, ei, feat, lab_np, ov_mask, N, nclass, seed,
                            os.path.join(args.save_dir, f'arxiv_{BTAG}_gats_seed{seed}.pth'))

        probs_all = _infer(base, feat, ei)
        run_res = {}
        u_id = 1. - probs_all[id_m].max(1)
        r_id = compute_split_metrics(probs_all[id_m], u_id, lab_np[id_m], nclass)
        print(f'  ID-test | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f}')
        run_res['ID-test'] = r_id

        for i, (ty0, ty1) in enumerate(ARXIV_TESTS):
            te_mask = (node_year >= ty0) & (node_year <= ty1)
            u_ood   = 1. - probs_all[te_mask].max(1)
            r_ood   = compute_split_metrics(probs_all[te_mask], u_ood, lab_np[te_mask], nclass)
            r_ood   = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            run_res[split_names[1+i]] = r_ood
        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, f'arxiv_{BTAG}_gats')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv',
              f'OGB-Arxiv — GATS [{args.backbone}]',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


def run_eerm():
    all_keys = build_all_keys(binary=False)
    print(f'\n[EERM-{args.eerm_dataset}] Loading...')
    (_, ei, feat_envs, lab_t, lab_np, nclass,
     tr_mask, val_mask, test_mask, ood_names, N) = load_eerm(args.eerm_root, args.eerm_dataset, device)
    feat_tr = feat_envs[0]; feat_oods = feat_envs[2:]
    tr_np = tr_mask.cpu().numpy(); val_np = val_mask.cpu().numpy(); te_np = test_mask.cpu().numpy()
    split_names = ['ID-test'] + ood_names
    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  [{args.backbone}] GATS EERM Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        base = _make_base(feat_tr.shape[1], nclass)
        _train_base(base, ei, feat_tr, lab_np, tr_np, val_np,
                    os.path.join(args.save_dir, f'eerm_{args.eerm_dataset}_{BTAG}_base_seed{seed}.pth'),
                    nclass, seed)
        gats = _train_gats(base, ei, feat_tr, lab_np, val_np, N, nclass, seed,
                            os.path.join(args.save_dir, f'eerm_{args.eerm_dataset}_{BTAG}_gats_seed{seed}.pth'))

        run_res = {}
        probs_id = _infer(base, feat_tr, ei)
        u_id = 1. - probs_id[te_np].max(1)
        r_id = compute_split_metrics(probs_id[te_np], u_id, lab_np[te_np], nclass)
        print(f'  ID-test | acc={r_id["acc"]:.4f}')
        run_res['ID-test'] = r_id

        for feat_ood, name in zip(feat_oods, ood_names):
            probs_ood = _infer(base, feat_ood, ei)
            u_ood = 1. - probs_ood[te_np].max(1)
            r_ood = compute_split_metrics(probs_ood[te_np], u_ood, lab_np[te_np], nclass)
            r_ood = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            run_res[name] = r_ood
        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    ds = args.eerm_dataset
    pref = os.path.join(args.save_dir, f'{ds}_{BTAG}_gats')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv',
              f'EERM-{ds} — GATS [{args.backbone}]',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


if __name__ == '__main__':
    {'elliptic': run_elliptic, 'arxiv': run_arxiv, 'eerm': run_eerm}[args.dataset]()
