"""
G-ΔUQ — 随机锚点不确定性估计
==============================
支持 --backbone GCN / GAT / GraphSAGE
不确定性 u = mean std over anchors（per-node, 跨类平均 sigmoid std）

用法:
    python experiments/run_gduq.py --dataset elliptic --data_dir ./elliptic --runs 5
    python experiments/run_gduq.py --dataset elliptic --data_dir ./elliptic --backbone GAT --runs 5
    python experiments/run_gduq.py --dataset elliptic --data_dir ./elliptic --backbone GraphSAGE --runs 5
    python experiments/run_gduq.py --dataset arxiv --data_path ./data.pkl --runs 5
    python experiments/run_gduq.py --dataset eerm --eerm_dataset cora --eerm_root ./cora --runs 5
"""
import sys; sys.path.insert(0, 'src')

import os, time, argparse, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data

from gnn_uq_bench.datasets import (
    load_elliptic, load_arxiv, load_eerm, stratified_split,
    ELLIPTIC_TRAIN, ELLIPTIC_TESTS, ARXIV_TRAIN_YEAR, ARXIV_TESTS, ARXIV_OODVAL_YEARS,
)
from gnn_uq_bench.models import BaseModelNode, GraphANTNode
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
parser.add_argument('--num_layers',    type=int,   default=2)
parser.add_argument('--dropout',       type=float, default=0.5)
parser.add_argument('--lr',            type=float, default=0.01)
parser.add_argument('--weight_decay',  type=float, default=5e-4)
parser.add_argument('--epochs',        type=int,   default=2000)
parser.add_argument('--patience',      type=int,   default=100)
parser.add_argument('--n_anchors',     type=int,   default=30)
parser.add_argument('--anchor_type',   type=str,   default='node', choices=['node', 'graph'])
parser.add_argument('--id_val_ratio',  type=float, default=0.1)
parser.add_argument('--id_test_ratio', type=float, default=0.1)
parser.add_argument('--class_weight',  type=float, default=10.0)
parser.add_argument('--base_seed',     type=int,   default=42)
parser.add_argument('--save_dir',      type=str,   default='./results/gduq')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BTAG = args.backbone.lower()
print(f'[设备] {device}  [backbone] {args.backbone}')


def _build_gant(nfeat, nclass, feat_np):
    mu  = torch.tensor(feat_np.mean(0), dtype=torch.float32)
    std = torch.tensor(feat_np.std(0) + 1e-6, dtype=torch.float32)
    base = BaseModelNode(nfeat * 2, args.hidden, nclass,
                         args.num_layers, args.dropout,
                         backbone=args.backbone)   # ← backbone 参数传入
    return GraphANTNode(base, mu, std,
                        anchor_type=args.anchor_type,
                        num_classes=nclass).to(device)


def _train(model, data, tr_mask, val_mask, lab_np, nclass, save_path, seed, class_weight=None):
    torch.manual_seed(seed)
    if class_weight and nclass == 2:
        crit = nn.CrossEntropyLoss(weight=torch.FloatTensor([1., class_weight]).to(device))
    else:
        crit = nn.CrossEntropyLoss()
    opt   = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lab_t = torch.tensor(lab_np, dtype=torch.long).to(device)
    tr_t  = torch.tensor(tr_mask, dtype=torch.bool).to(device)
    val_t = torch.tensor(val_mask, dtype=torch.bool).to(device)
    best, bad, bs = 1e9, 0, None
    data = data.to(device)
    for _ in range(args.epochs):
        model.train(); opt.zero_grad()
        logits = model(data)
        crit(logits[tr_t], lab_t[tr_t]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lv = F.cross_entropy(model(data)[val_t], lab_t[val_t]).item()
        if lv < best: bs = copy.deepcopy(model.state_dict()); best, bad = lv, 0
        else: bad += 1
        if bad >= args.patience: break
    model.load_state_dict(bs); torch.save(bs, save_path)
    print(f'    GDUQ val={best:.4f}')
    return model


def run_elliptic():
    all_keys = build_all_keys(binary=True)
    print('\n[Elliptic] Loading...')
    _, ei_tr, feat_tr, lab_tr, lab_tr_np, N_tr = load_elliptic(ELLIPTIC_TRAIN, args.data_dir, device)
    tr_base = (lab_tr_np >= 0)
    feat_np = feat_tr.cpu().numpy()

    test_graphs = []
    for i, steps in enumerate(ELLIPTIC_TESTS):
        _, ei_te, f_te, lb_te, lnp, N_te = load_elliptic(steps, args.data_dir, device)
        tm = (lnp >= 0)
        test_graphs.append((Data(x=f_te, edge_index=ei_te, y=lb_te), lnp, tm, N_te))

    split_names = ['ID-test'] + [
        f'OOD-test_{i}(s{ELLIPTIC_TESTS[i][0]}-{ELLIPTIC_TESTS[i][-1]})'
        for i in range(len(ELLIPTIC_TESTS))]
    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  [{args.backbone}] G-DUQ Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(lab_tr_np, tr_base,
                                              args.id_val_ratio, args.id_test_ratio, seed)
        data_tr = Data(x=feat_tr, edge_index=ei_tr, y=lab_tr).to(device)
        model = _build_gant(feat_tr.shape[1], 2, feat_np)
        _train(model, data_tr, tr_m, val_m, lab_tr_np, 2,
               os.path.join(args.save_dir, f'elliptic_{BTAG}_seed{seed}.pth'),
               seed, args.class_weight)

        probs, u_all = model.infer(data_tr, args.n_anchors)
        run_res = {}
        r_id = compute_split_metrics(probs[id_m], u_all[id_m], lab_tr_np[id_m], 2, binary=True)
        print(f'  ID-test | acc={r_id["acc"]:.4f} ue_auroc={r_id["ue_auroc"]:.4f}')
        run_res['ID-test'] = r_id

        for i, (data_te, lnp, tm, N_te) in enumerate(test_graphs):
            rng = np.random.default_rng(seed + i + 100)
            te_idx = np.where(tm)[0]; perm = rng.permutation(len(te_idx))
            nh = max(1, len(te_idx) // 2)
            tr_te = np.zeros(N_te, bool); tr_te[te_idx[perm[:nh]]] = True
            vl_te = np.zeros(N_te, bool); vl_te[te_idx[perm[nh:]]] = True
            if vl_te.sum() == 0: vl_te = tr_te.copy()

            fnp_te = data_te.x.cpu().numpy()
            model_te = _build_gant(data_te.x.shape[1], 2, fnp_te)
            _train(model_te, data_te, tr_te, vl_te, lnp, 2,
                   os.path.join(args.save_dir, f'elliptic_{BTAG}_seed{seed}_te{i}.pth'),
                   seed + i + 100, args.class_weight)
            probs_te, u_te = model_te.infer(data_te, args.n_anchors)
            r_ood = compute_split_metrics(probs_te[tm], u_te[tm], lnp[tm], 2, binary=True)
            r_ood = add_cross_split_metrics(r_id, r_ood, r_id['_u'], r_ood['_u'])
            run_res[split_names[1+i]] = r_ood
            print(f'  {split_names[1+i][:30]} | ood_auroc={r_ood["ood_auroc"]:.4f}')

        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, f'elliptic_{BTAG}_gduq')
    summarize(all_runs, split_names, all_keys, pref+'_results.csv',
              f'Elliptic — G-ΔUQ [{args.backbone}]',
              reliability_path=pref+'_reliability.csv',
              uncertainty_path=pref+'_uncertainty.csv')


def run_arxiv():
    all_keys = build_all_keys(binary=False)
    print('\n[Arxiv] Loading...')
    _, ei, feat, lab_t, lab_np, node_year, nclass, N = load_arxiv(args.data_path, device)
    base_mask = (node_year <= ARXIV_TRAIN_YEAR)
    feat_np = feat.cpu().numpy()
    data = Data(x=feat, edge_index=ei, y=lab_t)
    split_names = ['ID-test'] + [f'OOD-test_{i}({ty0}-{ty1})'
                                  for i, (ty0, ty1) in enumerate(ARXIV_TESTS)]
    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  [{args.backbone}] G-DUQ Arxiv Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(lab_np, base_mask,
                                              args.id_val_ratio, args.id_test_ratio, seed)
        model = _build_gant(feat.shape[1], nclass, feat_np)
        _train(model, data, tr_m, val_m, lab_np, nclass,
               os.path.join(args.save_dir, f'arxiv_{BTAG}_seed{seed}.pth'), seed)
        probs, u_all = model.infer(data, args.n_anchors)
        run_res = {}
        r_id = compute_split_metrics(probs[id_m], u_all[id_m], lab_np[id_m], nclass)
        print(f'  ID-test | acc={r_id["acc"]:.4f}')
        run_res['ID-test'] = r_id

        for i, (ty0, ty1) in enumerate(ARXIV_TESTS):
            te_mask = (node_year >= ty0) & (node_year <= ty1)
            r_ood = compute_split_metrics(probs[te_mask], u_all[te_mask], lab_np[te_mask], nclass)
            r_ood = add_cross_split_metrics(r_id, r_ood, r_id['_u'], r_ood['_u'])
            run_res[split_names[1+i]] = r_ood
        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, f'arxiv_{BTAG}_gduq')
    summarize(all_runs, split_names, all_keys, pref+'_results.csv',
              f'OGB-Arxiv — G-ΔUQ [{args.backbone}]',
              reliability_path=pref+'_reliability.csv',
              uncertainty_path=pref+'_uncertainty.csv')


def run_eerm():
    all_keys = build_all_keys(binary=False)
    print(f'\n[EERM-{args.eerm_dataset}] Loading...')
    (_, ei, feat_envs, lab_t, lab_np, nclass,
     tr_mask, val_mask, test_mask, ood_names, N) = load_eerm(args.eerm_root, args.eerm_dataset, device)
    feat_tr = feat_envs[0]; feat_oods = feat_envs[2:]
    tr_np = tr_mask.cpu().numpy(); val_np = val_mask.cpu().numpy(); te_np = test_mask.cpu().numpy()
    feat_np = feat_tr.cpu().numpy()
    split_names = ['ID-test'] + ood_names
    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  [{args.backbone}] G-DUQ EERM Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        data_tr = Data(x=feat_tr, edge_index=ei, y=lab_t).to(device)
        model = _build_gant(feat_tr.shape[1], nclass, feat_np)
        _train(model, data_tr, tr_np, val_np, lab_np, nclass,
               os.path.join(args.save_dir, f'eerm_{args.eerm_dataset}_{BTAG}_seed{seed}.pth'), seed)
        probs_id, u_id = model.infer(data_tr, args.n_anchors)
        run_res = {}
        r_id = compute_split_metrics(probs_id[te_np], u_id[te_np], lab_np[te_np], nclass)
        print(f'  ID-test | acc={r_id["acc"]:.4f}')
        run_res['ID-test'] = r_id

        for feat_ood, name in zip(feat_oods, ood_names):
            data_ood = Data(x=feat_ood, edge_index=ei, y=lab_t).to(device)
            probs_ood, u_ood = model.infer(data_ood, args.n_anchors)
            r_ood = compute_split_metrics(probs_ood[te_np], u_ood[te_np], lab_np[te_np], nclass)
            r_ood = add_cross_split_metrics(r_id, r_ood, r_id['_u'], r_ood['_u'])
            run_res[name] = r_ood
        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    ds = args.eerm_dataset
    pref = os.path.join(args.save_dir, f'{ds}_{BTAG}_gduq')
    summarize(all_runs, split_names, all_keys, pref+'_results.csv',
              f'EERM-{ds} — G-ΔUQ [{args.backbone}]',
              reliability_path=pref+'_reliability.csv',
              uncertainty_path=pref+'_uncertainty.csv')


if __name__ == '__main__':
    {'elliptic': run_elliptic, 'arxiv': run_arxiv, 'eerm': run_eerm}[args.dataset]()
