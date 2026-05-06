"""
S-BGCN-T-K (Vanilla GNN + entropy uncertainty)
===============================================
支持 --backbone GCN / GAT / GraphSAGE

用法:
    python experiments/run_ungnn.py --dataset elliptic --data_dir ./elliptic --runs 5
    python experiments/run_ungnn.py --dataset elliptic --data_dir ./elliptic --backbone GAT --runs 5
    python experiments/run_ungnn.py --dataset elliptic --data_dir ./elliptic --backbone GraphSAGE --runs 5
    python experiments/run_ungnn.py --dataset arxiv --data_path ./data.pkl --runs 5
    python experiments/run_ungnn.py --dataset eerm --eerm_dataset cora --eerm_root ./cora --runs 5
"""
import sys; sys.path.insert(0, 'src')

import os, time, argparse, copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from gnn_uq_bench.datasets import (
    load_elliptic, load_arxiv, load_eerm, stratified_split,
    ELLIPTIC_TRAIN, ELLIPTIC_TESTS, ARXIV_TRAIN_YEAR, ARXIV_TESTS,
)
from gnn_uq_bench.models import build_sparse_backbone
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
parser.add_argument('--id_val_ratio',  type=float, default=0.1)
parser.add_argument('--id_test_ratio', type=float, default=0.1)
parser.add_argument('--class_weight',  type=float, default=10.0)
parser.add_argument('--base_seed',     type=int,   default=42)
parser.add_argument('--save_dir',      type=str,   default='./results/ungnn')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BTAG = args.backbone.lower()
print(f'[设备] {device}  [backbone] {args.backbone}')


def _make_model(nfeat, nclass):
    return build_sparse_backbone(
        args.backbone, nfeat, args.hidden, nclass, args.dropout).to(device)


def _entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)


def _train(model, adj, feat, lab_np, tr_mask, val_mask, save_path,
           nclass, seed, class_weight=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model.reset_parameters()
    crit = (nn.CrossEntropyLoss(weight=torch.FloatTensor([1., class_weight]).to(device))
            if class_weight and nclass == 2 else nn.CrossEntropyLoss())
    opt   = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lab_t = torch.tensor(lab_np, dtype=torch.long).to(device)
    tr_t  = torch.tensor(tr_mask,  dtype=torch.bool).to(device)
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
    model.load_state_dict(bs); torch.save(bs, save_path)
    print(f'    best val={best:.4f}')
    return model


@torch.no_grad()
def _infer(model, feat, adj):
    model.eval()
    logits = model(feat, adj).cpu().numpy()
    ex = np.exp(logits - logits.max(1, keepdims=True))
    return ex / ex.sum(1, keepdims=True)


def run_elliptic():
    all_keys = build_all_keys(binary=True)
    print('\n[Elliptic] Loading...')
    adj_tr, _, feat_tr, _, lab_tr_np, N_tr = load_elliptic(ELLIPTIC_TRAIN, args.data_dir, device)
    tr_base = (lab_tr_np >= 0)

    test_graphs = []
    for i, steps in enumerate(ELLIPTIC_TESTS):
        adj_te, _, feat_te, _, lab_te_np, N_te = load_elliptic(steps, args.data_dir, device)
        te_mask = (lab_te_np >= 0)
        test_graphs.append((adj_te, feat_te, lab_te_np, te_mask, N_te))
        print(f'  OOD-test_{i}: labeled={te_mask.sum()}')

    split_names = ['ID-test'] + [
        f'OOD-test_{i}(s{ELLIPTIC_TESTS[i][0]}-{ELLIPTIC_TESTS[i][-1]})'
        for i in range(len(ELLIPTIC_TESTS))]
    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  [{args.backbone}] Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(lab_tr_np, tr_base,
                                              args.id_val_ratio, args.id_test_ratio, seed)
        model = _make_model(feat_tr.shape[1], 2)
        _train(model, adj_tr, feat_tr, lab_tr_np, tr_m, val_m,
               os.path.join(args.save_dir, f'elliptic_{BTAG}_seed{seed}.pth'),
               2, seed, args.class_weight)

        run_res = {}
        probs_all = _infer(model, feat_tr, adj_tr)
        u_id = _entropy(probs_all[id_m])
        r_id = compute_split_metrics(probs_all[id_m], u_id, lab_tr_np[id_m], 2, binary=True)
        print(f'  ID-test | acc={r_id["acc"]:.4f} f1={r_id["f1"]:.4f} '
              f'ece={r_id["ece"]:.4f} ue_auroc={r_id["ue_auroc"]:.4f}')
        run_res['ID-test'] = r_id

        for i, (adj_te, feat_te, lab_te_np, te_mask, N_te) in enumerate(test_graphs):
            rng = np.random.default_rng(seed + i + 100)
            te_idx = np.where(te_mask)[0]; perm = rng.permutation(len(te_idx))
            nh = max(1, len(te_idx) // 2)
            tr_te = np.zeros(N_te, bool); tr_te[te_idx[perm[:nh]]] = True
            vl_te = np.zeros(N_te, bool); vl_te[te_idx[perm[nh:]]] = True
            if vl_te.sum() == 0: vl_te = tr_te.copy()

            model_te = _make_model(feat_te.shape[1], 2)
            _train(model_te, adj_te, feat_te, lab_te_np, tr_te, vl_te,
                   os.path.join(args.save_dir, f'elliptic_{BTAG}_seed{seed}_te{i}.pth'),
                   2, seed + i + 100, args.class_weight)
            probs_te = _infer(model_te, feat_te, adj_te)
            u_ood    = _entropy(probs_te[te_mask])
            r_ood    = compute_split_metrics(probs_te[te_mask], u_ood,
                                             lab_te_np[te_mask], 2, binary=True)
            r_ood    = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            name     = split_names[1 + i]
            print(f'  {name[:32]} | ood_auroc={r_ood["ood_auroc"]:.4f}')
            run_res[name] = r_ood

        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, f'elliptic_{BTAG}_ungnn')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv',
              f'Elliptic — S-BGCN-T-K [{args.backbone}]',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


def run_arxiv():
    all_keys = build_all_keys(binary=False)
    print('\n[Arxiv] Loading...')
    adj_t, _, feat_t, _, lab_np, node_year, nclass, N = load_arxiv(args.data_path, device)
    base_mask   = (node_year <= ARXIV_TRAIN_YEAR)
    split_names = ['ID-test'] + [f'OOD-test_{i}({ty0}-{ty1})'
                                  for i, (ty0, ty1) in enumerate(ARXIV_TESTS)]
    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  [{args.backbone}] Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(lab_np, base_mask,
                                              args.id_val_ratio, args.id_test_ratio, seed)
        model = _make_model(feat_t.shape[1], nclass)
        _train(model, adj_t, feat_t, lab_np, tr_m, val_m,
               os.path.join(args.save_dir, f'arxiv_{BTAG}_seed{seed}.pth'), nclass, seed)

        probs_all = _infer(model, feat_t, adj_t)
        run_res = {}
        u_id = _entropy(probs_all[id_m])
        r_id = compute_split_metrics(probs_all[id_m], u_id, lab_np[id_m], nclass)
        print(f'  ID-test | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f}')
        run_res['ID-test'] = r_id

        for i, (ty0, ty1) in enumerate(ARXIV_TESTS):
            te_mask = (node_year >= ty0) & (node_year <= ty1)
            u_ood   = _entropy(probs_all[te_mask])
            r_ood   = compute_split_metrics(probs_all[te_mask], u_ood, lab_np[te_mask], nclass)
            r_ood   = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            run_res[split_names[1+i]] = r_ood
            print(f'  {split_names[1+i]} | ood_auroc={r_ood["ood_auroc"]:.4f}')

        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, f'arxiv_{BTAG}_ungnn')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv',
              f'OGB-Arxiv — S-BGCN-T-K [{args.backbone}]',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


def run_eerm():
    all_keys = build_all_keys(binary=False)
    print(f'\n[EERM-{args.eerm_dataset}] Loading...')
    (adj, _, feat_envs, lab_t, lab_np, nclass,
     tr_mask, val_mask, test_mask, ood_names, N) = load_eerm(args.eerm_root, args.eerm_dataset, device)

    feat_tr  = feat_envs[0]; feat_oods = feat_envs[2:]
    tr_np    = tr_mask.cpu().numpy(); val_np = val_mask.cpu().numpy()
    te_np    = test_mask.cpu().numpy()
    split_names = ['ID-test'] + ood_names
    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  [{args.backbone}] Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        model = _make_model(feat_tr.shape[1], nclass)
        _train(model, adj, feat_tr, lab_np, tr_np, val_np,
               os.path.join(args.save_dir, f'eerm_{args.eerm_dataset}_{BTAG}_seed{seed}.pth'),
               nclass, seed)

        run_res = {}
        probs_id = _infer(model, feat_tr, adj)
        u_id     = _entropy(probs_id[te_np])
        r_id     = compute_split_metrics(probs_id[te_np], u_id, lab_np[te_np], nclass)
        print(f'  ID-test | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f}')
        run_res['ID-test'] = r_id

        for feat_ood, name in zip(feat_oods, ood_names):
            probs_ood = _infer(model, feat_ood, adj)
            u_ood     = _entropy(probs_ood[te_np])
            r_ood     = compute_split_metrics(probs_ood[te_np], u_ood, lab_np[te_np], nclass)
            r_ood     = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            run_res[name] = r_ood
            print(f'  {name[:30]} | acc={r_ood["acc"]:.4f}')

        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    ds   = args.eerm_dataset
    pref = os.path.join(args.save_dir, f'{ds}_{BTAG}_ungnn')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv',
              f'EERM-{ds} — S-BGCN-T-K [{args.backbone}]',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


if __name__ == '__main__':
    {'elliptic': run_elliptic, 'arxiv': run_arxiv, 'eerm': run_eerm}[args.dataset]()
