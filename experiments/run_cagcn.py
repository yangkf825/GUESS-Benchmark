"""
CaGCN — 图卷积温度缩放校准
============================
用法:
    python experiments/run_cagcn.py --dataset elliptic --data_dir ./elliptic --runs 5
    python experiments/run_cagcn.py --dataset arxiv --data_path ./data.pkl --runs 5
    python experiments/run_cagcn.py --dataset eerm --eerm_dataset cora --eerm_root ./cora --runs 5
"""
import sys; sys.path.insert(0, 'src')

import os, time, argparse, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam

from gnn_uq_bench.datasets import (
    load_elliptic, load_arxiv, load_eerm, stratified_split,
    ELLIPTIC_TRAIN, ELLIPTIC_VAL, ELLIPTIC_TESTS, ARXIV_TRAIN_YEAR, ARXIV_TESTS, ARXIV_OODVAL_YEARS,
)
from gnn_uq_bench.models import GCNSparse, CaGCN
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
parser.add_argument('--runs',          type=int,   default=5)
parser.add_argument('--hidden',        type=int,   default=64)
parser.add_argument('--dropout',       type=float, default=0.5)
parser.add_argument('--lr',            type=float, default=0.01)
parser.add_argument('--lr_for_cal',    type=float, default=0.01)
parser.add_argument('--weight_decay',  type=float, default=5e-4)
parser.add_argument('--l2_for_cal',    type=float, default=5e-4)
parser.add_argument('--epochs',        type=int,   default=2000)
parser.add_argument('--epoch_for_st',  type=int,   default=200)
parser.add_argument('--patience',      type=int,   default=100)
parser.add_argument('--stage',         type=int,   default=1)
parser.add_argument('--threshold',     type=float, default=0.8)
parser.add_argument('--Lambda',        type=float, default=0.5)
parser.add_argument('--id_val_ratio',  type=float, default=0.1)
parser.add_argument('--id_test_ratio', type=float, default=0.1)
parser.add_argument('--class_weight',  type=float, default=10.0)
parser.add_argument('--base_seed',     type=int,   default=42)
parser.add_argument('--save_dir',      type=str,   default='./results/cagcn')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[设备] {device}')


def _intra_loss(output, labels):
    p = F.softmax(output, dim=1); pred = p.max(1)[1]
    ok  = (pred == labels).nonzero(as_tuple=True)[0]
    bad = (pred != labels).nonzero(as_tuple=True)[0]
    s = p.sort(1, descending=True)[0]; t1, t2 = s[:, 0], s[:, 1]
    return ((1 - t1[ok] + t2[ok]).sum() + (t1[bad] - t2[bad]).sum()) / labels.size(0)


def _train_base(model, adj, feat, lab_t, tr_mask, val_mask, save_path, crit,
                dyn_tr, pseudo_lab, seed):
    torch.manual_seed(seed); model.reset_parameters()
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, bad = 1e9, 0
    for _ in range(args.epochs):
        model.train(); opt.zero_grad()
        crit(model(feat, adj)[dyn_tr], pseudo_lab[dyn_tr]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lv = crit(model(feat, adj)[val_mask], lab_t[val_mask]).item()
        if lv < best: torch.save(model.state_dict(), save_path); best, bad = lv, 0
        else: bad += 1
        if bad == args.patience: break
    print(f'    base val={best:.4f}')


def _train_cagcn(adj_ov, feat_ov, lab_ov, ov_mask, nclass, base_path, save_path, crit, epochs, seed):
    torch.manual_seed(seed)
    base = GCNSparse(feat_ov.shape[1], args.hidden, nclass, args.dropout)
    base.load_state_dict(torch.load(base_path, map_location=device))
    model = CaGCN(nclass, base).to(device)
    opt = Adam(filter(lambda p: p.requires_grad, model.parameters()),
               lr=args.lr_for_cal, weight_decay=args.l2_for_cal)
    best, bad = 1e9, 0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(feat_ov, adj_ov)
        (crit(out[ov_mask], lab_ov[ov_mask]) +
         args.Lambda * _intra_loss(out[ov_mask], lab_ov[ov_mask])).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            lv = crit(model(feat_ov, adj_ov)[ov_mask], lab_ov[ov_mask]).item()
        if lv < best: torch.save(model.state_dict(), save_path); best, bad = lv, 0
        else: bad += 1
        if bad == args.patience: break
    print(f'    CaGCN val={best:.4f}')


def _gen_pseudo(adj, feat, lab, nclass, base_path, cagcn_path, dyn_tr, pseudo_lab, seed):
    torch.manual_seed(seed)
    base = GCNSparse(feat.shape[1], args.hidden, nclass, args.dropout)
    base.load_state_dict(torch.load(base_path, map_location=device))
    m = CaGCN(nclass, base).to(device)
    m.load_state_dict(torch.load(cagcn_path, map_location=device)); m.eval()
    with torch.no_grad():
        conf, pred = F.softmax(m(feat, adj), 1).max(1)
    in_tr = set(dyn_tr.nonzero(as_tuple=True)[0].tolist())
    added = 0
    for i in (conf > args.threshold).nonzero(as_tuple=True)[0].tolist():
        if i not in in_tr:
            pseudo_lab[i] = pred[i]; dyn_tr[i] = True; added += 1
    print(f'    伪标签+{added}, train={dyn_tr.sum().item()}')
    return dyn_tr, pseudo_lab


@torch.no_grad()
def _infer_cagcn(adj, feat, nclass, base_path, cagcn_path):
    base = GCNSparse(feat.shape[1], args.hidden, nclass, args.dropout)
    base.load_state_dict(torch.load(base_path, map_location=device))
    m = CaGCN(nclass, base).to(device); m.eval()
    logits = m(feat, adj).cpu().numpy()
    ex = np.exp(logits - logits.max(1, keepdims=True))
    return ex / ex.sum(1, keepdims=True)


def _run_one(seed, adj, feat, lab_t, lab_np, tr_mask, val_mask,
             adj_ov, feat_ov, lab_ov, ov_mask, nclass, crit, prefix):
    tr_np  = tr_mask.cpu().numpy(); val_np = val_mask.cpu().numpy()
    dyn_tr     = tr_mask.clone()
    pseudo_lab = lab_t.clone()
    base_path  = os.path.join(args.save_dir, f'{prefix}_seed{seed}_base.pth')
    cagcn_path = os.path.join(args.save_dir, f'{prefix}_seed{seed}_cagcn.pth')

    model = GCNSparse(feat.shape[1], args.hidden, nclass, args.dropout).to(device)
    for stage in range(1, args.stage + 1):
        last = (stage == args.stage)
        print(f'  [Stage {stage}/{args.stage}]')
        _train_base(model, adj, feat, lab_t, dyn_tr, val_mask, base_path, crit, dyn_tr, pseudo_lab, seed)
        cal_ep = args.epochs if last else args.epoch_for_st
        _train_cagcn(adj_ov, feat_ov, lab_ov, ov_mask, nclass, base_path, cagcn_path, crit, cal_ep, seed)
        if not last:
            dyn_tr, pseudo_lab = _gen_pseudo(adj, feat, lab_t, nclass,
                                              base_path, cagcn_path, dyn_tr, pseudo_lab, seed)
    return base_path, cagcn_path


def run_elliptic():
    all_keys = build_all_keys(binary=True)
    print('\n[Elliptic] Loading...')
    adj_tr, _, feat_tr, lab_tr, lab_tr_np, N_tr = load_elliptic(ELLIPTIC_TRAIN, args.data_dir, device)
    adj_ov, _, feat_ov, lab_ov, lab_ov_np, N_ov = load_elliptic(ELLIPTIC_VAL,   args.data_dir, device)
    ov_mask = (lab_ov_np >= 0)
    ov_mask_t = torch.tensor(ov_mask, dtype=torch.bool).to(device)
    tr_base = (lab_tr_np >= 0)

    crit = (nn.CrossEntropyLoss(weight=torch.FloatTensor([1., args.class_weight]).to(device))
            if args.class_weight else nn.CrossEntropyLoss())

    test_graphs = []
    for i, steps in enumerate(ELLIPTIC_TESTS):
        a, _, f, lb, lb_np, N = load_elliptic(steps, args.data_dir, device)
        tm = (lb_np >= 0)
        a_ov, _, f_ov, lb_ov, lb_ov_np2, _ = load_elliptic(steps, args.data_dir, device)
        test_graphs.append((a, f, lb_np, tm, N, a_ov, f_ov, lb_ov, (lb_ov_np2 >= 0)))

    split_names = ['ID-test'] + [
        f'OOD-test_{i}(s{ELLIPTIC_TESTS[i][0]}-{ELLIPTIC_TESTS[i][-1]})'
        for i in range(len(ELLIPTIC_TESTS))]
    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  CaGCN Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(lab_tr_np, tr_base,
                                              args.id_val_ratio, args.id_test_ratio, seed)
        tr_mask_t  = torch.tensor(tr_m,  dtype=torch.bool).to(device)
        val_mask_t = torch.tensor(val_m, dtype=torch.bool).to(device)

        bp, cp = _run_one(seed, adj_tr, feat_tr, lab_tr, lab_tr_np,
                          tr_mask_t, val_mask_t, adj_ov, feat_ov, lab_ov, ov_mask_t,
                          2, crit, 'elliptic')
        run_res = {}
        probs = _infer_cagcn(adj_tr, feat_tr, 2, bp, cp)
        u_id  = 1. - probs[id_m].max(1)
        r_id  = compute_split_metrics(probs[id_m], u_id, lab_tr_np[id_m], 2, binary=True)
        print(f'  ID-test | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f}')
        run_res['ID-test'] = r_id

        for i, (a_te, f_te, lb_np, tm, N_te, a_ov2, f_ov2, lb_ov2, ovm2) in enumerate(test_graphs):
            ovm_t = torch.tensor(ovm2, dtype=torch.bool).to(device)
            bp2, cp2 = _run_one(seed + i + 100, a_te, f_te,
                                 torch.tensor(lb_np, dtype=torch.long).to(device),
                                 lb_np, torch.tensor(tm, dtype=torch.bool).to(device),
                                 torch.tensor(tm, dtype=torch.bool).to(device),
                                 a_ov2, f_ov2, lb_ov2, ovm_t, 2, crit,
                                 f'elliptic_te{i}')
            probs_te = _infer_cagcn(a_te, f_te, 2, bp2, cp2)
            u_ood    = 1. - probs_te[tm].max(1)
            r_ood    = compute_split_metrics(probs_te[tm], u_ood, lb_np[tm], 2, binary=True)
            r_ood    = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            run_res[split_names[1+i]] = r_ood
            print(f'  {split_names[1+i][:30]} | ood_auroc={r_ood["ood_auroc"]:.4f}')

        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, 'elliptic_cagcn')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv',
              'Elliptic — CaGCN',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


def run_arxiv():
    all_keys = build_all_keys(binary=False)
    print('\n[Arxiv] Loading...')
    adj, _, feat, _, lab_np, node_year, nclass, N = load_arxiv(args.data_path, device)
    lab_t = torch.tensor(lab_np, dtype=torch.long).to(device)
    base_mask = (node_year <= ARXIV_TRAIN_YEAR)
    oy0, oy1  = ARXIV_OODVAL_YEARS
    ov_mask   = (node_year >= oy0) & (node_year <= oy1)
    ov_mask_t = torch.tensor(ov_mask, dtype=torch.bool).to(device)
    crit = nn.CrossEntropyLoss()
    split_names = ['ID-test'] + [f'OOD-test_{i}({ty0}-{ty1})'
                                  for i, (ty0, ty1) in enumerate(ARXIV_TESTS)]
    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  CaGCN Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(lab_np, base_mask,
                                              args.id_val_ratio, args.id_test_ratio, seed)
        tr_t  = torch.tensor(tr_m, dtype=torch.bool).to(device)
        val_t = torch.tensor(val_m, dtype=torch.bool).to(device)

        bp, cp = _run_one(seed, adj, feat, lab_t, lab_np, tr_t, val_t,
                          adj, feat, lab_t, ov_mask_t, nclass, crit, 'arxiv')
        run_res = {}
        probs = _infer_cagcn(adj, feat, nclass, bp, cp)
        u_id  = 1. - probs[id_m].max(1)
        r_id  = compute_split_metrics(probs[id_m], u_id, lab_np[id_m], nclass)
        print(f'  ID-test | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f}')
        run_res['ID-test'] = r_id

        for i, (ty0, ty1) in enumerate(ARXIV_TESTS):
            te_mask = (node_year >= ty0) & (node_year <= ty1)
            u_ood   = 1. - probs[te_mask].max(1)
            r_ood   = compute_split_metrics(probs[te_mask], u_ood, lab_np[te_mask], nclass)
            r_ood   = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            run_res[split_names[1+i]] = r_ood
        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, 'arxiv_cagcn')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv', 'OGB-Arxiv — CaGCN',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


def run_eerm():
    all_keys = build_all_keys(binary=False)
    print(f'\n[EERM-{args.eerm_dataset}] Loading...')
    (adj, _, feat_envs, lab_t, lab_np, nclass,
     tr_mask, val_mask, test_mask, ood_names, N) = load_eerm(args.eerm_root, args.eerm_dataset, device)
    feat_tr = feat_envs[0]; feat_oods = feat_envs[2:]
    crit = nn.CrossEntropyLoss()
    split_names = ['ID-test'] + ood_names
    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  CaGCN EERM Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        bp, cp = _run_one(seed, adj, feat_tr, lab_t, lab_np,
                          tr_mask, val_mask, adj, feat_tr, lab_t, val_mask,
                          nclass, crit, f'eerm_{args.eerm_dataset}')
        te_np = test_mask.cpu().numpy()
        run_res = {}
        probs_id = _infer_cagcn(adj, feat_tr, nclass, bp, cp)
        u_id  = 1. - probs_id[te_np].max(1)
        r_id  = compute_split_metrics(probs_id[te_np], u_id, lab_np[te_np], nclass)
        print(f'  ID-test | acc={r_id["acc"]:.4f}')
        run_res['ID-test'] = r_id

        for feat_ood, name in zip(feat_oods, ood_names):
            p_ood = _infer_cagcn(adj, feat_ood, nclass, bp, cp)
            u_ood = 1. - p_ood[te_np].max(1)
            r_ood = compute_split_metrics(p_ood[te_np], u_ood, lab_np[te_np], nclass)
            r_ood = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            run_res[name] = r_ood
        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    ds = args.eerm_dataset
    pref = os.path.join(args.save_dir, f'{ds}_cagcn')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv', f'EERM-{ds} — CaGCN',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


if __name__ == '__main__':
    {'elliptic': run_elliptic, 'arxiv': run_arxiv, 'eerm': run_eerm}[args.dataset]()
