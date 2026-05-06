"""
CalGNN — GCN/GAT/GraphSAGE + 6种后处理校准
============================================
支持 --backbone GCN / GAT / GraphSAGE

用法:
    python experiments/run_calgnn.py --dataset elliptic --data_dir ./elliptic --runs 5
    python experiments/run_calgnn.py --dataset elliptic --data_dir ./elliptic --backbone GAT --runs 5
    python experiments/run_calgnn.py --dataset elliptic --data_dir ./elliptic --backbone GraphSAGE --runs 5
    python experiments/run_calgnn.py --dataset arxiv --data_path ./data.pkl --runs 5
    python experiments/run_calgnn.py --dataset eerm --eerm_dataset cora --eerm_root ./cora --runs 5
"""
import sys; sys.path.insert(0, 'src')

import os, time, argparse, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import scipy.special
from torch.optim import Adam

from gnn_uq_bench.datasets import (
    load_elliptic, load_arxiv, load_eerm, stratified_split,
    ELLIPTIC_TRAIN, ELLIPTIC_VAL, ELLIPTIC_TESTS,
    ARXIV_TRAIN_YEAR, ARXIV_TESTS, ARXIV_OODVAL_YEARS,
)
from gnn_uq_bench.models import build_sparse_backbone
from gnn_uq_bench.calibration import (
    TemperatureScaling, HistogramBinning, IsotonicCalib, BBQ,
    MetaCalMisCoverage, compute_sm_conf, rbs_fit, apply_rbs,
)
from gnn_uq_bench.metrics import (
    compute_split_metrics, add_cross_split_metrics, build_all_keys, summarize_calgnn,
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
parser.add_argument('--add_cal_loss',  action='store_true', default=False)
parser.add_argument('--alpha',         type=float, default=0.5)
parser.add_argument('--lmbda',         type=float, default=0.1)
parser.add_argument('--num_bins_rbs',  type=int,   default=10)
parser.add_argument('--id_val_ratio',  type=float, default=0.1)
parser.add_argument('--id_test_ratio', type=float, default=0.1)
parser.add_argument('--class_weight',  type=float, default=10.0)
parser.add_argument('--base_seed',     type=int,   default=42)
parser.add_argument('--save_dir',      type=str,   default='./results/calgnn')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BTAG = args.backbone.lower()
print(f'[设备] {device}  [backbone] {args.backbone}')

CAL_METHODS     = ['Uncal', 'TS', 'HB', 'Iso', 'BBQ', 'MetaCal', 'RBS']
CAL_METHODS_OUT = ['Uncal', 'RBS']


def _make_model(nfeat, nclass):
    return build_sparse_backbone(
        args.backbone, nfeat, args.hidden, nclass, args.dropout).to(device)


def _cal_loss(y_true, logits, lmbda, epoch, epochs, bin_num=15):
    probs  = F.softmax(logits, dim=1)
    pred   = probs.max(1)[1]; conf = probs.max(1)[0]
    bsize  = torch.tensor(1.0 / bin_num)
    bounds = torch.arange(bsize, 1 + bsize, bsize)
    acc_vec = torch.zeros(len(y_true), device=logits.device)
    for b in bounds:
        lo = b - bsize; m = (conf > lo) & (conf <= b)
        if m.sum() > 0:
            acc_vec[m] = (pred[m] == y_true[m]).float().mean()
    cal = -(acc_vec * torch.log(conf.clamp(min=1e-8))).sum()
    anneal = torch.min(torch.tensor(lmbda), torch.tensor(lmbda * (epoch + 1) / epochs))
    return cal * anneal


def _train(nfeat, nclass, adj_or_ei, feat, lab_np, lab_t, tr_mask, val_mask,
           save_path, crit, seed, epoch_override=None):
    torch.manual_seed(seed)
    model = _make_model(nfeat, nclass)
    model.reset_parameters()
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lab_t2 = torch.tensor(lab_np, dtype=torch.long).to(device) if lab_t is None else lab_t
    tr_t  = torch.tensor(tr_mask, dtype=torch.bool).to(device)
    val_t = torch.tensor(val_mask, dtype=torch.bool).to(device)
    best, bad, bs = 1e9, 0, None
    ep = epoch_override or args.epochs
    for epoch in range(ep):
        model.train(); opt.zero_grad()
        logits = model(feat, adj_or_ei)
        loss   = crit(logits[tr_t], lab_t2[tr_t])
        if args.add_cal_loss:
            loss = args.alpha * loss + (1 - args.alpha) * _cal_loss(
                lab_t2[tr_t], logits[tr_t], args.lmbda, epoch, ep)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lv = crit(model(feat, adj_or_ei)[val_t], lab_t2[val_t]).item()
        if lv < best: bs = copy.deepcopy(model.state_dict()); best, bad = lv, 0
        else: bad += 1
        if bad >= args.patience: break
    model.load_state_dict(bs); torch.save(bs, save_path)
    print(f'    base val={best:.4f}')
    return model


@torch.no_grad()
def _all_logits_probs(model, feat, adj_or_ei):
    model.eval()
    logits = model(feat, adj_or_ei).cpu().numpy()
    return logits, scipy.special.softmax(logits, axis=1)


def _fit_calibrators(ov_logits, ov_probs, ov_labels, ei_ov, N_ov, probs_all_t):
    ts = TemperatureScaling().fit(ov_logits, ov_labels)
    hb = HistogramBinning().fit(ov_probs, ov_labels)
    iso= IsotonicCalib().fit(ov_probs, ov_labels)
    bbq= BBQ().fit(ov_probs, ov_labels)
    mc = MetaCalMisCoverage().fit(ov_logits, ov_labels)
    sm = compute_sm_conf(ei_ov, N_ov, probs_all_t, device)
    T_rbs, bins_rbs = rbs_fit(sm, ov_logits, ov_labels, args.num_bins_rbs)
    return ts, hb, iso, bbq, mc, T_rbs, bins_rbs, sm


def _eval_split(logits, probs, labels, nclass, binary,
                ts, hb, iso, bbq, mc, T_rbs, bins_rbs, sm_conf_split):
    all_p = {
        'Uncal':   probs,
        'TS':      ts.predict_proba(logits),
        'HB':      hb.predict_proba(probs),
        'Iso':     iso.predict_proba(probs),
        'BBQ':     bbq.predict_proba(probs),
        'MetaCal': mc.predict(logits),
        'RBS':     apply_rbs(T_rbs, bins_rbs, sm_conf_split, logits, device),
    }
    res = {}
    for cm, p in all_p.items():
        u_cm = 1. - p.max(1)
        res[cm] = compute_split_metrics(p, u_cm, labels, nclass, binary)
    return res


def run_elliptic():
    all_keys = build_all_keys(binary=True)
    print('\n[Elliptic] Loading...')
    adj_tr, ei_tr, feat_tr, _, lab_tr_np, N_tr = load_elliptic(ELLIPTIC_TRAIN, args.data_dir, device)
    adj_ov, ei_ov, feat_ov, _, lab_ov_np, N_ov = load_elliptic(ELLIPTIC_VAL,   args.data_dir, device)
    tr_base = (lab_tr_np >= 0); ov_mask = (lab_ov_np >= 0)
    crit = (nn.CrossEntropyLoss(weight=torch.FloatTensor([1., args.class_weight]).to(device))
            if args.class_weight else nn.CrossEntropyLoss())

    # GATSparse/SAGESparse 内部自动转 edge_index，外部统一传 adj
    test_graphs = []
    for i, steps in enumerate(ELLIPTIC_TESTS):
        a, ei, f, _, lnp, N = load_elliptic(steps, args.data_dir, device)
        test_graphs.append((a, ei, f, lnp, (lnp >= 0), N))

    split_names = ['ID-test'] + [
        f'OOD-test_{i}(s{ELLIPTIC_TESTS[i][0]}-{ELLIPTIC_TESTS[i][-1]})'
        for i in range(len(ELLIPTIC_TESTS))]
    all_runs_per_method = {m: [] for m in CAL_METHODS_OUT}

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  [{args.backbone}] CalGNN Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(lab_tr_np, tr_base,
                                              args.id_val_ratio, args.id_test_ratio, seed)
        model = _train(feat_tr.shape[1], 2, adj_tr, feat_tr, lab_tr_np, None,
                       tr_m, val_m,
                       os.path.join(args.save_dir, f'elliptic_{BTAG}_seed{seed}.pth'), crit, seed)

        ov_logits, ov_probs = _all_logits_probs(model, feat_ov, adj_ov)
        ov_logits = ov_logits[ov_mask]; ov_probs = ov_probs[ov_mask]
        ov_labels = lab_ov_np[ov_mask]

        probs_all_t = F.softmax(model(feat_ov, adj_ov), dim=1)
        ts, hb, iso, bbq, mc, T_rbs, bins_rbs, _ = _fit_calibrators(
            ov_logits, ov_probs, ov_labels, ei_ov, N_ov, probs_all_t)

        id_logits, id_probs = _all_logits_probs(model, feat_tr, adj_tr)
        probs_all_tr_t = F.softmax(model(feat_tr, adj_tr), dim=1)
        sm_tr = compute_sm_conf(ei_tr, N_tr, probs_all_tr_t, device)
        id_res = _eval_split(id_logits[id_m], id_probs[id_m], lab_tr_np[id_m],
                              2, True, ts, hb, iso, bbq, mc, T_rbs, bins_rbs, sm_tr[id_m])
        for cm in CAL_METHODS_OUT:
            print(f'  ID-test [{cm}] acc={id_res[cm]["acc"]:.4f} ece={id_res[cm]["ece"]:.4f}')

        run_per_m = {m: {'ID-test': id_res[m]} for m in CAL_METHODS_OUT}

        for i, (a_te, ei_te, f_te, lnp, tm, N_te) in enumerate(test_graphs):
            rng = np.random.default_rng(seed + i + 100)
            te_idx = np.where(tm)[0]; perm = rng.permutation(len(te_idx))
            nh = max(1, len(te_idx)//2)
            tr_te = np.zeros(N_te, bool); tr_te[te_idx[perm[:nh]]] = True
            vl_te = np.zeros(N_te, bool); vl_te[te_idx[perm[nh:]]] = True
            if vl_te.sum() == 0: vl_te = tr_te.copy()

            model_te = _train(f_te.shape[1], 2, a_te, f_te, lnp, None,
                               tr_te, vl_te,
                               os.path.join(args.save_dir, f'elliptic_{BTAG}_seed{seed}_te{i}.pth'),
                               crit, seed + i + 100)

            te_logits, te_probs = _all_logits_probs(model_te, f_te, a_te)
            pall_t = F.softmax(model_te(f_te, a_te), dim=1)
            sm_te  = compute_sm_conf(ei_te, N_te, pall_t, device)

            ov_l2 = te_logits[vl_te]; ov_p2 = te_probs[vl_te]; ov_lab2 = lnp[vl_te]
            pov_t  = F.softmax(model_te(f_te, a_te), dim=1)
            ts2, hb2, iso2, bbq2, mc2, T2, bins2, _ = _fit_calibrators(
                ov_l2, ov_p2, ov_lab2, ei_te, N_te, pov_t)

            te_res = _eval_split(te_logits[tm], te_probs[tm], lnp[tm],
                                  2, True, ts2, hb2, iso2, bbq2, mc2, T2, bins2, sm_te[tm])
            name = split_names[1+i]
            for cm in CAL_METHODS_OUT:
                r_ood = add_cross_split_metrics(
                    run_per_m[cm]['ID-test'], te_res[cm],
                    run_per_m[cm]['ID-test']['_u'], te_res[cm]['_u'])
                run_per_m[cm][name] = r_ood

        for m in CAL_METHODS_OUT:
            all_runs_per_method[m].append(run_per_m[m])
        print(f'  elapsed {time.time()-t0:.1f}s')

    for cm in CAL_METHODS_OUT:
        pref = os.path.join(args.save_dir, f'elliptic_{BTAG}_calgnn_{cm}')
        summarize_calgnn(all_runs_per_method[cm], split_names, all_keys,
                         pref + '_results.csv', f'Elliptic — CalGNN [{args.backbone}][{cm}]',
                         [cm],
                         reliability_path=pref + '_reliability.csv',
                         uncertainty_path=pref + '_uncertainty.csv')


def run_arxiv():
    all_keys = build_all_keys(binary=False)
    print('\n[Arxiv] Loading...')
    adj, ei, feat, _, lab_np, node_year, nclass, N = load_arxiv(args.data_path, device)
    lab_t = torch.tensor(lab_np, dtype=torch.long).to(device)
    base_mask = (node_year <= ARXIV_TRAIN_YEAR)
    oy0, oy1  = ARXIV_OODVAL_YEARS
    ov_mask   = (node_year >= oy0) & (node_year <= oy1)
    crit = nn.CrossEntropyLoss()
    split_names = ['ID-test'] + [f'OOD-test_{i}({ty0}-{ty1})'
                                  for i, (ty0, ty1) in enumerate(ARXIV_TESTS)]
    all_runs_pm = {m: [] for m in CAL_METHODS_OUT}

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  [{args.backbone}] CalGNN Arxiv Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(lab_np, base_mask,
                                              args.id_val_ratio, args.id_test_ratio, seed)
        model = _train(feat.shape[1], nclass, adj, feat, lab_np, lab_t,
                       tr_m, val_m,
                       os.path.join(args.save_dir, f'arxiv_{BTAG}_seed{seed}.pth'),
                       crit, seed)

        logits_all, probs_all = _all_logits_probs(model, feat, adj)
        pall_t  = F.softmax(model(feat, adj), dim=1)
        sm_all  = compute_sm_conf(ei, N, pall_t, device)

        ov_l = logits_all[ov_mask]; ov_p = probs_all[ov_mask]; ov_lab = lab_np[ov_mask]
        ts,hb,iso,bbq,mc,T_rbs,bins_rbs,_ = _fit_calibrators(ov_l, ov_p, ov_lab, ei, N, pall_t)

        id_res = _eval_split(logits_all[id_m], probs_all[id_m], lab_np[id_m],
                              nclass, False, ts,hb,iso,bbq,mc,T_rbs,bins_rbs, sm_all[id_m])
        for cm in CAL_METHODS_OUT:
            print(f'  ID-test [{cm}] acc={id_res[cm]["acc"]:.4f} ece={id_res[cm]["ece"]:.4f}')

        run_per_m = {m: {'ID-test': id_res[m]} for m in CAL_METHODS_OUT}

        for i, (ty0, ty1) in enumerate(ARXIV_TESTS):
            te_mask = (node_year >= ty0) & (node_year <= ty1)
            te_res  = _eval_split(logits_all[te_mask], probs_all[te_mask], lab_np[te_mask],
                                   nclass, False, ts,hb,iso,bbq,mc,T_rbs,bins_rbs, sm_all[te_mask])
            name = split_names[1+i]
            for cm in CAL_METHODS_OUT:
                r_ood = add_cross_split_metrics(
                    run_per_m[cm]['ID-test'], te_res[cm],
                    run_per_m[cm]['ID-test']['_u'], te_res[cm]['_u'])
                run_per_m[cm][name] = r_ood

        for m in CAL_METHODS_OUT: all_runs_pm[m].append(run_per_m[m])
        print(f'  elapsed {time.time()-t0:.1f}s')

    for cm in CAL_METHODS_OUT:
        pref = os.path.join(args.save_dir, f'arxiv_{BTAG}_calgnn_{cm}')
        summarize_calgnn(all_runs_pm[cm], split_names, all_keys,
                         pref+'_results.csv', f'OGB-Arxiv — CalGNN [{args.backbone}][{cm}]', [cm],
                         reliability_path=pref+'_reliability.csv',
                         uncertainty_path=pref+'_uncertainty.csv')


def run_eerm():
    all_keys = build_all_keys(binary=False)
    print(f'\n[EERM-{args.eerm_dataset}] Loading...')
    (adj, ei, feat_envs, lab_t, lab_np, nclass,
     tr_mask, val_mask, test_mask, ood_names, N) = load_eerm(args.eerm_root, args.eerm_dataset, device)
    feat_tr = feat_envs[0]; feat_oods = feat_envs[2:]
    tr_np = tr_mask.cpu().numpy(); val_np = val_mask.cpu().numpy(); te_np = test_mask.cpu().numpy()
    crit = nn.CrossEntropyLoss()
    split_names = ['ID-test'] + ood_names
    all_runs_pm = {m: [] for m in CAL_METHODS_OUT}

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  [{args.backbone}] CalGNN EERM Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        model = _train(feat_tr.shape[1], nclass, adj, feat_tr, lab_np, lab_t,
                       tr_np, val_np,
                       os.path.join(args.save_dir, f'eerm_{args.eerm_dataset}_{BTAG}_seed{seed}.pth'),
                       crit, seed)

        pall_t = F.softmax(model(feat_tr, adj), dim=1)
        logits_all, probs_all = _all_logits_probs(model, feat_tr, adj)
        sm_all = compute_sm_conf(ei, N, pall_t, device)

        ov_l = logits_all[val_np]; ov_p = probs_all[val_np]; ov_lab = lab_np[val_np]
        ts,hb,iso,bbq,mc,T_rbs,bins_rbs,_ = _fit_calibrators(ov_l, ov_p, ov_lab, ei, N, pall_t)

        id_res = _eval_split(logits_all[te_np], probs_all[te_np], lab_np[te_np],
                              nclass, False, ts,hb,iso,bbq,mc,T_rbs,bins_rbs, sm_all[te_np])
        run_per_m = {m: {'ID-test': id_res[m]} for m in CAL_METHODS_OUT}

        for feat_ood, name in zip(feat_oods, ood_names):
            pood_t = F.softmax(model(feat_ood, adj), dim=1)
            log_ood, p_ood = _all_logits_probs(model, feat_ood, adj)
            sm_ood = compute_sm_conf(ei, N, pood_t, device)
            te_res = _eval_split(log_ood[te_np], p_ood[te_np], lab_np[te_np],
                                  nclass, False, ts,hb,iso,bbq,mc,T_rbs,bins_rbs, sm_ood[te_np])
            for cm in CAL_METHODS_OUT:
                r_ood = add_cross_split_metrics(
                    run_per_m[cm]['ID-test'], te_res[cm],
                    run_per_m[cm]['ID-test']['_u'], te_res[cm]['_u'])
                run_per_m[cm][name] = r_ood

        for m in CAL_METHODS_OUT: all_runs_pm[m].append(run_per_m[m])
        print(f'  elapsed {time.time()-t0:.1f}s')

    ds = args.eerm_dataset
    for cm in CAL_METHODS_OUT:
        pref = os.path.join(args.save_dir, f'{ds}_{BTAG}_calgnn_{cm}')
        summarize_calgnn(all_runs_pm[cm], split_names, all_keys,
                         pref+'_results.csv', f'EERM-{ds} — CalGNN [{args.backbone}][{cm}]', [cm],
                         reliability_path=pref+'_reliability.csv',
                         uncertainty_path=pref+'_uncertainty.csv')


if __name__ == '__main__':
    {'elliptic': run_elliptic, 'arxiv': run_arxiv, 'eerm': run_eerm}[args.dataset]()
