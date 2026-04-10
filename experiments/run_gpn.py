"""
GPN — Graph Posterior Network
==============================
不确定性 u = 1 / alpha.sum(-1)（总 evidence 倒数，越低越不确定）

用法:
    python experiments/run_gpn.py --dataset elliptic --data_dir ./elliptic --runs 5
    python experiments/run_gpn.py --dataset arxiv --data_path ./data.pkl --runs 5
    python experiments/run_gpn.py --dataset eerm --eerm_dataset cora --eerm_root ./cora --runs 5
"""
import sys; sys.path.insert(0, 'src')

import os, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from gnn_uq_bench.datasets import (
    load_elliptic, load_arxiv, load_eerm, stratified_split,
    ELLIPTIC_TRAIN, ELLIPTIC_VAL, ELLIPTIC_TESTS,
    ARXIV_TRAIN_YEAR, ARXIV_TESTS, ARXIV_OODVAL_YEARS,
)
from gnn_uq_bench.models import (
    GPNModel, gpn_uce_loss, gpn_entropy_reg, gpn_ce_loss, sym_norm_edge,
)
from gnn_uq_bench.metrics import (
    compute_split_metrics, add_cross_split_metrics, build_all_keys, summarize,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',              type=str,   default='elliptic',
                    choices=['elliptic', 'arxiv', 'eerm'])
parser.add_argument('--data_dir',             type=str,   default='./elliptic')
parser.add_argument('--data_path',            type=str,   default='./data.pkl')
parser.add_argument('--eerm_dataset',         type=str,   default='cora', choices=['cora', 'amazon'])
parser.add_argument('--eerm_root',            type=str,   default=None)
parser.add_argument('--runs',                 type=int,   default=5)
parser.add_argument('--dim_hidden',           type=int,   default=64)
parser.add_argument('--dim_latent',           type=int,   default=10)
parser.add_argument('--radial_layers',        type=int,   default=10)
parser.add_argument('--K',                    type=int,   default=10)
parser.add_argument('--alpha_teleport',       type=float, default=0.2)
parser.add_argument('--dropout',              type=float, default=0.5)
parser.add_argument('--entropy_reg',          type=float, default=1e-5)
parser.add_argument('--alpha_evidence_scale', type=str,   default='latent-new-plus-classes')
parser.add_argument('--lr',                   type=float, default=0.01)
parser.add_argument('--flow_lr',              type=float, default=0.01)
parser.add_argument('--weight_decay',         type=float, default=5e-4)
parser.add_argument('--flow_weight_decay',    type=float, default=0.0)
parser.add_argument('--epochs',               type=int,   default=10000)
parser.add_argument('--warmup_epochs',        type=int,   default=5)
parser.add_argument('--patience',             type=int,   default=50)
parser.add_argument('--id_val_ratio',         type=float, default=0.1)
parser.add_argument('--id_test_ratio',        type=float, default=0.1)
parser.add_argument('--base_seed',            type=int,   default=42)
parser.add_argument('--save_dir',             type=str,   default='./results/gpn')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[设备] {device}')


def _make_data(feat, ei, labels_t, N):
    return Data(x=feat, edge_index=ei, y=labels_t).to(device)


def _train_gpn(data, tr_mask_np, val_mask_np, nfeat, nclass, N, save_path, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    data = data.to(device)
    ei, ew = sym_norm_edge(data.edge_index, N)
    ei = ei.to(device); ew = ew.to(device)
    tr_t  = torch.tensor(tr_mask_np,  dtype=torch.bool).to(device)
    val_t = torch.tensor(val_mask_np, dtype=torch.bool).to(device)

    model = GPNModel(
        dim_features=nfeat, num_classes=nclass,
        dim_hidden=args.dim_hidden, dim_latent=args.dim_latent,
        radial_layers=args.radial_layers, K=args.K,
        alpha_teleport=args.alpha_teleport, dropout_prob=args.dropout,
        alpha_evidence_scale=args.alpha_evidence_scale).to(device)

    opt, flow_opt = model.get_optimizer(
        args.lr, args.weight_decay, args.flow_lr, args.flow_weight_decay)

    best, bad, bs = float('inf'), 0, None
    for epoch in range(args.epochs):
        model.train()
        warmup = (epoch < args.warmup_epochs)
        cur    = flow_opt if warmup else opt
        cur.zero_grad()
        pred = model(data, tr_t, ei, ew)
        if warmup:
            loss = gpn_ce_loss(pred['log_soft'][tr_t], data.y[tr_t])
        else:
            n_tr = tr_t.float().sum()
            loss = (gpn_uce_loss(pred['alpha'][tr_t], data.y[tr_t])
                    + gpn_entropy_reg(pred['alpha'][tr_t], args.entropy_reg)) / n_tr
        loss.backward(); cur.step()

        model.eval()
        with torch.no_grad():
            pv   = model(data, tr_t, ei, ew)
            vloss = gpn_ce_loss(pv['log_soft'][val_t], data.y[val_t]).item()
        if vloss < best:
            bs = {k: v.clone() for k, v in model.state_dict().items()}
            best, bad = vloss, 0
        else:
            bad += 1
        if bad >= args.patience and epoch >= args.warmup_epochs:
            break

    model.load_state_dict(bs)
    torch.save(bs, save_path)
    print(f'    GPN val={best:.4f}')

    model.eval()
    with torch.no_grad():
        pf = model(data, tr_t, ei, ew)
    return pf['alpha'].cpu().numpy(), pf['soft'].cpu().numpy()


@torch.no_grad()
def _infer_gpn(model_state, feat_new, ei_orig, lab_t, tr_mask_np, nfeat, nclass, N):
    """用已有权重对新特征（EERM OOD 环境）推理"""
    data = Data(x=feat_new, edge_index=ei_orig, y=lab_t).to(device)
    ei, ew = sym_norm_edge(data.edge_index, N)
    ei = ei.to(device); ew = ew.to(device)
    tr_t = torch.tensor(tr_mask_np, dtype=torch.bool).to(device)
    model = GPNModel(nfeat, nclass,
                     args.dim_hidden, args.dim_latent, args.radial_layers,
                     args.K, args.alpha_teleport, args.dropout,
                     args.alpha_evidence_scale).to(device)
    model.load_state_dict(model_state); model.eval()
    pf = model(data, tr_t, ei, ew)
    return pf['alpha'].cpu().numpy(), pf['soft'].cpu().numpy()


def _metrics(alpha, soft, labels, nclass, binary):
    u = 1. / (alpha.sum(-1) + 1e-8)
    return compute_split_metrics(soft, u, labels, nclass, binary)


def run_elliptic():
    all_keys = build_all_keys(binary=True)
    print('\n[Elliptic] Loading...')
    _, ei_tr, feat_tr, lab_tr, lab_tr_np, N_tr = load_elliptic(ELLIPTIC_TRAIN, args.data_dir, device)
    _, ei_ov, feat_ov, lab_ov, lab_ov_np, N_ov = load_elliptic(ELLIPTIC_VAL,   args.data_dir, device)
    tr_base = (lab_tr_np >= 0)
    data_tr = Data(x=feat_tr, edge_index=ei_tr, y=lab_tr)

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
        print(f'\n{"="*60}\n  GPN Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(lab_tr_np, tr_base,
                                              args.id_val_ratio, args.id_test_ratio, seed)
        alpha_all, soft_all = _train_gpn(data_tr, tr_m, val_m, feat_tr.shape[1], 2,
                                          N_tr,
                                          os.path.join(args.save_dir, f'elliptic_seed{seed}'),
                                          seed)
        run_res = {}
        r_id = _metrics(alpha_all[id_m], soft_all[id_m], lab_tr_np[id_m], 2, binary=True)
        print(f'  ID-test | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f}')
        run_res['ID-test'] = r_id

        for i, (data_te, lnp, tm, N_te) in enumerate(test_graphs):
            rng = np.random.default_rng(seed + i + 100)
            te_idx = np.where(tm)[0]; perm = rng.permutation(len(te_idx))
            nh = max(1, len(te_idx) // 2)
            tr_te = np.zeros(N_te, bool); tr_te[te_idx[perm[:nh]]] = True
            vl_te = np.zeros(N_te, bool); vl_te[te_idx[perm[nh:]]] = True
            if vl_te.sum() == 0: vl_te = tr_te.copy()

            alpha_te, soft_te = _train_gpn(data_te, tr_te, vl_te, data_te.x.shape[1], 2,
                                            N_te,
                                            os.path.join(args.save_dir,
                                                          f'elliptic_seed{seed}_te{i}'),
                                            seed + i + 100)
            r_ood = _metrics(alpha_te[tm], soft_te[tm], lnp[tm], 2, binary=True)
            r_ood = add_cross_split_metrics(r_id, r_ood, r_id['_u'], r_ood['_u'])
            run_res[split_names[1+i]] = r_ood
            print(f'  {split_names[1+i][:30]} | ood_auroc={r_ood["ood_auroc"]:.4f}')

        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, 'elliptic_gpn')
    summarize(all_runs, split_names, all_keys, pref+'_results.csv',
              'Elliptic — GPN',
              reliability_path=pref+'_reliability.csv',
              uncertainty_path=pref+'_uncertainty.csv')


def run_arxiv():
    all_keys = build_all_keys(binary=False)
    print('\n[Arxiv] Loading...')
    _, ei, feat, lab_t, lab_np, node_year, nclass, N = load_arxiv(args.data_path, device)
    base_mask = (node_year <= ARXIV_TRAIN_YEAR)
    oy0, oy1  = ARXIV_OODVAL_YEARS
    data = Data(x=feat, edge_index=ei, y=lab_t)
    split_names = ['ID-test'] + [f'OOD-test_{i}({ty0}-{ty1})'
                                  for i, (ty0, ty1) in enumerate(ARXIV_TESTS)]
    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  GPN Arxiv Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        tr_m, val_m, id_m = stratified_split(lab_np, base_mask,
                                              args.id_val_ratio, args.id_test_ratio, seed)
        alpha_all, soft_all = _train_gpn(data, tr_m, val_m, feat.shape[1], nclass,
                                          N,
                                          os.path.join(args.save_dir, f'arxiv_seed{seed}'),
                                          seed)
        run_res = {}
        r_id = _metrics(alpha_all[id_m], soft_all[id_m], lab_np[id_m], nclass, False)
        print(f'  ID-test | acc={r_id["acc"]:.4f}')
        run_res['ID-test'] = r_id

        for i, (ty0, ty1) in enumerate(ARXIV_TESTS):
            te_mask = (node_year >= ty0) & (node_year <= ty1)
            r_ood = _metrics(alpha_all[te_mask], soft_all[te_mask], lab_np[te_mask], nclass, False)
            r_ood = add_cross_split_metrics(r_id, r_ood, r_id['_u'], r_ood['_u'])
            run_res[split_names[1+i]] = r_ood
        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, 'arxiv_gpn')
    summarize(all_runs, split_names, all_keys, pref+'_results.csv', 'OGB-Arxiv — GPN',
              reliability_path=pref+'_reliability.csv',
              uncertainty_path=pref+'_uncertainty.csv')


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
        print(f'\n{"="*60}\n  GPN EERM Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')
        data = Data(x=feat_tr, edge_index=ei, y=lab_t)
        alpha_id, soft_id = _train_gpn(data, tr_np, val_np, feat_tr.shape[1], nclass,
                                        N,
                                        os.path.join(args.save_dir,
                                                      f'eerm_{args.eerm_dataset}_seed{seed}'),
                                        seed)
        # save model state for OOD inference
        state = torch.load(os.path.join(args.save_dir,
                                         f'eerm_{args.eerm_dataset}_seed{seed}.pth'),
                            map_location=device)

        run_res = {}
        r_id = _metrics(alpha_id[te_np], soft_id[te_np], lab_np[te_np], nclass, False)
        print(f'  ID-test | acc={r_id["acc"]:.4f}')
        run_res['ID-test'] = r_id

        for feat_ood, name in zip(feat_oods, ood_names):
            alpha_ood, soft_ood = _infer_gpn(state, feat_ood, ei, lab_t,
                                               tr_np, feat_tr.shape[1], nclass, N)
            r_ood = _metrics(alpha_ood[te_np], soft_ood[te_np], lab_np[te_np], nclass, False)
            r_ood = add_cross_split_metrics(r_id, r_ood, r_id['_u'], r_ood['_u'])
            run_res[name] = r_ood
            print(f'  {name[:30]} | ood_auroc={r_ood["ood_auroc"]:.4f}')

        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    ds = args.eerm_dataset
    pref = os.path.join(args.save_dir, f'{ds}_gpn')
    summarize(all_runs, split_names, all_keys, pref+'_results.csv', f'EERM-{ds} — GPN',
              reliability_path=pref+'_reliability.csv',
              uncertainty_path=pref+'_uncertainty.csv')


if __name__ == '__main__':
    {'elliptic': run_elliptic, 'arxiv': run_arxiv, 'eerm': run_eerm}[args.dataset]()
