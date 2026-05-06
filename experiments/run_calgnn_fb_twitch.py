"""
CalGNN — Facebook100 & Twitch（跨域 OOD）
==========================================
支持 --backbone GCN / GAT / GraphSAGE

用法:
    python experiments/run_calgnn_fb_twitch.py --dataset twitch --data_root ./data --runs 5
    python experiments/run_calgnn_fb_twitch.py --dataset twitch --data_root ./data --backbone GAT --runs 5
    python experiments/run_calgnn_fb_twitch.py --dataset twitch --data_root ./data --backbone GraphSAGE --runs 5
"""
import sys; sys.path.insert(0, 'src')

import os, time, argparse, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import scipy.special
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import degree

from gnn_uq_bench.datasets_fb_twitch import load_facebook_twitch, DOMAIN_SETTINGS
from gnn_uq_bench.calibration import (
    TemperatureScaling, HistogramBinning, IsotonicCalib, BBQ,
    MetaCalMisCoverage, compute_sm_conf, rbs_fit, apply_rbs,
)
from gnn_uq_bench.metrics import (
    compute_split_metrics, add_cross_split_metrics, build_all_keys, summarize_calgnn,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',      type=str,   default='twitch',
                    choices=['facebook', 'twitch'])
parser.add_argument('--data_root',    type=str,   default='./data')
parser.add_argument('--backbone',     type=str,   default='GCN',
                    choices=['GCN', 'GAT', 'GraphSAGE'],
                    help='GNN backbone: GCN (default) / GAT / GraphSAGE')
parser.add_argument('--runs',         type=int,   default=5)
parser.add_argument('--hidden',       type=int,   default=8)
parser.add_argument('--dropout',      type=float, default=0.3)
parser.add_argument('--lr',           type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--epochs',       type=int,   default=100)
parser.add_argument('--patience',     type=int,   default=50)
parser.add_argument('--num_bins_rbs', type=int,   default=10)
parser.add_argument('--base_seed',    type=int,   default=42)
parser.add_argument('--save_dir',     type=str,   default='./results/calgnn_fb_twitch')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BTAG = args.backbone.lower()
print(f'[设备] {device}  [backbone] {args.backbone}')

CAL_METHODS_OUT = ['Uncal', 'RBS']


# ── 三种 backbone 模型（三层 + BN）──────────────────────────────
class GCNModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dp):
        super().__init__()
        self.c1  = GCNConv(nfeat, nhid); self.bn1 = nn.BatchNorm1d(nhid)
        self.c2  = GCNConv(nhid,  nhid); self.bn2 = nn.BatchNorm1d(nhid)
        self.c3  = GCNConv(nhid,  nclass); self.dp = dp

    def forward(self, x, ei):
        x = F.dropout(F.relu(self.bn1(self.c1(x, ei))), self.dp, self.training)
        x = F.dropout(F.relu(self.bn2(self.c2(x, ei)) + x), self.dp, self.training)
        return self.c3(x, ei)


class GATModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dp, heads=4):
        super().__init__()
        self.c1  = GATConv(nfeat, nhid, heads=heads, dropout=dp, concat=True)
        self.bn1 = nn.BatchNorm1d(nhid * heads)
        self.c2  = GATConv(nhid * heads, nhid, heads=1, dropout=dp, concat=False)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.c3  = GATConv(nhid, nclass, heads=1, dropout=dp, concat=False)
        self.dp  = dp

    def forward(self, x, ei):
        x = F.dropout(x, self.dp, self.training)
        x = F.dropout(F.elu(self.bn1(self.c1(x, ei))), self.dp, self.training)
        x = F.dropout(F.elu(self.bn2(self.c2(x, ei))), self.dp, self.training)
        return self.c3(x, ei)


class SAGEModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dp):
        super().__init__()
        self.c1  = SAGEConv(nfeat, nhid); self.bn1 = nn.BatchNorm1d(nhid)
        self.c2  = SAGEConv(nhid,  nhid); self.bn2 = nn.BatchNorm1d(nhid)
        self.c3  = SAGEConv(nhid,  nclass); self.dp = dp

    def forward(self, x, ei):
        x = F.dropout(F.relu(self.bn1(self.c1(x, ei))), self.dp, self.training)
        x = F.dropout(F.relu(self.bn2(self.c2(x, ei)) + x), self.dp, self.training)
        return self.c3(x, ei)


def _make_model(nfeat, nclass):
    bb = args.backbone.upper()
    if bb == 'GCN':
        return GCNModel(nfeat, args.hidden, nclass, args.dropout).to(device)
    elif bb == 'GAT':
        return GATModel(nfeat, args.hidden, nclass, args.dropout).to(device)
    else:
        return SAGEModel(nfeat, args.hidden, nclass, args.dropout).to(device)


def _train(train_data, val_data, nfeat, nclass, seed, save_path):
    torch.manual_seed(seed)
    model = _make_model(nfeat, nclass)
    opt   = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, bad, bs = 1e9, 0, None
    for epoch in range(args.epochs):
        model.train()
        for data in train_data:
            data = data.to(device); opt.zero_grad()
            F.cross_entropy(model(data.x, data.edge_index), data.y).backward()
            opt.step()
        model.eval(); v_loss = 0.
        with torch.no_grad():
            for data in val_data:
                data = data.to(device)
                v_loss += F.cross_entropy(model(data.x, data.edge_index), data.y).item()
        if v_loss < best:
            bs = copy.deepcopy(model.state_dict()); best, bad = v_loss, 0
        else:
            bad += 1
        if bad >= args.patience: break
    model.load_state_dict(bs); torch.save(bs, save_path)
    print(f'    val_loss={best:.4f}')
    return model


@torch.no_grad()
def _logits_probs_sm_conf(model, data):
    model.eval(); data = data.to(device)
    logits_t = model(data.x, data.edge_index)
    probs_t  = F.softmax(logits_t, dim=1)
    N   = data.x.size(0)
    sm  = compute_sm_conf(data.edge_index, N, probs_t, device)
    return logits_t.cpu().numpy(), probs_t.cpu().numpy(), sm


def _fit_calibrators(ov_logits, ov_probs, ov_labels, ov_sm, ov_ei, N_ov):
    ts  = TemperatureScaling().fit(ov_logits, ov_labels)
    hb  = HistogramBinning().fit(ov_probs,  ov_labels)
    iso = IsotonicCalib().fit(ov_probs,     ov_labels)
    bbq = BBQ().fit(ov_probs,               ov_labels)
    mc  = MetaCalMisCoverage().fit(ov_logits, ov_labels)
    T_rbs, bins_rbs = rbs_fit(ov_sm, ov_logits, ov_labels, args.num_bins_rbs)
    return ts, hb, iso, bbq, mc, T_rbs, bins_rbs


def _eval_all_cal(logits, probs, labels, nclass,
                  ts, hb, iso, bbq, mc, T_rbs, bins_rbs, sm):
    all_p = {
        'Uncal':   probs,
        'TS':      ts.predict_proba(logits),
        'HB':      hb.predict_proba(probs),
        'Iso':     iso.predict_proba(probs),
        'BBQ':     bbq.predict_proba(probs),
        'MetaCal': mc.predict(logits),
        'RBS':     apply_rbs(T_rbs, bins_rbs, sm, logits, device),
    }
    res = {}
    for cm, p in all_p.items():
        u = 1. - p.max(1)
        res[cm] = compute_split_metrics(p, u, labels, nclass)
    return res


def main():
    all_keys = build_all_keys(binary=False)
    print(f'\n[{args.dataset}] Loading...')
    (label_map, nclass, nfeat,
     train_data, val_data, test_data,
     domain_names, scaler) = load_facebook_twitch(args.dataset, args.data_root, device=None)

    id_dom   = domain_names['train'][-1]
    id_data  = train_data[-1]
    ood_doms  = domain_names['val'] + domain_names['test']
    ood_datas = val_data + test_data
    split_names = ['ID-test'] + [f'OOD-{d}' for d in ood_doms]

    all_runs_pm = {'_combined': []}

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  [{args.backbone}] CalGNN Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')

        model = _train(
            train_data, val_data, nfeat, nclass, seed,
            os.path.join(args.save_dir, f'{args.dataset}_{BTAG}_seed{seed}.pth'))

        ov_logits_l, ov_probs_l, ov_labels_l, ov_sm_l = [], [], [], []
        for data in val_data:
            lg, pb, sm = _logits_probs_sm_conf(model, data)
            ov_logits_l.append(lg); ov_probs_l.append(pb)
            ov_labels_l.append(data.y.cpu().numpy()); ov_sm_l.append(sm)
        ov_logits = np.concatenate(ov_logits_l)
        ov_probs  = np.concatenate(ov_probs_l)
        ov_labels = np.concatenate(ov_labels_l)
        ov_sm     = np.concatenate(ov_sm_l)

        ts, hb, iso, bbq, mc, T_rbs, bins_rbs = _fit_calibrators(
            ov_logits, ov_probs, ov_labels, ov_sm, None, None)

        id_lg, id_pb, id_sm = _logits_probs_sm_conf(model, id_data)
        id_labels = id_data.y.cpu().numpy()
        id_res = _eval_all_cal(id_lg, id_pb, id_labels, nclass,
                                ts, hb, iso, bbq, mc, T_rbs, bins_rbs, id_sm)
        for cm in CAL_METHODS_OUT:
            print(f'  ID-test [{cm}] acc={id_res[cm]["acc"]:.4f} ece={id_res[cm]["ece"]:.4f}')

        run_per_m = {m: {'ID-test': id_res[m]} for m in CAL_METHODS_OUT}

        for dom, data_ood in zip(ood_doms, ood_datas):
            ood_lg, ood_pb, ood_sm = _logits_probs_sm_conf(model, data_ood)
            ood_labels = data_ood.y.cpu().numpy()
            ood_res = _eval_all_cal(ood_lg, ood_pb, ood_labels, nclass,
                                     ts, hb, iso, bbq, mc, T_rbs, bins_rbs, ood_sm)
            name = f'OOD-{dom}'
            for cm in CAL_METHODS_OUT:
                r_ood = add_cross_split_metrics(
                    run_per_m[cm]['ID-test'], ood_res[cm],
                    run_per_m[cm]['ID-test']['_u'], ood_res[cm]['_u'])
                run_per_m[cm][name] = r_ood
            print(f'  {name:12s} | Uncal ood_auroc={run_per_m["Uncal"][name]["ood_auroc"]:.4f}')

        run_per_split = {}
        for split in ['ID-test'] + [f'OOD-{d}' for d in ood_doms]:
            run_per_split[split] = {}
            for m in CAL_METHODS_OUT:
                run_per_split[split][m] = run_per_m[m].get(split, {})
        all_runs_pm['_combined'].append(run_per_split)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref_base = os.path.join(args.save_dir, f'{args.dataset}_{BTAG}_calgnn')
    from gnn_uq_bench.metrics import summarize_calgnn
    summarize_calgnn(
        all_runs_pm['_combined'], split_names, all_keys,
        pref_base + '_results.csv',
        f'{args.dataset.capitalize()} — CalGNN [{args.backbone}]',
        CAL_METHODS_OUT,
        reliability_path=pref_base + '_reliability.csv',
        uncertainty_path=pref_base + '_uncertainty.csv')


if __name__ == '__main__':
    main()
