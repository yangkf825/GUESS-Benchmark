"""
CalGNN — Facebook100 & Twitch（跨域 OOD）
==========================================
6种后处理校准：Uncal / TS / HB / Iso / BBQ / MetaCal / RBS

用法:
    python experiments/run_calgnn_fb_twitch.py \
        --dataset twitch --data_root ./data --runs 5
    python experiments/run_calgnn_fb_twitch.py \
        --dataset facebook --data_root ./data --runs 5
"""
import sys; sys.path.insert(0, 'src')

import os, time, argparse, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import scipy.special
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree

from gnn_uq_bench.datasets_fb_twitch import (
    load_facebook_twitch, DOMAIN_SETTINGS,
)
from gnn_uq_bench.calibration import (
    TemperatureScaling, HistogramBinning, IsotonicCalib, BBQ,
    MetaCalMisCoverage, compute_sm_conf, rbs_fit, apply_rbs,
)
from gnn_uq_bench.metrics import (
    compute_split_metrics, add_cross_split_metrics, build_all_keys,
    summarize_calgnn,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',      type=str,   default='twitch',
                    choices=['facebook', 'twitch'])
parser.add_argument('--data_root',    type=str,   default='./data')
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
print(f'[设备] {device}')

CAL_METHODS     = ['Uncal', 'TS', 'HB', 'Iso', 'BBQ', 'MetaCal', 'RBS']
CAL_METHODS_OUT = ['Uncal', 'RBS']


class GCNModel(nn.Module):
    """三层 GCN + BN（与同事 CalGNN 代码一致）"""
    def __init__(self, nfeat, nhid, nclass, dp):
        super().__init__()
        self.c1  = GCNConv(nfeat, nhid); self.bn1 = nn.BatchNorm1d(nhid)
        self.c2  = GCNConv(nhid,  nhid); self.bn2 = nn.BatchNorm1d(nhid)
        self.c3  = GCNConv(nhid,  nclass); self.dp = dp

    def forward(self, x, ei):
        x = F.dropout(F.relu(self.bn1(self.c1(x, ei))), self.dp, self.training)
        x = F.dropout(F.relu(self.bn2(self.c2(x, ei)) + x), self.dp, self.training)
        return self.c3(x, ei)


def _train(train_data, val_data, nfeat, nclass, seed, save_path):
    torch.manual_seed(seed)
    model = GCNModel(nfeat, args.hidden, nclass, args.dropout).to(device)
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
    """返回 (logits_np, probs_np, sm_conf_np) 全部 numpy"""
    model.eval(); data = data.to(device)
    logits_t = model(data.x, data.edge_index)
    probs_t  = F.softmax(logits_t, dim=1)
    # smoothed confidence via neighbourhood average
    N   = data.x.size(0)
    ei  = data.edge_index
    sm  = compute_sm_conf(ei, N, probs_t, device)
    return logits_t.cpu().numpy(), probs_t.cpu().numpy(), sm


def _fit_calibrators(ov_logits, ov_probs, ov_labels,
                     ov_sm, ov_ei, N_ov):
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

    # 合并所有 val 域做校准
    val_logits_list, val_probs_list, val_labels_list, val_sm_list = [], [], [], []

    all_runs_pm = {m: [] for m in CAL_METHODS_OUT}
    all_runs_pm['_combined'] = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  CalGNN Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')

        model = _train(
            train_data, val_data, nfeat, nclass, seed,
            os.path.join(args.save_dir, f'{args.dataset}_seed{seed}.pth'))

        # 收集 val 域数据用于拟合校准器
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

        # ID-test
        id_lg, id_pb, id_sm = _logits_probs_sm_conf(model, id_data)
        id_labels = id_data.y.cpu().numpy()
        id_res = _eval_all_cal(id_lg, id_pb, id_labels, nclass,
                                ts, hb, iso, bbq, mc, T_rbs, bins_rbs, id_sm)
        for cm in CAL_METHODS_OUT:
            print(f'  ID-test [{cm}] acc={id_res[cm]["acc"]:.4f} '
                  f'ece={id_res[cm]["ece"]:.4f}')

        run_per_m = {m: {'ID-test': id_res[m]} for m in CAL_METHODS_OUT}

        # OOD splits
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

        # 转换结构：run_per_split[split][cm] = metric_dict
        # summarize_calgnn 期望 all_runs[i][split][cm]
        run_per_split = {}
        for split in ['ID-test'] + [f'OOD-{d}' for d in ood_doms]:
            run_per_split[split] = {}
            for m in CAL_METHODS_OUT:
                run_per_split[split][m] = run_per_m[m].get(split, {})
        all_runs_pm['_combined'].append(run_per_split)
        print(f'  elapsed {time.time()-t0:.1f}s')

    # DEBUG
    print("DEBUG: len(all_runs_pm['_combined'])=", len(all_runs_pm['_combined']))
    if all_runs_pm['_combined']:
        r0 = all_runs_pm['_combined'][0]
        print("DEBUG: r0.keys()=", list(r0.keys()))
        sname = list(r0.keys())[0]
        print(f"DEBUG: r0['{sname}'].keys()=", list(r0[sname].keys()))
        cm0 = list(r0[sname].keys())[0]
        m = r0[sname][cm0]
        print(f"DEBUG: r0['{sname}']['{cm0}'].keys()=", list(m.keys())[:5] if m else 'EMPTY')
        print(f"DEBUG: acc=", m.get('acc'))
    pref_base = os.path.join(args.save_dir, f'{args.dataset}_calgnn')
    summarize_calgnn(
        all_runs_pm['_combined'], split_names, all_keys,
        pref_base + '_results.csv',
        f'{args.dataset.capitalize()} — CalGNN',
        CAL_METHODS_OUT,
        reliability_path=pref_base + '_reliability.csv',
        uncertainty_path=pref_base + '_uncertainty.csv')


if __name__ == '__main__':
    main()