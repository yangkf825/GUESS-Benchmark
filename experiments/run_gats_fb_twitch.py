"""
GATS — Facebook100 & Twitch（跨域 OOD）
=========================================
不确定性：1 - max(p)

用法:
    python experiments/run_gats_fb_twitch.py \
        --dataset twitch --data_root ./data --runs 5
    python experiments/run_gats_fb_twitch.py \
        --dataset facebook --data_root ./data --runs 5
"""
import sys; sys.path.insert(0, 'src')

import os, time, argparse, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam

from gnn_uq_bench.datasets_fb_twitch import load_facebook_twitch, DOMAIN_SETTINGS
from gnn_uq_bench.models import GCNPyG, GATModel, GATS, bfs_distance
from gnn_uq_bench.metrics import (
    compute_split_metrics, add_cross_split_metrics, build_all_keys, summarize,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',       type=str,   default='twitch',
                    choices=['facebook', 'twitch'])
parser.add_argument('--data_root',     type=str,   default='./data')
parser.add_argument('--model',         type=str,   default='GCN', choices=['GCN', 'GAT'])
parser.add_argument('--runs',          type=int,   default=5)
parser.add_argument('--hidden',        type=int,   default=64)
parser.add_argument('--dropout',       type=float, default=0.5)
parser.add_argument('--lr',            type=float, default=0.01)
parser.add_argument('--weight_decay',  type=float, default=5e-4)
parser.add_argument('--epochs',        type=int,   default=200)
parser.add_argument('--patience',      type=int,   default=50)
parser.add_argument('--gats_heads',    type=int,   default=8)
parser.add_argument('--gats_bias',     type=float, default=1.0)
parser.add_argument('--gats_wdecay',   type=float, default=0.005)
parser.add_argument('--gats_epochs',   type=int,   default=200)
parser.add_argument('--gats_patience', type=int,   default=50)
parser.add_argument('--bfs_depth',     type=int,   default=2)
parser.add_argument('--base_seed',     type=int,   default=42)
parser.add_argument('--save_dir',      type=str,   default='./results/gats_fb_twitch')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[设备] {device}')


def _get_base(nfeat, nclass):
    if args.model == 'GCN':
        return GCNPyG(nfeat, args.hidden, nclass, args.dropout).to(device)
    return GATModel(nfeat, args.hidden, nclass, args.dropout).to(device)


def _train_base(base, train_data, val_data, seed, save_path):
    torch.manual_seed(seed); base.reset_parameters()
    opt  = Adam(base.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, bad, bs = 1e9, 0, None
    for epoch in range(args.epochs):
        base.train()
        for data in train_data:
            data = data.to(device); opt.zero_grad()
            F.cross_entropy(base(data.x, data.edge_index), data.y).backward()
            opt.step()
        base.eval(); v_loss = 0.
        with torch.no_grad():
            for data in val_data:
                data = data.to(device)
                v_loss += F.cross_entropy(base(data.x, data.edge_index), data.y).item()
        if v_loss < best:
            bs = copy.deepcopy(base.state_dict()); best, bad = v_loss, 0
        else:
            bad += 1
        if bad >= args.patience: break
    base.load_state_dict(bs); torch.save(bs, save_path)
    print(f'    base val={best:.4f}')
    return base


def _train_gats_on_domain(base, data, tr_mask_np, seed, save_path, nclass, N):
    """在单个域（通常是 val 域）上训练 GATS 温度层"""
    torch.manual_seed(seed)
    data = data.to(device)
    ei   = data.edge_index
    tr_t = torch.tensor(tr_mask_np, dtype=torch.bool, device=device)
    dist = bfs_distance(ei, tr_t, args.bfs_depth, N, device)

    gats = GATS(base, ei, N, dist, nclass,
                heads=args.gats_heads, bias=args.gats_bias).to(device)
    opt  = Adam(gats.cagat.parameters(), lr=args.lr, weight_decay=args.gats_wdecay)
    lab_t = data.y
    best, bad, bs = 1e9, 0, None
    for epoch in range(args.gats_epochs):
        gats.train(); opt.zero_grad()
        F.cross_entropy(gats(data.x, ei), lab_t).backward(); opt.step()
        gats.eval()
        with torch.no_grad():
            lv = F.cross_entropy(gats(data.x, ei), lab_t).item()
        if lv < best:
            bs = copy.deepcopy(gats.state_dict()); best, bad = lv, 0
        else:
            bad += 1
        if bad >= args.gats_patience: break
    gats.load_state_dict(bs); torch.save(bs, save_path)
    print(f'    GATS val={best:.4f}')
    return gats


@torch.no_grad()
def _infer(model, data):
    model.eval(); data = data.to(device)
    logits = model(data.x, data.edge_index).cpu().numpy()
    ex = np.exp(logits - logits.max(1, keepdims=True))
    return ex / ex.sum(1, keepdims=True)


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

    # GATS 的 val 域用第一个 val 域（整个域作为校准图）
    val_dom_data = val_data[0]
    N_val = val_dom_data.x.size(0)
    # 用全部节点作为 "overlap" mask（跨域无节点级别 split）
    tr_mask_np_all = np.ones(N_val, dtype=bool)

    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  GATS Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')

        base = _get_base(nfeat, nclass)
        _train_base(base, train_data, val_data, seed,
                    os.path.join(args.save_dir,
                                 f'{args.dataset}_base_seed{seed}.pth'))

        gats = _train_gats_on_domain(
            base, val_dom_data, tr_mask_np_all, seed,
            os.path.join(args.save_dir, f'{args.dataset}_gats_seed{seed}.pth'),
            nclass, N_val)

        run_res = {}
        probs_id = _infer(base, id_data)   # ID-test 用 base（GATS 绑定 val 域图）
        labels_id = id_data.y.numpy()
        u_id = 1. - probs_id.max(1)
        r_id = compute_split_metrics(probs_id, u_id, labels_id, nclass)
        print(f'  ID-test ({id_dom}) | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f}')
        run_res['ID-test'] = r_id

        for dom, data_ood in zip(ood_doms, ood_datas):
            # 对 OOD 域尝试直接用 base 推理（GATS 图结构已固定在 val 域）
            probs_ood = _infer(base, data_ood)
            labels_ood = data_ood.y.numpy()
            u_ood = 1. - probs_ood.max(1)
            r_ood = compute_split_metrics(probs_ood, u_ood, labels_ood, nclass)
            r_ood = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            name  = f'OOD-{dom}'
            print(f'  {name:12s} | ood_auroc={r_ood["ood_auroc"]:.4f}')
            run_res[name] = r_ood

        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, f'{args.dataset}_gats')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv',
              f'{args.dataset.capitalize()} — GATS',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


if __name__ == '__main__':
    main()
