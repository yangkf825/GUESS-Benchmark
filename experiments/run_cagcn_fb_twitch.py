"""
CaGCN — Facebook100 & Twitch（跨域 OOD）
==========================================
图卷积温度缩放校准，不确定性：1 - max(p)

用法:
    python experiments/run_cagcn_fb_twitch.py \
        --dataset twitch --data_root ./data --runs 5
    python experiments/run_cagcn_fb_twitch.py \
        --dataset facebook --data_root ./data --runs 5
"""
import sys; sys.path.insert(0, 'src')

import os, time, argparse, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv

from gnn_uq_bench.datasets_fb_twitch import load_facebook_twitch, DOMAIN_SETTINGS
from gnn_uq_bench.models import GCNSparse, CaGCN, GraphConvolution
from gnn_uq_bench.metrics import (
    compute_split_metrics, add_cross_split_metrics, build_all_keys, summarize,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',      type=str,   default='twitch',
                    choices=['facebook', 'twitch'])
parser.add_argument('--data_root',    type=str,   default='./data')
parser.add_argument('--runs',         type=int,   default=5)
parser.add_argument('--hidden',       type=int,   default=64)
parser.add_argument('--dropout',      type=float, default=0.5)
parser.add_argument('--lr',           type=float, default=0.01)
parser.add_argument('--lr_for_cal',   type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--l2_for_cal',   type=float, default=5e-3)
parser.add_argument('--epochs',       type=int,   default=200)
parser.add_argument('--patience',     type=int,   default=50)
parser.add_argument('--Lambda',       type=float, default=0.5)
parser.add_argument('--base_seed',    type=int,   default=42)
parser.add_argument('--save_dir',     type=str,   default='./results/cagcn_fb_twitch')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[设备] {device}')


# ── PyG GCN base（与其他 fb_twitch 脚本统一）─────────────────
class GCNBase(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dp):
        super().__init__()
        self.c1 = GCNConv(nfeat, nhid); self.bn1 = nn.BatchNorm1d(nhid)
        self.c2 = GCNConv(nhid,  nhid); self.bn2 = nn.BatchNorm1d(nhid)
        self.c3 = GCNConv(nhid,  nclass); self.dp = dp

    def forward(self, x, ei):
        x = F.dropout(F.relu(self.bn1(self.c1(x, ei))), self.dp, self.training)
        x = F.dropout(F.relu(self.bn2(self.c2(x, ei)) + x), self.dp, self.training)
        return self.c3(x, ei)

    def reset_parameters(self):
        for m in [self.c1, self.c2, self.c3]:
            m.reset_parameters()
        for bn in [self.bn1, self.bn2]:
            bn.reset_parameters()


# ── CaGCN for PyG（用 PyG GCNConv 替代稀疏 adj）────────────────
class CaGCNPyG(nn.Module):
    """图卷积温度缩放：base 冻结，两层 GCN 学 per-node T"""
    def __init__(self, nclass, base_model):
        super().__init__()
        self.base = base_model
        self.s1   = GCNConv(nclass, 16)
        self.s2   = GCNConv(16, 1)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x, ei):
        logits = self.base(x, ei)
        t = torch.log(torch.exp(self.s2(F.relu(self.s1(logits, ei)), ei)) + 1.1)
        return logits * t


def _intra_loss(output, labels):
    p = F.softmax(output, dim=1); pred = p.max(1)[1]
    ok  = (pred == labels).nonzero(as_tuple=True)[0]
    bad = (pred != labels).nonzero(as_tuple=True)[0]
    s = p.sort(1, descending=True)[0]; t1, t2 = s[:, 0], s[:, 1]
    return ((1 - t1[ok] + t2[ok]).sum() + (t1[bad] - t2[bad]).sum()) / labels.size(0)


def _train_base(nfeat, nclass, train_data, val_data, seed, save_path):
    torch.manual_seed(seed)
    model = GCNBase(nfeat, args.hidden, nclass, args.dropout).to(device)
    model.reset_parameters()
    opt  = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    print(f'    base val={best:.4f}')
    return model


def _train_cagcn(base, nclass, val_data, seed, save_path):
    torch.manual_seed(seed)
    model = CaGCNPyG(nclass, base).to(device)
    opt   = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=args.lr_for_cal, weight_decay=args.l2_for_cal)
    best, bad, bs = 1e9, 0, None
    for epoch in range(args.epochs):
        model.train()
        for data in val_data:
            data = data.to(device); opt.zero_grad()
            out = model(data.x, data.edge_index)
            (F.cross_entropy(out, data.y) +
             args.Lambda * _intra_loss(out, data.y)).backward()
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
    print(f'    CaGCN val={best:.4f}')
    return model


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

    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  CaGCN Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')

        base = _train_base(
            nfeat, nclass, train_data, val_data, seed,
            os.path.join(args.save_dir, f'{args.dataset}_base_seed{seed}.pth'))
        cagcn = _train_cagcn(
            base, nclass, val_data, seed,
            os.path.join(args.save_dir, f'{args.dataset}_cagcn_seed{seed}.pth'))

        run_res = {}
        probs_id = _infer(cagcn, id_data)
        labels_id = id_data.y.numpy()
        u_id = 1. - probs_id.max(1)
        r_id = compute_split_metrics(probs_id, u_id, labels_id, nclass)
        print(f'  ID-test ({id_dom}) | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f}')
        run_res['ID-test'] = r_id

        for dom, data_ood in zip(ood_doms, ood_datas):
            probs_ood = _infer(cagcn, data_ood)
            labels_ood = data_ood.y.numpy()
            u_ood = 1. - probs_ood.max(1)
            r_ood = compute_split_metrics(probs_ood, u_ood, labels_ood, nclass)
            r_ood = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            name  = f'OOD-{dom}'
            print(f'  {name:12s} | ood_auroc={r_ood["ood_auroc"]:.4f}')
            run_res[name] = r_ood

        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, f'{args.dataset}_cagcn')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv',
              f'{args.dataset.capitalize()} — CaGCN',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


if __name__ == '__main__':
    main()
