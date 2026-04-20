"""
S-BGCN-T-K — Facebook100 & Twitch（跨域 OOD）
================================================
不确定性：预测熵 H[p]

用法:
    python experiments/run_ungnn_fb_twitch.py \
        --dataset twitch --data_root ./data --runs 5
    python experiments/run_ungnn_fb_twitch.py \
        --dataset facebook --data_root ./data --runs 5
"""
import sys; sys.path.insert(0, 'src')

import os, time, argparse, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv

from gnn_uq_bench.datasets_fb_twitch import load_facebook_twitch, DOMAIN_SETTINGS
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
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--epochs',       type=int,   default=200)
parser.add_argument('--patience',     type=int,   default=50)
parser.add_argument('--base_seed',    type=int,   default=42)
parser.add_argument('--save_dir',     type=str,   default='./results/ungnn_fb_twitch')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[设备] {device}')


# ── 模型（三层 GCN + BatchNorm，与同事代码一致）─────────────────
class GCNModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.c1  = GCNConv(nfeat, nhid); self.bn1 = nn.BatchNorm1d(nhid)
        self.c2  = GCNConv(nhid,  nhid); self.bn2 = nn.BatchNorm1d(nhid)
        self.c3  = GCNConv(nhid,  nclass)
        self.dp  = dropout

    def forward(self, x, ei):
        x = F.dropout(F.relu(self.bn1(self.c1(x, ei))), self.dp, self.training)
        x = F.dropout(F.relu(self.bn2(self.c2(x, ei)) + x), self.dp, self.training)
        return self.c3(x, ei)


def _entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)


def _train_on_domains(train_data, val_data, nfeat, nclass, seed, save_path):
    """在所有训练域上联合训练，验证域做 early stopping。"""
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

        model.eval(); v_loss = 0.0
        with torch.no_grad():
            for data in val_data:
                data = data.to(device)
                v_loss += F.cross_entropy(model(data.x, data.edge_index), data.y).item()
        if v_loss < best:
            bs = copy.deepcopy(model.state_dict()); best, bad = v_loss, 0
        else:
            bad += 1
        if bad >= args.patience: break

    model.load_state_dict(bs)
    torch.save(bs, save_path)
    print(f'    val_loss={best:.4f}')
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

    # ID 域: 取最后一个训练域做 ID-test（与同事代码一致）
    id_dom   = domain_names['train'][-1]
    id_data  = train_data[-1]

    # OOD 域：val + test
    ood_doms  = domain_names['val'] + domain_names['test']
    ood_datas = val_data + test_data
    split_names = ['ID-test'] + [f'OOD-{d}' for d in ood_doms]

    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')

        model = _train_on_domains(
            train_data, val_data, nfeat, nclass, seed,
            os.path.join(args.save_dir, f'{args.dataset}_seed{seed}.pth'))

        run_res = {}

        # ID-test
        probs_id = _infer(model, id_data)
        labels_id = id_data.y.cpu().numpy()
        u_id = _entropy(probs_id)
        r_id = compute_split_metrics(probs_id, u_id, labels_id, nclass)
        print(f'  ID-test ({id_dom}) | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f} '
              f'ue_auroc={r_id["ue_auroc"]:.4f}')
        run_res['ID-test'] = r_id

        # OOD-test
        for dom, data_ood in zip(ood_doms, ood_datas):
            probs_ood = _infer(model, data_ood)
            labels_ood = data_ood.y.cpu().numpy()
            u_ood = _entropy(probs_ood)
            r_ood = compute_split_metrics(probs_ood, u_ood, labels_ood, nclass)
            r_ood = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            name  = f'OOD-{dom}'
            print(f'  {name:12s} | acc={r_ood["acc"]:.4f} '
                  f'ood_auroc={r_ood["ood_auroc"]:.4f}')
            run_res[name] = r_ood

        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, f'{args.dataset}_ungnn')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv',
              f'{args.dataset.capitalize()} — S-BGCN-T-K',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


if __name__ == '__main__':
    main()
