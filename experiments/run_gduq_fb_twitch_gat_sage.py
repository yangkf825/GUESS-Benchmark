"""
G-ΔUQ — Facebook100 & Twitch（跨域 OOD）
==========================================
不确定性：multi-anchor sigmoid std 均值

用法:
    python experiments/run_gduq_fb_twitch.py \
        --dataset twitch --data_root ./data --runs 5
    python experiments/run_gduq_fb_twitch.py \
        --dataset facebook --data_root ./data --runs 5
"""
import sys; sys.path.insert(0, 'src')
from gnn_uq_bench.model_gat_sage import (canonical_backbone_name, get_pyg_backbone, get_pyg_backbone_bn, get_sparse_backbone, GraphANTNodeBackbone, GPNBackboneModel)

import os, time, argparse, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data

from gnn_uq_bench.datasets_fb_twitch import load_facebook_twitch, DOMAIN_SETTINGS
from gnn_uq_bench.models import BaseModelNode, GraphANTNode
from gnn_uq_bench.metrics import (
    compute_split_metrics, add_cross_split_metrics, build_all_keys, summarize,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',      type=str,   default='twitch',
                    choices=['facebook', 'twitch'])
parser.add_argument('--data_root',    type=str,   default='./data')
parser.add_argument('--runs',         type=int,   default=5)
parser.add_argument('--model',         type=str,   default='GAT',
                    choices=['GCN', 'GAT', 'SAGE', 'GraphSAGE'],
                    help='backbone: GCN, GAT, SAGE/GraphSAGE')
parser.add_argument('--backbone_heads', type=int,   default=8,
                    help='GAT attention heads for the new backbone')
parser.add_argument('--hidden',       type=int,   default=64)
parser.add_argument('--num_layers',   type=int,   default=2)
parser.add_argument('--dropout',      type=float, default=0.5)
parser.add_argument('--lr',           type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--epochs',       type=int,   default=100)
parser.add_argument('--patience',     type=int,   default=20)
parser.add_argument('--n_anchors',    type=int,   default=10)
parser.add_argument('--anchor_type',  type=str,   default='node',
                    choices=['node', 'graph'])
parser.add_argument('--base_seed',    type=int,   default=42)
parser.add_argument('--save_dir',     type=str,   default='./results/gduq_fb_twitch_gat_sage')
args = parser.parse_args()

def _backbone_name():
    return canonical_backbone_name(args.model)


def _model_tag():
    return _backbone_name().lower()


def _tagged_prefix(prefix):
    return f'{prefix}_{_model_tag()}'


def _tagged_title(title):
    return f'{title} [{_backbone_name()}]'


os.makedirs(args.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[设备] {device}')


def _build_gant(nfeat, nclass, all_train_x_np):
    """Build G-DUQ with GCN/GAT/GraphSAGE graph backbone."""
    mu  = torch.tensor(all_train_x_np.mean(0), dtype=torch.float32)
    std = torch.tensor(all_train_x_np.std(0)  + 1e-6, dtype=torch.float32)
    base = get_pyg_backbone(args.model, nfeat * 2, args.hidden, nclass,
                            args.dropout, heads=getattr(args, 'backbone_heads', 8))
    return GraphANTNodeBackbone(base, mu, std,
                                anchor_type=args.anchor_type,
                                num_classes=nclass).to(device)


def _train(model, train_data, val_data, seed, save_path):
    torch.manual_seed(seed)
    opt  = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, bad, bs = float('inf'), 0, None

    for epoch in range(args.epochs):
        model.train()
        for data in train_data:
            data = data.to(device); opt.zero_grad()
            # GraphANTNode.forward 返回 logits（单锚点）
            logits = model(data)
            F.cross_entropy(logits, data.y).backward()
            opt.step()

        model.eval(); v_loss = 0.
        with torch.no_grad():
            for data in val_data:
                data = data.to(device)
                v_loss += F.cross_entropy(model(data), data.y).item()
        if v_loss < best:
            bs = copy.deepcopy(model.state_dict()); best, bad = v_loss, 0
        else:
            bad += 1
        if bad >= args.patience: break

    model.load_state_dict(bs); torch.save(bs, save_path)
    print(f'    GDUQ val={best:.4f}')
    return model


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

    # 合并训练域特征估计锚点分布
    all_train_x = np.concatenate([d.x.numpy() for d in train_data], axis=0)

    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  G-ΔUQ Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')

        model = _build_gant(nfeat, nclass, all_train_x)
        _train(model, train_data, val_data, seed,
               os.path.join(args.save_dir, f'{args.dataset}_gduq_seed{seed}.pth'))

        run_res = {}

        # ID-test
        probs_id, u_id = model.infer(id_data.to(device), args.n_anchors)
        labels_id = id_data.y.cpu().numpy()
        r_id = compute_split_metrics(probs_id, u_id, labels_id, nclass)
        print(f'  ID-test ({id_dom}) | acc={r_id["acc"]:.4f} '
              f'ue_auroc={r_id["ue_auroc"]:.4f}')
        run_res['ID-test'] = r_id

        # OOD-test
        for dom, data_ood in zip(ood_doms, ood_datas):
            probs_ood, u_ood = model.infer(data_ood.to(device), args.n_anchors)
            labels_ood = data_ood.y.cpu().numpy()
            r_ood = compute_split_metrics(probs_ood, u_ood, labels_ood, nclass)
            r_ood = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            name  = f'OOD-{dom}'
            print(f'  {name:12s} | ood_auroc={r_ood["ood_auroc"]:.4f}')
            run_res[name] = r_ood

        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, f'{args.dataset}_gduq')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv',
              f'{args.dataset.capitalize()} — G-ΔUQ',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


if __name__ == '__main__':
    main()
