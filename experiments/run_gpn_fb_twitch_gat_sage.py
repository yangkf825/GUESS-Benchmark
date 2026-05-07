"""
GPN — Facebook100 & Twitch（跨域 OOD）
========================================
不确定性：1 / alpha.sum()（Dirichlet evidence 倒数）

用法:
    python experiments/run_gpn_fb_twitch.py \
        --dataset twitch --data_root ./data --runs 3
    python experiments/run_gpn_fb_twitch.py \
        --dataset facebook --data_root ./data --runs 3
"""
import sys; sys.path.insert(0, 'src')
from gnn_uq_bench.model_gat_sage import (canonical_backbone_name, get_pyg_backbone, get_pyg_backbone_bn, get_sparse_backbone, GraphANTNodeBackbone, GPNBackboneModel)

import os, time, argparse, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data

from gnn_uq_bench.datasets_fb_twitch import load_facebook_twitch, DOMAIN_SETTINGS
from gnn_uq_bench.models import (
    GPNModel, gpn_uce_loss, gpn_entropy_reg, gpn_ce_loss, sym_norm_edge,
)
from gnn_uq_bench.metrics import (
    compute_split_metrics, add_cross_split_metrics, build_all_keys, summarize,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',              type=str,   default='twitch',
                    choices=['facebook', 'twitch'])
parser.add_argument('--data_root',            type=str,   default='./data')
parser.add_argument('--runs',                 type=int,   default=3)
parser.add_argument('--model',         type=str,   default='GAT',
                    choices=['GCN', 'GAT', 'SAGE', 'GraphSAGE'],
                    help='backbone: GCN, GAT, SAGE/GraphSAGE')
parser.add_argument('--dim_hidden',           type=int,   default=64)
parser.add_argument('--dim_latent',           type=int,   default=16)
parser.add_argument('--radial_layers',        type=int,   default=6)
parser.add_argument('--K',                    type=int,   default=10)
parser.add_argument('--alpha_teleport',       type=float, default=0.1)
parser.add_argument('--dropout',              type=float, default=0.2)
parser.add_argument('--entropy_reg',          type=float, default=1e-4)
parser.add_argument('--alpha_evidence_scale', type=str,   default='latent-new-plus-classes')
parser.add_argument('--lr',                   type=float, default=0.001)
parser.add_argument('--flow_lr',              type=float, default=0.001)
parser.add_argument('--flow_wd',              type=float, default=5e-4)
parser.add_argument('--epochs',               type=int,   default=100)
parser.add_argument('--warmup_epochs',        type=int,   default=20)
parser.add_argument('--patience',             type=int,   default=20)
parser.add_argument('--base_seed',            type=int,   default=42)
parser.add_argument('--save_dir',             type=str,   default='./results/gpn_fb_twitch_gat_sage')
parser.add_argument('--backbone_heads', type=int,   default=8,
                    help='GAT attention heads for the new backbone')
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


def _train(train_data, val_data, nfeat, nclass, seed, save_path):
    torch.manual_seed(seed)
    model = GPNBackboneModel(
        nfeat, nclass, args.dim_hidden, args.dim_latent,
        args.radial_layers, args.K, args.alpha_teleport,
        args.dropout, args.alpha_evidence_scale,
        backbone=args.model, heads=getattr(args, 'backbone_heads', 8)).to(device)
    opt = model.get_optimizer(args.lr, 5e-4, args.flow_lr, args.flow_wd)[0]
    best, bad, bs = float('inf'), 0, None

    for epoch in range(args.epochs):
        model.train()
        for data in train_data:
            data = data.to(device)
            ei, ew = sym_norm_edge(data.edge_index, data.x.size(0))
            opt.zero_grad()
            pred = model(data, data.train_mask if hasattr(data, 'train_mask') else
                         torch.ones(data.x.size(0), dtype=torch.bool, device=device),
                         ei, ew)
            if epoch < args.warmup_epochs:
                loss = gpn_ce_loss(pred['log_soft'], data.y)
            else:
                n = data.x.size(0)
                loss = (gpn_uce_loss(pred['alpha'], data.y)
                        + gpn_entropy_reg(pred['alpha'], args.entropy_reg)) / n
            loss.backward(); opt.step()

        model.eval(); v_loss = 0.
        with torch.no_grad():
            for data in val_data:
                data = data.to(device)
                ei, ew = sym_norm_edge(data.edge_index, data.x.size(0))
                # create a dummy all-true train_mask for GPN forward
                tr_m = torch.ones(data.x.size(0), dtype=torch.bool, device=device)
                pred = model(data, tr_m, ei, ew)
                v_loss += gpn_ce_loss(pred['log_soft'], data.y).item()

        if v_loss < best:
            bs = {k: v.clone() for k, v in model.state_dict().items()}
            best, bad = v_loss, 0
        else:
            bad += 1
        if bad >= args.patience: break

    model.load_state_dict(bs); torch.save(bs, save_path)
    print(f'    GPN val={best:.4f}')
    return model


@torch.no_grad()
def _infer(model, data):
    model.eval(); data = data.to(device)
    ei, ew = sym_norm_edge(data.edge_index, data.x.size(0))
    tr_m = torch.ones(data.x.size(0), dtype=torch.bool, device=device)
    pred = model(data, tr_m, ei, ew)
    alpha = pred['alpha'].cpu().numpy()
    soft  = pred['soft'].cpu().numpy()
    u     = 1. / (alpha.sum(-1) + 1e-10)
    return soft, u


def main():
    all_keys = build_all_keys(binary=False)
    print(f'\n[{args.dataset}] Loading...')
    (label_map, nclass, nfeat,
     train_data, val_data, test_data,
     domain_names, scaler) = load_facebook_twitch(args.dataset, args.data_root, device=None)

    # GPN 的 forward 需要 data.x — 直接用 PyG Data，不需要 train_mask
    # 这里 train_mask 传全 True（跨域训练没有节点级别 split）

    id_dom   = domain_names['train'][-1]
    id_data  = train_data[-1]
    ood_doms  = domain_names['val'] + domain_names['test']
    ood_datas = val_data + test_data
    split_names = ['ID-test'] + [f'OOD-{d}' for d in ood_doms]

    all_runs = []

    for r in range(args.runs):
        seed = args.base_seed + r; t0 = time.time()
        print(f'\n{"="*60}\n  GPN Run {r+1}/{args.runs}  seed={seed}\n{"="*60}')

        model = _train(
            train_data, val_data, nfeat, nclass, seed,
            os.path.join(args.save_dir, f'{args.dataset}_gpn_seed{seed}.pth'))

        run_res = {}
        probs_id, u_id = _infer(model, id_data)
        labels_id = id_data.y.cpu().numpy()
        r_id = compute_split_metrics(probs_id, u_id, labels_id, nclass)
        print(f'  ID-test ({id_dom}) | acc={r_id["acc"]:.4f} ece={r_id["ece"]:.4f}')
        run_res['ID-test'] = r_id

        for dom, data_ood in zip(ood_doms, ood_datas):
            probs_ood, u_ood = _infer(model, data_ood)
            labels_ood = data_ood.y.cpu().numpy()
            r_ood = compute_split_metrics(probs_ood, u_ood, labels_ood, nclass)
            r_ood = add_cross_split_metrics(r_id, r_ood, u_id, u_ood)
            name  = f'OOD-{dom}'
            print(f'  {name:12s} | ood_auroc={r_ood["ood_auroc"]:.4f}')
            run_res[name] = r_ood

        all_runs.append(run_res)
        print(f'  elapsed {time.time()-t0:.1f}s')

    pref = os.path.join(args.save_dir, f'{args.dataset}_gpn')
    summarize(all_runs, split_names, all_keys, pref + '_results.csv',
              f'{args.dataset.capitalize()} — GPN',
              reliability_path=pref + '_reliability.csv',
              uncertainty_path=pref + '_uncertainty.csv')


if __name__ == '__main__':
    main()