"""
gnn_uq_bench.datasets.facebook_twitch
======================================
Facebook100 和 Twitch 数据集加载器。

数据格式：
  Facebook100 — .mat 文件（local_info 矩阵 + 邻接矩阵 A）
  Twitch      — musae_<DOM>_target.csv / _edges.csv / _features.json

域分割（跨域 OOD）：
  Facebook100: train=8个学校 / val=3个 / test=3个
  Twitch:      train=ES/FR/PTBR/RU / val=DE / test=ENGB/TW

返回接口统一为 PyG Data 对象，保持与其他数据集一致。
"""

import os
import json
import warnings

import numpy as np
import scipy.io as sio
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── 域分割常量 ────────────────────────────────────────────────
DOMAIN_SETTINGS = {
    'facebook': {
        'train': ['Amherst41', 'Brandeis99', 'Brown11', 'Carnegie49',
                  'Cornell5', 'Johns Hopkins55', 'Princeton12', 'WashU32'],
        'val':   ['Bingham82', 'Texas80', 'Yale4'],
        'test':  ['Caltech36', 'Duke14', 'Penn94'],
    },
    'twitch': {
        'train': ['ES', 'FR', 'PTBR', 'RU'],
        'val':   ['DE'],
        'test':  ['ENGB', 'TW'],
    },
}

TWITCH_FEAT_DIM = 3170   # musae Twitch 特征维度（固定）


# ── 全局标签映射（保证跨域一致）────────────────────────────────

def get_global_label_map(dataset: str, data_root: str) -> dict:
    """
    扫描所有域，建立统一的 label_value → class_index 映射。
    Facebook100: 取 local_info[:,0] 中 0 < v < 1000 的学生年级标签。
    Twitch:      取 'mature' 列的二值标签 {0, 1}。
    """
    all_doms = (DOMAIN_SETTINGS[dataset]['train'] +
                DOMAIN_SETTINGS[dataset]['val']   +
                DOMAIN_SETTINGS[dataset]['test'])
    unique_labels = set()
    for dom in all_doms:
        if dataset == 'facebook':
            mat   = sio.loadmat(os.path.join(data_root, 'facebook100', f'{dom}.mat'))
            y_raw = mat['local_info'][:, 0].astype(np.int64)
            unique_labels.update(y_raw[(y_raw > 0) & (y_raw < 1000)].tolist())
        else:
            df = pd.read_csv(
                os.path.join(data_root, 'twitch', dom, f'musae_{dom}_target.csv'))
            unique_labels.update(df['mature'].unique().tolist())
    return {val: i for i, val in enumerate(sorted(unique_labels))}


# ── 单域数据加载 ───────────────────────────────────────────────

def load_domain(dataset: str, dom: str, data_root: str,
                label_map: dict, scaler=None, device=None) -> Data:
    """
    加载一个域的图数据，返回 PyG Data 对象（在 CPU 上）。

    Parameters
    ----------
    dataset   : 'facebook' 或 'twitch'
    dom       : 域名称，如 'DE' / 'Caltech36'
    data_root : 数据根目录
    label_map : get_global_label_map() 返回的映射
    scaler    : 可选 StandardScaler（Facebook100 需要）
    device    : 不传则返回 CPU 张量，调用方自行 .to(device)
    """
    if dataset == 'facebook':
        mat = sio.loadmat(os.path.join(data_root, 'facebook100', f'{dom}.mat'))
        x   = mat['local_info'][:, 1:].astype(np.float32)
        if scaler is not None:
            x = scaler.transform(x)
        y_raw = mat['local_info'][:, 0].astype(np.int64)
        y     = torch.tensor([label_map.get(int(v), 0) for v in y_raw],
                              dtype=torch.long)
        row, col = mat['A'].nonzero()
        ei = to_undirected(
            torch.stack([torch.from_numpy(row.copy()),
                         torch.from_numpy(col.copy())], dim=0),
            num_nodes=x.shape[0])
        data = Data(x=torch.from_numpy(x), y=y, edge_index=ei)

    else:   # twitch
        d_dir = os.path.join(data_root, 'twitch', dom)
        y = torch.tensor(
            pd.read_csv(os.path.join(d_dir, f'musae_{dom}_target.csv'))
            ['mature'].values.astype(np.int64),
            dtype=torch.long)
        edges = pd.read_csv(
            os.path.join(d_dir, f'musae_{dom}_edges.csv')).values.T.astype(np.int64)
        ei = to_undirected(
            torch.from_numpy(edges), num_nodes=y.size(0))
        with open(os.path.join(d_dir, f'musae_{dom}_features.json'), 'r') as f:
            feats = json.load(f)
        x = torch.zeros((y.size(0), TWITCH_FEAT_DIM))
        for node_id, feat_list in feats.items():
            idx = int(node_id)
            if idx < y.size(0):
                x[idx, feat_list] = 1.0
        x = x / x.sum(1, keepdim=True).clamp(min=1e-8)
        data = Data(x=x, y=y, edge_index=ei)

    if device is not None:
        data = data.to(device)
    return data


# ── 完整数据集加载（train / val / test 三组）──────────────────

def load_facebook_twitch(dataset: str, data_root: str, device=None):
    """
    一次性加载全部域数据。

    Returns
    -------
    label_map  : dict — 全局标签映射
    nclass     : int  — 类别数
    nfeat      : int  — 特征维度
    train_data : list[Data] — 训练域（ID 域）
    val_data   : list[Data] — 验证域（校准用）
    test_data  : list[Data] — 测试域（OOD）
    domain_names: dict{'train','val','test'} — 各组域名称列表
    scaler     : StandardScaler 或 None
    """
    label_map = get_global_label_map(dataset, data_root)
    nclass    = len(label_map)

    scaler = None
    if dataset == 'facebook':
        scaler = StandardScaler()
        all_tr_x = [
            sio.loadmat(os.path.join(data_root, 'facebook100', f'{d}.mat'))
            ['local_info'][:, 1:].astype(np.float32)
            for d in DOMAIN_SETTINGS['facebook']['train']
        ]
        scaler.fit(np.concatenate(all_tr_x, axis=0))

    def _load_all(split):
        return [load_domain(dataset, d, data_root, label_map, scaler, device)
                for d in DOMAIN_SETTINGS[dataset][split]]

    train_data = _load_all('train')
    val_data   = _load_all('val')
    test_data  = _load_all('test')

    nfeat = train_data[0].x.size(1)

    domain_names = {
        'train': DOMAIN_SETTINGS[dataset]['train'],
        'val':   DOMAIN_SETTINGS[dataset]['val'],
        'test':  DOMAIN_SETTINGS[dataset]['test'],
    }

    print(f'  {dataset}: nclass={nclass}  nfeat={nfeat}')
    print(f'  train={[d for d in domain_names["train"]]}')
    print(f'  val  ={domain_names["val"]}  test={domain_names["test"]}')

    return label_map, nclass, nfeat, train_data, val_data, test_data, domain_names, scaler
