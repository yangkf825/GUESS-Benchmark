"""
gnn_uq_bench.datasets
=====================
数据加载：Elliptic / OGB-Arxiv / EERM-Cora / EERM-Amazon

两套 adj 格式：
  - sparse_adj  : torch.sparse_coo_tensor（GCN/CaGCN 用，稀疏矩阵乘）
  - edge_index  : (2, E) torch.LongTensor  （PyG GCN/GAT/GATS/GPN/GDUQ 用）
各加载函数均返回两者，调用方按需取用。
"""

import os
import pickle

import numpy as np
import scipy.sparse as sp
import torch

# ── 时序 / 环境 分割常量 ─────────────────────────────────────
ELLIPTIC_TRAIN = list(range(7,  12))
ELLIPTIC_VAL   = list(range(12, 17))
ELLIPTIC_TESTS = [
    list(range(17, 21)), list(range(21, 25)), list(range(25, 29)),
    list(range(29, 33)), list(range(33, 37)), list(range(37, 41)),
    list(range(41, 44)), list(range(44, 47)), list(range(47, 49)),
]

ARXIV_TRAIN_YEAR   = 2013
ARXIV_OODVAL_YEARS = (2014, 2015)
ARXIV_TESTS        = [(2014, 2016), (2016, 2018), (2018, 2020)]


# ────────────────────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────────────────────

def _sym_norm_sparse(adj_sp, N, device):
    """scipy CSR → 对称归一化 torch sparse (D^{-1/2} A D^{-1/2})，加自环"""
    adj_sp = (adj_sp + sp.eye(N)).tocoo().astype(np.float32)
    rowsum = np.array(adj_sp.sum(1)).flatten()
    d_inv  = np.power(rowsum, -0.5); d_inv[np.isinf(d_inv)] = 0.
    D      = sp.diags(d_inv)
    An     = (D @ adj_sp @ D).tocoo()
    idx    = torch.from_numpy(np.vstack([An.row, An.col]).astype(np.int64))
    val    = torch.from_numpy(An.data)
    return torch.sparse_coo_tensor(idx, val, (N, N), dtype=torch.float32).to(device)


def _make_undirected_ei(edge_index_np, N):
    """numpy (2, E) → 无向 edge_index (2, E') torch.LongTensor"""
    src, dst = edge_index_np[0], edge_index_np[1]
    ei = np.unique(np.vstack([np.concatenate([src, dst]),
                               np.concatenate([dst, src])]), axis=1)
    return torch.from_numpy(ei.astype(np.int64))


def _adj_sp_from_ei_np(edge_index_np, N):
    """numpy edge_index → scipy CSR（已去重）"""
    src, dst = edge_index_np[0], edge_index_np[1]
    su = np.concatenate([src, dst]); du = np.concatenate([dst, src])
    A  = sp.coo_matrix((np.ones(len(su), np.float32), (su, du)), shape=(N, N)).tocsr()
    A.data[:] = 1.
    return A


def _row_norm(feat_np):
    rs = feat_np.sum(1, keepdims=True); rs[rs == 0] = 1.
    return (feat_np / rs).astype(np.float32)


def stratified_split(labels_np, base_mask_np, val_ratio, test_ratio, seed):
    """
    在 base_mask 内做分层 train/val/test 划分。
    labels_np, base_mask_np : numpy (N,)
    返回 (train_mask, val_mask, test_mask) numpy bool (N,)
    """
    rng      = np.random.default_rng(seed)
    N        = len(labels_np)
    base_idx = base_mask_np.nonzero()[0]
    base_lab = labels_np[base_idx]
    val_lst, test_lst, train_lst = [], [], []
    for cls in np.unique(base_lab):
        if cls < 0: continue
        idx  = base_idx[base_lab == cls]
        perm = rng.permutation(len(idx)); idx = idx[perm]
        n    = len(idx)
        nv   = max(1, int(n * val_ratio))
        nt   = max(1, int(n * test_ratio))
        while nv + nt >= n and (nv > 1 or nt > 1):
            if nv >= nt: nv -= 1
            else:        nt -= 1
        if nv + nt >= n: nv = nt = 0
        val_lst  .extend(idx[:nv].tolist())
        test_lst .extend(idx[nv:nv + nt].tolist())
        train_lst.extend(idx[nv + nt:].tolist())

    def to_mask(lst):
        m = np.zeros(N, dtype=bool)
        if lst: m[np.array(lst)] = True
        return m

    return to_mask(train_lst), to_mask(val_lst), to_mask(test_lst)


# ────────────────────────────────────────────────────────────────────────────
# Elliptic Bitcoin
# ────────────────────────────────────────────────────────────────────────────

def _load_elliptic_step(step, data_dir):
    with open(os.path.join(data_dir, f'{step}.pkl'), 'rb') as f:
        d = pickle.load(f)
    return d[0], np.array(d[1]), np.array(d[2], dtype=np.float32)


def load_elliptic(steps, data_dir, device, row_normalize=True):
    """
    合并多个时间步 → 块对角图。
    返回:
      sparse_adj  : torch sparse (N, N)
      edge_index  : torch LongTensor (2, E)
      features    : torch FloatTensor (N, F)
      labels      : torch LongTensor (N,)
      labels_np   : numpy (N,)
      N           : int
    """
    adjs, labs, feats = [], [], []
    for s in steps:
        adj_s, lab, feat = _load_elliptic_step(s, data_dir)
        adjs.append(adj_s); labs.append(lab); feats.append(feat)

    adj_csr = sp.block_diag(adjs, format='csr')
    lab_np  = np.concatenate(labs)
    feat_np = np.concatenate(feats)
    N       = feat_np.shape[0]

    if row_normalize:
        feat_np = _row_norm(feat_np)

    sparse_adj = _sym_norm_sparse(adj_csr, N, device)

    coo = adj_csr.tocoo()
    ei_np = np.unique(np.vstack([np.concatenate([coo.row, coo.col]),
                                  np.concatenate([coo.col, coo.row])]), axis=1)
    edge_index = torch.from_numpy(ei_np.astype(np.int64)).to(device)

    features = torch.tensor(feat_np, dtype=torch.float32).to(device)
    labels   = torch.tensor(lab_np,  dtype=torch.long).to(device)

    return sparse_adj, edge_index, features, labels, lab_np, N


# ────────────────────────────────────────────────────────────────────────────
# OGB-Arxiv
# ────────────────────────────────────────────────────────────────────────────

def load_arxiv(data_path, device, row_normalize=True):
    """
    返回:
      sparse_adj, edge_index, features, labels, labels_np, node_year, nclass, N
    """
    with open(data_path, 'rb') as f:
        raw = pickle.load(f)
    graph  = raw['graph']  if isinstance(raw, dict) else raw[0]
    labels = raw['labels'] if isinstance(raw, dict) else raw[1]

    def _get(d, *keys):
        for k in keys:
            if k in d: return d[k]
        raise KeyError(keys)

    edge_index_np = np.array(_get(graph, 'edge_index'), dtype=np.int64)
    node_feat     = np.array(_get(graph, 'node_feat', 'node_feature', 'x'), dtype=np.float32)
    node_year     = np.array(_get(graph, 'node_year', 'year')).flatten().astype(np.int64)
    N             = int(_get(graph, 'num_nodes', 'num_node'))
    labels_np     = np.array(labels).flatten().astype(np.int64)
    nclass        = int(labels_np.max()) + 1
    print(f'  Arxiv: N={N}  nclass={nclass}  nfeat={node_feat.shape[1]}')

    if row_normalize:
        node_feat = _row_norm(node_feat)

    adj_csr    = _adj_sp_from_ei_np(edge_index_np, N)
    sparse_adj = _sym_norm_sparse(adj_csr, N, device)
    edge_index = _make_undirected_ei(edge_index_np, N).to(device)
    features   = torch.tensor(node_feat,  dtype=torch.float32).to(device)
    labels_t   = torch.tensor(labels_np,  dtype=torch.long).to(device)

    return sparse_adj, edge_index, features, labels_t, labels_np, node_year, nclass, N


# ────────────────────────────────────────────────────────────────────────────
# EERM-Cora / EERM-Amazon-Photo
# ────────────────────────────────────────────────────────────────────────────

def load_eerm(eerm_root, dataset, device):
    """
    加载 EERM 格式数据集（Cora / Amazon-Photo）。
    格式：gen/{i}-gcn.pkl = (x_tensor, y_tensor)，环境 0=train 1=val 2~9=OOD

    返回:
      sparse_adj   : torch sparse (N, N)
      edge_index   : torch LongTensor (2, E)
      feat_envs    : list[10] of FloatTensor (N, F)，env0~9
      labels_t     : torch LongTensor (N,)
      labels_np    : numpy (N,)
      nclass       : int
      tr_mask      : torch bool (N,)
      val_mask     : torch bool (N,)
      test_mask    : torch bool (N,)
      ood_names    : list[8] of str
      N            : int
    """
    gen_dir    = os.path.join(eerm_root, 'gen')
    feat_envs  = []
    labels_np  = None

    for i in range(10):
        with open(os.path.join(gen_dir, f'{i}-gcn.pkl'), 'rb') as f:
            d = pickle.load(f)
        x = d[0].detach().numpy() if isinstance(d[0], torch.Tensor) else np.array(d[0])
        y = d[1].detach().numpy() if isinstance(d[1], torch.Tensor) else np.array(d[1])
        feat_envs.append(torch.tensor(x.astype(np.float32), dtype=torch.float32).to(device))
        if labels_np is None:
            labels_np = y.astype(np.int64)

    N      = feat_envs[0].shape[0]
    nclass = int(labels_np.max()) + 1
    print(f'  EERM-{dataset}: N={N}  nclass={nclass}  nfeat={feat_envs[0].shape[1]}')

    # 读边和 split
    raw_dir = os.path.join(eerm_root, 'raw')
    if dataset == 'cora':
        def _lr(name):
            with open(os.path.join(raw_dir, f'ind.cora.{name}'), 'rb') as f:
                return pickle.load(f, encoding='latin1')
        allx  = _lr('allx'); y_raw = _lr('y'); graph = _lr('graph')
        with open(os.path.join(raw_dir, 'ind.cora.test.index')) as f:
            test_idx = np.array([int(i) for i in f.read().split()])
        rows, cols = [], []
        for s, ds in graph.items():
            for d in ds:
                rows.append(s); cols.append(d)
        edge_index_np = np.vstack([rows, cols]).astype(np.int64)
        train_idx = np.arange(y_raw.shape[0])
        val_idx   = np.arange(allx.shape[0], allx.shape[0] + 500)
    else:  # amazon
        npz = np.load(os.path.join(raw_dir, 'amazon_electronics_photo.npz'), allow_pickle=True)
        A   = sp.csr_matrix((npz['adj_data'], npz['adj_indices'], npz['adj_indptr']),
                             shape=tuple(npz['adj_shape'])).tocoo()
        edge_index_np = np.vstack([A.row, A.col]).astype(np.int64)
        rng = np.random.default_rng(42)
        tr  = []
        for c in np.unique(labels_np):
            ic = np.where(labels_np == c)[0]
            tr.extend(rng.choice(ic, min(20, len(ic)), replace=False).tolist())
        train_idx = np.array(tr)
        rest = np.setdiff1d(np.arange(N), train_idx); rng.shuffle(rest)
        val_idx  = rest[:500]; test_idx = rest[500:1500]

    adj_csr    = _adj_sp_from_ei_np(edge_index_np, N)
    sparse_adj = _sym_norm_sparse(adj_csr, N, device)
    edge_index = _make_undirected_ei(edge_index_np, N).to(device)

    def idx2mask(idx):
        m = torch.zeros(N, dtype=torch.bool); m[idx] = True; return m

    labels_t  = torch.tensor(labels_np, dtype=torch.long).to(device)
    tr_mask   = idx2mask(train_idx).to(device)
    val_mask  = idx2mask(val_idx).to(device)
    test_mask = idx2mask(test_idx).to(device)
    ood_names = [f'OOD-test_{i}(env{i+2})' for i in range(8)]

    return (sparse_adj, edge_index, feat_envs, labels_t, labels_np,
            nclass, tr_mask, val_mask, test_mask, ood_names, N)
