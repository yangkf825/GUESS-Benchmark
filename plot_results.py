"""
plot_results.py — 结果可视化入口
===================================
读取 results/ 下各算法输出的 CSV，生成论文图表。

用法:
    python plot_results.py --data_dir ./results --out_dir ./figures --dataset elliptic
    python plot_results.py --data_dir ./results --out_dir ./figures --dataset arxiv
    python plot_results.py --data_dir ./results --out_dir ./figures --dataset cora
    python plot_results.py --data_dir ./results --out_dir ./figures --dataset amazon
"""

import os, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./results')
parser.add_argument('--out_dir',  type=str, default='./figures')
parser.add_argument('--dataset',  type=str, default='elliptic',
                    choices=['elliptic', 'arxiv', 'cora', 'amazon'])
parser.add_argument('--dpi',      type=int, default=300)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
DS = args.dataset

# ── 算法名映射（CSV tag → 显示名）──────────────────────────
ALGO_TAG_TO_DISPLAY = {
    'ungnn':         'S-BGCN-T-K',
    'gats':          'GATS',
    'cagcn':         'CaGCN',
    'calgnn_Uncal':  'Vanilla',
    'calgnn_RBS':    'RBS',
    'gpn':           'GPN',
    'gduq':          'G-ΔUQ',
}

PALETTE = {
    'Vanilla':    '#1f77b4',
    'CaGCN':      '#ff7f0e',
    'GATS':       '#2ca02c',
    'RBS':        '#d62728',
    'S-BGCN-T-K': '#9467bd',
    'GPN':        '#8c564b',
    'G-ΔUQ':      '#e377c2',
}

MARKERS = {
    'Vanilla':    'o', 'CaGCN': 's', 'GATS': '^',
    'RBS':        'v', 'S-BGCN-T-K': 'D', 'GPN': 'P', 'G-ΔUQ': 'X',
}

# OOD split 顺序（按 shift 强度）
OOD_ORDER = {
    'elliptic': [f'OOD-test_{i}' for i in range(9)],
    'arxiv':    ['OOD-test_0(2014-2016)', 'OOD-test_1(2016-2018)', 'OOD-test_2(2018-2020)'],
    'cora':     [f'OOD-test_{i}(env{i+2})' for i in [3,7,0,1,6,8,4,5,2]],
    'amazon':   [f'OOD-test_{i}(env{i+2})' for i in [4,6,1,7,0,2,5,3,8]],
}


# ── 加载 CSV ──────────────────────────────────────────────────

def _find_csv(alg_dir, ds, tag):
    """在 results/{tag}/ 下找到对应数据集的 CSV"""
    d = os.path.join(args.data_dir, tag)
    if not os.path.isdir(d):
        return None
    for fn in os.listdir(d):
        if fn.endswith('_results.csv') and ds in fn:
            return os.path.join(d, fn)
    return None


def load_all():
    """返回 {display_name: DataFrame}"""
    dfs = {}
    for tag, name in ALGO_TAG_TO_DISPLAY.items():
        path = _find_csv(args.data_dir, DS, tag)
        if path is None:
            # CalGNN 的两个方法放在同一目录
            base_tag = tag.split('_')[0]
            path = _find_csv(args.data_dir, base_tag, DS)
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            dfs[name] = df
            print(f'  Loaded {name}: {path}')
        else:
            print(f'  [skip] {name}: no CSV found for dataset={DS}')
    return dfs


def get_val(df, split, col):
    """从 DataFrame 中取指定 split 和指标的均值"""
    col_mean = col + '_mean'
    if col_mean not in df.columns:
        return float('nan')
    rows = df[df['split'] == split]
    if rows.empty:
        return float('nan')
    return float(rows[col_mean].iloc[0])


def savefig(name):
    path = os.path.join(args.out_dir, f'{DS}_{name}.pdf')
    plt.savefig(path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {path}')


# ── 图1: ID-test 校准对比（ECE bar chart）───────────────────

def plot_calibration_bar(dfs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    metrics = ['ece', 'nll']
    titles  = ['ECE (↓)', 'NLL (↓)']
    algos   = [n for n in ALGO_TAG_TO_DISPLAY.values() if n in dfs]

    for ax, metric, title in zip(axes, metrics, titles):
        vals = [get_val(dfs[a], 'ID-test', metric) for a in algos]
        colors = [PALETTE.get(a, '#333') for a in algos]
        bars = ax.bar(algos, vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(metric.upper())
        ax.set_xticklabels(algos, rotation=30, ha='right', fontsize=9)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle(f'{DS.capitalize()} — ID-test Calibration', fontsize=13)
    plt.tight_layout()
    savefig('fig_calibration_bar')


# ── 图2: OOD ECE 趋势线 ──────────────────────────────────────

def plot_ood_trend(dfs, metric='ece', ylabel='ECE (↓)'):
    ood_splits = OOD_ORDER.get(DS, [])
    # filter to splits actually in the data
    if dfs:
        sample_df = next(iter(dfs.values()))
        avail = set(sample_df['split'].tolist())
        ood_splits = [s for s in ood_splits if any(s in av for av in avail)]

    fig, ax = plt.subplots(figsize=(9, 4))
    algos = [n for n in ALGO_TAG_TO_DISPLAY.values() if n in dfs]
    for name in algos:
        df = dfs[name]
        ys = []
        for sp in ood_splits:
            # fuzzy match
            rows = df[df['split'].str.contains(sp.split('(')[0], regex=False)]
            if rows.empty:
                ys.append(float('nan'))
            else:
                col = metric + '_mean'
                ys.append(float(rows[col].iloc[0]) if col in rows.columns else float('nan'))
        xs = list(range(len(ood_splits)))
        ax.plot(xs, ys, marker=MARKERS.get(name, 'o'),
                color=PALETTE.get(name, '#333'), label=name, linewidth=1.5)

    ax.set_xticks(range(len(ood_splits)))
    ax.set_xticklabels([s.split('(')[0] for s in ood_splits],
                       rotation=30, ha='right', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{DS.capitalize()} — OOD {metric.upper()} trend')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    savefig(f'fig_ood_{metric}_trend')


# ── 图3: UE-AUROC 对比 ───────────────────────────────────────

def plot_ue_auroc(dfs):
    algos = [n for n in ALGO_TAG_TO_DISPLAY.values() if n in dfs]
    splits = ['ID-test'] + [s for s in OOD_ORDER.get(DS, [])
                             if any(s in sp for sp in
                                    next(iter(dfs.values()))['split'].tolist())][:3]

    fig, axes = plt.subplots(1, len(splits), figsize=(4 * len(splits), 4), sharey=True)
    if len(splits) == 1:
        axes = [axes]

    for ax, sp in zip(axes, splits):
        vals   = [get_val(dfs[a], sp, 'ue_auroc') for a in algos]
        colors = [PALETTE.get(a, '#333') for a in algos]
        ax.bar(algos, vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(sp[:20], fontsize=9)
        ax.set_xticklabels(algos, rotation=45, ha='right', fontsize=7)
        ax.set_ylim(0, 1)
        if ax == axes[0]:
            ax.set_ylabel('UE-AUROC (↑)')

    fig.suptitle(f'{DS.capitalize()} — UE-AUROC', fontsize=12)
    plt.tight_layout()
    savefig('fig_ue_auroc')


# ── 图4: AURC (selective classification) ─────────────────────

def plot_aurc(dfs):
    algos = [n for n in ALGO_TAG_TO_DISPLAY.values() if n in dfs]
    vals  = [get_val(dfs[a], 'ID-test', 'aurc') for a in algos]
    colors = [PALETTE.get(a, '#333') for a in algos]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(algos, vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('AURC (↓)')
    ax.set_title(f'{DS.capitalize()} — Selective Classification (AURC, ID-test)')
    ax.set_xticklabels(algos, rotation=30, ha='right')
    plt.tight_layout()
    savefig('fig_aurc')


# ── 图5: OOD-AUROC 趋势 ──────────────────────────────────────

def plot_ood_auroc(dfs):
    plot_ood_trend(dfs, metric='ood_auroc', ylabel='OOD-AUROC (↑)')
    # rename
    src = os.path.join(args.out_dir, f'{DS}_fig_ood_ood_auroc_trend.pdf')
    dst = os.path.join(args.out_dir, f'{DS}_fig_ood_auroc_trend.pdf')
    if os.path.exists(src):
        os.rename(src, dst)


# ── 主入口 ────────────────────────────────────────────────────

def main():
    print(f'\n[plot_results] dataset={DS}  data_dir={args.data_dir}')
    dfs = load_all()
    if not dfs:
        print('[ERROR] No result CSVs found. Run experiments first.')
        return

    print(f'\nGenerating figures → {args.out_dir}')
    plot_calibration_bar(dfs)
    plot_ood_trend(dfs, metric='ece',      ylabel='ECE (↓)')
    plot_ood_trend(dfs, metric='delta_ece', ylabel='Δ-ECE (↓)')
    plot_ue_auroc(dfs)
    plot_aurc(dfs)
    plot_ood_auroc(dfs)
    print('\nDone.')


if __name__ == '__main__':
    main()
