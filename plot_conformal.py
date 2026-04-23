"""
绘制共形预测对比图
==================
图1：Coverage vs Efficiency (Set Size)
图2：Coverage vs Singleton Hit Ratio
图3：SHR vs Set Size 散点图（每个点 = 一个 alpha 值）

使用方法：
    python plot_conformal.py \
        --cfgnn_dir    ./results/cfgnn \
        --confgnn_dir  ./results/confgnn \
        --graphcp_dir  ./results/graph_cp \
        --daps_dir     ./results/daps \
        --split        all \
        --out          ./figures
"""

import os, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── 参数 ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--cfgnn_dir',   type=str, default='./results/cfgnn')
parser.add_argument('--confgnn_dir', type=str, default='./results/confgnn')
parser.add_argument('--graphcp_dir', type=str, default='./results/graph_cp')
parser.add_argument('--daps_dir',    type=str, default='./results/daps')
parser.add_argument('--split',       type=str, default='all')
parser.add_argument('--out',         type=str, default='./figures')
parser.add_argument('--dataset',     type=str, default='elliptic')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ── 样式 ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         12,
    'axes.linewidth':    1.2,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'legend.framealpha': 0.9,
    'legend.fontsize':   11,
})

METHODS = {
    'APS':    {'color': '#4C72B0', 'ls': '-',  'marker': 'o', 'ms': 6},
    'RAPS':   {'color': '#DD8452', 'ls': '-',  'marker': 's', 'ms': 6},
    'CF-GNN': {'color': '#55A868', 'ls': '-',  'marker': '^', 'ms': 7},
    'NAPS':   {'color': '#C44E52', 'ls': '-',  'marker': 'D', 'ms': 6},
    'DAPS':   {'color': '#8172B2', 'ls': '-',  'marker': 'P', 'ms': 7},
}

NEAR_OOD = ('OOD-test_0', 'OOD-test_1', 'OOD-test_2')
FAR_OOD  = ('OOD-test_3', 'OOD-test_4', 'OOD-test_5',
             'OOD-test_6', 'OOD-test_7', 'OOD-test_8')


# ════════════════════════════════════════════════════════════════════════
# 1. 数据加载
# ════════════════════════════════════════════════════════════════════════

def _extract_alpha(fname):
    m = re.search(r'_a([\d]+[_\.][\d]+)', fname)
    if m:
        return float(m.group(1).replace('_', '.'))
    return None


def _load_dir(directory, suffix, method_name):
    rows = []
    if not os.path.isdir(directory):
        print(f'  [警告] 目录不存在: {directory}')
        return pd.DataFrame()
    for fname in sorted(os.listdir(directory)):
        if not (fname.endswith('.csv') and suffix in fname):
            continue
        alpha = _extract_alpha(fname)
        if alpha is None:
            continue
        try:
            df = pd.read_csv(os.path.join(directory, fname))
            df['alpha']  = alpha
            df['method'] = method_name
            rows.append(df)
        except Exception as e:
            print(f'  [警告] 读取失败 {fname}: {e}')
    if not rows:
        print(f'  [警告] {directory} 中未找到匹配 "{suffix}" 的文件')
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_all():
    dfs = []
    for d, suffix, name in [
        (args.cfgnn_dir,   'cfgnn_aps',  'APS'),
        (args.cfgnn_dir,   'cfgnn_raps', 'RAPS'),
        (args.confgnn_dir, 'confgnn_aps','CF-GNN'),
        (args.graphcp_dir, 'weighted',   'NAPS'),
        (args.daps_dir,    'daps',       'DAPS'),
    ]:
        df = _load_dir(d, suffix, name)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        raise RuntimeError('未加载到任何数据，请检查目录路径')
    return pd.concat(dfs, ignore_index=True)


# ════════════════════════════════════════════════════════════════════════
# 2. 聚合工具
# ════════════════════════════════════════════════════════════════════════

METRIC_COLS = ['coverage_mean', 'set_size_mean', 'shr_mean',
               'coverage_std',  'set_size_std',  'shr_std']


def agg_by_method_alpha(df):
    """对每个 (method, alpha) 取所有 split 的均值"""
    avail = [c for c in METRIC_COLS if c in df.columns]
    return (df.groupby(['method', 'alpha'])[avail]
              .mean().reset_index()
              .sort_values(['method', 'alpha']))


def filter_splits(df, mode):
    if mode == 'all':
        return df
    elif mode == 'ID-test':
        return df[df['split'] == 'ID-test']
    elif mode == 'OOD':
        return df[df['split'] != 'ID-test']
    elif mode == 'near_ood':
        return df[df['split'].str.startswith(NEAR_OOD)]
    elif mode == 'far_ood':
        return df[df['split'].str.startswith(FAR_OOD)]
    else:
        return df[df['split'] == mode]


# ════════════════════════════════════════════════════════════════════════
# 3. 图1：Coverage vs Efficiency
# ════════════════════════════════════════════════════════════════════════

def plot_coverage_efficiency(agg, out_path, title=''):
    fig, ax = plt.subplots(figsize=(6, 5))
    for method, style in METHODS.items():
        sub = agg[agg['method'] == method].sort_values('coverage_mean')
        if sub.empty:
            continue
        ax.plot(sub['coverage_mean'], sub['set_size_mean'],
                color=style['color'], ls=style['ls'],
                marker=style['marker'], ms=style['ms'],
                label=method, linewidth=2, zorder=3)
        if 'set_size_std' in sub.columns:
            ax.fill_between(sub['coverage_mean'],
                            sub['set_size_mean'] - sub['set_size_std'],
                            sub['set_size_mean'] + sub['set_size_std'],
                            color=style['color'], alpha=0.12)
    ax.set_xlabel('Coverage', fontsize=13)
    ax.set_ylabel('Efficiency (Set Size)', fontsize=13)
    ax.set_xlim(0.75, 1.00)
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  图1 → {out_path}')


# ════════════════════════════════════════════════════════════════════════
# 4. 图2：Coverage vs SHR
# ════════════════════════════════════════════════════════════════════════

def plot_coverage_shr(agg, out_path, title=''):
    fig, ax = plt.subplots(figsize=(6, 5))
    for method, style in METHODS.items():
        sub = agg[agg['method'] == method].sort_values('coverage_mean')
        if sub.empty:
            continue
        ax.plot(sub['coverage_mean'], sub['shr_mean'],
                color=style['color'], ls=style['ls'],
                marker=style['marker'], ms=style['ms'],
                label=method, linewidth=2, zorder=3)
        if 'shr_std' in sub.columns:
            ax.fill_between(sub['coverage_mean'],
                            sub['shr_mean'] - sub['shr_std'],
                            sub['shr_mean'] + sub['shr_std'],
                            color=style['color'], alpha=0.12)
    ax.set_xlabel('Coverage', fontsize=13)
    ax.set_ylabel('Singleton Hit Ratio', fontsize=13)
    ax.set_xlim(0.75, 1.00)
    ax.set_ylim(bottom=0)
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  图2 → {out_path}')


# ════════════════════════════════════════════════════════════════════════
# 5. 图1+图2 并排
# ════════════════════════════════════════════════════════════════════════

def plot_combined(agg, out_path, title=''):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method, style in METHODS.items():
        sub = agg[agg['method'] == method].sort_values('coverage_mean')
        if sub.empty:
            continue
        kw = dict(color=style['color'], ls=style['ls'],
                  marker=style['marker'], ms=style['ms'],
                  label=method, linewidth=2, zorder=3)
        axes[0].plot(sub['coverage_mean'], sub['set_size_mean'], **kw)
        axes[1].plot(sub['coverage_mean'], sub['shr_mean'],      **kw)
        if 'set_size_std' in sub.columns:
            axes[0].fill_between(sub['coverage_mean'],
                                 sub['set_size_mean'] - sub['set_size_std'],
                                 sub['set_size_mean'] + sub['set_size_std'],
                                 color=style['color'], alpha=0.12)
        if 'shr_std' in sub.columns:
            axes[1].fill_between(sub['coverage_mean'],
                                 sub['shr_mean'] - sub['shr_std'],
                                 sub['shr_mean'] + sub['shr_std'],
                                 color=style['color'], alpha=0.12)

    axes[0].set_xlabel('Coverage', fontsize=13)
    axes[0].set_ylabel('Efficiency (Set Size)', fontsize=13)
    axes[0].set_xlim(0.75, 1.00)
    axes[0].set_title('Coverage vs Efficiency', fontsize=13)
    axes[0].legend(loc='upper left')

    axes[1].set_xlabel('Coverage', fontsize=13)
    axes[1].set_ylabel('Singleton Hit Ratio', fontsize=13)
    axes[1].set_xlim(0.75, 1.00)
    axes[1].set_ylim(bottom=0)
    axes[1].set_title('Coverage vs Singleton Hit Ratio', fontsize=13)
    axes[1].legend(loc='upper right')

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  合并图 → {out_path}')


# ════════════════════════════════════════════════════════════════════════
# 6. 图3：SHR vs Set Size 散点图
# ════════════════════════════════════════════════════════════════════════

def plot_shr_vs_setsize(df, out_path, title=''):
    """
    每个点 = 一个 alpha 值下所有 OOD split 的均值
    连线表示随 alpha 变化的轨迹（从低覆盖率到高覆盖率）
    """
    ood_df = df[df['split'] != 'ID-test'].copy()
    if ood_df.empty:
        print('  [警告] 无 OOD 数据，跳过散点图')
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    for method, style in METHODS.items():
        sub = ood_df[ood_df['method'] == method]
        if sub.empty:
            continue
        pts = (sub.groupby('alpha')[['shr_mean', 'set_size_mean']]
                  .mean().reset_index()
                  .sort_values('alpha', ascending=False))
        ax.scatter(pts['shr_mean'], pts['set_size_mean'],
                   color=style['color'], marker=style['marker'],
                   s=60, zorder=4, label=method)
        ax.plot(pts['shr_mean'], pts['set_size_mean'],
                color=style['color'], ls='-',
                linewidth=1.5, alpha=0.7, zorder=3)

    ax.set_xlabel('Singleton Hit Ratio', fontsize=13)
    ax.set_ylabel('Effective Set Size',  fontsize=13)
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  散点图 → {out_path}')


# ════════════════════════════════════════════════════════════════════════
# 7. 主入口
# ════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Loading CSVs...')
    df = load_all()
    print(f'  加载完成: {len(df)} 行, 方法: {df["method"].unique().tolist()}')
    print(f'  Alpha 值: {sorted(df["alpha"].unique().tolist())}')
    print(f'  Splits: {df["split"].unique().tolist()[:5]} ...')

    ds = args.dataset

    # ── 按 split 模式绘图 ──────────────────────────────────────────────
    print(f'\n绘图 (split={args.split})...')
    sub  = filter_splits(df, args.split)
    agg  = agg_by_method_alpha(sub)
    tag  = f'{ds}_{args.split}'
    plot_coverage_efficiency(agg, os.path.join(args.out, f'{tag}_efficiency.pdf'))
    plot_coverage_shr(       agg, os.path.join(args.out, f'{tag}_shr.pdf'))
    plot_combined(           agg, os.path.join(args.out, f'{tag}_combined.pdf'), title=args.split)

    # ── Near / Far OOD 分组 ────────────────────────────────────────────
    print('\n按 Near/Far OOD 分组绘图...')
    for mode, label in [('all',      'All Splits'),
                         ('ID-test',  'ID-test'),
                         ('near_ood', 'Near OOD (s17-28)'),
                         ('far_ood',  'Far OOD (s29-48)')]:
        sub = filter_splits(df, mode)
        if sub.empty:
            continue
        agg = agg_by_method_alpha(sub)
        tag = f'{ds}_{mode}'
        plot_combined(agg,
                      os.path.join(args.out, f'{tag}_combined.pdf'),
                      title=label)

    # ── 散点图（图3）──────────────────────────────────────────────────
    print('\n绘制 SHR vs Set Size 散点图...')
    plot_shr_vs_setsize(
        df,
        os.path.join(args.out, f'{ds}_shr_vs_setsize.pdf'),
        title='OOD Splits')

    near = filter_splits(df, 'near_ood')
    far  = filter_splits(df, 'far_ood')
    if not near.empty:
        plot_shr_vs_setsize(near,
                            os.path.join(args.out, f'{ds}_near_ood_shr_vs_setsize.pdf'),
                            title='Near OOD (s17-28)')
    if not far.empty:
        plot_shr_vs_setsize(far,
                            os.path.join(args.out, f'{ds}_far_ood_shr_vs_setsize.pdf'),
                            title='Far OOD (s29-48)')

    print('\n全部完成！')
