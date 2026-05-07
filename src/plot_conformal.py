"""
绘制共形预测对比图
==================
图1：Coverage vs Efficiency (Set Size)
图2：Coverage vs Singleton Hit Ratio

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
parser.add_argument('--split',       type=str, default='all',
                    help='使用哪些 split 的平均值: all / ID-test / OOD / 指定名称')
parser.add_argument('--out',         type=str, default='./figures')
parser.add_argument('--dataset',     type=str, default='elliptic')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ── 样式 ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      12,
    'axes.linewidth': 1.2,
    'axes.grid':      True,
    'grid.alpha':     0.3,
    'grid.linestyle': '--',
    'legend.framealpha': 0.9,
    'legend.fontsize': 11,
})

METHODS = {
    'APS':     {'color': '#4C72B0', 'ls': '-',  'marker': 'o', 'ms': 6},
    'RAPS':    {'color': '#DD8452', 'ls': '-',  'marker': 's', 'ms': 6},
    'CF-GNN':  {'color': '#55A868', 'ls': '-',  'marker': '^', 'ms': 7},
    'NAPS':    {'color': '#C44E52', 'ls': '-',  'marker': 'D', 'ms': 6},
    'DAPS':    {'color': '#8172B2', 'ls': '-',  'marker': 'P', 'ms': 7},
}


# ════════════════════════════════════════════════════════════════════════
# 1. 文件加载工具
# ════════════════════════════════════════════════════════════════════════

def _extract_alpha(fname):
    """从文件名中提取 alpha 值，例如 _a0.1_ 或 _a0_1_"""
    m = re.search(r'_a([\d]+[_\.][\d]+)', fname)
    if m:
        return float(m.group(1).replace('_', '.'))
    return None


def _load_dir(directory, suffix, method_name):
    """
    读取某个目录下所有匹配 suffix 的 CSV，
    按 alpha 汇总，返回 DataFrame。
    columns: alpha, split, coverage_mean, set_size_mean, shr_mean,
             coverage_std, set_size_std, shr_std
    """
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
        fpath = os.path.join(directory, fname)
        try:
            df = pd.read_csv(fpath)
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
    """加载五个算法的数据"""
    dfs = []

    # APS（cfgnn 目录，score=aps）
    d = _load_dir(args.cfgnn_dir, 'cfgnn_aps', 'APS')
    if not d.empty: dfs.append(d)

    # RAPS（cfgnn 目录，score=raps）
    d = _load_dir(args.cfgnn_dir, 'cfgnn_raps', 'RAPS')
    if not d.empty: dfs.append(d)

    # CF-GNN（confgnn 目录）
    d = _load_dir(args.confgnn_dir, 'confgnn_aps', 'CF-GNN')
    if not d.empty: dfs.append(d)

    # NAPS（graph_cp 目录，weighted 模式）
    d = _load_dir(args.graphcp_dir, 'weighted', 'NAPS')
    if not d.empty: dfs.append(d)

    # DAPS（daps 目录）
    d = _load_dir(args.daps_dir, 'daps', 'DAPS')
    if not d.empty: dfs.append(d)

    if not dfs:
        raise RuntimeError('未加载到任何数据，请检查目录路径')

    return pd.concat(dfs, ignore_index=True)


# ════════════════════════════════════════════════════════════════════════
# 2. Split 过滤与聚合
# ════════════════════════════════════════════════════════════════════════

def filter_and_agg(df, split_mode):
    """
    按 split_mode 过滤节点，然后对每个 (method, alpha) 取均值。
    split_mode:
      'all'     — 所有 split 的均值
      'ID-test' — 只用 ID-test
      'OOD'     — 只用 OOD splits（排除 ID-test）
      其他字符串 — 精确匹配 split 名称
    """
    if split_mode == 'all':
        sub = df
    elif split_mode == 'ID-test':
        sub = df[df['split'] == 'ID-test']
    elif split_mode == 'OOD':
        sub = df[df['split'] != 'ID-test']
    else:
        sub = df[df['split'] == split_mode]

    if sub.empty:
        raise ValueError(f'过滤后数据为空，split_mode={split_mode}')

    agg = (sub.groupby(['method', 'alpha'])
             [['coverage_mean', 'set_size_mean', 'shr_mean',
               'coverage_std',  'set_size_std',  'shr_std']]
             .mean()
             .reset_index())
    return agg.sort_values(['method', 'alpha'])


# ════════════════════════════════════════════════════════════════════════
# 3. 绘图
# ════════════════════════════════════════════════════════════════════════

def plot_coverage_efficiency(agg, out_path):
    """图1：Coverage vs Efficiency（Set Size）"""
    fig, ax = plt.subplots(figsize=(6, 5))

    for method, style in METHODS.items():
        sub = agg[agg['method'] == method].sort_values('coverage_mean')
        if sub.empty:
            continue
        ax.plot(sub['coverage_mean'], sub['set_size_mean'],
                color=style['color'], ls=style['ls'],
                marker=style['marker'], ms=style['ms'],
                label=method, linewidth=2, zorder=3)
        # 误差带
        ax.fill_between(
            sub['coverage_mean'],
            sub['set_size_mean'] - sub['set_size_std'],
            sub['set_size_mean'] + sub['set_size_std'],
            color=style['color'], alpha=0.12)

    ax.set_xlabel('Coverage', fontsize=13)
    ax.set_ylabel('Efficiency (Set Size)', fontsize=13)
    ax.set_xlim(0.75, 1.00)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  图1 已保存: {out_path}')


def plot_coverage_shr(agg, out_path):
    """图2：Coverage vs Singleton Hit Ratio"""
    fig, ax = plt.subplots(figsize=(6, 5))

    for method, style in METHODS.items():
        sub = agg[agg['method'] == method].sort_values('coverage_mean')
        if sub.empty:
            continue
        ax.plot(sub['coverage_mean'], sub['shr_mean'],
                color=style['color'], ls=style['ls'],
                marker=style['marker'], ms=style['ms'],
                label=method, linewidth=2, zorder=3)
        ax.fill_between(
            sub['coverage_mean'],
            sub['shr_mean'] - sub['shr_std'],
            sub['shr_mean'] + sub['shr_std'],
            color=style['color'], alpha=0.12)

    ax.set_xlabel('Coverage', fontsize=13)
    ax.set_ylabel('Singleton Hit Ratio', fontsize=13)
    ax.set_xlim(0.75, 1.00)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  图2 已保存: {out_path}')


def plot_both_subplots(agg, out_path):
    """将图1和图2并排输出为一张图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for method, style in METHODS.items():
        sub = agg[agg['method'] == method].sort_values('coverage_mean')
        if sub.empty:
            continue

        # 图1
        axes[0].plot(sub['coverage_mean'], sub['set_size_mean'],
                     color=style['color'], ls=style['ls'],
                     marker=style['marker'], ms=style['ms'],
                     label=method, linewidth=2, zorder=3)
        axes[0].fill_between(
            sub['coverage_mean'],
            sub['set_size_mean'] - sub['set_size_std'],
            sub['set_size_mean'] + sub['set_size_std'],
            color=style['color'], alpha=0.12)

        # 图2
        axes[1].plot(sub['coverage_mean'], sub['shr_mean'],
                     color=style['color'], ls=style['ls'],
                     marker=style['marker'], ms=style['ms'],
                     label=method, linewidth=2, zorder=3)
        axes[1].fill_between(
            sub['coverage_mean'],
            sub['shr_mean'] - sub['shr_std'],
            sub['shr_mean'] + sub['shr_std'],
            color=style['color'], alpha=0.12)

    axes[0].set_xlabel('Coverage', fontsize=13)
    axes[0].set_ylabel('Efficiency (Set Size)', fontsize=13)
    axes[0].set_xlim(0.75, 1.00)
    axes[0].legend(loc='upper left')
    axes[0].set_title('Coverage vs Efficiency', fontsize=13)

    axes[1].set_xlabel('Coverage', fontsize=13)
    axes[1].set_ylabel('Singleton Hit Ratio', fontsize=13)
    axes[1].set_xlim(0.75, 1.00)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc='upper right')
    axes[1].set_title('Coverage vs Singleton Hit Ratio', fontsize=13)

    fig.suptitle(f'Elliptic — {args.split} splits', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  合并图 已保存: {out_path}')


# ════════════════════════════════════════════════════════════════════════
# 4. 分 split 类型输出（Near OOD / Far OOD / All）
# ════════════════════════════════════════════════════════════════════════

def plot_by_ood_group(df):
    """
    Elliptic 按时间步远近分组：
      Near OOD: OOD-test_0 ~ OOD-test_2（步骤17-28，较近）
      Far OOD:  OOD-test_3 ~ OOD-test_8（步骤29-48，较远）
    """
    near_pattern = ['OOD-test_0', 'OOD-test_1', 'OOD-test_2']
    far_pattern  = ['OOD-test_3', 'OOD-test_4', 'OOD-test_5',
                    'OOD-test_6', 'OOD-test_7', 'OOD-test_8']

    groups = {
        'all':      df,
        'ID-test':  df[df['split'] == 'ID-test'],
        'near_ood': df[df['split'].str.startswith(tuple(near_pattern))],
        'far_ood':  df[df['split'].str.startswith(tuple(far_pattern))],
    }

    for gname, gdf in groups.items():
        if gdf.empty:
            continue
        agg = (gdf.groupby(['method', 'alpha'])
                  [['coverage_mean', 'set_size_mean', 'shr_mean',
                    'coverage_std',  'set_size_std',  'shr_std']]
                  .mean().reset_index()
                  .sort_values(['method', 'alpha']))

        tag = f'{args.dataset}_{gname}'
        plot_coverage_efficiency(
            agg, os.path.join(args.out, f'{tag}_coverage_efficiency.pdf'))
        plot_coverage_shr(
            agg, os.path.join(args.out, f'{tag}_coverage_shr.pdf'))
        plot_both_subplots(
            agg, os.path.join(args.out, f'{tag}_combined.pdf'))


# ════════════════════════════════════════════════════════════════════════
# 5. 主入口
# ════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Loading CSVs...')
    df = load_all()
    print(f'  加载完成: {len(df)} 行, 方法: {df["method"].unique().tolist()}')
    print(f'  Alpha 值: {sorted(df["alpha"].unique().tolist())}')
    print(f'  Splits: {df["split"].unique().tolist()[:5]} ...')

    # 按指定 split 模式输出
    print(f'\n绘图 (split={args.split})...')
    agg = filter_and_agg(df, args.split)
    tag = f'{args.dataset}_{args.split}'
    plot_coverage_efficiency(
        agg, os.path.join(args.out, f'{tag}_coverage_efficiency.pdf'))
    plot_coverage_shr(
        agg, os.path.join(args.out, f'{tag}_coverage_shr.pdf'))
    plot_both_subplots(
        agg, os.path.join(args.out, f'{tag}_combined.pdf'))

    # 同时按 OOD 分组输出
    print('\n按 Near/Far OOD 分组绘图...')
    plot_by_ood_group(df)

    print('\n全部完成！')
