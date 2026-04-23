import os, argparse, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',  type=str, default='./results')
parser.add_argument('--out_dir',   type=str, default='./figures')
parser.add_argument('--dataset',   type=str, default='elliptic',
                    choices=['elliptic', 'arxiv', 'cora', 'amazon', 'twitch', 'facebook100'])
parser.add_argument('--dpi',       type=int, default=1200)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
DS = args.dataset


ALGOS = ['Vanilla', 'GPN', 'G-ΔUQ', 'S-BGCN-T-K', 'CaGCN', 'GATS', 'RBS']

DISPLAY = {
    'calGNN-Uncal': 'Vanilla',
    'calGNN-RBS':   'RBS',
    'CaGCN':        'CaGCN',
    'GATS':         'GATS',
    'UnGNN':        'S-BGCN-T-K',
    'GPN':          'GPN',
    'GDUQ':         'G-ΔUQ',
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
    'Vanilla':    'o',
    'CaGCN':      's',
    'GATS':       '^',
    'RBS':        'v',
    'S-BGCN-T-K': 'D',
    'GPN':        'P',
    'G-ΔUQ':      'X',
}

# ── OOD splits
if DS == 'elliptic':
    _OOD_SPLITS_RAW = [
        'OOD-test_0(s17-20)', 'OOD-test_1(s21-24)', 'OOD-test_2(s25-28)',
        'OOD-test_3(s29-32)', 'OOD-test_4(s33-36)', 'OOD-test_5(s37-40)',
        'OOD-test_6(s41-43)', 'OOD-test_7(s44-46)', 'OOD-test_8(s47-48)',
    ]
    _OOD_LABELS_RAW = [f'T{i+1}' for i in range(9)]
    # shift intensity 
    _OOD_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    BINARY = False
elif DS == 'arxiv':
    _OOD_SPLITS_RAW = ['OOD-test_0(2014-2016)', 'OOD-test_1(2016-2018)',
                       'OOD-test_2(2018-2020)']
    _OOD_LABELS_RAW = ['2014-2016', '2016-2018', '2018-2020']
    # shift intensity 
    _OOD_ORDER = [0, 1, 2]
    BINARY = False
elif DS == 'cora':

    _OOD_SPLITS_RAW = [f'OOD-test_{i}(env{i+2})' for i in range(8)]
    _OOD_LABELS_RAW = [f'OOD-{i}' for i in range(8)]
    _OOD_ORDER = [3, 7, 0, 1, 6, 5]
    _OOD_LABELS_RAW_CORA = [f'T{k}' for k in range(len(_OOD_ORDER))]
    BINARY = False
else:  # amazon
    # EERM Amazon: env2~9 → OOD-test_0~7

    _OOD_SPLITS_RAW = [f'OOD-test_{i}(env{i+2})' for i in range(8)]
    _OOD_LABELS_RAW = [f'OOD-{i}' for i in range(8)]
    _OOD_ORDER = list(range(8))
    _OOD_LABELS_RAW_AMAZON = [f'T{k}' for k in range(8)]
    BINARY = False

if DS == 'twitch':
    OOD_SPLITS = ['OOD-DE', 'OOD-ENGB', 'OOD-TW']
    OOD_LABELS = ['DE', 'ENGB', 'TW']
    BINARY = False
elif DS == 'facebook100':
    OOD_SPLITS = ['OOD-Bingham82', 'OOD-Texas80', 'OOD-Yale4',
                  'OOD-Caltech36', 'OOD-Duke14', 'OOD-Penn94']
    OOD_LABELS = ['Bingham82', 'Texas80', 'Yale4',
                  'Caltech36', 'Duke14', 'Penn94']
    BINARY = False
elif DS == 'cora':
    OOD_SPLITS = [_OOD_SPLITS_RAW[i] for i in _OOD_ORDER]
    OOD_LABELS = _OOD_LABELS_RAW_CORA
elif DS == 'amazon':
    OOD_SPLITS = [_OOD_SPLITS_RAW[i] for i in _OOD_ORDER]
    OOD_LABELS = _OOD_LABELS_RAW_AMAZON
else:
    OOD_SPLITS = [_OOD_SPLITS_RAW[i] for i in _OOD_ORDER]
    OOD_LABELS = [_OOD_LABELS_RAW[i] for i in _OOD_ORDER]

ALL_SPLITS  = ['ID-test'] + OOD_SPLITS
ALL_LABELS  = ['ID-test'] + OOD_LABELS

COV_TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

METRIC_ARROW = {
    'acc':    True,  'f1':     True,  'prec':   True,  'rec':    True,
    'ece':    False, 'nll':    False, 'brier':  False,
    'ue_auroc': True, 'ue_aupr': True,
    'ood_auroc': True,
    'aurc':   False, 'aurc_ood': False,
    'srtr_auc': False,
    'coverage@risk01': True, 'coverage@risk05': True, 'coverage@risk10': True,
}

def metric_label(m):
    arrow = '↑' if METRIC_ARROW.get(m, True) else '↓'
    nice = {'ece':'ECE','nll':'NLL','brier':'Brier Score',
            'acc':'Accuracy','f1':'F1','ue_auroc':'UE-AUROC',
            'ue_aupr':'UE-AUPR','ood_auroc':'OOD-AUROC',
            'aurc':'AURC','aurc_ood':'AURC','srtr_auc':'SRTR-AUC',
            'coverage@risk01':'Coverage@Risk≤1%',
            'coverage@risk05':'Coverage@Risk≤5%',
            'coverage@risk10':'Coverage@Risk≤10%',}
    return f'{nice.get(m, m)} {arrow}'


plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          20,
    'axes.titlesize':     21,
    'axes.labelsize':     20,
    'legend.fontsize':    11,
    'xtick.labelsize':    17,
    'ytick.labelsize':    17,
    'axes.spines.top':    True,
    'axes.spines.right':  True,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
    'figure.dpi':         args.dpi,
})

def set_box_axes(ax):
    for sp in ax.spines.values():
        sp.set_visible(True)

# ══════════════════════════════════════════════════════════════
# 1. data load
# ══════════════════════════════════════════════════════════════
ALGO_FILES = {
    'CaGCN':  'cagcn',
    'GATS':   'gats',
    'UnGNN':  'ungnn',
    'GPN':    'gpn',
    'GDUQ':   'gduq',
}

def load_results():
    dfs = {}
    for internal, fname in ALGO_FILES.items():
        path = os.path.join(args.data_dir, f'{DS}_{fname}_results.csv')
        if os.path.exists(path):
            dfs[DISPLAY[internal]] = pd.read_csv(path)
        else:
            print(f'  [warning] : {path}')
    cal_path = os.path.join(args.data_dir, f'{DS}_calgnn_results.csv')
    if os.path.exists(cal_path):
        cal = pd.read_csv(cal_path)
        for cm_key, disp in [('Uncal','Vanilla'), ('RBS','RBS')]:
            sub = cal[cal['cal_method'] == cm_key].drop(columns=['cal_method']).reset_index(drop=True)
            dfs[disp] = sub
    else:
        print(f'  [warning] : {cal_path}')
    return dfs

def load_reliability():
    dfs = {}
    for internal, fname in ALGO_FILES.items():
        path = os.path.join(args.data_dir, f'{DS}_{fname}_reliability.csv')
        if os.path.exists(path):
            dfs[DISPLAY[internal]] = pd.read_csv(path)
    cal_path = os.path.join(args.data_dir, f'{DS}_calgnn_reliability.csv')
    if os.path.exists(cal_path):
        cal = pd.read_csv(cal_path)
        for cm_key, disp in [('Uncal','Vanilla'), ('RBS','RBS')]:
            sub = cal[cal['cal_method'] == cm_key].drop(columns=['cal_method']).reset_index(drop=True)
            dfs[disp] = sub
    return dfs

def load_uncertainty():
    dfs = {}
    for internal, fname in ALGO_FILES.items():
        path = os.path.join(args.data_dir, f'{DS}_{fname}_uncertainty_samples.csv')
        if os.path.exists(path):
            dfs[DISPLAY[internal]] = pd.read_csv(path)
    cal_path = os.path.join(args.data_dir, f'{DS}_calgnn_uncertainty_samples.csv')
    if os.path.exists(cal_path):
        cal = pd.read_csv(cal_path)
        for cm_key, disp in [('Uncal','Vanilla'), ('RBS','RBS')]:
            sub = cal[cal['cal_method'] == cm_key].drop(columns=['cal_method']).reset_index(drop=True)
            dfs[disp] = sub
    return dfs

def get_val(df, split, col):
    row = df[df['split'] == split]
    if row.empty:
        return float('nan'), float('nan')
    mu  = float(row[f'{col}_mean'].values[0]) if f'{col}_mean' in row.columns else float('nan')
    std = float(row[f'{col}_std'].values[0])  if f'{col}_std'  in row.columns else float('nan')
    return mu, std

def savefig(name):
    for ext in ('pdf', 'png'):
        path = os.path.join(args.out_dir, f'{DS}_{name}.{ext}')
        plt.savefig(path, bbox_inches='tight',
                    dpi=args.dpi if ext == 'png' else None)
    plt.close()
    print(f'  ✓ {DS}_{name}.png')

def savefig_fig(fig, name):
    """ figure save pdf+png， close。"""
    for ext in ('pdf', 'png'):
        path = os.path.join(args.out_dir, f'{DS}_{name}.{ext}')
        fig.savefig(path, bbox_inches='tight',
                    dpi=args.dpi if ext == 'png' else None)
    print(f'  ✓ {DS}_{name}.png')

def legend_handles(algos=None):
    if algos is None: algos = ALGOS
    return [Line2D([0],[0], color=PALETTE[a], marker=MARKERS[a],
                   markersize=7, linewidth=2.0, label=a)
            for a in algos if a in PALETTE]

# ══════════════════════════════════════════════════════════════
# fig1：Calibration Overall Comparison
# ══════════════════════════════════════════════════════════════
def plot_fig1(dfs):
    print('fig1...')
    metrics = [
        ('ece',   'ECE ↓',         'fig1a_calibration_ece'),
        ('nll',   'NLL ↓',         'fig1b_calibration_nll'),
        ('brier', 'Brier Score ↓', 'fig1c_calibration_brier'),
    ]
    x = np.arange(len(ALGOS)); width = 0.35
    id_patch  = mpatches.Patch(color='gray', alpha=0.9,  label='ID-test')
    ood_patch = mpatches.Patch(color='gray', alpha=0.45, label='OOD-avg')

    for metric, mlabel, fname in metrics:
        fig, ax = plt.subplots(figsize=(7, 5))
        for j, (skey, alpha) in enumerate([('ID-test', 0.9), ('OOD-avg', 0.45)]):
            vals, errs = [], []
            for algo in ALGOS:
                if algo not in dfs: vals.append(np.nan); errs.append(0); continue
                mu, std = get_val(dfs[algo], skey, metric)
                vals.append(mu); errs.append(std)
            offset = (j - 0.5) * width
            ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
                   color=[PALETTE[a] for a in ALGOS], alpha=alpha,
                   edgecolor='white', linewidth=0.5,
                   error_kw=dict(elinewidth=1, ecolor='gray'))
        ax.set_ylabel(mlabel, fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels(ALGOS, rotation=35, ha='right', fontsize=17)
        ax.set_ylim(bottom=0)
        ax.legend(handles=[id_patch, ood_patch],
                  loc='upper right', fontsize=11, framealpha=0.85)
        set_box_axes(ax)
        fig.tight_layout()
        savefig_fig(fig, fname)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════
# fig2：Calibration under Shift
# ══════════════════════════════════════════════════════════════
def plot_fig2(dfs):
    print('fig2...')
    fig, axes = plt.subplots(2, 1, figsize=(9, 8),
                             sharex=True, gridspec_kw={'hspace': 0.08})

    for ax, (metric, mlabel) in zip(axes,
            [('acc', 'Accuracy ↑'), ('brier', 'Brier Score ↓')]):
        for algo in ALGOS:
            if algo not in dfs: continue
            vals, errs = [], []
            for sp in ALL_SPLITS:
                mu, std = get_val(dfs[algo], sp, metric)
                vals.append(mu); errs.append(std)
            lw = 2.5 if algo == 'G-ΔUQ' else 1.8
            ax.plot(ALL_LABELS, vals, marker=MARKERS[algo],
                    color=PALETTE[algo], linewidth=lw, markersize=6, label=algo)
            lo = [v-e if not math.isnan(v) else np.nan for v,e in zip(vals,errs)]
            hi = [v+e if not math.isnan(v) else np.nan for v,e in zip(vals,errs)]
            ax.fill_between(ALL_LABELS, lo, hi, color=PALETTE[algo], alpha=0.12)
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
        ax.set_ylabel(mlabel, fontsize=20)
        set_box_axes(ax)

    axes[1].set_xticks(range(len(ALL_LABELS)))
    axes[1].set_xticklabels(ALL_LABELS, rotation=35, ha='right', fontsize=17)
    axes[1].set_xlabel('', fontsize=20)
    axes[0].legend(handles=legend_handles(), loc='lower left',
                   fontsize=11, ncol=2, framealpha=0.85)
    plt.tight_layout()
    savefig('fig2_calibration_shift')


# ══════════════════════════════════════════════════════════════
# fig3：Reliability Diagram
# ══════════════════════════════════════════════════════════════
def plot_fig3(rel_dfs):
    print('fig3...')
    mid_ood = OOD_SPLITS[len(OOD_SPLITS)//2]
    mid_lab = OOD_LABELS[len(OOD_LABELS)//2]
    scenarios = [
        ('ID-test', 'ID-test', 'fig3a_reliability_id_test'),
        (mid_ood,   mid_lab,   f'fig3b_reliability_{mid_lab.lower().replace("-","_")}'),
    ]

    for split, slabel, fname in scenarios:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.plot([0,1],[0,1], 'k--', linewidth=2.5, label='Perfect', alpha=0.7, zorder=0)
        for algo in ALGOS:
            if algo not in rel_dfs: continue
            df = rel_dfs[algo]
            sub = df[df['split'] == split].copy()
            if sub.empty: continue
            grp = sub.groupby('bin').agg(
                avg_conf_mean=('avg_confidence','mean'),
                acc_mean=('accuracy','mean')).reset_index()
            grp = grp.dropna(subset=['acc_mean'])
            lw = 2.5 if algo == 'G-ΔUQ' else 1.5
            ax.plot(grp['avg_conf_mean'], grp['acc_mean'],
                    marker=MARKERS[algo], color=PALETTE[algo],
                    linewidth=lw, markersize=5, label=algo)
        ax.set_xlim(0.4, 1.01); ax.set_ylim(0.4, 1.01)
        ax.set_xlabel('Confidence', fontsize=20)
        ax.set_ylabel('Accuracy', fontsize=20)
        ax.legend(
            handles=[Line2D([0],[0],color='k',ls='--',lw=2.5,label='Perfect')]
                    + legend_handles(),
            loc='lower right', fontsize=11, framealpha=0.85)
        set_box_axes(ax)
        fig.tight_layout()
        savefig_fig(fig, fname)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════
# fig4：Accuracy–ECE Trade-off
# ══════════════════════════════════════════════════════════════
def plot_fig4(dfs):
    print('fig4...')
    fig, ax = plt.subplots(figsize=(7, 6))

    for algo in ALGOS:
        if algo not in dfs: continue
        ece_id, ece_id_s = get_val(dfs[algo], 'ID-test', 'ece')
        acc_id, acc_id_s = get_val(dfs[algo], 'ID-test', 'acc')
        ece_oo, ece_oo_s = get_val(dfs[algo], 'OOD-avg', 'ece')
        acc_oo, acc_oo_s = get_val(dfs[algo], 'OOD-avg', 'acc')
        if any(math.isnan(v) for v in [ece_id, acc_id, ece_oo, acc_oo]):
            continue
        ax.scatter(ece_id, acc_id, color=PALETTE[algo], marker=MARKERS[algo],
                   s=100, zorder=4, label=algo)
        ax.errorbar(ece_id, acc_id, xerr=ece_id_s, yerr=acc_id_s,
                    fmt='none', color=PALETTE[algo], alpha=0.35, capsize=3)
        ax.scatter(ece_oo, acc_oo, color=PALETTE[algo], marker=MARKERS[algo],
                   s=100, zorder=4, facecolors='none', edgecolors=PALETTE[algo],
                   linewidths=1.5)
        ax.errorbar(ece_oo, acc_oo, xerr=ece_oo_s, yerr=acc_oo_s,
                    fmt='none', color=PALETTE[algo], alpha=0.35, capsize=3)
        ax.annotate('', xy=(ece_oo, acc_oo), xytext=(ece_id, acc_id),
                    arrowprops=dict(arrowstyle='->', color=PALETTE[algo],
                                   lw=1.2, alpha=0.6))

    id_handles = legend_handles()
    id_patch   = Line2D([0],[0], marker='o', color='gray', lw=0,
                        markersize=9, label='● ID-test (filled)')
    ood_patch  = Line2D([0],[0], marker='o', color='gray', lw=0,
                        markersize=9, markerfacecolor='none',
                        markeredgecolor='gray', label='○ OOD-avg (open)')
    ax.set_xlabel('ECE ↓', fontsize=20)
    ax.set_ylabel('Accuracy ↑', fontsize=20)
    ax.legend(handles=id_handles + [id_patch, ood_patch],
              loc='upper right', fontsize=11, framealpha=0.85)
    set_box_axes(ax)
    plt.tight_layout()
    savefig('fig4_acc_ece_tradeoff')


# ══════════════════════════════════════════════════════════════
# fig5：UE-AUROC / UE-AUPR
# ══════════════════════════════════════════════════════════════
def plot_fig5(dfs):
    print('fig5...')
    metrics = [
        ('ue_auroc', 'UE-AUROC ↑', 'fig5a_ue_auroc'),
        ('ue_aupr',  'UE-AUPR ↑',  'fig5b_ue_aupr'),
    ]
    x = np.arange(len(ALGOS)); width = 0.35
    id_patch  = mpatches.Patch(color='gray', alpha=0.9,  label='ID-test')
    ood_patch = mpatches.Patch(color='gray', alpha=0.45, label='OOD-avg')

    for metric, mlabel, fname in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        for j, (skey, alpha) in enumerate([('ID-test', 0.9), ('OOD-avg', 0.45)]):
            vals, errs = [], []
            for algo in ALGOS:
                if algo not in dfs: vals.append(np.nan); errs.append(0); continue
                mu, std = get_val(dfs[algo], skey, metric)
                vals.append(mu); errs.append(std)
            offset = (j - 0.5) * width
            ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
                   color=[PALETTE[a] for a in ALGOS], alpha=alpha,
                   edgecolor='white', linewidth=0.5,
                   error_kw=dict(elinewidth=1, ecolor='gray'))
        ax.set_ylabel(mlabel, fontsize=20)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(ALGOS, rotation=35, ha='right', fontsize=17)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        # 每个子图独立图例：仅 ID/OOD 透明度说明
        ax.legend(handles=[id_patch, ood_patch],
                  loc='upper right', fontsize=11, framealpha=0.85)
        set_box_axes(ax)
        fig.tight_layout()
        savefig_fig(fig, fname)
        plt.close(fig)


# ═# ══════════════════════════════════════════════════════════════
# fig6：Correct vs Incorrect violin
# ══════════════════════════════════════════════════════════════
def plot_fig6(unc_dfs):
    print('fig6...')
    mid_ood = OOD_SPLITS[len(OOD_SPLITS)//2]
    mid_lab = OOD_LABELS[len(OOD_LABELS)//2]
    scenarios = [('ID-test', 'ID-test'), (mid_ood, mid_lab)]

    avail = [a for a in ALGOS if a in unc_dfs]
    n = len(avail)
    if n == 0:
        print('  [跳过] 无数据'); return

    N_SAMPLE = 800
    rng = np.random.default_rng(42)

    import matplotlib.colors as mcolors
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(2, n, figsize=(3.4*n, 11), sharey=False, sharex=False)
    if n == 1:
        axes = [[axes[0]], [axes[1]]]


    col_ymax = {}
    for col, algo in enumerate(avail):
        df = unc_dfs[algo]
        all_vals = []
        for split, _ in scenarios:
            sub = df[df['split'] == split]
            for correct in [0, 1]:
                v = sub[sub['correct'] == correct]['u'].values
                if len(v) > 0:
                    all_vals.extend(v.tolist())
        col_ymax[algo] = np.percentile(all_vals, 99) * 1.05 if all_vals else 1.0

    for row, (split, slabel) in enumerate(scenarios):
        for col, algo in enumerate(avail):
            ax = axes[row][col]
            df = unc_dfs[algo]
            color = PALETTE[algo]
            sub = df[df['split'] == split]

            data = {}
            for correct, side in [(1, 'Correct'), (0, 'Incorrect')]:
                vals = sub[sub['correct'] == correct]['u'].values
                if len(vals) == 0:
                    data[side] = np.array([]); continue
                idx = rng.choice(len(vals), min(N_SAMPLE, len(vals)), replace=False)
                data[side] = vals[idx]

            all_v = np.concatenate([v for v in data.values() if len(v) > 0])
            if len(all_v) == 0:
                ax.set_visible(False); continue

            y_max = col_ymax[algo]
            y_min = 0
            half_w = 0.38
            base_rgb = mcolors.to_rgb(color)

            for side, x_sign, alpha in [
                ('Correct',   -1, 0.85),
                ('Incorrect', +1, 0.45),
            ]:
                vals = data.get(side, np.array([]))
                if len(vals) < 5: continue
                side_color = color if side == 'Correct' \
                             else tuple(min(1, c*0.5 + 0.5) for c in base_rgb)
                try:
                    kde = gaussian_kde(vals, bw_method='scott')
                    y_pts = np.linspace(y_min, y_max, 200)
                    density = kde(y_pts)
                    density = density / density.max() * half_w
                    xs = x_sign * density
                    ax.fill_betweenx(y_pts, 0, xs, color=side_color, alpha=alpha)
                    ax.plot(xs, y_pts, color=side_color, linewidth=0.8, alpha=0.9)
                except Exception:
                    pass
                if len(vals) >= 4:
                    q25, q50, q75 = np.percentile(vals, [25, 50, 75])
                    w_val = x_sign * half_w * 0.35
                    ax.plot([0, w_val], [q50, q50],
                            color='white', linewidth=2.5, zorder=5)
                    for q in [q25, q75]:
                        ax.plot([0, w_val*0.7], [q, q],
                                color='gray', linewidth=1.0, zorder=4)

            ax.axvline(0, color='gray', linewidth=0.6, alpha=0.5)
            ax.set_xlim(-half_w*1.2, half_w*1.2)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([-half_w*0.6, half_w*0.6])
            ax.set_xticklabels(['Correct', 'Incorrect'], fontsize=13)
            ax.tick_params(axis='y', labelleft=True, labelsize=19)

            if row == 0:
                ax.text(0.04, 0.97, algo, transform=ax.transAxes,
                        fontsize=13, color=color, fontweight='bold',
                        va='top', ha='left')
            ax.set_ylabel('Uncertainty (u)' if col == 0 else '', fontsize=20)
            if col == 0:
                ax.text(-0.52, 0.5, slabel, transform=ax.transAxes,
                        fontsize=13, va='center', ha='right', rotation=90,
                        fontweight='bold')
            set_box_axes(ax)

    plt.tight_layout()
    savefig('fig6_uncertainty_correct_incorrect')


# ══════════════════════════════════════════════════════════════
# fig7：OOD-AUROC 
# ══════════════════════════════════════════════════════════════
def plot_fig7(dfs):
    print('fig7...')
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ALGOS))
    width = 0.08
    colors_ood = plt.cm.Blues(np.linspace(0.35, 0.9, len(OOD_SPLITS)))

    for j, (split, slabel) in enumerate(zip(OOD_SPLITS, OOD_LABELS)):
        vals, errs = [], []
        for algo in ALGOS:
            if algo not in dfs: vals.append(np.nan); errs.append(0); continue
            mu, std = get_val(dfs[algo], split, 'ood_auroc')
            vals.append(mu); errs.append(std)
        offset = (j - len(OOD_SPLITS)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, yerr=errs, capsize=2,
               color=colors_ood[j], label=slabel,
               edgecolor='white', linewidth=0.3,
               error_kw=dict(elinewidth=0.8, ecolor='gray'))

    ax.set_xticks(x)
    ax.set_xticklabels(ALGOS, rotation=35, ha='right', fontsize=17)
    ax.set_ylabel('OOD-AUROC ↑', fontsize=20); ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.legend(title='OOD split', loc='upper left', ncol=3, fontsize=9)
    set_box_axes(ax)
    plt.tight_layout()
    savefig('fig7_ood_auroc_comparison')


# ══════════════════════════════════════════════════════════════
# fig8a：OOD-AUROC 
# ══════════════════════════════════════════════════════════════
def plot_fig8a(dfs):
    print('fig8a...')
    fig, ax = plt.subplots(figsize=(10, 5))

    for algo in ALGOS:
        if algo not in dfs: continue
        vals, errs = [], []
        for sp in OOD_SPLITS:
            mu, std = get_val(dfs[algo], sp, 'ood_auroc')
            vals.append(mu); errs.append(std)
        lw = 2.5 if algo == 'G-ΔUQ' else 1.8
        ax.plot(OOD_LABELS, vals, marker=MARKERS[algo],
                color=PALETTE[algo], linewidth=lw, markersize=6, label=algo)
        lo = [v-e if not math.isnan(v) else np.nan for v,e in zip(vals,errs)]
        hi = [v+e if not math.isnan(v) else np.nan for v,e in zip(vals,errs)]
        ax.fill_between(OOD_LABELS, lo, hi, color=PALETTE[algo], alpha=0.12)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Shift severity', fontsize=20)
    ax.set_ylabel('OOD-AUROC ↑', fontsize=20)
    ax.set_xticks(range(len(OOD_LABELS)))
    ax.set_xticklabels(OOD_LABELS, rotation=0, ha='center', fontsize=17)
    ax.legend(handles=legend_handles(), loc='best', ncol=2, fontsize=10, framealpha=0.85)
    set_box_axes(ax)
    plt.tight_layout()
    savefig('fig8a_ood_auroc_shift')


# ══════════════════════════════════════════════════════════════
# fig8b： Violin
# ══════════════════════════════════════════════════════════════
def plot_fig8b(unc_dfs):
    print('fig8b...')
  
    if DS == 'cora':
        strongest_ood = 'OOD-test_2(env4)'
    else:
        strongest_ood = OOD_SPLITS[-1]

    avail = [a for a in ALGOS if a in unc_dfs]
    n = len(avail)
    if n == 0:
        print('  [跳过] 无数据'); return

    N_SAMPLE = 800
    rng = np.random.default_rng(42)

    import matplotlib.colors as mcolors
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, n, figsize=(3.4*n, 6.5), sharey=False)
    if n == 1: axes = [axes]

    for ax, algo in zip(axes, avail):
        df = unc_dfs[algo]
        color = PALETTE[algo]

        data = {}
        for split_key, side in [('ID-test', 'ID'), (strongest_ood, 'OOD')]:
            sub = df[df['split'] == split_key]['u'].values
            if len(sub) == 0: data[side] = np.array([]); continue
            idx = rng.choice(len(sub), min(N_SAMPLE, len(sub)), replace=False)
            data[side] = sub[idx]

        all_vals = np.concatenate([v for v in data.values() if len(v)>0])
        if len(all_vals) == 0: ax.set_visible(False); continue
        y_max = np.percentile(all_vals, 99) * 1.05
        y_min = 0

        x_center = 0
        half_w = 0.38
        base_rgb = mcolors.to_rgb(color)

        for side, x_sign, side_color in [
            ('ID',  -1, color),
            ('OOD', +1, tuple(min(1,c*0.6+0.4) for c in base_rgb))
        ]:
            vals = data.get(side, np.array([]))
            if len(vals) < 5: continue
            try:
                kde = gaussian_kde(vals, bw_method='scott')
                y_pts = np.linspace(y_min, y_max, 200)
                density = kde(y_pts)
                density = density / density.max() * half_w
                xs = x_center + x_sign * density
                xs_base = np.full_like(xs, x_center)
                ax.fill_betweenx(y_pts, xs_base, xs,
                                 color=side_color,
                                 alpha=0.80 if side=='ID' else 0.55)
                ax.plot(xs, y_pts, color=side_color, linewidth=0.8, alpha=0.8)
            except Exception:
                pass

            if len(vals) >= 4:
                q25, q50, q75 = np.percentile(vals, [25, 50, 75])
                w_val = x_sign * half_w * 0.35
                ax.plot([x_center, x_center + w_val], [q50, q50],
                        color='white', linewidth=2.5, zorder=5)
                ax.plot([x_center, x_center + w_val*0.7],
                        [q25, q25], color='gray', linewidth=1.0, zorder=4)
                ax.plot([x_center, x_center + w_val*0.7],
                        [q75, q75], color='gray', linewidth=1.0, zorder=4)

        ax.axvline(x_center, color='gray', linewidth=0.6, alpha=0.5)
        ax.set_xlim(-half_w*1.2, half_w*1.2)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([-half_w*0.6, half_w*0.6])
        ax.set_xticklabels(['ID', 'OOD'], fontsize=13)
        ax.tick_params(axis='y', labelsize=19)
  
        ax.text(0.05, 0.97, algo, transform=ax.transAxes,
                fontsize=13, color=color, fontweight='bold',
                va='top', ha='left')
        ax.set_ylabel('Uncertainty (u)' if algo == avail[0] else '', fontsize=20)
        set_box_axes(ax)

    plt.tight_layout()
    savefig('fig8b_id_ood_uncertainty_dist')


# ══════════════════════════════════════════════════════════════
# fig9：Risk-Coverage Curve
# ══════════════════════════════════════════════════════════════
def plot_fig9(dfs):
    print('fig9...')
    scenarios = [
        ('ID-test',  'ID-test',  'fig9a_risk_coverage_id_test'),
        ('OOD-avg',  'OOD-avg',  'fig9b_risk_coverage_ood_avg'),
    ]
    for split, slabel, fname in scenarios:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for algo in ALGOS:
            if algo not in dfs: continue
            risks = []
            for tau in COV_TAUS:
                mu, _ = get_val(dfs[algo], split, f'risk@{tau}')
                risks.append(mu)
            valid = [(t,r) for t,r in zip(COV_TAUS, risks) if not math.isnan(r)]
            if not valid: continue
            ts, rs = zip(*valid)
            lw = 2.5 if algo == 'G-ΔUQ' else 1.8
            ax.plot(ts, rs, marker=MARKERS[algo], color=PALETTE[algo],
                    linewidth=lw, markersize=6, label=algo)
        ax.set_xlabel('Coverage (τ)', fontsize=20)
        ax.set_ylabel('Risk (Error Rate) ↓', fontsize=20)
        ax.set_xlim(0.05, 1.05); ax.set_ylim(bottom=0)
        ax.set_xticks(COV_TAUS)
        ax.set_xticklabels([f'{t:.1f}' for t in COV_TAUS], fontsize=15)
        ax.legend(handles=legend_handles(), loc='upper left',
                  fontsize=11, framealpha=0.85, ncol=1)
        set_box_axes(ax)
        fig.tight_layout()
        savefig_fig(fig, fname)
        plt.close(fig)


# ═# ══════════════════════════════════════════════════════════════
# fig10：AURC
# ══════════════════════════════════════════════════════════════
def plot_fig10(dfs):
    print('fig10...')
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(ALGOS)); width = 0.35

    for j, (split, col, slabel, alpha) in enumerate([
            ('ID-test', 'aurc',     'ID-test',  0.9),
            ('OOD-avg', 'aurc_ood', 'OOD-avg',  0.45),
    ]):
        vals, errs = [], []
        for algo in ALGOS:
            if algo not in dfs: vals.append(np.nan); errs.append(0); continue
            mu, std = get_val(dfs[algo], split, col)
            vals.append(mu); errs.append(std)
        offset = (j - 0.5) * width
        ax.bar(x + offset, vals, width, yerr=errs, capsize=4,
               color=[PALETTE[a] for a in ALGOS], alpha=alpha,
               edgecolor='white', linewidth=0.5,
               error_kw=dict(elinewidth=1, ecolor='gray'),
               label=slabel)

    ax.set_xticks(x)
    ax.set_xticklabels(ALGOS, rotation=35, ha='right', fontsize=17)
    ax.set_ylabel('AURC ↓', fontsize=20); ax.set_ylim(bottom=0)
    id_p  = mpatches.Patch(color='gray', alpha=0.9,  label='ID-test')
    ood_p = mpatches.Patch(color='gray', alpha=0.45, label='OOD-avg')
    ax.legend(handles=[id_p, ood_p], loc='upper right', fontsize=11)
    set_box_axes(ax)
    plt.tight_layout()
    savefig('fig10_aurc_comparison')


# ══════════════════════════════════════════════════════════════
# fig11：SRTR Curve
# ══════════════════════════════════════════════════════════════
def plot_fig11(dfs):
    print('fig11...')
    n_show = min(3, len(OOD_SPLITS))
    ood_show = OOD_SPLITS[:n_show]
    lab_show = OOD_LABELS[:n_show]

    leg_loc = 'upper right' if DS in ('arxiv', 'cora', 'amazon') else 'upper left'

    for idx, (split, slabel) in enumerate(zip(ood_show, lab_show)):
        safe_label = slabel.lower().replace('-', '_')
        fname = f'fig11_{chr(97+idx)}_srtr_{safe_label}'
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for algo in ALGOS:
            if algo not in dfs: continue
            srtrs = [get_val(dfs[algo], split, f'srtr@{tau}')[0]
                     for tau in COV_TAUS]
            valid = [(t,s) for t,s in zip(COV_TAUS, srtrs) if not math.isnan(s)]
            if not valid: continue
            ts, ss = zip(*valid)
            lw = 2.5 if algo == 'G-ΔUQ' else 1.8
            ax.plot(ts, ss, marker=MARKERS[algo], color=PALETTE[algo],
                    linewidth=lw, markersize=6, label=algo)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_xlabel('Coverage (τ)', fontsize=20)
        ax.set_ylabel('SRTR ↓', fontsize=20)
        ax.set_xlim(0.05, 1.05)
        ax.set_xticks(COV_TAUS)
        ax.set_xticklabels([f'{t:.1f}' for t in COV_TAUS], fontsize=15)
        all_vals = []
        for algo in ALGOS:
            if algo not in dfs: continue
            for tau in COV_TAUS:
                v, _ = get_val(dfs[algo], split, f'srtr@{tau}')
                if not math.isnan(v): all_vals.append(v)
        if all_vals:
            y_top = min(max(all_vals)*1.05, np.percentile(all_vals, 95)*1.5)
            y_bot = 0 if DS == 'elliptic' else max(0, np.percentile(all_vals, 1) * 0.9)  # cora/amazon 同 arxiv
            ax.set_ylim(y_bot, y_top)
        ax.legend(handles=legend_handles(), loc=leg_loc,
                  fontsize=11, framealpha=0.85)
        set_box_axes(ax)
        fig.tight_layout()
        savefig_fig(fig, fname)
        plt.close(fig)


# ═# ══════════════════════════════════════════════════════════════
# fig12：Coverage under Shift
# ══════════════════════════════════════════════════════════════
def plot_fig12(dfs):
    print('画图12...')
    targets = [
        ('coverage@risk01', 'Coverage@Risk≤1% ↑',  'fig12a_coverage_risk01'),
        ('coverage@risk05', 'Coverage@Risk≤5% ↑',  'fig12b_coverage_risk05'),
        ('coverage@risk10', 'Coverage@Risk≤10% ↑', 'fig12c_coverage_risk10'),
    ]

    for col, clabel, fname in targets:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for algo in ALGOS:
            if algo not in dfs: continue
            vals, errs = [], []
            for sp in OOD_SPLITS:
                mu, std = get_val(dfs[algo], sp, col)
                vals.append(mu); errs.append(std)
            lw   = 3.0 if algo == 'G-ΔUQ' else 1.8
            ms   = 7   if algo == 'G-ΔUQ' else 5
            zord = 5   if algo == 'G-ΔUQ' else 3
            color = PALETTE[algo]
            ax.plot(OOD_LABELS, vals, marker=MARKERS[algo],
                    color=color, linewidth=lw, markersize=ms,
                    label=algo, zorder=zord)
            lo = [v-e if not (math.isnan(v) or math.isnan(e)) else np.nan
                  for v,e in zip(vals,errs)]
            hi = [v+e if not (math.isnan(v) or math.isnan(e)) else np.nan
                  for v,e in zip(vals,errs)]
            ax.fill_between(OOD_LABELS, lo, hi, color=color, alpha=0.12)
        ax.set_ylabel(clabel, fontsize=20)
        ax.set_xlabel('', fontsize=20)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(OOD_LABELS)))
        ax.set_xticklabels(OOD_LABELS, rotation=35, ha='right', fontsize=17)
        handles = []
        for algo in ALGOS:
            lw_leg = 3.0 if algo == 'G-ΔUQ' else 1.5
            handles.append(Line2D([0],[0], color=PALETTE[algo], marker=MARKERS[algo],
                                   markersize=6, linewidth=lw_leg, label=algo))
        ax.legend(handles=handles, loc='upper left',
                  fontsize=11, framealpha=0.85, ncol=1)
        set_box_axes(ax)
        fig.tight_layout()
        savefig_fig(fig, fname)
        plt.close(fig)


# ═# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════
def main():
    print(f'\n[Dataset] {DS.upper()}')
    print(f'[Data Directory] {args.data_dir}')
    print(f'[Output Directory] {args.out_dir}\n')

    dfs     = load_results()
    rel_dfs = load_reliability()
    unc_dfs = load_uncertainty()

    loaded = sorted(dfs.keys())
    print(f'Loaded: {loaded}\n')

    plot_fig1(dfs)
    plot_fig2(dfs)
    plot_fig3(rel_dfs)
    plot_fig4(dfs)
    plot_fig5(dfs)
    plot_fig6(unc_dfs)
    plot_fig7(dfs)
    plot_fig8a(dfs)
    plot_fig8b(unc_dfs)
    plot_fig9(dfs)
    plot_fig10(dfs)
    plot_fig11(dfs)
    plot_fig12(dfs)

    print(f'\nAll images have been saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
