"""
gnn_uq_bench.metrics
====================
RQ1: acc, ece, nll, brier, f1/prec/rec (binary)
     OOD: delta_ece, delta_nll, delta_brier
RQ2: ue_auroc, ue_aupr, ood_auroc, delta_ue_auroc
RQ3: aurc, risk@tau, srtr@tau, srtr_auc, aurc_ood
     coverage@risk01/05/10
"""

import math
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

COV_FULL = [round(0.1 * i, 1) for i in range(1, 11)]


# ─────────────────────────────────────────────────────────────
# 1. 基础指标
# ─────────────────────────────────────────────────────────────

def reliability_bins(probs, labels, n_bins=15):
    conf  = probs.max(1)
    pred  = probs.argmax(1)
    acc_a = (pred == labels).astype(float)
    edges = np.linspace(0., 1., n_bins + 1)
    bins  = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            bins.append((float(conf[m].mean()), float(acc_a[m].mean()), int(m.sum())))
        else:
            bins.append((float((lo + hi) / 2), float('nan'), 0))
    return bins


def ece(probs, labels, n_bins=15):
    bins = reliability_bins(probs, labels, n_bins)
    N = len(labels)
    return float(sum(abs(c - a) * (cnt / N)
                     for c, a, cnt in bins if cnt > 0 and not math.isnan(a)))


def nll(probs, labels):
    return float(-np.log(probs[np.arange(len(labels)), labels] + 1e-10).mean())


def brier(probs, labels, nclass):
    oh = np.eye(nclass)[labels]
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def f1_binary(probs, labels):
    pred = probs.argmax(1)
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    return 2 * p * r / (p + r + 1e-8), p, r


# ─────────────────────────────────────────────────────────────
# 2. 不确定性评估
# ─────────────────────────────────────────────────────────────

def ue_auroc(u, probs, labels):
    wrong = (probs.argmax(1) != labels).astype(int)
    try:
        return float(roc_auc_score(wrong, u)), float(average_precision_score(wrong, u))
    except Exception:
        return float('nan'), float('nan')


def ood_auroc(u_id, u_ood):
    scores = np.concatenate([u_id, u_ood])
    domain = np.concatenate([np.zeros(len(u_id)), np.ones(len(u_ood))])
    try:
        return float(roc_auc_score(domain, scores))
    except Exception:
        return float('nan')


# ─────────────────────────────────────────────────────────────
# 3. Risk-Coverage (RQ3)
# ─────────────────────────────────────────────────────────────

def risk_curve(probs, u, labels):
    N = len(labels)
    wrong = (probs.argmax(1) != labels).astype(float)
    ws = wrong[np.argsort(u)]
    return {tau: float(ws[:max(1, int(math.ceil(tau * N)))].mean()) for tau in COV_FULL}


def aurc(rc):
    taus = sorted(rc.keys())
    risks = [rc[t] for t in taus]
    return float(sum((risks[j] + risks[j + 1]) / 2 * (taus[j + 1] - taus[j])
                     for j in range(len(taus) - 1)))


# ─────────────────────────────────────────────────────────────
# 4. 组合：单 split 全量指标
# ─────────────────────────────────────────────────────────────

def compute_split_metrics(probs, u, labels, nclass, binary=False):
    """
    probs   : (N, C) numpy softmax 概率
    u       : (N,)   per-node 不确定性分数（各方法自定义）
    labels  : (N,)   整数标签
    返回 dict，_probs/_u/_correct/_reliability_bins 为内部字段（不写 CSV）
    """
    res = dict(
        acc=float((probs.argmax(1) == labels).mean()),
        ece=ece(probs, labels),
        nll=nll(probs, labels),
        brier=brier(probs, labels, nclass),
    )
    if binary:
        f1, pr, re = f1_binary(probs, labels)
        res.update(f1=f1, prec=pr, rec=re)

    res['ue_auroc'], res['ue_aupr'] = ue_auroc(u, probs, labels)

    rc = risk_curve(probs, u, labels)
    res['aurc'] = aurc(rc)
    for tau in COV_FULL:
        res[f'risk@{tau}'] = rc[tau]

    # Coverage @ target risk
    N = len(labels)
    wrong = (probs.argmax(1) != labels).astype(float)
    ws = wrong[np.argsort(u)]
    for target, key in [(0.01, 'coverage@risk01'), (0.05, 'coverage@risk05'), (0.10, 'coverage@risk10')]:
        cov = float('nan')
        for tau in COV_FULL:
            if float(ws[:max(1, int(math.ceil(tau * N)))].mean()) <= target:
                cov = tau
        res[key] = cov

    # 内部字段
    res['_probs']            = probs
    res['_u']                = u
    res['_correct']          = (probs.argmax(1) == labels).astype(int)
    res['_reliability_bins'] = reliability_bins(probs, labels)
    return res


def add_cross_split_metrics(id_res, ood_res, u_id, u_ood):
    """追加跨 split OOD 指标到 ood_res 副本中"""
    out = dict(ood_res)
    for k in ('ece', 'nll', 'brier'):
        out[f'delta_{k}'] = ood_res.get(k, float('nan')) - id_res.get(k, float('nan'))
    out['delta_ue_auroc'] = (ood_res.get('ue_auroc', float('nan'))
                              - id_res.get('ue_auroc', float('nan')))
    out['ood_auroc'] = ood_auroc(u_id, u_ood)

    srtr = {}
    for tau in COV_FULL:
        ri = id_res.get(f'risk@{tau}', float('nan'))
        ro = ood_res.get(f'risk@{tau}', float('nan'))
        v  = (ro / ri) if (not math.isnan(ri) and ri > 0) else float('nan')
        srtr[tau] = v
        out[f'srtr@{tau}'] = v

    valid = [(t, v) for t, v in sorted(srtr.items()) if not math.isnan(v)]
    out['srtr_auc'] = (
        float(sum((valid[j][1] + valid[j + 1][1]) / 2 * (valid[j + 1][0] - valid[j][0])
                  for j in range(len(valid) - 1)))
        if len(valid) >= 2 else float('nan')
    )
    out['aurc_ood'] = ood_res.get('aurc', float('nan'))
    for k in ('_probs', '_u', '_correct', '_reliability_bins'):
        if k in ood_res:
            out[k] = ood_res[k]
    return out


def build_all_keys(binary):
    base  = (['acc', 'f1', 'prec', 'rec'] if binary else ['acc'])
    base += ['ece', 'nll', 'brier', 'ue_auroc', 'ue_aupr', 'aurc']
    base += [f'risk@{t}' for t in COV_FULL]
    extra  = ['delta_ece', 'delta_nll', 'delta_brier', 'delta_ue_auroc',
               'ood_auroc', 'aurc_ood']
    extra += [f'srtr@{t}' for t in COV_FULL]
    extra += ['srtr_auc', 'coverage@risk01', 'coverage@risk05', 'coverage@risk10']
    return base + extra


# ─────────────────────────────────────────────────────────────
# 5. 汇总输出
# ─────────────────────────────────────────────────────────────

import csv, os

SHOW_KEYS = ['acc', 'ece', 'nll', 'brier', 'ue_auroc', 'aurc']


def summarize(all_runs, split_names, all_keys, csv_path, title,
              reliability_path=None, uncertainty_path=None):
    """
    all_runs : list of {split_name: metric_dict}
    CalGNN 的 all_runs 结构为 {split_name: {cal_method: metric_dict}}，
    那种情况请直接调用 summarize_calgnn。
    """
    col_w = 18
    sep   = '═' * (26 + col_w * len(SHOW_KEYS))
    print(f'\n{sep}')
    print(f'  {title}  ({len(all_runs)} runs)')
    print(sep)
    print(f'  {"split":<24}' + ''.join(f'{k:>{col_w}}' for k in SHOW_KEYS))
    print('  ' + '─' * (24 + col_w * len(SHOW_KEYS)))

    mean_rows = [['split'] + [f'{k}_mean' for k in all_keys] + [f'{k}_std' for k in all_keys]]

    for sname in split_names:
        vals = defaultdict(list)
        for r in all_runs:
            for k in all_keys:
                v = r.get(sname, {}).get(k)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    vals[k].append(v)
        mu = {k: np.mean(vals[k]) if vals[k] else float('nan') for k in all_keys}
        sd = {k: np.std(vals[k])  if vals[k] else float('nan') for k in all_keys}
        print(f'  {sname[:24]:<24}' +
              ''.join(f'{mu[k]:.4f}±{sd[k]:.4f}'.rjust(col_w) for k in SHOW_KEYS))
        mean_rows.append([sname] + [f'{mu[k]:.6f}' for k in all_keys]
                                  + [f'{sd[k]:.6f}' for k in all_keys])

    ood_names = [n for n in split_names if n.startswith('OOD')]
    if ood_names:
        vals = defaultdict(list)
        for r in all_runs:
            for n in ood_names:
                for k in all_keys:
                    v = r.get(n, {}).get(k)
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        vals[k].append(v)
        mu = {k: np.mean(vals[k]) if vals[k] else float('nan') for k in all_keys}
        sd = {k: np.std(vals[k])  if vals[k] else float('nan') for k in all_keys}
        print(f'  {"OOD-avg":<24}' +
              ''.join(f'{mu[k]:.4f}±{sd[k]:.4f}'.rjust(col_w) for k in SHOW_KEYS))
        mean_rows.append(['OOD-avg'] + [f'{mu[k]:.6f}' for k in all_keys]
                                     + [f'{sd[k]:.6f}' for k in all_keys])

    print(f'\n{sep}')
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerows(mean_rows)
    print(f'  结果 → {csv_path}')

    if reliability_path:
        rows = [['run', 'split', 'bin', 'avg_confidence', 'accuracy', 'count']]
        for ri, r in enumerate(all_runs):
            for sname in split_names:
                for bi, (c, a, cnt) in enumerate(r.get(sname, {}).get('_reliability_bins', [])):
                    rows.append([ri, sname, bi + 1, f'{c:.6f}',
                                 f'{a:.6f}' if not math.isnan(a) else 'nan', cnt])
        with open(reliability_path, 'w', newline='') as f:
            csv.writer(f).writerows(rows)
        print(f'  Reliability → {reliability_path}')

    if uncertainty_path:
        rows = [['run', 'split', 'u', 'correct']]
        for ri, r in enumerate(all_runs):
            for sname in split_names:
                m = r.get(sname, {})
                u_arr, c_arr = m.get('_u'), m.get('_correct')
                if u_arr is not None:
                    for uv, cv in zip(u_arr.tolist(), c_arr.tolist()):
                        rows.append([ri, sname, f'{uv:.6f}', int(cv)])
        with open(uncertainty_path, 'w', newline='') as f:
            csv.writer(f).writerows(rows)
        print(f'  不确定性样本 → {uncertainty_path}')


def summarize_calgnn(all_runs, split_names, all_keys, csv_path, title,
                     cal_methods_out, reliability_path=None, uncertainty_path=None):
    """CalGNN 专用汇总：all_runs[i][split][cal_method] = metric_dict"""
    col_w = 18
    sep   = '═' * (30 + col_w * len(SHOW_KEYS))
    print(f'\n{sep}')
    print(f'  {title}  ({len(all_runs)} runs)')
    print(sep)

    mean_rows = [['split', 'cal_method'] +
                 [f'{k}_mean' for k in all_keys] + [f'{k}_std' for k in all_keys]]

    all_split_names = list(split_names) + (['OOD-avg'] if any(
        n.startswith('OOD') for n in split_names) else [])

    for sname in split_names:
        print(f'\n  [{sname}]')
        print(f'  {"cal_method":<14}' + ''.join(f'{k:>{col_w}}' for k in SHOW_KEYS))
        print('  ' + '─' * (14 + col_w * len(SHOW_KEYS)))
        for cm in cal_methods_out:
            vals = defaultdict(list)
            for r in all_runs:
                m = r.get(sname, {}).get(cm, {})
                for k in all_keys:
                    v = m.get(k)
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        vals[k].append(v)
            mu = {k: np.mean(vals[k]) if vals[k] else float('nan') for k in all_keys}
            sd = {k: np.std(vals[k])  if vals[k] else float('nan') for k in all_keys}
            print(f'  {cm:<14}' +
                  ''.join(f'{mu[k]:.4f}±{sd[k]:.4f}'.rjust(col_w) for k in SHOW_KEYS))
            mean_rows.append([sname, cm] + [f'{mu[k]:.6f}' for k in all_keys]
                                          + [f'{sd[k]:.6f}' for k in all_keys])

    ood_names = [n for n in split_names if n.startswith('OOD')]
    if ood_names:
        print(f'\n  [OOD-avg]')
        print(f'  {"cal_method":<14}' + ''.join(f'{k:>{col_w}}' for k in SHOW_KEYS))
        print('  ' + '─' * (14 + col_w * len(SHOW_KEYS)))
        for cm in cal_methods_out:
            vals = defaultdict(list)
            for r in all_runs:
                for n in ood_names:
                    m = r.get(n, {}).get(cm, {})
                    for k in all_keys:
                        v = m.get(k)
                        if v is not None and not (isinstance(v, float) and math.isnan(v)):
                            vals[k].append(v)
            mu = {k: np.mean(vals[k]) if vals[k] else float('nan') for k in all_keys}
            sd = {k: np.std(vals[k])  if vals[k] else float('nan') for k in all_keys}
            print(f'  {cm:<14}' +
                  ''.join(f'{mu[k]:.4f}±{sd[k]:.4f}'.rjust(col_w) for k in SHOW_KEYS))
            mean_rows.append(['OOD-avg', cm] + [f'{mu[k]:.6f}' for k in all_keys]
                                              + [f'{sd[k]:.6f}' for k in all_keys])

    print(f'\n{sep}')
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerows(mean_rows)
    print(f'  结果 → {csv_path}')
