"""
gnn_uq_bench.calibration
=========================
后处理校准方法：TemperatureScaling / HistogramBinning / IsotonicCalib /
               BBQ / MetaCalMisCoverage / RBS (图拓扑感知)

所有 class 均实现 .fit() / .predict_proba() 或 .predict() 接口。
"""

import numpy as np
import scipy.special
import scipy.optimize
import torch
from sklearn.isotonic import IsotonicRegression as SklearnIso
from sklearn.model_selection import train_test_split
from scipy.stats import entropy as scipy_entropy


# ─────────────────────────────────────────────────────────────
class TemperatureScaling:
    """全局温度缩放（Guo et al., ICML 2017）"""
    def __init__(self): self.T = 1.0

    def fit(self, logits_np, labels_np):
        def obj(T):
            P = scipy.special.softmax(logits_np / T, axis=1)
            return -np.sum(np.log(P[np.arange(len(labels_np)), labels_np] + 1e-30))
        def grad(T):
            E  = np.exp(logits_np / T)
            li = logits_np[np.arange(len(labels_np)), labels_np]
            dT = (np.sum(E * (logits_np - li.reshape(-1, 1)), 1) / np.sum(E, 1)).sum()
            return -dT / T ** 2
        T_opt = scipy.optimize.fmin_bfgs(obj, x0=1.0, fprime=grad, gtol=1e-6, disp=False)[0]
        self.T = max(T_opt, 1e-3); return self

    def predict_proba(self, logits_np):
        return scipy.special.softmax(logits_np / self.T, axis=1)


# ─────────────────────────────────────────────────────────────
class HistogramBinning:
    """分位数直方图 binning（每类独立）"""
    def __init__(self, n_bins=20): self.n_bins = n_bins

    def fit(self, probs_np, labels_np):
        self.calibrators_ = []
        for c in range(probs_np.shape[1]):
            p1 = probs_np[:, c]; y1 = (labels_np == c).astype(float)
            bins = np.quantile(p1, np.linspace(0, 1, self.n_bins + 1))
            bins[0] = 0.; bins[-1] = 1.; bins = np.unique(bins)
            dig  = np.digitize(p1, bins).clip(1, len(bins) - 1)
            freq = np.array([y1[dig == i].mean() if (dig == i).sum() > 0 else 0.
                             for i in range(1, len(bins))])
            self.calibrators_.append((bins, freq))
        return self

    def predict_proba(self, probs_np):
        out = np.zeros_like(probs_np)
        for c, (bins, freq) in enumerate(self.calibrators_):
            dig = np.digitize(probs_np[:, c], bins).clip(1, len(bins) - 1)
            out[:, c] = np.array([freq[i - 1] for i in dig])
        return out / out.sum(1, keepdims=True).clip(min=1e-8)


# ─────────────────────────────────────────────────────────────
class IsotonicCalib:
    """等渗回归校准（每类独立）"""
    def fit(self, probs_np, labels_np):
        self.calibrators_ = []
        for c in range(probs_np.shape[1]):
            ir = SklearnIso(out_of_bounds='clip')
            ir.fit(probs_np[:, c], (labels_np == c).astype(float))
            self.calibrators_.append(ir)
        return self

    def predict_proba(self, probs_np):
        out = np.stack([ir.predict(probs_np[:, c])
                        for c, ir in enumerate(self.calibrators_)], axis=1)
        return (out / out.sum(1, keepdims=True).clip(min=1e-8)).clip(0, 1)


# ─────────────────────────────────────────────────────────────
class BBQ:
    """贝叶斯 Binning into Quantiles（Naeini et al., AAAI 2015）"""
    def __init__(self, C=10): self.C = C

    def _fit_binary(self, p1, y):
        import scipy.special as ss
        N = len(y)
        n_min = max(1, int(np.floor(N ** (1 / 3) / self.C)))
        n_max = min(max(int(np.ceil(N / 5)), int(np.ceil(self.C * N ** (1 / 3)))), 50)
        n_min = min(n_min, n_max)
        T = n_max - n_min + 1
        binnings, scores, freqs = [], [], []
        for nb in range(n_min, n_max + 1):
            bins = np.quantile(p1, np.linspace(0, 1, nb + 1))
            bins[0] = 0.; bins[-1] = 1.
            bins = np.maximum.accumulate(bins)
            dig  = np.digitize(p1, bins).clip(1, len(bins) - 1)
            N_b  = np.bincount(dig, minlength=len(bins))[1:]
            m_b  = np.array([y[dig == i].sum() for i in range(1, len(bins))])
            n_b  = N_b - m_b
            p_b  = np.clip((bins[1:] + bins[:-1]) / 2, 1e-9, 1 - 1e-9)
            al   = np.clip(N / T * p_b, 1e-9, None)
            bt   = np.clip(N / T * (1 - p_b), 1e-9, None)
            ll   = (ss.gammaln(N / T) + ss.gammaln(m_b + al) + ss.gammaln(n_b + bt)
                    - ss.gammaln(N_b + N / T) - ss.gammaln(al) - ss.gammaln(bt)).sum()
            scores.append(-np.log(T) + ll); binnings.append(bins)
            freqs.append([y[dig == i].mean() if (dig == i).sum() > 0 else p_b[i - 1]
                          for i in range(1, len(bins))])
        self._binnings = binnings; self._scores = scores; self._freqs = freqs

    def fit(self, probs_np, labels_np):
        self._models = []
        for c in range(probs_np.shape[1]):
            self._fit_binary(probs_np[:, c], (labels_np == c).astype(float))
            self._models.append((self._binnings, self._scores, self._freqs))
            self._binnings = self._scores = self._freqs = []
        return self

    def predict_proba(self, probs_np):
        import scipy.special as ss
        out = np.zeros_like(probs_np)
        for c, (binnings, scores, freqs) in enumerate(self._models):
            w   = np.exp(np.array(scores) - ss.logsumexp(scores))
            col = np.zeros(len(probs_np))
            for bins, freq, wi in zip(binnings, freqs, w):
                dig = np.searchsorted(bins, probs_np[:, c]).clip(0, len(bins) - 2)
                col += wi * np.array([freq[d] for d in dig])
            out[:, c] = col
        return (out / out.sum(1, keepdims=True).clip(min=1e-8)).clip(0, 1)


# ─────────────────────────────────────────────────────────────
class MetaCalTS:
    """MetaCal with Temperature Scaling base"""
    def fit(self, logits_np, labels_np):
        ts = TemperatureScaling(); ts.fit(logits_np, labels_np); self.T = ts.T; return self
    def predict(self, logits_np):
        return scipy.special.softmax(logits_np / self.T, axis=1)


class MetaCalMisCoverage:
    """MetaCal with mis-coverage correction（Zhao et al., ICML 2021）"""
    def __init__(self, alpha=0.05): self.alpha = alpha

    def fit(self, logits_np, labels_np):
        neg = logits_np.argmax(1) == labels_np
        xs_neg, ys_neg = logits_np[neg],  labels_np[neg]
        xs_pos, ys_pos = logits_np[~neg], labels_np[~neg]
        if len(xs_neg) <= 1:
            self.threshold = float('inf')
            self.base_model = MetaCalTS().fit(logits_np, labels_np)
            return self
        n1 = min(max(1, len(xs_neg) // 10), 500)
        x1, x2, _, y2 = train_test_split(xs_neg, ys_neg,
                                          train_size=min(n1, len(xs_neg) - 1), random_state=0)
        x2 = np.r_[x2, xs_pos]; y2 = np.r_[y2, ys_pos]
        s1 = scipy_entropy(scipy.special.softmax(x1, axis=1), axis=1)
        self.threshold = np.quantile(s1, 1 - self.alpha)
        s2   = scipy_entropy(scipy.special.softmax(x2, axis=1), axis=1)
        cond = s2 < self.threshold
        self.base_model = MetaCalTS()
        self.base_model.fit(x2[cond] if cond.sum() > 1 else logits_np,
                            y2[cond] if cond.sum() > 1 else labels_np)
        return self

    def predict(self, logits_np):
        s       = scipy_entropy(scipy.special.softmax(logits_np, axis=1), axis=1)
        neg_ind = s < self.threshold
        out     = np.full_like(logits_np, 1.0 / logits_np.shape[1], dtype=float)
        if neg_ind.sum() > 0:
            out[neg_ind] = self.base_model.predict(logits_np[neg_ind])
        return out


# ─────────────────────────────────────────────────────────────
# RBS (Relative Binning Scaling) — 图拓扑感知校准
# ─────────────────────────────────────────────────────────────

def compute_sm_conf(edge_index, N, probs_t, device):
    """
    Smoothed confidence：邻域平均 softmax 置信度。
    edge_index : (2, E) LongTensor on device
    probs_t    : (N, C) FloatTensor on device
    返回 numpy (N,)
    """
    src, dst = edge_index[0], edge_index[1]
    vals = torch.ones(src.size(0), dtype=torch.float32, device=device)
    A    = torch.sparse_coo_tensor(
        torch.stack([dst, src], 0), vals, (N, N)).coalesce()
    AP   = torch.sparse.mm(A, probs_t)
    deg  = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.)
    AP   = AP / deg.unsqueeze(1)
    pred = probs_t.argmax(1)
    sm   = AP[torch.arange(N, device=device), pred]
    return sm.cpu().numpy()


def rbs_fit(sm_conf_sub, logits_np, labels_np, num_bins):
    """拟合 RBS per-bin 温度列表"""
    bins   = [1. / num_bins * (i + 1) for i in range(num_bins)]
    T_list = []
    for i, b in enumerate(bins):
        lo   = 0. if i == 0 else bins[i - 1]
        mask = (sm_conf_sub > lo) & (sm_conf_sub <= b)
        if mask.sum() > 1:
            ts = TemperatureScaling(); ts.fit(logits_np[mask], labels_np[mask])
            T_list.append(ts.T)
        else:
            T_list.append(1.0)
    return T_list, bins


def apply_rbs(T_list, bins, sm_conf_sub, logits_np, device):
    """应用 RBS 输出校准后 softmax 概率"""
    T_vec = np.ones(len(logits_np), dtype=np.float32)
    for i, b in enumerate(bins):
        lo = 0. if i == 0 else bins[i - 1]
        m  = (sm_conf_sub > lo) & (sm_conf_sub <= b)
        T_vec[m] = T_list[i]
    with torch.no_grad():
        t = torch.tensor(logits_np, dtype=torch.float32, device=device)
        v = torch.tensor(T_vec, dtype=torch.float32, device=device).unsqueeze(1)
        return torch.softmax(t / v, dim=1).cpu().numpy()
