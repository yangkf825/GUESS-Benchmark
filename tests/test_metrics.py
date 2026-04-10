"""Unit tests for gnn_uq_bench.metrics."""

import math
import numpy as np
import pytest

from gnn_uq_bench.metrics.calibration import ece, nll, brier, reliability_bins
from gnn_uq_bench.metrics.classification import accuracy, f1_binary
from gnn_uq_bench.metrics.uncertainty import entropy_vec, ue_auroc, ood_auroc
from gnn_uq_bench.metrics.risk_coverage import risk_curve, aurc
from gnn_uq_bench.metrics.composite import (
    compute_split_metrics, add_cross_split_metrics, build_all_keys,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_binary():
    """Perfect binary classifier."""
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1, 0, 1])
    return probs, labels


@pytest.fixture
def random_binary():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1], size=200)
    labels = rng.integers(0, 2, size=200)
    return probs, labels


# ── Calibration ───────────────────────────────────────────────────────────────

def test_ece_perfect(perfect_binary):
    probs, labels = perfect_binary
    assert ece(probs, labels) == pytest.approx(0.0, abs=1e-6)


def test_ece_returns_float(random_binary):
    probs, labels = random_binary
    result = ece(probs, labels)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_nll_perfect(perfect_binary):
    probs, labels = perfect_binary
    assert nll(probs, labels) == pytest.approx(0.0, abs=1e-6)


def test_brier_perfect(perfect_binary):
    probs, labels = perfect_binary
    assert brier(probs, labels, nclass=2) == pytest.approx(0.0, abs=1e-6)


def test_reliability_bins_length(random_binary):
    probs, labels = random_binary
    bins = reliability_bins(probs, labels, n_bins=15)
    assert len(bins) == 15
    for avg_c, acc, cnt in bins:
        assert 0.0 <= avg_c <= 1.0
        assert cnt >= 0


# ── Classification ────────────────────────────────────────────────────────────

def test_accuracy_perfect(perfect_binary):
    probs, labels = perfect_binary
    assert accuracy(probs, labels) == pytest.approx(1.0)


def test_f1_perfect(perfect_binary):
    probs, labels = perfect_binary
    f1, pr, re = f1_binary(probs, labels)
    assert f1 == pytest.approx(1.0, abs=1e-6)
    assert pr == pytest.approx(1.0, abs=1e-6)
    assert re == pytest.approx(1.0, abs=1e-6)


# ── Uncertainty ───────────────────────────────────────────────────────────────

def test_entropy_uniform():
    probs = np.full((10, 2), 0.5)
    u = entropy_vec(probs)
    assert u.shape == (10,)
    expected = -2 * 0.5 * math.log(0.5)
    assert np.allclose(u, expected, atol=1e-5)


def test_entropy_deterministic():
    probs = np.eye(3)[[0, 1, 2]]
    u = entropy_vec(probs)
    assert np.allclose(u, 0.0, atol=1e-5)


def test_ue_auroc_shape(random_binary):
    probs, labels = random_binary
    u = entropy_vec(probs)
    auroc_val, aupr_val = ue_auroc(u, probs, labels)
    assert isinstance(auroc_val, float)
    assert isinstance(aupr_val, float)


def test_ood_auroc_perfect():
    u_id  = np.zeros(100)
    u_ood = np.ones(100)
    assert ood_auroc(u_id, u_ood) == pytest.approx(1.0)


# ── Risk-coverage ─────────────────────────────────────────────────────────────

def test_risk_curve_keys(random_binary):
    probs, labels = random_binary
    u  = entropy_vec(probs)
    rc = risk_curve(probs, u, labels)
    expected_keys = [round(0.1 * i, 1) for i in range(1, 11)]
    assert set(rc.keys()) == set(expected_keys)


def test_aurc_range(random_binary):
    probs, labels = random_binary
    u = entropy_vec(probs)
    rc = risk_curve(probs, u, labels)
    a  = aurc(rc)
    assert 0.0 <= a <= 1.0


# ── Composite ─────────────────────────────────────────────────────────────────

def test_compute_split_metrics_keys(random_binary):
    probs, labels = random_binary
    u = entropy_vec(probs)
    res = compute_split_metrics(probs, u, labels, nclass=2, binary=True)
    for key in ["acc", "f1", "ece", "nll", "brier", "ue_auroc", "aurc"]:
        assert key in res


def test_build_all_keys_binary():
    keys = build_all_keys(binary=True)
    assert "f1" in keys
    assert "acc" in keys
    assert "ood_auroc" in keys
    assert "srtr_auc" in keys


def test_build_all_keys_multiclass():
    keys = build_all_keys(binary=False)
    assert "f1" not in keys
    assert "acc" in keys


def test_add_cross_split_metrics(random_binary):
    probs, labels = random_binary
    u = entropy_vec(probs)
    r_id  = compute_split_metrics(probs, u, labels, nclass=2)
    r_ood = compute_split_metrics(probs, u, labels, nclass=2)
    out   = add_cross_split_metrics(r_id, r_ood, u, u)
    assert "delta_ece" in out
    assert "ood_auroc" in out
    assert "srtr_auc" in out
