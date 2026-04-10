"""Unit tests for post-hoc calibration methods."""

import numpy as np
import pytest

from gnn_uq_bench.post_processing import (
    TemperatureScaling, HistogramBinning, IsotonicCalib,
    BBQ, MetaCalTS, MetaCalMisCoverage,
)


@pytest.fixture
def logits_labels():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 1, (200, 3))
    labels = rng.integers(0, 3, 200)
    return logits, labels


def _is_valid_probs(p: np.ndarray) -> bool:
    return (
        p.shape[1] >= 2
        and np.allclose(p.sum(1), 1.0, atol=1e-5)
        and (p >= 0).all()
    )


def test_temperature_scaling(logits_labels):
    logits, labels = logits_labels
    ts = TemperatureScaling().fit(logits, labels)
    assert ts.T > 0
    probs = ts.predict_proba(logits)
    assert _is_valid_probs(probs)


def test_histogram_binning(logits_labels):
    logits, labels = logits_labels
    import scipy.special
    probs = scipy.special.softmax(logits, axis=1)
    hb = HistogramBinning().fit(probs, labels)
    out = hb.predict_proba(probs)
    assert _is_valid_probs(out)


def test_isotonic(logits_labels):
    logits, labels = logits_labels
    import scipy.special
    probs = scipy.special.softmax(logits, axis=1)
    iso = IsotonicCalib().fit(probs, labels)
    out = iso.predict_proba(probs)
    assert _is_valid_probs(out)


def test_bbq(logits_labels):
    logits, labels = logits_labels
    import scipy.special
    probs = scipy.special.softmax(logits, axis=1)
    bbq = BBQ().fit(probs, labels)
    out = bbq.predict_proba(probs)
    assert _is_valid_probs(out)


def test_metacal_ts(logits_labels):
    logits, labels = logits_labels
    mc = MetaCalTS().fit(logits, labels)
    out = mc.predict(logits)
    assert _is_valid_probs(out)


def test_metacal_miscoverage(logits_labels):
    logits, labels = logits_labels
    mc = MetaCalMisCoverage().fit(logits, labels)
    out = mc.predict(logits)
    assert out.shape == logits.shape
