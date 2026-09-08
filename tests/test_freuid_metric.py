"""Unit tests for the exact competition metric (`pytest tests/`).

Strategy: (1) tiny HAND-COMPUTED cases -- the only guard against a wrong-but-
self-consistent implementation; (2) invariants (identity vs trapezoid sweep,
tie convention, monotone invariance) on seeded data. Needs numpy + scikit-learn.
"""
import numpy as np
import pytest

from freuid_metric import audet, _audet_trapz, apcer_at_bpcer, freuid_score


@pytest.mark.parametrize("y, s, expected", [
    ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.0),    # perfect ranking
    ([0, 1], [0.9, 0.1], 1.0),                    # perfectly inverted
    ([0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9], 0.25),   # one swapped pair: 3/4 correct
    ([0, 1], [0.5, 0.5], 0.5),                    # shared score -> half credit
], ids=["perfect", "inverted", "one-swap", "tie"])
def test_audet_hand_computed(y, s, expected):
    assert abs(audet(y, s) - expected) < 1e-12


def test_perfect_ranking_is_zero_everywhere():
    f, a, p = freuid_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert (f, a, p) == (0.0, 0.0, 0.0)


def test_all_tied_is_chance():
    y = np.array([0, 1] * 100)
    assert audet(y, np.full(200, 7.0)) == 0.5


def test_apcer_at_bpcer_hand_computed():
    # 101 genuine at 0.00..1.00: the 99% quantile is 0.99; attacks below it are misses.
    gen = np.linspace(0, 1, 101)
    att = np.array([0.5, 0.98, 0.995, 1.0])         # 2 of 4 below t*=0.99
    y = np.r_[np.zeros(101), np.ones(4)].astype(int)
    assert abs(apcer_at_bpcer(y, np.r_[gen, att], 0.01) - 0.5) < 1e-12


def test_apcer_zero_when_separable():
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    s = np.r_[np.random.default_rng(0).random(100) * 0.4, 0.6 + np.zeros(100)]
    assert apcer_at_bpcer(y, s, 0.01) == 0.0


def test_freuid_formula():
    # freuid_score must equal the documented harmonic-mean formula of its own parts.
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 5000)
    s = rng.normal(loc=y * 1.0, scale=1.0)
    f, a, p = freuid_score(y, s)
    g1, g2 = 1 - a, 1 - p
    assert abs(f - (1 - 2 * g1 * g2 / (g1 + g2))) < 1e-12


def test_identity_vs_trapz():
    # AuDET via 1-AUC must match the direct DET-area sweep on CONTINUOUS scores.
    # (On coarse/tied score grids the trapezoid sweep is only an approximation --
    # measured ~5e-3 off -- so ties are pinned exactly in the hand cases instead.)
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 4000)
    sc = rng.normal(0, 1, 4000) + y * 0.8
    assert abs(audet(y, sc) - _audet_trapz(y, sc)) < 2e-3


@pytest.mark.parametrize("transform", [
    lambda s: 1 / (1 + np.exp(-3 * s)),   # sigmoid
    lambda s: s * 7.0 - 100.0,            # affine
], ids=["sigmoid", "affine"])
def test_monotone_invariance(transform):
    # The metric may only depend on ranking: monotone transforms are no-ops.
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 4000)
    s = rng.normal(loc=y * 1.2, scale=1.0)
    assert np.allclose(freuid_score(y, s), freuid_score(y, transform(s)), atol=1e-9)
