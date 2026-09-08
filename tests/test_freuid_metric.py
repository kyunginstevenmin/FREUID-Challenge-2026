"""Unit tests for the exact competition metric. No framework: `python tests/test_freuid_metric.py`.

Strategy: (1) tiny HAND-COMPUTED cases -- the only guard against a wrong-but-
self-consistent implementation; (2) the invariants from freuid_metric's own
__main__ block (identity vs trapezoid sweep, monotone invariance) on seeded data.
Needs only numpy + scikit-learn.
"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from freuid_metric import audet, _audet_trapz, apcer_at_bpcer, freuid_score


def test_hand_computed():
    # Perfect ranking: every attack above every genuine -> AuDET 0, FREUID 0.
    f, a, p = freuid_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert a == 0.0 and p == 0.0 and f == 0.0, (f, a, p)

    # Perfectly inverted ranking -> AuDET 1 (worst).
    assert audet([0, 1], [0.9, 0.1]) == 1.0

    # One swapped pair: 4 (gen, att) pairs, 3 ranked correctly -> AUC 0.75, AuDET 0.25.
    assert abs(audet([0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9]) - 0.25) < 1e-12

    # Chance-level: interleaved identical rankings -> AuDET 0.5.
    y = np.array([0, 1] * 500)
    s = np.tile(np.arange(500).repeat(2), 1)[:1000].astype(float)  # gen/att share every score
    assert abs(audet(y, s) - 0.5) < 1e-12


def test_apcer_at_bpcer():
    # 101 genuine at 0.00..1.00: the 99% quantile is 0.99; attacks below it are misses.
    gen = np.linspace(0, 1, 101)
    att = np.array([0.5, 0.98, 0.995, 1.0])         # 2 of 4 below t*=0.99
    y = np.r_[np.zeros(101), np.ones(4)].astype(int)
    s = np.r_[gen, att]
    assert abs(apcer_at_bpcer(y, s, 0.01) - 0.5) < 1e-12

    # All attacks above every genuine -> APCER 0 at any target.
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
    # measured ~5e-3 off -- so tie handling is pinned exactly in test_ties instead.)
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 4000)
    sc = rng.normal(0, 1, 4000) + y * 0.8
    assert abs(audet(y, sc) - _audet_trapz(y, sc)) < 2e-3


def test_ties():
    # Exact tie-credit convention of the production audet(): a shared score
    # counts half, so one gen + one att at the same value -> AuDET 0.5 ...
    assert audet([0, 1], [0.5, 0.5]) == 0.5
    # ... and everything tied at one value is exactly chance.
    y = np.array([0, 1] * 100)
    assert audet(y, np.full(200, 7.0)) == 0.5


def test_monotone_invariance():
    # The metric may only depend on ranking: sigmoid/affine transforms are no-ops.
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 4000)
    s = rng.normal(loc=y * 1.2, scale=1.0)
    f1 = freuid_score(y, s)
    f2 = freuid_score(y, 1 / (1 + np.exp(-3 * s)))
    f3 = freuid_score(y, s * 7.0 - 100.0)
    assert np.allclose(f1, f2, atol=1e-9) and np.allclose(f1, f3, atol=1e-9)


if __name__ == "__main__":
    test_hand_computed()
    test_apcer_at_bpcer()
    test_freuid_formula()
    test_identity_vs_trapz()
    test_ties()
    test_monotone_invariance()
    print("freuid_metric: all tests passed")
