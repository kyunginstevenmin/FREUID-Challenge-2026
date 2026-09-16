"""Unit tests for evaluate.py's pure parts (bootstrap CI, checkpoint hashing, cache
naming). Inference itself needs a GPU + real data and is exercised at REF anchoring
(TASKS.md 3b)."""
import numpy as np
import pytest

from evaluate import bootstrap_ci, ckpt_sha, cache_path
from freuid_metric import freuid_score


@pytest.fixture(scope="module")
def scored_sample():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    s = rng.normal(loc=y * 1.2, scale=1.0)
    return y, s


def test_bootstrap_deterministic(scored_sample):
    y, s = scored_sample
    a = bootstrap_ci(y, s, n_boot=200, seed=7)
    b = bootstrap_ci(y, s, n_boot=200, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    c = bootstrap_ci(y, s, n_boot=200, seed=8)
    assert not np.array_equal(a[0], c[0])          # seed actually matters


def test_bootstrap_brackets_point_estimate(scored_sample):
    y, s = scored_sample
    point = np.asarray(freuid_score(y, s))
    lo, hi = bootstrap_ci(y, s, n_boot=300, seed=0)
    assert (lo <= point).all() and (point <= hi).all()
    assert (lo < hi).all()                          # nonzero width


def test_bootstrap_survives_tiny_minority():
    # 1990 vs 10: many resamples lose the minority class; the redraw loop must cope.
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(1990), np.ones(10)].astype(int)
    s = rng.normal(loc=y * 2.0, scale=1.0)
    lo, hi = bootstrap_ci(y, s, n_boot=50, seed=0)
    assert (lo <= hi).all()


def test_ckpt_sha(tmp_path):
    p1, p2, p3 = (tmp_path / n for n in ["a.pt", "b.pt", "c.pt"])
    p1.write_bytes(b"same bytes"); p2.write_bytes(b"same bytes"); p3.write_bytes(b"other")
    assert ckpt_sha(str(p1)) == ckpt_sha(str(p2))   # content-addressed, not path-addressed
    assert ckpt_sha(str(p1)) != ckpt_sha(str(p3))
    assert len(ckpt_sha(str(p1))) == 12


def test_cache_path_distinguishes_all_axes():
    keys = {cache_path(sha, prot, ro)
            for sha in ["aaa", "bbb"] for prot in ["tier1", "external"] for ro in ["raw", "deploy"]}
    assert len(keys) == 8                           # every (ckpt, protocol, readout) is distinct


# ---- two-level macro (EVALUATION.md gauge-reporting rule) ----
from evaluate import macro_scores, bootstrap_macro_ci


def _two_source_sample():
    # source A: type a1 perfect (FREUID 0), type a2 inverted (FREUID 1); source B: b1 perfect.
    y = [0, 0, 1, 1,  0, 1,  0, 0, 1, 1]
    s = [.1, .2, .8, .9,  .9, .1,  .1, .2, .8, .9]
    t = ["a1"] * 4 + ["a2"] * 2 + ["b1"] * 4
    src = ["A"] * 6 + ["B"] * 4
    return np.array(y), np.array(s), np.array(t), np.array(src)


def test_macro_scores_hand_computed():
    per_type, per_source, mt, ms = macro_scores(*_two_source_sample())
    assert per_type == {"a1": 0.0, "a2": 1.0, "b1": 0.0}
    assert per_source == {"A": 0.5, "B": 0.0}
    assert abs(mt - 1 / 3) < 1e-12          # 3 types weighted equally
    assert ms == 0.25                        # 2 sources weighted equally: A's two types share A's weight


def test_macro_source_differs_from_macro_type_when_sources_are_unbalanced():
    # Same per-type scores, but the two macros disagree exactly when one source has more types.
    _, _, mt, ms = macro_scores(*_two_source_sample())
    assert mt != ms


def test_bootstrap_macro_ci_deterministic_and_brackets():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    s = rng.normal(loc=y * 1.2, scale=1.0)
    t = np.repeat(["a1", "a2", "b1"], 200)
    src = np.where(t == "b1", "B", "A")
    point = np.asarray(macro_scores(y, s, t, src)[2:])
    a = bootstrap_macro_ci(y, s, t, src, n_boot=100, seed=3)
    b = bootstrap_macro_ci(y, s, t, src, n_boot=100, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert (a[0] <= point).all() and (point <= a[1]).all() and (a[0] < a[1]).all()


def test_bootstrap_macro_ci_survives_tiny_type():
    # A 2-row type loses a class in half of its resamples; the redraw loop must cope.
    y, s, t, src = _two_source_sample()
    lo, hi = bootstrap_macro_ci(y, s, t, src, n_boot=30, seed=0)
    assert (lo <= hi).all()
