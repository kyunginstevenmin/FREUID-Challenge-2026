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
