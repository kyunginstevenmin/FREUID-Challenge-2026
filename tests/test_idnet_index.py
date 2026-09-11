"""Tests for make_idnet_index.py.

Two tiers:
  1. Synthetic-fixture tests -- run anywhere (CI included): build a tiny fake IDNet
     tree in tmp_path and verify construction, determinism, filtering, collisions.
  2. Real-data acceptance tests -- run only where external/idnet exists (the ingest
     box); skipped cleanly elsewhere. Counts per the plan's acceptance criteria.
"""
import os
import subprocess
import sys

import pandas as pd
import pytest

from make_idnet_index import build_rows, row_hash, REPO

FRAUDS = ["fraud1_copy_and_move", "fraud2_face_morphing"]
CONFIG_TYPES = {"ESP_scanned", "ALB_scanned", "AZE_scanned", "FIN_scanned", "GRC_scanned",
                "LVA_scanned", "RUS_scanned", "SRB_scanned", "EST_scanned", "SVK_scanned"}


@pytest.fixture
def fake_tree(tmp_path):
    # 2 countries x (positive + 2 fraud folders) x 3 images; fraud reuses positive
    # basenames (the real release's collision trap) + a meta/ folder to be ignored.
    for country in ["est", "fin"]:
        for folder in ["positive"] + FRAUDS:
            d = tmp_path / country / folder
            d.mkdir(parents=True)
            for i in range(3):
                (d / f"doc_{i}.png").write_bytes(b"\xff\xd8fakejpeg")
        meta = tmp_path / country / "meta"
        meta.mkdir()
        (meta / "annotation.json").write_text("{}")
    return tmp_path


def test_build_structure_and_labels(fake_tree):
    df = build_rows(str(fake_tree))
    assert len(df) == 2 * 3 * 3                      # meta/ and .json ignored
    assert list(df.columns) == ["id", "path", "label", "type", "source"]
    assert df.id.is_unique                           # collision trap solved by relpath ids
    assert set(df.type) == {"EST_scanned", "FIN_scanned"}
    assert (df.source == "idnet").all()
    pos = df.id.str.contains("/positive/")
    assert (df.label[pos] == 0).all() and (df.label[~pos] == 1).all()
    assert (df.path.map(os.path.isabs)).all()


def test_same_basename_different_folders_distinct_rows(fake_tree):
    df = build_rows(str(fake_tree))
    doc0 = df[df.id.str.endswith("doc_0.png") & df.id.str.startswith("est/")]
    assert len(doc0) == 3                            # positive + 2 fraud variants
    assert set(doc0.label) == {0, 1}


def test_determinism_and_row_hash(fake_tree):
    a, b = build_rows(str(fake_tree)), build_rows(str(fake_tree))
    assert a.equals(b)
    assert row_hash(a) == row_hash(b)
    # path is machine-specific and must NOT affect identity: rebuild under a
    # different root with identical content -> same hash.
    import shutil
    clone = fake_tree.parent / "clone"
    shutil.copytree(fake_tree, clone)
    assert row_hash(build_rows(str(clone))) == row_hash(a)


def test_fraud_folder_filter_is_subset(fake_tree, tmp_path):
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, os.path.join(REPO, "src", "make_idnet_index.py"),
                        "--root", str(fake_tree), "--out-dir", str(out),
                        "--fraud-folders", "fraud1_copy_and_move"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    full = pd.read_csv(out / "idnet_full_index.csv")
    crop = pd.read_csv(out / "idnet_cropped_index.csv")
    assert len(crop) == 2 * 2 * 3                    # positive + 1 fraud folder
    assert set(crop.id) <= set(full.id)              # strict row-subset
    assert list(crop.id) == [i for i in full.id if i in set(crop.id)]   # order preserved
    assert not crop.id.str.contains("fraud2").any()


def test_unknown_fraud_folder_rejected(fake_tree, tmp_path):
    r = subprocess.run([sys.executable, os.path.join(REPO, "src", "make_idnet_index.py"),
                        "--root", str(fake_tree), "--out-dir", str(tmp_path / "o"),
                        "--fraud-folders", "fraud9_typo"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "fraud9_typo" in (r.stderr + r.stdout)


# ---------------- real-data tier (ingest box only) ----------------

DATA_ROOT = os.path.join(REPO, "external", "idnet")
needs_data = pytest.mark.skipif(not os.path.isdir(DATA_ROOT),
                                reason="external/idnet not present (ingest box only)")


@needs_data
def test_real_tree_counts_and_types():
    df = build_rows(DATA_ROOT)
    assert set(df.type) <= CONFIG_TYPES              # exact frozen config strings
    folder = df.id.str.split("/").str[1]
    per = df.groupby([df.type, folder]).size()
    assert (per == 5979).all(), per[per != 5979]     # every extracted folder complete


@needs_data
def test_real_sample_images_open():
    import cv2
    df = build_rows(DATA_ROOT)
    sample = df.sample(min(50, len(df)), random_state=0)
    dims_by_country = {}
    for t in sample.itertuples(index=False):
        img = cv2.imread(t.path)
        assert img is not None, t.path
        c = t.id.split("/")[0]
        dims_by_country.setdefault(c, set()).add(img.shape[:2])
    for c, dims in dims_by_country.items():
        assert len(dims) == 1, f"{c}: non-uniform dims {dims}"   # per-country uniformity
