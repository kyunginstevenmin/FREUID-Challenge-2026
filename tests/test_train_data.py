"""--split_dir mode of train.py (`pytest tests/`): the dataset_final contract is read into
(tr, va, gen), paths resolve against data_root, legacy data flags are rejected loudly, and
--limit keeps every source / gen-val type. Fixture = a 3-source miniature dataset_final
with real (tiny) jpgs under tmp_path, so the path-existence assert is exercised."""
import os

import cv2
import numpy as np
import pandas as pd
import pytest

from train import build_frames_split, load_image, resolve_args, ValDS

SOURCES = {"freuid": "dataset_cls/FREUID/FREUID_train", "sidt": "dataset_cls/SIDT/x",
           "fantasyid": "dataset_cls/FantasyID/y"}


def _write(root, rel):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), np.full((32, 48, 3), 127, np.uint8))
    return rel


@pytest.fixture
def split_dir(tmp_path):
    """<root>/dataset_final/{train,val,test}.csv; train/val: 3 sources x (2 genuine + 2 fraud);
    test: 2 unseen types from a 4th source x (1 genuine + 1 fraud)."""
    root = tmp_path; sd = root / "dataset_final"; sd.mkdir()
    def rows(split, spec):
        out = []
        for src, typ, n_each in spec:
            for lab in (0, 1):
                for k in range(n_each):
                    idv = f"{split}_{src}_{typ}_{lab}_{k}"
                    rel = _write(root, f"{SOURCES.get(src, 'dataset_cls/IDNet_v2/Images')}/{idv}.jpg")
                    out.append(dict(id=idv, image_path=rel, label=lab, type=typ, source=src))
        return pd.DataFrame(out)
    tv = [(s, f"{s}_type", 2) for s in SOURCES]
    rows("train", tv).to_csv(sd / "train.csv", index=False)
    rows("val", tv).to_csv(sd / "val.csv", index=False)
    rows("test", [("idnet", "idnet_esp", 1), ("idnet", "idnet_fin", 1)]).to_csv(sd / "test.csv", index=False)
    return sd


def test_frames_follow_the_contract(split_dir):
    tr, va, gen, vname = build_frames_split(resolve_args(["--split_dir", str(split_dir)]))
    assert (len(tr), len(va), len(gen)) == (12, 12, 4)
    assert list(tr.columns) == ["id", "label", "type", "source", "path"]
    assert all(os.path.exists(p) for d in (tr, va, gen) for p in d.path)   # data_root defaulted to parent
    assert set(tr.source) == set(SOURCES) and set(gen.type) == {"idnet_esp", "idnet_fin"}
    assert vname == "dataset_final"


def test_explicit_data_root_wins(split_dir, tmp_path):
    bad = tmp_path / "elsewhere"; bad.mkdir()
    with pytest.raises(AssertionError, match="check --data_root"):
        build_frames_split(resolve_args(["--split_dir", str(split_dir), "--data_root", str(bad)]))


@pytest.mark.parametrize("legacy", [["--full_data"], ["--holdout", "EGYPT/DL"], ["--fold", "0"],
                                    ["--idnet_countries", ""], ["--lim_idn", "10"]])
def test_legacy_flags_are_rejected(split_dir, legacy):
    with pytest.raises(SystemExit, match="exclusive with the legacy data flags"):
        build_frames_split(resolve_args(["--split_dir", str(split_dir)] + legacy))


def test_limit_keeps_every_source_and_gen_type(split_dir):
    tr, va, gen, _ = build_frames_split(resolve_args(["--split_dir", str(split_dir), "--limit", "6"]))
    assert set(tr.source) == set(SOURCES) and len(tr) == 6          # 3 sources x 2 labels x 1
    assert set(gen.type) == {"idnet_esp", "idnet_fin"} and set(gen.label) == {0, 1}


def test_id_overlap_across_splits_is_an_error(split_dir):
    va = pd.read_csv(split_dir / "val.csv"); tr = pd.read_csv(split_dir / "train.csv")
    va.loc[0, "id"] = tr.id[0]; va.to_csv(split_dir / "val.csv", index=False)
    with pytest.raises(AssertionError, match="id overlap"):
        build_frames_split(resolve_args(["--split_dir", str(split_dir)]))


def test_load_image_prefers_path(split_dir):
    tr, *_ = build_frames_split(resolve_args(["--split_dir", str(split_dir)]))
    img = load_image("freuid", "not-a-real-kaggle-id", tr.path[0])   # id route would fail
    assert img.shape == (32, 48, 3)
    x, y = ValDS(tr.id.tolist(), tr.label.values, 16, 24, paths=tr.path.tolist())[0]
    assert tuple(x.shape) == (3, 16, 24) and y == float(tr.label[0])


# ---- step 2: gen-val selection inputs ----
import torch
from train import genval_metrics, lean_state


def test_genval_metrics_matches_hand_computed_macros():
    # source A: type a1 perfect (0), a2 inverted (1); source B: b1 perfect -> macro_type 1/3, macro_source 1/4
    gen = pd.DataFrame({"label": [0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
                        "type": ["a1"] * 4 + ["a2"] * 2 + ["b1"] * 4,
                        "source": ["A"] * 6 + ["B"] * 4})
    m = genval_metrics(gen, np.array([.1, .2, .8, .9, .9, .1, .1, .2, .8, .9]))
    assert m["per_source"] == {"A": 0.5, "B": 0.0} and m["macro_source"] == 0.25
    assert abs(m["macro_type"] - 1 / 3) < 1e-12
    assert 0.0 < m["freuid"] < 1.0                         # pooled is a different number


def test_lean_state_keeps_trainable_and_head_only():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2); self.backbone.weight.requires_grad_(False)
            self.backbone.bias.requires_grad_(False)
            self.head = torch.nn.BatchNorm1d(2)             # has running-stat buffers
            self.lora = torch.nn.Linear(2, 2)              # trainable, outside head
    keys = set(lean_state(M()))
    assert "backbone.weight" not in keys and "backbone.bias" not in keys
    assert {"lora.weight", "lora.bias", "head.weight", "head.bias", "head.running_mean"} <= keys


# ---- step 2: offline epoch freeze (tie rule) ----
from freeze_epoch import epoch_table, pick_frozen_epoch


def test_frozen_epoch_is_earliest_inside_best_ci():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    gen = pd.DataFrame({"label": y, "type": np.repeat(["a1", "a2", "b1"], 200),
                        "source": np.repeat(["A", "A", "B"], 200)})
    sep = {0: 0.3, 1: 1.2, 2: 1.25, 3: 1.22}        # ep1..3 are indistinguishable, ep0 is bad
    preds = {ep: rng.normal(loc=y * s, scale=1.0) for ep, s in sep.items()}
    t = epoch_table(gen, preds, n_boot=100)
    assert list(t.epoch) == [0, 1, 2, 3] and (t.ci_lo <= t.macro_source).all()
    assert t.loc[t.epoch == 0, "macro_source"].item() > t.ci_hi[1:].max()   # ep0 clearly worse
    assert pick_frozen_epoch(t) == 1                                          # not the argmax epoch


def test_one_se_band_is_narrower_than_ci95():
    from freeze_epoch import tie_threshold
    t = pd.DataFrame({"epoch": [0, 1, 2], "macro_source": [0.50, 0.40, 0.41],
                      "ci_lo": [0.45, 0.36, 0.37], "ci_hi": [0.55, 0.44, 0.45]})
    best = t.loc[t.macro_source.idxmin()]
    assert tie_threshold(best, "1se") < tie_threshold(best, "ci95") == 0.44
    assert abs(tie_threshold(best, "1se") - (0.40 + 0.08 / 3.92)) < 1e-12
    assert pick_frozen_epoch(t, "ci95") == 1 and pick_frozen_epoch(t, "1se") == 1   # ep2 (0.41) tied under ci95 only if it were earlier
    t.loc[2, "epoch"], t.loc[1, "epoch"] = 1, 2                                    # swap order: the 0.41 epoch is now earlier
    assert pick_frozen_epoch(t, "ci95") == 1 and pick_frozen_epoch(t, "1se") == 2  # 1se excludes it (0.41 > 0.4204)
