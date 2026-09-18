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
