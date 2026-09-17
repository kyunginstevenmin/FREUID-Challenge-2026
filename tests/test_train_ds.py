"""TrainDS synthetic-fraud injection (`pytest tests/`): OFF by default, label-flipping when
on, and reproducible for a run seed (the pre-2026-09-17 code used an unseeded Generator per
item, so the same seed produced different synthetic fraud every run)."""
import cv2
import numpy as np
import pytest
import torch

from train import TrainDS, build_parser


@pytest.fixture
def img_path(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (192, 288, 3), dtype=np.uint8)
    p = str(tmp_path / "genuine.jpg")
    cv2.imwrite(p, img)
    return p


def _ds(img_path, sbi):
    # groups=() -> no photometric transform, so only the injection can change the tensor
    return TrainDS(["g"], [0], ["x"], 96, 144, groups=(), sbi=sbi,
                   sources=["path"], paths=[img_path])


def test_injection_off_by_default():
    assert build_parser().get_default("sbi") == 0.0


def test_sbi_zero_keeps_genuine_label(img_path):
    x, y = _ds(img_path, 0.0)[0]
    assert y == 0.0 and tuple(x.shape) == (3, 96, 144)


def test_sbi_one_flips_label_to_fraud(img_path):
    _, y = _ds(img_path, 1.0)[0]
    assert y == 1.0


def test_injection_reproducible_per_seed(img_path):
    outs = []
    for _ in range(2):
        torch.manual_seed(7)
        outs.append(_ds(img_path, 1.0)[0][0])
    assert torch.equal(outs[0], outs[1])              # same seed -> same synthetic fraud
    torch.manual_seed(8)
    assert not torch.equal(outs[0], _ds(img_path, 1.0)[0][0])   # seed actually matters
