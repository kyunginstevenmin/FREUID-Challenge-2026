"""Unit tests for data.py preprocessing (`pytest tests/`). Needs numpy, cv2, torch."""
import numpy as np
import pytest

from data import letterbox, to_tensor_norm, IMAGENET_MEAN, IMAGENET_STD


def test_letterbox_wide_image_pads_rows():
    # Width binds (200->728 is the smaller scale): full-width content, rows padded evenly.
    out = letterbox(np.full((100, 200, 3), 200, np.uint8), 448, 728)
    assert out.shape == (448, 728, 3) and out.dtype == np.uint8
    nh = round(100 * 728 / 200)                       # 364
    y0 = (448 - nh) // 2                              # 42
    assert (out[:y0] == 0).all() and (out[y0 + nh:] == 0).all()
    assert (out[y0:y0 + nh] == 200).all()


def test_letterbox_tall_image_pads_cols():
    out = letterbox(np.full((300, 100, 3), 50, np.uint8), 448, 728)
    nw = round(100 * 448 / 300)
    x0 = (728 - nw) // 2
    assert (out[:, :x0] == 0).all() and (out[:, x0 + nw:] == 0).all()
    assert (out[:, x0:x0 + nw] == 50).all()


def test_letterbox_exact_fit_is_identity():
    img = np.random.default_rng(0).integers(0, 256, (448, 728, 3), dtype=np.uint8)
    assert (letterbox(img, 448, 728) == img).all()


def test_letterbox_pad_value():
    out = letterbox(np.zeros((10, 728, 3), np.uint8), 448, 728, pad_value=255)
    assert (out[0] == 255).all()


@pytest.mark.parametrize("h, w", [(875, 1387), (1000, 1585), (33, 517)],
                         ids=["EGYPT/DL", "MAURITIUS/ID", "extreme"])
def test_letterbox_never_distorts(h, w):
    # Aspect ratio of the content region must match the input (within rounding).
    out = letterbox(np.full((h, w, 3), 128, np.uint8), 448, 728)
    rows = np.where((out != 0).any(axis=(1, 2)))[0]
    cols = np.where((out != 0).any(axis=(0, 2)))[0]
    assert abs(len(rows) / len(cols) - h / w) < 0.02


def test_to_tensor_norm():
    img = np.zeros((4, 6, 3), np.uint8)
    img[..., 0] = 255                                  # pure red
    x = to_tensor_norm(img)
    assert tuple(x.shape) == (3, 4, 6) and str(x.dtype) == "torch.float32"
    assert abs(float(x[0, 0, 0]) - (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]) < 1e-6
    assert abs(float(x[1, 0, 0]) - (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]) < 1e-6
