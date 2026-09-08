"""Unit tests for data.py preprocessing (letterbox geometry + normalization).
No framework: `python tests/test_data.py`. Needs numpy, cv2, torch.
"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data import letterbox, to_tensor_norm, IMAGENET_MEAN, IMAGENET_STD


def test_letterbox_geometry():
    # Wide image into the training canvas: width binds (200->728 is the smaller scale),
    # content fills full width, vertical padding split evenly.
    img = np.full((100, 200, 3), 200, np.uint8)
    out = letterbox(img, 448, 728)
    assert out.shape == (448, 728, 3) and out.dtype == np.uint8
    nh = round(100 * 728 / 200)                       # 364
    y0 = (448 - nh) // 2                              # 42
    assert (out[:y0] == 0).all() and (out[y0 + nh:] == 0).all()      # pad rows
    assert (out[y0:y0 + nh] == 200).all()                            # content rows, full width

    # Tall image: height binds, horizontal padding.
    out = letterbox(np.full((300, 100, 3), 50, np.uint8), 448, 728)
    nw = round(100 * 448 / 300)
    x0 = (728 - nw) // 2
    assert (out[:, :x0] == 0).all() and (out[:, x0 + nw:] == 0).all()
    assert (out[:, x0:x0 + nw] == 50).all()


def test_letterbox_exact_fit_and_pad_value():
    # Already at target aspect+size -> unchanged, no padding anywhere.
    img = np.random.default_rng(0).integers(0, 256, (448, 728, 3), dtype=np.uint8)
    assert (letterbox(img, 448, 728) == img).all()
    # pad_value is respected.
    out = letterbox(np.zeros((10, 728, 3), np.uint8), 448, 728, pad_value=255)
    assert (out[0] == 255).all()


def test_letterbox_never_distorts():
    # Aspect ratio of the content region must match the input (within rounding).
    for h, w in [(875, 1387), (1000, 1585), (33, 517)]:
        img = np.full((h, w, 3), 128, np.uint8)
        out = letterbox(img, 448, 728)
        rows = np.where((out != 0).any(axis=(1, 2)))[0]
        cols = np.where((out != 0).any(axis=(0, 2)))[0]
        ch, cw = len(rows), len(cols)
        assert abs(ch / cw - h / w) < 0.02, (h, w, ch, cw)


def test_to_tensor_norm():
    img = np.zeros((4, 6, 3), np.uint8)
    img[..., 0] = 255                                  # pure red
    x = to_tensor_norm(img)
    assert tuple(x.shape) == (3, 4, 6) and x.dtype.__str__() == "torch.float32"
    exp_r = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    exp_g = (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    assert abs(float(x[0, 0, 0]) - exp_r) < 1e-6
    assert abs(float(x[1, 0, 0]) - exp_g) < 1e-6


if __name__ == "__main__":
    test_letterbox_geometry()
    test_letterbox_exact_fit_and_pad_value()
    test_letterbox_never_distorts()
    test_to_tensor_norm()
    print("data: all tests passed")
