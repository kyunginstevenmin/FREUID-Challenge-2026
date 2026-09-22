"""build_transform must not pass arguments albumentations ignores (2.x drops unknown kwargs
with a UserWarning, so a typo or a 1.x name silently changes the recipe -- happened 2026-09-19)."""
import warnings
import numpy as np
import augment

ALL_GROUPS = {"degrade", "color", "noise", "geometry", "dropout", "moire", "pcapture", "pcapture_heavy", "orient"}


def test_build_transform_uses_no_ignored_arguments():
    assert set(augment.DEFAULT_GROUPS) <= ALL_GROUPS
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tf = augment.build_transform(ALL_GROUPS)
    img = np.random.default_rng(0).integers(0, 256, (128, 192, 3), np.uint8)
    for _ in range(5):
        out = tf(image=img)["image"]
        assert out.dtype == np.uint8 and out.ndim == 3
