"""perf.py readouts (PROFILING.md always-on metrics). CPU-only: no timm/GPU needed, so
this file runs everywhere the others do and also on a bare torch install."""
import time

import pytest
import torch
from torch.utils.data import DataLoader

from perf import PEAK_TFLOPS, TimedLoader, count_forward_flops, gpu_peak_tflops, step_metrics


@pytest.mark.parametrize("name,expect", [
    ("NVIDIA RTX A4500", PEAK_TFLOPS["RTX A4500"]),
    ("NVIDIA A10G", PEAK_TFLOPS["A10G"]),          # must NOT match "A10" -> 125
    ("NVIDIA L40S", PEAK_TFLOPS["L40S"]),          # longest key wins over "L4"
    ("Some Future GPU", None),
])
def test_peak_lookup(name, expect):
    assert gpu_peak_tflops(name) == expect


def test_peak_override_wins():
    assert gpu_peak_tflops("NVIDIA RTX A4500", override=42.0) == 42.0


def test_forward_flops_linear():
    # one Linear(in=8,out=4) on batch 2: 2 * B * in * out multiply-adds = 128 FLOPs
    m = torch.nn.Linear(8, 4)
    assert count_forward_flops(m, torch.randn(2, 8)) == 128.0
    assert m.training   # mode restored


def test_step_metrics_uses_3x_forward():
    tflops, mfu = step_metrics(fwd_flops=1e12, n_steps=10, seconds=1.0, peak_tflops=100.0)
    assert tflops == pytest.approx(30.0)
    assert mfu == pytest.approx(0.30)
    assert step_metrics(1e12, 1, 1.0, None)[1] is None


class _Slow(torch.utils.data.Dataset):
    def __len__(self): return 4
    def __getitem__(self, i):
        time.sleep(0.02); return torch.tensor(i)


def test_timed_loader_accumulates_wait_and_resets():
    tl = TimedLoader(DataLoader(_Slow(), batch_size=2, num_workers=0))
    assert len(tl) == 2
    got = [b.tolist() for b in tl]
    assert got == [[0, 1], [2, 3]] and tl.n == 2
    assert tl.wait >= 0.06          # 4 items * 20ms, measured while blocked in next()
    first = tl.wait
    list(tl)                        # second epoch starts from zero
    assert tl.n == 2 and tl.wait < first + 0.06
