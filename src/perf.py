"""Always-on training-cost readouts (PROFILING.md): FLOP counting -> MFU, GPU peak lookup,
and a data-wait timer. Pure measurement -- nothing here changes what a run computes.

MFU (model FLOPs utilization) = achieved FLOP/s / GPU peak FLOP/s. It is the number that
makes img/s comparable across GPUs (local A4500 vs EC2 candidates). Training FLOPs per
step are approximated as 3x the forward count (fwd + ~2x for backward), the standard
convention (PaLM App. B); with a frozen-backbone LoRA the true backward multiplier is a
bit under 2x, so the MFU printed here is a slight UNDER-estimate. Fine for comparisons,
which is all it is for.
"""
from __future__ import annotations
import time

import torch

# Dense (no-sparsity) fp16/bf16 tensor-core peak, TFLOPS, from vendor spec sheets.
# Substring match on torch.cuda.get_device_name(); longest key wins. Unknown GPU ->
# pass --gpu_peak_tflops explicitly, otherwise MFU is omitted and only TFLOPS is logged.
PEAK_TFLOPS = {
    "RTX A4500": 94.6,
    "A10G": 70.0,      # AWS g5; NVIDIA A10G brief. A10 (non-G) is 125.
    "L4": 121.0,       # g6
    "L40S": 362.0,     # g6e
    "A100": 312.0,     # p4d (both 40/80GB)
    "H100": 989.0,     # p5 (SXM); PCIe is 756
    "T4": 65.0,        # g4dn
    "V100": 125.0,     # p3
}


def gpu_peak_tflops(name: str | None = None, override: float = 0.0) -> float | None:
    """Peak dense fp16/bf16 TFLOPS for the current (or named) GPU, or None if unknown."""
    if override and override > 0:
        return float(override)
    if name is None:
        name = torch.cuda.get_device_name() if torch.cuda.is_available() else ""
    hits = [k for k in PEAK_TFLOPS if k in name]
    if not hits:
        return None
    return PEAK_TFLOPS[max(hits, key=len)]


def count_forward_flops(model, x, dtype=torch.float16) -> float:
    """FLOPs of ONE forward pass on batch x (same autocast as training). Counted once at
    startup; the batch shape is fixed by drop_last=True so one count covers the run."""
    from torch.utils.flop_counter import FlopCounterMode
    was_training = model.training
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        if x.is_cuda:
            with torch.autocast("cuda", dtype=dtype):
                model(x)
        else:
            model(x)
    model.train(was_training)
    return float(fc.get_total_flops())


def step_metrics(fwd_flops: float, n_steps: int, seconds: float, peak_tflops: float | None):
    """Achieved TFLOPS and MFU for n_steps training steps in `seconds`."""
    tflops = 3.0 * fwd_flops * n_steps / max(seconds, 1e-9) / 1e12
    mfu = (tflops / peak_tflops) if peak_tflops else None
    return tflops, mfu


class TimedLoader:
    """Iterate a DataLoader while accumulating wall time spent BLOCKED in next().

    Measures input starvation directly: `wait` / epoch wall time = data-wait fraction.
    Accurate as long as the training step synchronizes CPU<->GPU each iteration (train.py
    does, via loss.item()); the CPU then reaches next() only after the GPU has finished
    the previous forward/backward, so time blocked here is time the GPU sits idle, up to
    at most one optimizer-step's worth of overestimate (those kernels are still queued).
    Without any per-step sync this would be optimistic (CPU runs ahead) -- don't remove the
    sync without also fixing this.
    """
    def __init__(self, loader):
        self.loader = loader
        self.wait = 0.0
        self.n = 0

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.wait = 0.0; self.n = 0
        it = iter(self.loader)
        while True:
            t = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                return
            self.wait += time.perf_counter() - t
            self.n += 1
            yield batch
