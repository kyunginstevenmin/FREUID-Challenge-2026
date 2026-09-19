"""GPU-ceiling microbenchmark (PROFILING.md): fwd+bwd+opt on ONE cached batch, same
res/bs/precision/loss as train.py -- so the only thing missing vs. real training is the
input pipeline. Gap between this img/s and train.py's steady-state img/s = input
starvation. Trains nothing; measures only.

Sweeps: --bs and --res accept comma lists, every combination is run; each row prints
img/s, ms/step, peak memory, TFLOPS and MFU, and is appended to --out (CSV) so cost
curves persist. OOM rows are recorded as such, not skipped -- the memory wall IS a result.

  # cycle-zero baseline, one backbone
  python src/gpu_ceiling.py --backbone vit_base_patch14_reg4_dinov2 --res 448x728 --bs 6
  # family-1 speed knobs, cost side only (score parity is verified by ONE train.py run)
  python src/gpu_ceiling.py ... --compile --fused_opt --precision bf16
  # family-2 cost curves
  python src/gpu_ceiling.py ... --bs 6,12,24,48            # batch-size curve (+ memory wall)
  python src/gpu_ceiling.py ... --res 322x518,448x728,518x840   # resolution curve
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT
from model import FreuidModel
from perf import count_forward_flops, gpu_peak_tflops, step_metrics
from train import focal_bce

FIELDS = ["date", "gpu", "backbone", "head_type", "lora_r", "res", "bs", "precision", "compile",
          "fused_opt", "grad_ckpt", "img_per_s", "ms_per_step", "peak_mem_gb", "tflops", "mfu", "note"]


def bench(args, H, W, bs, peak):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = FreuidModel(backbone=args.backbone, lora_r=args.lora_r,
                        head_type=args.head_type).cuda()
    model.trainable_params()
    if args.grad_ckpt:
        model.backbone.set_grad_checkpointing(True)
    model.train()
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    fwd = torch.compile(model, dynamic=False) if args.compile else model
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3,
                            fused=args.fused_opt)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"))

    x = torch.randn(bs, 3, H, W, device="cuda")
    y = torch.randint(0, 2, (bs,), device="cuda").float()
    fwd_flops = count_forward_flops(model, x, amp_dtype)

    def step():
        with torch.autocast("cuda", dtype=amp_dtype):
            loss = focal_bce(fwd(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)
        loss.item()   # mirror train.py's per-step sync so ms/step is comparable

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops, mfu = step_metrics(fwd_flops, args.steps, dt, peak)
    row = dict(img_per_s=args.steps * bs / dt, ms_per_step=1000 * dt / args.steps,
               peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30, tflops=tflops, mfu=mfu, note="")
    del model, fwd, opt, x, y
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=str, default="vit_large_patch14_reg4_dinov2")
    ap.add_argument("--res", type=str, default="448x728", help="HxW, comma list sweeps")
    ap.add_argument("--bs", type=str, default="6", help="int, comma list sweeps")
    ap.add_argument("--head_type", type=str, default="patch")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--fused_opt", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--gpu_peak_tflops", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "results", "gpu_ceiling.csv"),
                    help="CSV to append rows to ('' = don't persist)")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    gpu = torch.cuda.get_device_name()
    peak = gpu_peak_tflops(gpu, args.gpu_peak_tflops)
    print(f"[ceiling] gpu={gpu} peak={peak or 'unknown'} TFLOPS {args.backbone} head={args.head_type} "
          f"precision={args.precision} compile={args.compile} fused_opt={args.fused_opt} "
          f"grad_ckpt={args.grad_ckpt}", flush=True)

    rows = []
    for res in args.res.split(","):
        H, W = (int(v) for v in res.lower().split("x"))
        for bs in (int(b) for b in args.bs.split(",")):
            try:
                r = bench(args, H, W, bs, peak)
            except torch.cuda.OutOfMemoryError:
                r = dict(img_per_s=None, ms_per_step=None, peak_mem_gb=None, tflops=None, mfu=None,
                         note="OOM")
                torch.cuda.empty_cache()
            row = dict(date=datetime.now().strftime("%Y-%m-%d %H:%M"), gpu=gpu, backbone=args.backbone,
                       head_type=args.head_type, lora_r=args.lora_r, res=f"{H}x{W}", bs=bs,
                       precision=args.precision, compile=args.compile, fused_opt=args.fused_opt,
                       grad_ckpt=args.grad_ckpt, **r)
            rows.append(row)
            if r["note"] == "OOM":
                print(f"  res={H}x{W} bs={bs}: OOM", flush=True)
            else:
                print(f"  res={H}x{W} bs={bs}: {r['img_per_s']:.1f} img/s  {r['ms_per_step']:.0f} ms/step  "
                      f"peak_mem={r['peak_mem_gb']:.1f}GB  {r['tflops']:.1f} TFLOPS"
                      + (f"  MFU={r['mfu']:.0%}" if r["mfu"] is not None else ""), flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"[ceiling] appended {len(rows)} row(s) -> {args.out}")


if __name__ == "__main__":
    main()
