"""GPU-ceiling microbenchmark (PROFILING.md method): fwd+bwd+opt on ONE cached batch,
same res/bs/AMP/loss as train.py -- so the only thing missing vs. real training is the
input pipeline. The gap between this img/s and train.py's steady-state img/s = input
starvation. Math-affecting knobs are frozen by the ablation protocols; this script
only measures, it trains nothing.

Usage (per backbone, matching the run's config):
  python src/gpu_ceiling.py --backbone vit_base_patch14_reg4_dinov2 --res 448x728 --bs 6
"""
from __future__ import annotations
import argparse, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FreuidModel
from train import focal_bce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=str, default="vit_large_patch14_reg4_dinov2")
    ap.add_argument("--res", type=str, default="448x728")
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--head_type", type=str, default="patch")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()
    H, W = (int(v) for v in args.res.lower().split("x"))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    model = FreuidModel(backbone=args.backbone, lora_r=args.lora_r,
                        head_type=args.head_type).cuda()
    model.trainable_params()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    x = torch.randn(args.bs, 3, H, W, device="cuda")
    y = torch.randint(0, 2, (args.bs,), device="cuda").float()

    def step():
        with torch.autocast("cuda", dtype=torch.float16):
            loss = focal_bce(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    ips = args.steps * args.bs / dt
    print(f"[ceiling] {args.backbone} res={H}x{W} bs={args.bs} head={args.head_type}: "
          f"{ips:.1f} img/s ({1000*dt/args.steps:.0f} ms/step, {args.steps} steps)")


if __name__ == "__main__":
    main()
