"""Unified checkpoint evaluation (TASKS.md 2e): any checkpoint x named protocol x readout,
with per-image prediction caching so bootstrap/slicing/sweeps never redo inference.

Protocols:
  tier1     the 4,000-row disjoint IDNet val -- row-identical to the set train.py builds
            under the cv5 flags (--idn_val_from_unused, lim_idn 80000, idn_val_n 4000):
            same pandas seeds, independent of --seed/--backbone (the ablation's
            row-identity guarantee). MUST stay in lockstep with train.py's branch.
  external  the full EST+SVK IDNet pool (eval_external.py's set).

Readouts:
  raw       single-scale image logit at the checkpoint's training res.
  deploy    patch re-aggregation (top-5%, w=0.25) + 4-scale logit-avg TTA
            (0.85/0.9/1.0/1.1) -- the submitted configuration; PatchHead only.
            Mirrors infer_patchagg.py exactly.

Cache: preds/<ckpt-sha12>_<protocol>_<readout>.csv (path,label,score). Cache hits need
no GPU: bootstrap/re-analysis runs anywhere. Results append to results/eval_results.csv.

Usage:
  python src/evaluate.py --ckpt weights/cv5_full_ep2.pt --protocol tier1 --readout deploy
"""
from __future__ import annotations
import argparse, hashlib, os, sys, time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT
from freuid_metric import freuid_score

CV5_COUNTRIES = ["ESP_scanned", "ALB_scanned", "AZE_scanned", "FIN_scanned", "GRC_scanned",
                 "LVA_scanned", "RUS_scanned", "SRB_scanned", "EST_scanned", "SVK_scanned"]
LIM_IDN, IDN_VAL_N = 80000, 4000
DEPLOY_SCALES = (0.85, 0.9, 1.0, 1.1)
TOPK_FRAC, W_TOPK = 0.05, 0.25
PREDS_DIR = os.path.join(ROOT, "preds")
RESULTS_CSV = os.path.join(ROOT, "results", "eval_results.csv")


def ckpt_sha(path: str, n_hex: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n_hex]


def cache_path(sha: str, protocol: str, readout: str) -> str:
    return os.path.join(PREDS_DIR, f"{sha}_{protocol}_{readout}.csv")


def tier1_df() -> pd.DataFrame:
    """The 4k disjoint IDNet val. MUST mirror train.py's --idn_val_from_unused branch
    byte-for-byte (same groupby/sample calls, random_state 0 then 1)."""
    idn = pd.read_csv(os.path.join(ROOT, "external", "idnet_cropped_index.csv"))
    pool = idn[idn.type.isin(CV5_COUNTRIES)]
    idn_tr = pool.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), LIM_IDN // 2), random_state=0))
    unused = pool[~pool.id.isin(set(idn_tr.id))]
    val = unused.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), IDN_VAL_N // 2), random_state=1))
    return val.reset_index(drop=True)


def external_df() -> pd.DataFrame:
    idn = pd.read_csv(os.path.join(ROOT, "external", "idnet_cropped_index.csv"))
    return idn[idn.type.isin(["EST_scanned", "SVK_scanned"])].reset_index(drop=True)


PROTOCOLS = {"tier1": tier1_df, "external": external_df}


def bootstrap_ci(y, scores, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap 95% CI for (freuid, audet, apcer). Resamples rows with
    replacement; deterministic for a given seed."""
    y = np.asarray(y); scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    stats = []
    n = len(y)
    while len(stats) < n_boot:
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():        # resample lost a class; redraw
            continue
        stats.append(freuid_score(yb, scores[idx]))
    stats = np.asarray(stats)           # (n_boot, 3)
    lo = np.quantile(stats, alpha / 2, axis=0)
    hi = np.quantile(stats, 1 - alpha / 2, axis=0)
    return lo, hi


def predict(ckpt: str, df: pd.DataFrame, readout: str, eval_bs: int, workers: int) -> np.ndarray:
    """Run inference (GPU). Imports of torch/model live here so cache-hit runs need neither."""
    import torch
    from torch.utils.data import DataLoader
    from data import IDNetEvalDS
    from model import FreuidModel

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cargs = ck.get("args") or {}
    H, W = (int(v) for v in cargs.get("res", "448x728").lower().split("x"))
    model = FreuidModel(backbone=cargs.get("backbone", "vit_large_patch14_reg4_dinov2"),
                        pretrained="model_lean" in ck, lora_r=cargs.get("lora_r", 16),
                        head_type=cargs.get("head_type", "patch")).cuda().eval()
    if "model_lean" in ck:
        missing, unexpected = model.load_state_dict(ck["model_lean"], strict=False)
        assert not unexpected, unexpected[:3]
    else:
        model.load_state_dict(ck["model"])

    def one_pass(Hs, Ws, pagg):
        dl = DataLoader(IDNetEvalDS(df, Hs, Ws), batch_size=eval_bs, shuffle=False,
                        num_workers=workers, pin_memory=True)
        ps = []
        t0 = time.time()
        with torch.no_grad():
            for x, _ in dl:
                x = x.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    if pagg:                       # infer_patchagg.py's re-aggregation
                        head = model.head
                        tok = model.backbone.forward_features(x)
                        patch = head.norm(tok[:, head.n_prefix:])
                        plog = head.scorer(patch).squeeze(-1)
                        k = max(1, int(TOPK_FRAC * plog.shape[1]))
                        topk = plog.topk(k, dim=1).values.mean(1)
                        aw = torch.softmax(head.attn(patch).squeeze(-1), dim=1)
                        img = W_TOPK * topk + (1 - W_TOPK) * (plog * aw).sum(1)
                    else:
                        img = model(x)
                    ps.append(torch.sigmoid(img).float().cpu())
        print(f"  {Hs}x{Ws}: {len(df)} imgs in {time.time()-t0:.0f}s", flush=True)
        return torch.cat(ps).numpy()

    if readout == "raw":
        return one_pass(H, W, pagg=False)
    assert cargs.get("head_type", "patch") == "patch", "deploy readout needs PatchHead"
    eps = 1e-6
    lg = np.zeros(len(df), np.float64)
    for s in DEPLOY_SCALES:
        p = np.clip(one_pass(int(H * s) // 14 * 14, int(W * s) // 14 * 14, pagg=True),
                    eps, 1 - eps)
        lg += np.log(p / (1 - p))
    return (1 / (1 + np.exp(-lg / len(DEPLOY_SCALES)))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--protocol", required=True, choices=sorted(PROTOCOLS))
    ap.add_argument("--readout", default="raw", choices=["raw", "deploy"])
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples (0 = skip CI)")
    ap.add_argument("--force", action="store_true", help="ignore an existing prediction cache")
    args = ap.parse_args()

    df = PROTOCOLS[args.protocol]()
    sha = ckpt_sha(args.ckpt)
    cpath = cache_path(sha, args.protocol, args.readout)
    cached = os.path.exists(cpath) and not args.force
    if cached:
        cdf = pd.read_csv(cpath)
        assert len(cdf) == len(df) and (cdf.path.values == df.path.values).all(), \
            f"cache {cpath} does not match the current protocol dataframe -- rerun with --force"
        scores = cdf.score.values
        print(f"cache hit: {cpath}", flush=True)
    else:
        scores = predict(args.ckpt, df, args.readout, args.eval_bs, args.workers)
        os.makedirs(PREDS_DIR, exist_ok=True)
        pd.DataFrame({"path": df.path.values, "label": df.label.values,
                      "score": scores}).to_csv(cpath, index=False)
        print(f"cached -> {cpath}", flush=True)

    y = df.label.values.astype(np.float32)
    f, a, p = freuid_score(y, scores)
    line = (f"[{os.path.basename(args.ckpt)} sha={sha} {args.protocol}/{args.readout}] "
            f"n={len(df)} FREUID={f:.4f} AuDET={a:.4f} APCER@1%BPCER={p:.4f}")
    row = {"ckpt": args.ckpt, "sha": sha, "protocol": args.protocol, "readout": args.readout,
           "n": len(df), "freuid": f, "audet": a, "apcer": p, "cached": cached}
    if args.boot:
        lo, hi = bootstrap_ci(y, scores, n_boot=args.boot)
        line += f"  FREUID 95% CI [{lo[0]:.4f}, {hi[0]:.4f}] (boot={args.boot})"
        row.update({"freuid_ci_lo": lo[0], "freuid_ci_hi": hi[0], "n_boot": args.boot})
    print(line, flush=True)

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a", index=False,
                               header=not os.path.exists(RESULTS_CSV))
    print(f"appended -> {RESULTS_CSV}", flush=True)


if __name__ == "__main__":
    main()
