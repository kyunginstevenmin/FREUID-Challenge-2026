"""Identify which 2 of the 4 IDNet fraud folders the original index used (plan exit b).

Method: ONE inference pass of the frozen REF checkpoint over ALL EST+SVK images
(59,790 rows of the superset index), cached via evaluate.py's prediction cache;
then score each of the C(4,2)=6 candidate pools (positives + a folder pair) from
the cached per-image scores -- no re-inference per candidate.

Reference numbers from the winning solution's report (his pool = 35,874 imgs):
  raw full-pool      0.0154   (tex ~321: pre-TTA full-pool score)   <- primary, --readout raw
  deploy full-pool   0.0116   (tex ~321: pagg+TTA confirmation)     <- tie-break, --readout deploy
Decision rule (plan 5b): rank candidates by |score - reference|; accept the winner
only if the runner-up gap is large vs bootstrap-CI half-widths (GPU float drift and
eval noise mean we match by separation, not exactness).

Usage (GPU box, superset index present):
  python src/fingerprint_folders.py --ckpt weights/cv5_full_ep2.pt
"""
from __future__ import annotations
import argparse, itertools, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import PREDS_DIR, bootstrap_ci, cache_path, ckpt_sha, predict
from freuid_metric import freuid_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAUDS = ["fraud1_copy_and_move", "fraud2_face_morphing",
          "fraud3_face_replacement", "fraud4_combined"]
REFERENCE = {"raw": 0.0154, "deploy": 0.0116}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(REPO, "weights", "cv5_full_ep2.pt"))
    ap.add_argument("--index", default=os.path.join(REPO, "external", "idnet_full_index.csv"))
    ap.add_argument("--readout", default="raw", choices=["raw", "deploy"])
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    full = pd.read_csv(args.index)
    pool = full[full.type.isin(["EST_scanned", "SVK_scanned"])].reset_index(drop=True)
    assert len(pool) == 59790, f"expected 59,790 EST+SVK rows, got {len(pool)}"

    sha = ckpt_sha(args.ckpt)
    cpath = cache_path(sha, "estsvk_all", args.readout)
    if os.path.exists(cpath) and not args.force:
        cdf = pd.read_csv(cpath)
        assert (cdf.path.values == pool.path.values).all(), "cache/pool mismatch -- --force"
        scores = cdf.score.values
        print(f"cache hit: {cpath}")
    else:
        scores = predict(args.ckpt, pool, args.readout, args.eval_bs, args.workers)
        os.makedirs(PREDS_DIR, exist_ok=True)
        pd.DataFrame({"path": pool.path.values, "label": pool.label.values,
                      "score": scores}).to_csv(cpath, index=False)
        print(f"cached -> {cpath}")

    ref = REFERENCE[args.readout]
    folder = pool.id.str.split("/").str[1]
    rows = []
    for pair in itertools.combinations(FRAUDS, 2):
        mask = folder.isin({"positive", *pair}).values
        y, s = pool.label.values[mask].astype(np.float32), scores[mask]
        assert mask.sum() == 35874, (pair, mask.sum())
        f, a, apc = freuid_score(y, s)
        lo, hi = bootstrap_ci(y, s, n_boot=args.boot)
        rows.append({"pair": "+".join(p.split("_")[0] for p in pair), "freuid": f,
                     "audet": a, "apcer": apc, "ci_lo": lo[0], "ci_hi": hi[0],
                     "dist": abs(f - ref)})
        print(f"{rows[-1]['pair']:15s} FREUID={f:.4f} [{lo[0]:.4f},{hi[0]:.4f}] "
              f"AuDET={a:.4f} APCER@1%={apc:.4f}  |Δref|={abs(f-ref):.4f}")

    res = pd.DataFrame(rows).sort_values("dist").reset_index(drop=True)
    win, second = res.iloc[0], res.iloc[1]
    hw = (win.ci_hi - win.ci_lo) / 2
    sep = (second.dist - win.dist) / hw if hw > 0 else float("inf")
    print(f"\nreference ({args.readout}) = {ref}")
    print(f"winner: {win.pair} (|Δ|={win.dist:.4f}); runner-up {second.pair} "
          f"(|Δ|={second.dist:.4f}); separation = {sep:.1f}x CI half-width")
    verdict = ("CLEAR" if sep >= 4 else "AMBIGUOUS -- run --readout deploy tie-break "
               "or fall back to exit (c)")
    print(f"verdict: {verdict}")
    out = os.path.join(REPO, "results", f"fingerprint_{args.readout}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res.to_csv(out, index=False)
    print(f"table -> {out}")


if __name__ == "__main__":
    main()
