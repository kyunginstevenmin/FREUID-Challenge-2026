"""Epoch freeze (EVALUATION.md §3.1) from a --split_dir run's per-epoch gen-val dumps.

For every epoch: source-macro + type-stratified bootstrap CI (evaluate.bootstrap_macro_ci).
Best = lowest source-macro. Tie rule: the FROZEN epoch is the EARLIEST epoch whose
source-macro lies inside the best epoch's tie band (less training = less template
memorization). Band (EVALUATION.md §3.1): --band ci95 (default) = the 95% bootstrap
interval; --band 1se = one bootstrap SE (the literal one-standard-error rule), the
option to switch to when the 95% band ties most epochs. Prints both picks; nothing is
written -- record the pick and the band used in EVALUATION.md §6.

  python src/freeze_epoch.py --tag own_v2_ctrl_s42 --vname dataset_final [--boot 1000] [--band 1se]
"""
from __future__ import annotations
import argparse, glob, os, re, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT
from evaluate import macro_scores, bootstrap_macro_ci

OOF_DIR = os.path.join(ROOT, "oof")


def epoch_table(gen: pd.DataFrame, preds_by_epoch: dict[int, np.ndarray], n_boot=1000, seed=0):
    rows = []
    for ep in sorted(preds_by_epoch):
        p = preds_by_epoch[ep]
        _, per_source, mt, ms = macro_scores(gen.label.values, p, gen.type.values, gen.source.values)
        lo, hi = bootstrap_macro_ci(gen.label.values, p, gen.type.values, gen.source.values,
                                    n_boot=n_boot, seed=seed)
        rows.append({"epoch": ep, "macro_source": ms, "ci_lo": lo[1], "ci_hi": hi[1],
                     "macro_type": mt, **{f"src/{k}": v for k, v in per_source.items()}})
    return pd.DataFrame(rows)


def tie_threshold(best_row, band="ci95") -> float:
    """Upper edge of the tie band around the best epoch. 1se uses the normal approximation
    SE = (ci_hi - ci_lo) / (2 * 1.96) of the bootstrap distribution (ponytail: exact SE would
    need the raw bootstrap draws; the approximation is within a few % for these sizes)."""
    if band == "ci95":
        return float(best_row.ci_hi)
    if band == "1se":
        return float(best_row.macro_source + (best_row.ci_hi - best_row.ci_lo) / (2 * 1.96))
    raise ValueError(band)


def pick_frozen_epoch(table: pd.DataFrame, band="ci95") -> int:
    """Earliest epoch whose source-macro is <= the best epoch's tie threshold."""
    best = table.loc[table.macro_source.idxmin()]
    tied = table[table.macro_source <= tie_threshold(best, band)]
    return int(tied.epoch.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--vname", default="dataset_final")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--band", default="ci95", choices=["ci95", "1se"])
    a = ap.parse_args()
    gen = pd.read_csv(os.path.join(OOF_DIR, f"genval_{a.tag}_{a.vname}.csv"))
    files = glob.glob(os.path.join(OOF_DIR, f"genvalpred_{a.tag}_{a.vname}_ep*.npy"))
    preds = {int(re.search(r"_ep(\d+)\.npy$", f).group(1)): np.load(f) for f in files}
    assert preds, "no per-epoch gen-val dumps found"
    t = epoch_table(gen, preds, n_boot=a.boot)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    best_ep = int(t.loc[t.macro_source.idxmin(), "epoch"])
    n_tied = {b: int((t.macro_source <= tie_threshold(t.loc[t.macro_source.idxmin()], b)).sum()) for b in ("ci95", "1se")}
    print(f"\nbest epoch {best_ep}; tied under ci95: {n_tied['ci95']}/{len(t)} epochs, under 1se: {n_tied['1se']}/{len(t)}")
    print(f"FROZEN epoch under --band {a.band}: {pick_frozen_epoch(t, a.band)}"
          f"   (ci95 -> {pick_frozen_epoch(t, 'ci95')}, 1se -> {pick_frozen_epoch(t, '1se')})")


if __name__ == "__main__":
    main()
