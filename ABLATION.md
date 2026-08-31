# Head-type ablation protocol (pre-registered)

Written before any run, per evaluation-first discipline. Any change to this protocol
after the first training run must be recorded here with a dated note, and applies to
ALL arms equally.

## Question

Does feature-level attention pooling (`attnpool`, aggregate-then-score) match or beat
per-patch MIL scoring (`patch`, score-then-aggregate) on unseen document types, given
identical backbone, LoRA budget, and training recipe? The two heads are
capacity-matched (6.82M trainable each); the only variable is aggregation order.

## Arms

| arm | head_type | note |
|-----|-----------|------|
| A (baseline) | `patch`    | current default, competition-winning head |
| B            | `attnpool` | aggregate-then-score counterpart |
| C (optional) | `mac`      | global-pooling reference point (8.40M, not capacity-matched) |

Phase 2 (only after A vs B is read out): `patch` + per-patch auxiliary BCE
(Chai et al. 2020, Eq. 1) — kept out of phase 1 so the aggregation-order question
stays unconfounded.

## Controls

- **Same starting weights.** Backbone: identical pretrained DINOv2-L from the timm
  cache (deterministic). LoRA-A and head init: fixed by `--seed` (added to train.py;
  seeds python/numpy/torch). Heads differ by construction, so cross-arm init identity
  is impossible; init variance is covered by the multi-seed design instead.
- **Same recipe.** One command line, differing only in `--head_type`, `--seed`, `--tag`.
  FREUID-only data regime (the cv3 regime): no IDNet countries in training, so the
  EST/SVK readout stays a blind test (see the caveat in eval_external.py).
- **Same selection rule.** `--select_on idnet` (the leak-free signal), default
  `--heldout_idnet EST_scanned,SVK_scanned`, default `--idn_val_n`.
- **Seeds.** 3 per arm: 42, 43, 44. Report per-seed numbers and mean ± std; never a
  single lucky number.

Run matrix (2 arms × 3 seeds = 6 runs; +3 if arm C is included):

```bash
for HT in patch attnpool; do for S in 42 43 44; do
  python src/train.py --fold 0 --select_on idnet \
      --head_type $HT --seed $S --tag headabl_${HT}_s${S}
done; done
```

(Exact shared flags — epochs/bs/res/aug/sbi — pinned at launch time by copying them
into this file; the constraint is that all 6+ runs use byte-identical flags apart
from head_type/seed/tag.)

Note: `cudnn.benchmark = True` stays on, so seeded runs are reproducible in
distribution, not bit-identical. Every run logged to W&B including seeds.

## Evaluation tiers and touch budget

| tier | set | metric | touch policy |
|------|-----|--------|--------------|
| 1 | in-training HELDOUT-IDNet sample (~4k) | FREUID | free — used for checkpoint selection |
| 2 **(primary)** | full EST+SVK pool via `eval_external.py` (~5k) | freuid_score + bootstrap 95% CI | once per finished run |
| 3 | public leaderboard (late submission, 7,821 rows) | LB score | ≤1 submission per arm, sanity only — this probe was already heavily queried during the competition and is partially spent |
| 4 | private leaderboard (late submission) | LB score | **one shot per arm, after everything is frozen — see policy below** |

Tier 2 is the decision-making readout: report per-country (EST vs SVK) and per-label
slices, and per-seed spread. A conclusion in favor of either head requires the mean
gap to exceed the across-seed std and the bootstrap CIs to separate; otherwise report
"no detectable difference," which is itself a useful result (head architecture may
not matter under LoRA adaptation — cf. Zhai et al. 2021, Fig. 5).

## Private test set policy

Evaluating on the private set is **not data leakage** in the train-on-test sense
(nothing trains on it) and is no longer a fairness issue (the competition is closed).
The real risk is **adaptive overfitting**: every time a modeling decision is informed
by a private score, the private set degrades from an unbiased estimate into a
validation set, and subsequent numbers on it are optimistic. This repo's own history
is the cautionary tale — cv3 was the best public-probe model and the public→private
reversal is the central finding of the write-up. Do not recreate that dynamic with
the private set.

Rules:
1. Arms and seeds are pre-registered above, **before** any private evaluation.
2. All training, checkpoint selection, and head comparison happens on tiers 1–2 only.
3. When phase 1 is complete and frozen: one private submission per arm (best
   tier-2 checkpoint, fixed in advance as: median-seed run by tier-2 score).
4. **All** private numbers get reported, whatever they show. No second round of
   private submissions informed by the first; if phase 2 arms are trained later,
   their design must not depend on phase-1 private scores beyond what tiers 1–2
   already showed.

## Failure-mode analysis (after the numbers)

For the winning and losing arm alike: export per-patch heat-maps (`patch`) and
attention maps (`attnpool`) on the same EST/SVK error cases; check whether attention
concentrates on tamper artifacts or on country-specific layout. Verify the
preprocessing-confound guard (Chai et al., Supp. 7.1): real and fake eval images must
share one resize/save pipeline, or both arms inherit inflated generalization numbers.
