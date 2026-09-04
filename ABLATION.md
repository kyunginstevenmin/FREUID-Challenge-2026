# Head-type ablation protocol (pre-registered)

**Revision history.** v1 (2026-08-31): cv3/FREUID-only regime, EST/SVK blind readout.
v2 (2026-08-31, same day, **before any run was launched**): switched to the cv5
(winning-submission) regime at the project owner's direction — the question is now
"does attnpool beat the winning solution in its own regime," a deployment question
rather than a pure unseen-type generalization question. No training runs happened
under v1, so this is a clean replacement, not a post-hoc change. Any further change
after the first run must be an appended dated amendment, applied to all arms equally.
v3 (2026-09-03, **before any run was launched — headabl runs were queued for the
rig but not started**): re-sequenced behind the backbone-size ablation
(BACKBONE_ABLATION.md) at the project owner's direction, to bank the ~3.5×
per-run saving first if ViT-B proves non-inferior. Consequences: (1) arm A
(seeds 42/43) is satisfied by the backbone study's config-identical
`bbabl_cv5_L_s42/43` runs — no retraining; (2) the backbone this study's
trained arms use is set by BACKBONE_ABLATION.md decision rule 1 (L, or B if
adopted) before any head-arm run launches; if B is adopted, arm A is instead
`bbabl_cv5_B_s42/43` and only the two attnpool runs train (≈15 h); (3) if B is
adopted, the question weakens from "beats the winning solution in its own (L)
regime" to "beats patch on the adopted backbone" — REF remains the frozen
L-based reference and the write-up must state the head×backbone interaction
caveat. Arms, recipe, seeds, readouts, tiers, and decision rule are otherwise
unchanged.
v4 (2026-09-03, **before any run was launched**): single-seed descope at the
project owner's direction — seed 42 only, halving trained-run cost. A-vs-B is
now a paired single-run comparison (identical seed ⇒ identical data order and
augmentation draws, so the difference cancels shared-seed noise). Run-to-run
variance is now UNQUANTIFIED anywhere in the program; the bootstrap CI covers
eval-set sampling only, and the write-up must say so. Seed 43 on both arms is
the pre-registered escalation if the decision rule lands ambiguous.

## Question

Can feature-level attention pooling (`attnpool`) beat the competition-winning
`patch` head when trained with the exact Pick-2 (cv5_ep2) recipe, judged on
(a) the leak-free 4,000-image IDNet validation set and (b) the private test set?

## Arms

| arm | head_type | trained? | role |
|-----|-----------|----------|------|
| REF | `patch` | no — frozen `weights/cv5_full_ep2.pt` | winning solution, fixed reference (private LB 0.0582) |
| A | `patch` | yes, single run, seed 42 | retrained baseline — same-seed pair for B; controls what REF can't (REF is unseeded and epoch-picked with LB help) |
| B | `attnpool` | yes, single run, seed 42 | candidate |

`mac` is dropped (not competitive with either, and GPU budget is the binding
constraint in this regime). Phase 2 (only after A vs B is read out): `patch` +
per-patch auxiliary BCE (Chai et al. 2020, Eq. 1).

## Training recipe (verbatim cv5, from the write-up)

```bash
python src/train.py \
    --full_data --epochs 5 --bs 6 --accum 4 --eval_bs 8 \
    --res 448x728 --sbi 0.25 --attacks full \
    --idnet_countries ESP_scanned,ALB_scanned,AZE_scanned,FIN_scanned,GRC_scanned,LVA_scanned,RUS_scanned,SRB_scanned,EST_scanned,SVK_scanned \
    --heldout_idnet "" --idn_val_from_unused \
    --lim_idn 80000 --idn_val_n 4000 --select_on idnet \
    --workers 16 --save_every 250 \
    --head_type {patch|attnpool} --seed 42 --tag headabl_cv5_{HT}_s42
```

Only `--head_type` and `--tag` differ between runs. Run matrix: 2 arms × 1 seed
(42) = 2 runs ≈ 50 h on the RTX A4500 (≈15 h if the backbone study adopts B).
Escalation (pre-registered): add seed 43 to BOTH arms only if the decision rule
below lands ambiguous; never to just one arm.

**Row-identity guarantee (verified in code):** the 80k IDNet training cap samples
with `random_state=0` and the 4k validation with `random_state=1`
(train.py, idn sampling block) — fixed pandas seeds independent of `--seed`. Every
run in every arm therefore trains on identical IDNet rows and is validated on the
identical 4,000 disjoint rows. `--seed` controls LoRA-A/head init, data order, and
augmentation draws only.

**Checkpoint selection:** best epoch by `--select_on idnet` on the 4k val — and
nothing else. Note the historical asymmetry: the original cv5_ep2 was picked at
epoch 2 partly via a leaderboard probe (epoch 3 was better locally, worse on LB).
The ablation arms do NOT get leaderboard-assisted epoch selection; REF keeps its
historical advantage, and we accept that asymmetry rather than spend LB touches.

## Inference parity

The winning submission's pipeline is patch re-aggregation (top-5%, w=0.25) +
4-scale TTA (0.85/0.9/1.0/1.1). Patch re-aggregation consumes per-patch logits and
**cannot apply to attnpool**. Pre-registered readouts:

1. **Raw**: plain image logits, no TTA, no pagg — the architecture-only comparison.
2. **Deployment**: each arm with its best applicable pipeline — patch: pagg + TTA;
   attnpool: TTA only. This is what a private submission would actually ship, and
   the asymmetry is reported as such (pagg is a real advantage of the patch design,
   not a confound to hide).

## Evaluation tiers and touch budget

| tier | set | metric | touch policy |
|------|-----|--------|--------------|
| 1 **(primary)** | the 4,000-row disjoint IDNet val | freuid_score + bootstrap 95% CI | free during training (it is the selection signal); final numbers reported per run with CI |
| 2 | public leaderboard (late submission) | LB score | ≤1 per arm, sanity only — heavily queried during the competition, partially spent |
| 3 | private leaderboard (late submission) | LB score | **one shot per arm, after freeze** |

**Honest limitation of tier 1:** same countries, same IDNet pipeline as training —
it measures in-regime detection, not unseen-type generalization. In this design the
private set is the *only* unseen-distribution readout, which makes its one-shot
discipline more important, not less.

**Private-set policy (unchanged from v1):** not leakage, but adaptive-overfitting
risk. (1) Arms/seeds pre-registered above before any private evaluation. (2) All
selection on tier 1 only. (3) After freeze: one private submission per arm — the
deployment-readout checkpoint of its single run (if the seed-43 escalation fired,
the better-tier-1 run, fixed by that rule in advance). REF's private number (0.0582) is already known and is the bar. (4) All
private numbers reported, whatever they show; no second round informed by the
first. The repo's own cv3 public→private reversal is the cautionary tale.

**Decision rule:** B beats the winning solution only if it beats REF *and* arm A
on tier 1 with non-overlapping bootstrap CIs, and beats both on tier 3. The A–B
comparison is paired (same seed ⇒ same data order/aug draws), which cancels
shared-seed noise; what it cannot rule out is seed-specific luck — with one seed,
run-to-run variance is unquantified, and any positive claim carries that caveat
verbatim. If A and B's CIs overlap on tier 1, the result is "no detectable
difference" unless the seed-43 escalation is invoked; beating REF but not arm A
means run/selection variance, not architecture.

## Failure-mode analysis (after the numbers)

Per-patch heat-maps (patch) vs attention maps (attnpool) on the same tier-1 error
cases; check artifact-focus vs layout-focus. Every run logged to W&B with seed,
head_type, and full args.
