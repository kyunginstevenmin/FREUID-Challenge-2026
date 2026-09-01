# Head-type ablation protocol (pre-registered)

**Revision history.** v1 (2026-08-31): cv3/FREUID-only regime, EST/SVK blind readout.
v2 (2026-08-31, same day, **before any run was launched**): switched to the cv5
(winning-submission) regime at the project owner's direction — the question is now
"does attnpool beat the winning solution in its own regime," a deployment question
rather than a pure unseen-type generalization question. No training runs happened
under v1, so this is a clean replacement, not a post-hoc change. Any further change
after the first run must be an appended dated amendment, applied to all arms equally.

## Question

Can feature-level attention pooling (`attnpool`) beat the competition-winning
`patch` head when trained with the exact Pick-2 (cv5_ep2) recipe, judged on
(a) the leak-free 4,000-image IDNet validation set and (b) the private test set?

## Arms

| arm | head_type | trained? | role |
|-----|-----------|----------|------|
| REF | `patch` | no — frozen `weights/cv5_full_ep2.pt` | winning solution, fixed reference (private LB 0.0582) |
| A | `patch` | yes, seeded re-runs | seeded baseline — controls for run-to-run variance the unseeded REF can't |
| B | `attnpool` | yes, seeded re-runs | candidate |

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
    --head_type {patch|attnpool} --seed {42|43} --tag headabl_cv5_{HT}_s{SEED}
```

Only `--head_type`, `--seed`, `--tag` differ between runs. Run matrix: 2 arms ×
2 seeds (42, 43) = 4 runs. At ≈5 h/epoch × 5 epochs on the RTX A4500 that is
≈100 h GPU total; extend to seed 44 only if the A-vs-B gap is within the 2-seed
spread and a decision is still needed.

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
| 1 **(primary)** | the 4,000-row disjoint IDNet val | freuid_score + bootstrap 95% CI | free during training (it is the selection signal); final per-run numbers reported per seed, mean ± spread |
| 2 | public leaderboard (late submission) | LB score | ≤1 per arm, sanity only — heavily queried during the competition, partially spent |
| 3 | private leaderboard (late submission) | LB score | **one shot per arm, after freeze** |

**Honest limitation of tier 1:** same countries, same IDNet pipeline as training —
it measures in-regime detection, not unseen-type generalization. In this design the
private set is the *only* unseen-distribution readout, which makes its one-shot
discipline more important, not less.

**Private-set policy (unchanged from v1):** not leakage, but adaptive-overfitting
risk. (1) Arms/seeds pre-registered above before any private evaluation. (2) All
selection on tier 1 only. (3) After freeze: one private submission per arm — the
deployment-readout checkpoint of the best-tier-1 seed, fixed by that rule in
advance. REF's private number (0.0582) is already known and is the bar. (4) All
private numbers reported, whatever they show; no second round informed by the
first. The repo's own cv3 public→private reversal is the cautionary tale.

**Decision rule:** B beats the winning solution only if it beats REF *and* arm A's
seeded runs on both tier 1 (CI-separated, gap > seed spread) and tier 3. Beating
REF but not arm A means the gain is run-to-run variance, not architecture.

## Failure-mode analysis (after the numbers)

Per-patch heat-maps (patch) vs attention maps (attnpool) on the same tier-1 error
cases; check artifact-focus vs layout-focus. Every run logged to W&B with seed,
head_type, and full args.
