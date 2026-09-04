# Backbone-size ablation protocol (pre-registered) — DRAFT v1

**Revision history.** v0 (2026-09-03): drafted as a follow-up gated behind the
head-type ablation, reusing its winning-head runs as the L arm. v1 (2026-09-03,
same day, **before any run of either study was launched**): re-sequenced to run
FIRST at the project owner's direction — rationale: if a smaller backbone proves
non-inferior, the head ablation itself (and all later work) runs ~3.5× cheaper.
The reuse direction flips: this study's L runs become the head ablation's arm A
(see ABLATION.md v3 note). No training runs happened under v0, so this is a
clean replacement. Any change after the first run must be an appended dated
amendment, applied to all arms equally.

**Becomes binding (drop DRAFT) when:** the `TBD-at-launch` margin below is
filled in and the prerequisite code changes have landed. First launch freezes
the protocol.

## Question

How much of the deployed model's in-regime performance requires the ViT-Large
backbone? Specifically: is DINOv2 ViT-B/14 **non-inferior** to ViT-L/14 under
the identical cv5 recipe — and where does ViT-S/14 sit on the capacity curve?

This is a deployment-and-iteration question, not a superiority question. The
payoff of a non-inferior smaller backbone is concrete: ~3.5× (B) / ~10× (S)
cheaper training for the upcoming head ablation and every experiment after it,
and proportionally cheaper inference (the 6 h hidden-set budget currently costs
~6.3 h of A4500 time on L).

## Arms

All arms use the incumbent `patch` head (the winning solution's head). The head
question is deliberately deferred to the head ablation, which runs second, on
the backbone this study adopts.

| arm | backbone (timm) | params | embed dim | trained? | role |
|-----|-----------------|--------|-----------|----------|------|
| REF | `vit_large_patch14_reg4_dinov2` | ~304M | 1024 | no — frozen `weights/cv5_full_ep2.pt` | winning solution, fixed reference (private LB 0.0582) |
| L | `vit_large_patch14_reg4_dinov2` | ~304M | 1024 | yes, seeds 42/43 | incumbent, seeded — controls run-to-run variance; **doubles as head-ablation arm A** (identical config) |
| B | `vit_base_patch14_reg4_dinov2` | ~87M | 768 | yes, seeds 42/43 | candidate (non-inferiority) |
| S | `vit_small_patch14_reg4_dinov2` | ~22M | 384 | yes, seeds 42/43 | capacity-curve point; prospective screening vehicle |

All three backbones share patch size 14 and 4 registers, so at fixed
`--res 448x728` the token grid (32×52) and pixels-per-patch geometry — the
forensic-artifact resolution argument — are identical across arms. This
ablation isolates capacity, not resolution.

## Training recipe (verbatim cv5, one new lever)

```bash
python src/train.py \
    --full_data --epochs 5 --bs 6 --accum 4 --eval_bs 8 \
    --res 448x728 --sbi 0.25 --attacks full \
    --idnet_countries ESP_scanned,ALB_scanned,AZE_scanned,FIN_scanned,GRC_scanned,LVA_scanned,RUS_scanned,SRB_scanned,EST_scanned,SVK_scanned \
    --heldout_idnet "" --idn_val_from_unused \
    --lim_idn 80000 --idn_val_n 4000 --select_on idnet \
    --workers 16 --save_every 250 \
    --head_type patch --seed {42|43} \
    --backbone {vit_large|vit_base|vit_small}_patch14_reg4_dinov2 \
    --tag bbabl_cv5_{L|B|S}_s{SEED}
```

Only `--backbone`, `--seed`, `--tag` differ between runs. Run matrix: 3 arms ×
2 seeds = 6 runs. Estimated cost on the RTX A4500 (FLOPs ∝ params at fixed
tokens): L ≈ 25 h/run, B ≈ 7–8 h/run, S ≈ 2–4 h/run → **≈ 70–75 h GPU total**.
Launch order: **B and S seeds first (~20 h), L seeds last** — if a B run
crashes or smokes out a recipe problem, it is discovered before the expensive
L runs are spent. Extend to seed 44 only if the decision rule lands within the
2-seed spread and a decision is still needed.

**Batch-size allowance (pre-registered, not a confound):** smaller backbones
free VRAM, so `--bs` may be raised **only** with `--accum` lowered to keep the
effective batch at 24 (e.g. B: `--bs 12 --accum 2`). At matched effective batch
with identical data order this is gradient-equivalent, purely a wall-clock
optimization. The actual bs/accum used is recorded per run in W&B. The L runs
use the verbatim `--bs 6 --accum 4` so they remain byte-identical in config to
head-ablation arm A.

**Hyperparameter confound (stated up front):** `lora_r 16`, `lora_alpha 32`,
`lr_lora 2e-4`, `lr_head`, dropout — all inherited unchanged from the L recipe,
where they were tuned. A narrow B loss under L's hyperparameters means "B loses
under this recipe," not "B is worse." Contingency (pre-registered, see decision
rule): exactly one lr probe on B, seed 42 only, `--lr_lora 4e-4`, tag
`bbabl_cv5_B_s42_lr4e4`. No other hyperparameter exploration without a dated
amendment. S gets no probe (its role doesn't require winning).

**Row-identity guarantee:** unchanged from ABLATION.md — the 80k IDNet training
cap and 4k validation sample with fixed pandas seeds independent of `--seed` and
of `--backbone`; every run in every arm trains on identical rows and validates
on the identical disjoint 4,000.

**Checkpoint selection:** best epoch by `--select_on idnet` on the 4k val,
nothing else — same rule the head ablation pre-registered, keeping the shared L
runs selection-symmetric across both studies.

## Prerequisite code changes (before any run)

1. `train.py`: add `--backbone` (default `vit_large_patch14_reg4_dinov2`), pass
   to `FreuidModel`, ensure it is saved in the checkpoint's `args` dict.
2. Eval/infer scripts (`infer*.py`, `tta_*.py`, `patch_agg_grid.py`,
   `pagg_tta_combo.py`, `cv3_postproc_eval.py`, `eval_external.py`,
   `make_lean_ckpt.py`): construct the model with
   `ck["args"].get("backbone", "vit_large_patch14_reg4_dinov2")` — the same
   config-in-checkpoint pattern already used for `lora_r`. The default preserves
   every existing checkpoint.
3. Sanity check per new backbone before launching: forward + backward smoke test
   at 448×728 (as in `model.py.__main__`); verify token count 32×52, embed dim,
   and trainable = LoRA+head only.

## Inference parity

Per arm, two readouts (same definitions as ABLATION.md):

1. **Raw**: plain image logits — the capacity-only comparison.
2. **Deployment**: pagg (top-5%, w=0.25) + 4-scale TTA — applicable to all arms
   identically since every arm uses the `patch` head. Cleaner than the head
   ablation's asymmetric readout.

**Latency readout (secondary, reported for all arms):** training throughput
(img/s from the training log) and deployment-pipeline inference cost (ms/image,
4-scale TTA, A4500, same measurement harness as the README's 160 ms/image
figure). This is the benefit side of the non-inferiority trade and is reported
alongside the score, not instead of it. Bottleneck diagnosis and profiling
methodology live in [PROFILING.md](PROFILING.md) (engineering doc, not part of
this pre-registered protocol).

## Evaluation tiers and touch budget

| tier | set | metric | touch policy |
|------|-----|--------|--------------|
| 1 **(primary)** | the 4,000-row disjoint IDNet val | freuid_score + bootstrap 95% CI | free (selection signal); reported per seed, mean ± spread |
| 2 | public leaderboard | LB score | **0 planned** — amendment required to use |
| 3 | private leaderboard | LB score | **0 in this study.** All private touches are deferred to the end of the two-phase program (backbone → head): the final deployment candidate(s) chosen after the head ablation get the shots ABLATION.md already reserves. Adopting a backbone here is an *iteration* decision made on tier 1 alone. |

Honest limitation, inherited from ABLATION.md: tier 1 measures in-regime
detection, not unseen-type generalization. A smaller backbone could be tier-1
non-inferior yet generalize worse to unseen document types — capacity often
matters most out-of-distribution. Because this study spends no private touches,
that risk is carried forward, not resolved: the end-of-program private shot is
the only unseen-distribution readout, and if B was adopted mid-program the
whole downstream chain inherits the bet. This is accepted as the price of the
GPU savings and stated here so the final write-up reports it.

## Decision rule (pre-registered)

Let `spread(L)` = |seed 42 − seed 43| tier-1 gap of the L arm, and `Δ(X)` =
mean-over-seeds tier-1 deficit of arm X vs the L arm mean (positive = worse).

1. **Adopt B for iteration** (the head ablation and future experiments run on
   B) if `Δ(B) ≤ spread(L)` and the per-seed bootstrap CIs of B and L overlap.
   Otherwise the head ablation proceeds on L as originally registered.
2. **lr probe trigger:** if `spread(L) < Δ(B) ≤ 2·spread(L)`, run the one
   pre-registered lr probe; re-apply rule 1 with the probe replacing B-s42.
   If `Δ(B) > 2·spread(L)` (clear loss), stop: L's capacity is earning its
   cost, and the 25 h/run price is now evidence-backed.
3. **S is not adoption-eligible** for iteration or deployment regardless of
   score. Readout: where `Δ(S)` lands relative to `Δ(B)` maps the capacity
   curve. S becomes the default screening vehicle for cheap recipe-level
   pilots only if `Δ(S) ≤ 3·spread(L)`; screening validity is then checked
   opportunistically — any future intervention run on both S and the adopted
   backbone is logged as a rank-agreement data point.
4. **Deployment margin (used at end of program):** the final candidate is
   submitted privately only if its tier-1 score is within `TBD-at-launch: ____`
   of the best incumbent number; the incumbent REF private score (0.0582)
   remains the bar to beat or match.

All numbers reported whatever they show, including a B/S win over L (possible —
smaller models sometimes win under a recipe tuned at higher capacity and 5
epochs of LoRA; a win triggers the same rules, not celebration-driven scope
creep).

## Consequence for the head ablation (recorded here, mirrored in ABLATION.md v3)

- This study's L runs (`bbabl_cv5_L_s42/43`) are config-identical to
  head-ablation arm A (`patch`, L, cv5 recipe, seeds 42/43) and satisfy it
  without retraining.
- If rule 1 adopts B: the head ablation's trained arms run on backbone B; its
  patch arm is this study's B runs (also free), so the head ablation shrinks to
  **2 attnpool runs ≈ 15 h** (vs ≈ 100 h as originally budgeted).
- **Accepted weakening:** deciding the head on B answers "does attnpool beat
  patch on the adopted backbone," not "does attnpool beat the winning solution
  in its own (L) regime." REF stays the frozen L-based reference either way.
  Risk of a head×backbone interaction is accepted as low (Zhai et al. 2021:
  head architecture is second-order under full training) but is a real caveat
  the write-up must state.

## Failure-mode analysis (after the numbers)

Same tier-1 error cases across all three trained arms: per-patch heat-maps on
identical images — does reduced capacity lose artifact-focus (misses
high-frequency cues) or layout-focus (misses document structure)? Slice tier-1
scores by document type and by attack type (`self_blend` vs annotation-driven).
Every run logged to W&B with backbone, head_type, seed, bs/accum, and full args.
