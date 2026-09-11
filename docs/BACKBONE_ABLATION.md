# Backbone-size ablation protocol (pre-registered) — DRAFT v1

**Revision history.** v0 (2026-09-03): drafted as a follow-up gated behind the
head-type ablation, reusing its winning-head runs as the L arm. v1 (2026-09-03,
same day, **before any run of either study was launched**): re-sequenced to run
FIRST at the project owner's direction — rationale: if a smaller backbone proves
non-inferior, the head ablation itself (and all later work) runs ~3.5× cheaper.
The reuse direction flips: this study's L runs become the head ablation's arm A
(see ABLATION.md v3 note). No training runs happened under v0, so this is a
clean replacement. Any change after the first run must be an appended dated
amendment, applied to all arms equally. v2 (2026-09-03, **before any run was
launched**): single-seed descope at the project owner's direction — seed 42
only, cutting the matrix from 6 to 3 runs (≈35 h). Within-arm seed spread no
longer exists, so the decision rules' noise yardstick changes to the per-run
bootstrap CI on the 4k val (eval-sampling noise only; run-to-run variance is
unquantified — stated limitation carried to the write-up). Seed 43 on the
arms being compared is the pre-registered escalation for ambiguous outcomes.
v3 (2026-09-03, **before any run was launched**): trained-L arm dropped at the
project owner's direction — the frozen `weights/cv5_full_ep2.pt` (the report's
ViT-L result) is now the sole L reference. Trained matrix: **B and S only, 2
runs ≈ 10–12 h**. Consequences: (1) REF must be evaluated ONCE on the 4k
tier-1 val to anchor its score and bootstrap CI (deterministic set, fixed
pandas seeds — reproducible); (2) B-vs-REF is no longer a paired same-seed
comparison, and REF's epoch was historically LB-assisted, making it a slightly
optimistic bar — conservative for non-inferiority, stated in the write-up;
(3) the head ablation no longer inherits an arm from this study — it is ON
HOLD (see ABLATION.md status) and out of the runbook scope.

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
| REF (= L point) | `vit_large_patch14_reg4_dinov2` | ~304M | 1024 | no — frozen `weights/cv5_full_ep2.pt` | winning solution AND the L capacity point; evaluated once on tier 1 to anchor score + CI. Caveats: unseeded, epoch historically LB-assisted (optimistic bar → conservative non-inferiority test) |
| B | `vit_base_patch14_reg4_dinov2` | ~87M | 768 | yes, seed 42 | candidate (non-inferiority) |
| S | `vit_small_patch14_reg4_dinov2` | ~22M | 384 | yes, seed 42 | capacity-curve point; prospective screening vehicle (paired with B: same seed) |

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
    --head_type patch --seed 42 \
    --backbone {vit_base|vit_small}_patch14_reg4_dinov2 \
    --tag bbabl_cv5_{B|S}_s42
```

Only `--backbone` and `--tag` differ between runs. Run matrix: **2 trained runs**
(B, S; seed 42) ≈ 10–12 h on the RTX A4500 (B ≈ 7–8 h, S ≈ 2–4 h); the L point
is the frozen REF checkpoint, no L training. Before the runs: evaluate REF once
on the 4k tier-1 val (`eval_external`-style, same freuid_score + bootstrap) to
anchor the reference score and CI.
Escalation (pre-registered): add seed 43 to B (and S if its readout matters to
the decision) only if the decision rule lands ambiguous (rule 2's probe band).

**Batch-size allowance (pre-registered, not a confound):** smaller backbones
free VRAM, so `--bs` may be raised **only** with `--accum` lowered to keep the
effective batch at 24 (e.g. B: `--bs 12 --accum 2`). At matched effective batch
with identical data order this is gradient-equivalent, purely a wall-clock
optimization. The actual bs/accum used is recorded per run in W&B.

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
| 1 **(primary)** | the 4,000-row disjoint IDNet val | freuid_score + bootstrap 95% CI | free (selection signal); reported per run with CI |
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

B and S share seed 42 (paired with each other: identical data order/aug
draws). B-vs-REF is UNPAIRED — REF is a different, unseeded historical run;
additionally REF's epoch pick was LB-assisted, so REF is a mildly optimistic
bar (which makes the non-inferiority test conservative). Let `Δ(X)` = tier-1
deficit of arm X vs REF's anchored tier-1 score (positive = worse), and
`CIhw(REF)` = half-width of REF's tier-1 bootstrap 95% CI (the pre-registered
noise yardstick; covers eval-set sampling, NOT run-to-run variance — that
limitation rides with every conclusion below).

1. **Adopt B for iteration** (future experiments, incl. any head ablation, run
   on B) if the bootstrap CIs of B and REF overlap. Otherwise iteration stays
   on L.
2. **lr probe trigger:** if the CIs are disjoint but `Δ(B) ≤ 4·CIhw(REF)`
   (narrow loss), run the one pre-registered lr probe; re-apply rule 1 with the
   probe replacing B-s42. If `Δ(B) > 4·CIhw(REF)` (clear loss), stop: L's
   capacity is earning its cost, and the 25 h/run price is now evidence-backed.
3. **S is not adoption-eligible** for iteration or deployment regardless of
   score. Readout: where `Δ(S)` lands relative to `Δ(B)` maps the capacity
   curve. S becomes the default screening vehicle for cheap recipe-level
   pilots only if `Δ(S) ≤ 6·CIhw(REF)`; screening validity is then checked
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

## Consequence for the head ablation

The head ablation is **ON HOLD** (ABLATION.md carries the status note) and no
longer inherits any arm from this study — with no trained L run, there is no
free patch baseline. If/when it is revived, its arms are decided then: if rule
1 adopted B, its patch arm is this study's B run (free) and only attnpool
trains (≈ 8 h); on L, both arms would need training. Head×backbone interaction
caveat (Zhai et al. 2021) applies if the head is decided on B.

## Failure-mode analysis (after the numbers)

Same tier-1 error cases across all three trained arms: per-patch heat-maps on
identical images — does reduced capacity lose artifact-focus (misses
high-frequency cues) or layout-focus (misses document structure)? Slice tier-1
scores by document type and by attack type (`self_blend` vs annotation-driven).
Every run logged to W&B with backbone, head_type, seed, bs/accum, and full args.
