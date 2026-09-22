# Plan: first own model under EVALUATION.md (2026-09-17)

Goal: train the whole-image ViT recipe on `dataset_final` v2 and get its first
gen-val number under `id-fraud-detection/EVALUATION.md`. That run freezes the
protocol and is the §2.4 control arm. Everything downstream (ablations, final
pick) reuses what this plan builds.

Protocol facts this plan must satisfy (EVALUATION.md): selection number =
gen-val **source-macro** with stratified bootstrap CI; in-dist val = divergence
alarm only; epoch by gen-val, tie → earliest, then **frozen per recipe**;
private score end-only, one final pick; public-only probes uncapped but
pre-registered; every run logged (W&B); reproducible (seeded, pinned, one
command).

## Overview

| # | step | where | output | est. |
|---|---|---|---|---|
| 0 | REF baseline + harness shakedown | EC2 | REF's gen-val table + macros, cached preds | ½ day, ~1 GPU-h |
| 1 | `train.py` reads the dataset_final contract | fork | `--split_dir` mode, tests | 1–2 days |
| 2 | gen-val epoch selection inside training | fork | `--select_on genval`, per-source W&B logs | ½ day |
| 3 | run config + smoke run | fork / EC2 | `configs/own_v2_ctrl_s42.yaml`, `--limit` pass | ½ day |
| 4 | **first own run (control arm)** | EC2 | checkpoint, per-epoch gen-val table, W&B run | ~1 day wall, ~$15 |
| 5 | analysis + epoch freeze | local | ledger entries (§6), failure-mode slices | ½ day |
| 6 | pre-registered ablations | EC2 | one flag flip each | ~1 day each |
| 7 | final pick + submission | local / Kaggle | 1 submission | ½ day |

Steps 0 and 1 are independent — run 0 on EC2 while 1 is written locally.

**Profiling (added 2026-09-18):** no separate profiling step. The always-on
readouts in train.py (data-wait fraction, peak mem, TFLOPS/MFU — see
[PROFILING.md](../PROFILING.md)) make the control arm itself the cycle-zero
measurement; the speed-knob decision is gated at step 3 on a minutes-long
ceiling-script number, before the $15 run. Details under steps 3, 4 and 6.

## Step 0 — REF baseline on gen-val (no submission) — DONE 2026-09-17

Result in id-fraud-detection EVALUATION.md §6 "Baselines": deploy source-macro
0.4389 [0.4014, 0.4730]; unseen-only (3 sources) 0.5853. Numbers to beat.
Run log: `results/ref_genval.log`; g5.2xlarge on-demand (no spot capacity),
~45 min GPU. Lesson: the DL AMI's torch lives in /opt/pytorch (script fixed).

Why: the number to beat locally (CLAUDE.md HIGH #6), the first real exercise of
`evaluate.py --protocol genval` (path resolution, letterbox, deploy readout,
per-source printing), and REF's failure-mode map (expect idnet_* ≈ chance —
known; the print-capture sources are the informative rows).

1. Instance per `docs/RUNBOOK-EC2.md` §1–3 (g5.4xlarge spot). Clone BOTH repos
   side by side (`evaluate.py` resolves gen-val at
   `../../id-fraud-detection/dataset_final/test.csv`; else set `$GENVAL_CSV`).
2. Pull data per `id-fraud-detection/docs/ec2-training-runbook.md` step 7
   (~33 GB) and `weights/cv5_full_ep2.pt` from `$BUCKET/weights`.
3. `python src/evaluate.py --ckpt weights/cv5_full_ep2.pt --protocol genval --readout deploy`
   (raw readout too — one extra pass; both cached under `preds/`).
4. Sync `preds/` + `results/eval_results.csv` to S3; record REF's per-source
   table and both macros ± CI in EVALUATION.md §6 (new "Baselines" table).

Cost: 21,900 images × (1 raw + 4 TTA scales) ≈ 30–40 min on an A10G.

## Step 1 — `train.py` reads dataset_final (the engineering step) — DONE 2026-09-17

Implemented as `--split_dir` / `--data_root` + `build_frames_split()` (train.py), path-aware `ValDS`, `load_image` prefers `path`; legacy flags rejected; `--hardval` forced off; `--limit` stratified by (source,label) / (type,label). Real dataset_final: 63,382 / 15,961 / 21,900 rows in 0.3 s. Tests: tests/test_train_data.py (6). `--select_on genval` exists but is the POOLED gen-val score until step 2.

Current state: `main()` reads `splits/folds.csv` (FREUID only), bolts IDNet on
via `--idnet_countries/--lim_idn/...` from `external/idnet_cropped_index.csv`,
and `load_image()` dispatches FREUID rows to the Kaggle dir by id. None of that
matches `dataset_final`.

Contract (from `dataset_final/readme.md`): `id,image_path,label,type,source`;
`image_path` relative to the id-fraud-detection repo root; `label` 0/1;
`type` unique per doc type across sources (e.g. `idnet_est`); `source` ∈
{freuid, sidt, fantasyid, idnet}.

Changes (keep the legacy branch intact — the parked ablation configs and
`test_resolve_args` still exercise it):

- `--split_dir PATH` (dir holding train/val/test.csv) and `--data_root PATH`
  (defaults to split_dir's parent). When `--split_dir` is set:
  - `tr` = train.csv, `va` = val.csv, `gen` = test.csv, each with
    `path = data_root/image_path`; the `--full_data/--holdout/--fold/--idnet_*`
    flags are rejected (`validate_cfg`), not silently ignored.
  - `load_image`: use `path` whenever it is non-empty (FREUID rows now carry one).
  - `--hardval` forced off (the corrupted-val public-LB proxy is outside the
    protocol); `--limit` keeps working (stratified by source so a smoke run
    touches all four).
- Asserts at the boundary: the five columns present, labels ⊆ {0,1}, no `id`
  overlap across the three CSVs, a sampled path exists.
- Tests (`tests/test_train_data.py`): a 12-row fixture `dataset_final` in
  `tmp_path` (3 rows per source, tiny jpgs) → `build_frames(args)` returns the
  expected shapes/paths; legacy flags + `--split_dir` → `validate_cfg` raises;
  `--limit` keeps every source.

Practice note (SWE roadmap: config-driven runs, testing): the loading branch and
its fixture test are a good Kyungin rep; Claude adds the boundary asserts and
reviews.

## Step 2 — gen-val selection inside the run — DONE 2026-09-17

`genval_metrics()` per epoch (pooled + per-type + per-source + both macros), printed as a GEN-VAL line and logged to W&B (`genval/<source>`, `genval/type/<t>`, `genval/macro_*`); `--select_on genval` = source-macro; per-epoch dumps `oof/genvalpred_<tag>_<vname>_ep<k>.npy` + `oof/genval_<tag>_<vname>.csv`; per-epoch lean checkpoints `checkpoints/<tag>_<vname>_ep<k>_lean.pt` (`lean_state()`, the layout evaluate.py loads). Tie rule offline: `src/freeze_epoch.py --tag ...` prints the per-epoch CI table and the frozen epoch. Tests: 3 more in tests/test_train_data.py.

- Each epoch: evaluate on `gen` (raw readout, same `IDNetEvalDS` — it only
  needs `path,label`), then `from evaluate import macro_scores` →
  per-type, per-source, type-macro, source-macro. Print the per-source block
  (same format as `evaluate.py`), log to W&B as `genval/<source>`,
  `genval/macro_type`, `genval/macro_source`; in-dist val stays as
  `clean/*` for divergence only.
- `--select_on genval` (new default in split_dir mode): `sel = macro_source`.
- Tie rule (§3.1): save per-epoch gen-val predictions to `oof/genval_<tag>_ep<k>.npy`
  so `bootstrap_macro_ci` can be run offline over epochs after the run; the
  earliest epoch within the best epoch's CI is the frozen epoch. (No CI inside
  the training loop — 30 s/epoch of bootstrap is fine but keep the loop simple.)
- Checkpoint: keep `best` (by sel) **and** every epoch's lean weights
  (`make_lean_ckpt.py` pattern) so the frozen epoch can be a non-best one.
- Cost: 21,900 raw-readout images ≈ 6–8 min/epoch on A10G, vs ≈ 2.5–3 h/epoch
  of training (63k rows; his 5 h/epoch was on ~120k).
- Test: `select_on=genval` picks the epoch with the lowest source-macro from a
  fake per-epoch table (pure function, no GPU).

## Step 3 — config + smoke run

`configs/own_v2_ctrl_s42.yaml` (first own run = §2.4 control arm):
```yaml
split_dir: ../../id-fraud-detection/dataset_final   # v2; override on EC2 if laid out differently
epochs: 5            # winner trained 5, shipped ep2; §3.1 freezes the count after this run
bs: 6
accum: 4             # effective 24, his allowance
eval_bs: 8
res: 448x728
aug: core
sbi: 0.0             # injection OFF (ARCHITECTURE.md open decision 5)
backbone: vit_large_patch14_reg4_dinov2
head_type: patch
lora_r: 16
select_on: genval
workers: 16
save_every: 250
wandb: true
wandb_project: freuid-own
seed: 42
tag: own_v2_ctrl_s42
```
Smoke: `--config ... --limit 48 --epochs 1 --workers 4` on the instance (no
CUDA locally) — proves loading, the gen-val eval path, W&B, checkpoint +
resume. Then bake the AMI (runbook §3).

Profiling ride-alongs on the same instance session (added 2026-09-18; all
minutes, not hours):

- The smoke run is the first CUDA exercise of `src/perf.py` (FLOP count under
  autocast, the data-wait timer) and of `--compile --fused_opt`: run the smoke
  once plain and once with both flags. Both must complete; the resolved dump
  must show the flags.
- Cost side of the family-1 knobs: `python src/gpu_ceiling.py --backbone
  vit_large_patch14_reg4_dinov2 --res 448x728 --bs 6` plain, then with
  `--compile --fused_opt`. **Decision rule (basis: prior, 2026-09-18):** if the
  compiled ceiling is ≥ 15 % faster, `compile: true` and `fused_opt: true` go
  into `own_v2_ctrl_s42.yaml` and become part of the frozen recipe — there is no
  fp16/eager control to be at parity with yet, so this is the one moment the
  bundle is free to adopt; after launch it would cost a full parity run.
  bf16 stays OUT of the control (no speed gain on Ampere; historical recipe is
  fp16) unless the smoke log shows GradScaler overflow skips.
  *Note (2026-09-22):* the 2026-09-19 smoke logs cannot show skips — GradScaler
  skips silently and `train.py` never prints the scale (the iteration line
  fires every 100 its; the smokes ran 8–83). Metrics were finite, which is the
  only evidence. Possible fix, not implemented: log `scaler.get_scale()` and a
  skip count (scale dropped after `update()`) on the epoch line + W&B. Until
  then the rule is unmeasurable and fp16 stays by default.
- Data-wait check: a `--limit 500 --workers 16` pass (the 3a-style smoke) and
  read `wait=` from the console line. ≥ 15 % → sweep `--workers` (and check
  the NVMe mount) before launch; the 3c smoke hinted at input starvation
  (ceiling 26.4 vs real 15.1 img/s on ViT-B, profiler-polluted).
- *Added 2026-09-22 (from the 2026-09-19 smoke/ceiling review):* **eval is
  ≈ ¼ of an epoch and was invisible.** The epoch readouts time training only,
  but every epoch also scores val + gen-val (≈ 37.9k images; REF's single-scale
  pass ran ≈ 24 img/s on this GPU ⇒ ≈ 26 min, against ≈ 85 min of compiled
  training). `train.py` now prints `eval=<s>` on the epoch line and logs
  `eval/sec`; `eval_bs` goes 8 → 32 in the control config (eval-only,
  `no_grad`, plumbing — per-image outputs unchanged; gain read from `eval/sec`
  on the run, not assumed).
- Record the ceiling rows (`results/gpu_ceiling.csv`) and the verdict in
  PROFILING.md's results table — Kyungin writes the verdict rows.

**Final-config smoke (added 2026-09-22, the last item before step 4).** One
`bash own_run_ec2.sh wait` on the committed config — a crash check, not a
measurement and not a score. Three things have changed since the 2026-09-19
smoke and none has run on the box: the augmentation port (intended values
under 2.x names — only CPU-tested on a random image, never inside 16 workers
on real crops), `compile`/`fused_opt` read from the YAML instead of the CLI,
and `eval_bs: 32` (an eval OOM would otherwise surface after ~85 min of
training, at the end of epoch 0). ~10 min on the instance. Pass = the epoch
line prints with `eval=`, no `not valid for transform` warnings in the log
(the script greps for them), `wait=` still < 15 % under the real
augmentations, and the resolved dump shows `compile: true`, `fused_opt: true`,
`eval_bs: 32`. Then go straight to step 4 in the same session.

## Step 4 — first own run

- Pre-register in EVALUATION.md §6 before launch: recipe (config path + git
  SHAs of both repos), seed, epoch budget, "selection = source-macro, tie →
  earliest". This launch **freezes the protocol** (DRAFT → binding; bump the
  status line).
- Launch under tmux with the S3 escrow loop (runbook §4). Spot death →
  resume from `last.pt` (runbook §5).
- Wall (revised 2026-09-22 from the 2026-09-19 measurements): train split
  63,382 images at 12.4 img/s compiled ≈ 85 min/epoch, plus ≈ 26 min eval ⇒
  ≈ 9–10 h for 5 epochs (plain would be ≈ 12 h); epoch 0 also pays ≈ 75 s of
  compile. The earlier 16–20 h estimate was a prior.
- The run's `perf/*` readouts are PROFILING.md's cycle-zero row for ViT-L on
  this GPU (A10G). If the local A4500 is in play, the same config there gives
  the A4500 row and the first entry of the cost-per-instance table for free.

## Step 5 — analysis and epoch freeze

1. Offline: `bootstrap_macro_ci` over the saved per-epoch gen-val preds →
   choose the frozen epoch (earliest inside the best CI). Ledger: "Recipe epoch
   freezes" row.
2. Deploy readout on the frozen epoch via `evaluate.py --protocol genval
   --readout deploy` → the run's official table (per-type, per-source, both
   macros ± CI). Compare with REF's (step 0). No winner inside overlapping CIs.
3. Failure modes (HIGH #3): worst types, per-source gap, print-capture vs
   digital; attention maps on the worst 20 images (PatchHead gives them free).
4. Write it up short in `docs/RESULTS.md` (running log, one section per run).

## Step 6 — pre-registered ablations (one flag each, everything else frozen)

| arm | flag delta | hypothesis | where registered |
|---|---|---|---|
| scan-sim IDNet (§2.4) | swap IDNet_v2 pixels for the transformed copies (`scripts/IDNet_v2/scan_sim.py`, offline, seeded, hash-pinned) | gain on sidt/fantasyid, idnet may drop | EVALUATION.md §2.4 |
| synthetic fraud injection | `sbi: 0.25, attacks: full` | tests whether the winner's lever pays with real multi-source fraud | ARCHITECTURE.md #5 |
| backbone B | `backbone: vit_base_patch14_reg4_dinov2` | CI-overlap → keep the cheaper (§3.2) | BACKBONE_ABLATION.md pattern |
| retrain to frozen count | `epochs: k` (k = frozen epoch) | annealed epoch-k beats the mid-schedule epoch-k checkpoint | EVALUATION.md §3.1 alt (a) |
| average tied checkpoints | uniform average of the CI-tied lean ckpts (script TBD, ~20 lines) | soup beats the default pick | EVALUATION.md §3.1 alt (b) |
| LP-FT (linear probe, then fine-tune) | train the head on frozen DINOv2 features first (LoRA off, ~1 short epoch), then the normal recipe from that head; new `--lp_epochs N` flag | Kumar et al. 2022: a random head distorts pretrained features early under a large shift — our exact setup; gain should show on the print-capture sources | ARCHITECTURE.md open decision 6 |
| resolution (PROFILING.md family 2) | `res:` one step up or down (e.g. 518×840 / 322×518) | fine print-capture artefacts favour more tokens; cost curve from `gpu_ceiling.py --res a,b,c` decides which point is affordable BEFORE the score run | PROFILING.md knob table; register here when chosen |
| batch size + LR rule (family 2) | `bs:` up with `accum:` down, LR rule stated in advance | mostly a speed lever (MFU); score should hold — a cheap arm only if the ceiling `--bs` sweep shows a real img/s gain | PROFILING.md knob table |
| LoRA depth (family 2) | `only_last: N` (needs a CLI flag) | backward stops at the first adapted block → large speed gain; score hypothesis weak (less adaptation) — run only if the cost curve is compelling | PROFILING.md knob table |

Order: run the one the control-arm failure-mode table points at first. Each
arm: 1 seed (decided) → directional; add a second seed only if the CIs are
close. Family-2 arms get their cost curve from the ceiling script first
(minutes); a score run is spent only on points the curve makes affordable —
one knob per cycle (PROFILING.md rule 2).

## Step 7 — final pick + submission

1. Harness verification first, zero private information: REF checkpoint through
   `infer_private.py` with private rows constant-filled → public score should
   match his published one (EVALUATION.md §4.2 pre-registered probe).
2. Pick the final model on gen-val source-macro; write the pick in §6
   **before** submitting.
3. One real submission. Its private score is the campaign result. Ledger row.

## Runbook deltas (do with step 3)

`docs/RUNBOOK-EC2.md` is written for the parked ablation matrix. Add a short
"own-run" section: clone both repos side by side; pull per the
id-fraud-detection runbook step 7; `weights/` pull; launch line with the config;
escrow includes `oof/` and `preds/`.

## Open items this plan does not decide

- Exact `--limit` stratification and the fixture size (tests) — implementer's call.
- Whether the frozen-epoch rule needs per-epoch deploy-readout numbers or raw is
  enough for the tie test (default: raw, CI over raw preds).
- `docs/RESULTS.md` vs W&B report as the running write-up (default: markdown in
  repo, W&B links inside).
