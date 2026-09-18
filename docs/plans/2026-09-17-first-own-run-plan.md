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

## Step 2 — gen-val selection inside the run

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

## Step 4 — first own run

- Pre-register in EVALUATION.md §6 before launch: recipe (config path + git
  SHAs of both repos), seed, epoch budget, "selection = source-macro, tie →
  earliest". This launch **freezes the protocol** (DRAFT → binding; bump the
  status line).
- Launch under tmux with the S3 escrow loop (runbook §4). Spot death →
  resume from `last.pt` (runbook §5).
- Wall: ~5 × 3 h + evals ≈ 16–20 h; spot ≈ $12–15.

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

Order: run the one the control-arm failure-mode table points at first. Each
arm: 1 seed (decided) → directional; add a second seed only if the CIs are
close.

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
