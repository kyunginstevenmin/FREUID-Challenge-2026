# Task list — backbone ablation prep

Working order agreed 2026-09-04. Steps 1–3 are pipeline/skeleton work (Kyungin
drives, Claude reviews — SWE/ML roadmap practice: config-driven runs, experiment
tracking, testing). Step 4 is the launch gate. See
[BACKBONE_ABLATION.md](BACKBONE_ABLATION.md) ("Prerequisite code changes") for
the authoritative spec of items 1a–1c.

## 1. `--backbone` plumbing (prerequisite for any run)

- [x] 1a. `train.py`: add `--backbone` (default `vit_large_patch14_reg4_dinov2`),
      pass through to `FreuidModel`, confirm it lands in the checkpoint `args` dict.
      (2026-09-04: verified on g5.xlarge — 16-image micro-run, backbone present in
      both last + best ckpt args; model rebuilt from ckpt args alone.)
- [x] 1b. Eval/infer scripts (`infer*.py`, `tta_*.py`, `patch_agg_grid.py`,
      `pagg_tta_combo.py`, `cv3_postproc_eval.py`, `eval_external.py`,
      `make_lean_ckpt.py`): construct model with
      `ck["args"].get("backbone", "vit_large_patch14_reg4_dinov2")` so every
      existing checkpoint keeps loading.
      (2026-09-05, commit 903d376: all 12 sites, `head_type` threaded alongside;
      docker/ left untouched — frozen submission path. Also fixed make_lean_ckpt's
      module-level ViT-L constant: strip set + verify reference now come from the
      checkpoint's own args. py_compile clean; GPU round-trip of a B/S ckpt rides
      along with the 3a smoke run.)
- [x] 1c. Per new backbone (B, S): forward+backward smoke test at 448×728;
      verify token grid 32×52, embed dim (768 / 384), trainable = LoRA+head only.
      (2026-09-04, g5.xlarge: B 2.76M trainable / S 1.38M, 48 LoRA modules each,
      grads on LoRA A/B + head only. Note: LoRA params are named `.A`/`.B`,
      not `lora_*` — filter accordingly in any future trainable-param checks.)

## 2. Run-spec + tracking infrastructure

- [ ] 2a. Config-per-run — hand-rolled `--config <yaml>`, spec:
      - Precedence: argparse defaults < YAML < flags explicitly typed on the CLI.
        (Detect "explicitly typed" by parsing with `argparse.SUPPRESS` defaults
        first, or by comparing `sys.argv` — pick one, document it in the help text.)
      - Validation: a YAML key that doesn't match a known argparse dest is a hard
        error (typo protection), not silently ignored.
      - Resolved-config artifact: the fully-merged args are already saved into
        checkpoints via `vars(args)`; additionally dump them as
        `checkpoints/<tag>_resolved.yaml` and log to W&B, so a run's exact spec
        is readable without loading a .pt.
      - Commit one YAML per arm (`configs/bbabl_cv5_B_s42.yaml`,
        `configs/bbabl_cv5_S_s42.yaml`) encoding the recipe verbatim from
        BACKBONE_ABLATION.md; the launch command becomes
        `python src/train.py --config configs/bbabl_cv5_B_s42.yaml`.
      - No new dependency beyond pyyaml; NOT Hydra (decided 2026-09-04 —
        wrong size for this repo).
- [ ] 2b. W&B wiring in `train.py`: per-epoch clean/hard/idnet metrics, full
      resolved args (backbone, head_type, seed, bs/accum), tag per arm.
      (Protocol already promises this — BACKBONE_ABLATION.md failure-mode section.)
- [ ] 2c. (ride-along, roadmap table stakes) First unit tests: `freuid_metric`,
      `letterbox`, lean-ckpt round-trip, config loading. CI on push if cheap.
- [ ] 2d. Opt-in `--profile` flag in `train.py`: `torch.profiler` per
      [PROFILING.md](PROFILING.md) Method (schedule wait=5 warmup=2 active=5,
      `tensorboard_trace_handler`), off by default — ablation runs stay clean.
      Plus the GPU-ceiling microbenchmark script (fwd/bwd on one cached batch).
- [ ] 2e. Unified eval entry point (`src/evaluate.py`) — minimal version, spec:
      - Input: any checkpoint (backbone/head/res read from `ck["args"]`, per 1b)
        + a named protocol: `tier1` (the 4k disjoint IDNet val),
        `external` (full EST+SVK pool, today's eval_external.py), and a
        readout mode: `raw` | `deploy` (pagg top-5%/w=0.25 + 4-scale TTA).
      - Prediction caching: per-image scores written to
        `preds/<ckpt-sha>_<protocol>_<readout>.csv`; bootstrap CIs, slicing,
        and any post-hoc sweep reuse the cache instead of re-running inference.
      - Output: freuid_score + bootstrap 95% CI, printed and appended to a
        results CSV; this is the script that anchors REF (3b) and produces the
        per-arm readouts + failure-mode slices.
      - Descope guard: legacy sweep scripts (tta_grid, patch_agg_grid, ...) are
        NOT consolidated now — they stay as-is; folding them in is parked until
        after the ablation ships.

## 3. End-to-end smoke run

- [ ] 3a. `--limit 500 --epochs 1` run of the full loop (train → checkpoint →
      per-epoch eval → W&B log → best-ckpt JSON) on ViT-B config.
- [ ] 3b. Anchor REF: evaluate frozen `weights/cv5_full_ep2.pt` once on the 4k
      tier-1 val (freuid_score + bootstrap CI) — required before B/S comparisons.
      Runs through `evaluate.py` (2e); doubles as its first real-checkpoint test.
- [ ] 3c. Exercise `--profile` during the smoke run (GPU box): verify the trace
      opens in TensorBoard's trace viewer and `nvidia-smi dmon`/htop procedure
      works. Tooling shakeout only — real measurements happen at launch.

## 4. Launch gate

- [ ] 4a. Fill in the `TBD-at-launch` deployment margin in BACKBONE_ABLATION.md.
- [ ] 4b. Drop DRAFT (protocol freezes at first launch).
- [ ] 4c. Launch B (seed 42), then S (seed 42) on the A4500 / EC2 per
      [RUNBOOK-EC2.md](RUNBOOK-EC2.md).
- [ ] 4d. During/around the launches: fill PROFILING.md's results table (one
      trace + steady-state img/s + GPU ceiling per backbone; L via a short
      measurement-only window). Findings may drive plumbing changes ONLY
      (workers/pin_memory/disk) — bs/accum/precision/compile are frozen by the
      protocol (PROFILING.md Method).

## Parked (needs no decision now)

- CNN / simpler-architecture study → [FUTURE.md](FUTURE.md).
- Head ablation → ON HOLD, see ABLATION.md; arms decided after backbone verdict.
