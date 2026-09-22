# Training profiling: cost side of every recipe decision

Engineering companion to [BACKBONE_ABLATION.md](BACKBONE_ABLATION.md) and
[ABLATION.md](ABLATION.md). Not pre-registered — this doc records measurements and
the decisions they drove; edit freely.

**Status change (2026-09-18, basis: prior).** The recipe is no longer frozen: the goal
is a new model that scores higher, so batch size, precision, accumulation,
resolution, and the model itself are all open. Profiling therefore stops being a
diagnostic study and becomes the *cost side* of each recipe decision. The old
"math-affecting knobs are off-limits" rule is replaced by the pairing rule below.

## Rules

1. **Pairing rule.** Any knob that changes the math (family 2 below) is adopted only
   with a gen-val score + bootstrap 95% CI in the same cycle, logged in W&B and the
   EVALUATION.md ledger. Cost numbers alone never adopt a math-affecting change.
2. **One knob per cycle** for family 2. Family 1 knobs may be bundled into a single
   parity run because their cost side is measured individually and cheaply first.
3. **Always-on readouts** (below) are logged on every run, so every score ablation
   also produces its cost row for free. Deep dives (trace, py-spy) are for explaining
   a bad number, not for routine runs.
4. **Cycle zero is a measurement with no knobs changed.** If the data-wait fraction
   is high, none of the GPU knobs will show their real gain; fix input first.

## Always-on readouts (every train.py run)

Logged per epoch to the console line and W&B under `perf/`:

| readout | meaning | how it's measured |
|---|---|---|
| `img/s` | pipeline throughput | wall-clock over the epoch |
| `wait` / `perf/data_wait_frac` | fraction of epoch the GPU sat idle waiting for a batch | wall time blocked in `next(loader)`; valid because the step syncs every iteration via `loss.item()` (see `perf.TimedLoader`) |
| `perf/peak_mem_gb` | peak allocated memory | `torch.cuda.max_memory_allocated`, reset per epoch |
| `perf/tflops`, `perf/mfu` | achieved TFLOPS and model-FLOPs utilization | forward FLOPs counted once (`torch.utils.flop_counter`) × 3 per step; peak from `perf.PEAK_TFLOPS` by GPU name, or `--gpu_peak_tflops` |
| `eval=` / `eval/sec` | wall time of the per-epoch eval passes (val + gen-val ≈ 37.9k images on the control recipe) | wall-clock around the `evaluate()` calls. Added 2026-09-22: eval was ≈ ¼ of an epoch and outside every other readout |

MFU is what makes img/s comparable across GPUs (A4500 vs EC2 candidates), so it is
the number for the instance-choice question, not raw img/s. The 3× convention
**over**-estimates MFU for frozen-backbone LoRA (frozen Linears skip their weight-grad
matmul; true multiplier ~2–2.5×, so reported MFU is up to ~1.5× too high — corrected
2026-09-19). The per-GPU peak constants come from vendor spec sheets and were not
verified on hardware. Treat MFU as a relative number between configs, never absolute.

**Verdict rule (basis: prior, 2026-09-18; revise when a number exists):**
data-wait ≥ 15 % → input-bound, plumbing (workers, decode, disk) goes first.
Otherwise compute-bound; compare MFU against ~35–45 % (a reasonable band for ViT
training on a workstation GPU) to judge remaining GPU headroom.

## Tools

- **`src/gpu_ceiling.py`** — fwd+bwd+opt on one cached batch, same recipe minus the
  input pipeline. `--bs` / `--res` take comma lists and sweep every combination;
  rows append to `results/gpu_ceiling.csv` (OOM is recorded as a row: the memory wall
  is a result). Gap vs real img/s = input starvation. Minutes per point, trains nothing.
- **`train.py --profile`** — one `torch.profiler` trace (wait=5 warmup=2 active=5,
  first trained epoch) into `profiles/<tag>/`, plus `key_averages.txt` (top-40 CUDA
  ops) so a run can be diagnosed from the log without opening TensorBoard. Deep dive
  only. Read: GPU-stream gaps aligned with DataLoader = input-bound; confirm the
  attention kernel is the fused SDPA one.
- **`nvidia-smi dmon`, `htop`, `py-spy dump --pid <worker>`** — sanity checks;
  py-spy on a DataLoader worker shows where CPU time goes if wait is high.

## Knobs

| knob | family | flag | cost measured by | verification | status |
|---|---|---|---|---|---|
| DataLoader workers / pin_memory / disk | plumbing | `--workers` | data-wait frac | none (no math change) | measured 2026-09-19: wait 1 % at 16 workers on g5.4xlarge — no change |
| eval batch size | plumbing | `--eval_bs` | `eval/sec` | none (`no_grad`; per-image outputs unchanged) | 8 → 32 in the control config 2026-09-22; gain unmeasured until the run |
| torch.compile | 1 | `--compile` | ceiling script | parity run (bundled) | measured 2026-09-19: +35 % with fused_opt — adopted into control config |
| fused AdamW | 1 | `--fused_opt` | ceiling script | parity run (bundled) | measured 2026-09-19 (bundled with compile; 6.8 M trainable params ⇒ negligible alone) — adopted |
| bf16 (no GradScaler) | 1 | `--precision bf16` | ceiling script | parity run (bundled) | not measured |
| batch size (+ LR rule) | 2 | `--bs`, `--accum`, `--lr_*` | ceiling `--bs` sweep (img/s, memory wall) | gen-val + CI | not measured |
| resolution | 2 | `--res` | ceiling `--res` sweep | gen-val + CI | not measured |
| LoRA depth | 2 | `only_last` (model.py; no CLI flag yet) | ceiling script | gen-val + CI | not measured |
| gradient checkpointing | 2 (enabler) | `--grad_ckpt` | ceiling script (memory vs ms/step) | gen-val + CI of the bs/res it enables | not measured |
| backbone size | 2 | `--backbone` | always-on readouts of the ablation runs | BACKBONE_ABLATION.md protocol | planned |

Family 1 = expected score-neutral (numerics only). Family 2 = changes the model or
optimization. Plumbing = input pipeline; no verification needed.

## Suggested order

1. **Cycle zero** = the own-run plan's step 3 session + control arm
   ([plans/2026-09-17-first-own-run-plan.md](plans/2026-09-17-first-own-run-plan.md)):
   `gpu_ceiling.py` for ViT-L at the control recipe (minutes), then the control
   arm's always-on readouts. Fills the baseline rows below.
2. **Family 1 bundle, decided before launch.** Ceiling script plain vs
   `--compile --fused_opt`; ≥ 15 % faster ⇒ into the control config (basis: prior).
   No parity run needed at that moment because the control arm *defines* the
   recipe; after launch, adopting the bundle costs a full parity run.
3. **Resolution curve** (most likely to move the score), then **batch size**, then
   **LoRA depth**. Backbone size rides on the ablation already planned.

## Results

One row per cycle. `real` columns come from the always-on readouts of a training run;
`ceiling` columns from `results/gpu_ceiling.csv`.

| date | cycle / knob | GPU | backbone | res | bs | img/s real | img/s ceiling | wait | MFU | peak mem GB | score (gen-val, CI) | verdict | action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-19 | cycle 0, none (step 3c ceiling + 3d wait probe, `--limit 500 --workers 16`) | A10G | ViT-L | 448x728 | 6 | 9.0 | 9.2 | 1 % | 50 % real / 51 % ceiling | 12.4 | n/a (probe, 1 epoch) | GPU-bound: real = 98 % of ceiling, wait ≪ 15 % | no `--workers` sweep; `--workers 16` stays |
| 2026-09-19 | family 1: `--compile --fused_opt` (step 3c, ceiling only) | A10G | ViT-L | 448x728 | 6 | n/a | 12.4 | n/a | 69 % ceiling | 12.5 | n/a (family 1, no parity run needed pre-launch) | +35 % img/s (485 vs 653 ms/step), same memory | adopted: `compile: true` + `fused_opt: true` into `own_v2_ctrl_s42.yaml` |
| | cycle 0 real side, control arm (step 4; epoch ≥ 1 for steady state under compile) | A10G | ViT-L | 448x728 | 6 | | 12.4 | | | | | | |
| | cycle 0 on the A4500, if reachable | A4500 | ViT-L | 448x728 | 6 | | | | | | | | |

### Family 1 ceiling: plain vs `--compile --fused_opt` (2026-09-19, A10G, ViT-L 448x728 bs 6 fp16)

From `results/gpu_ceiling.csv` (50 timed steps after 10 warmup, one cached batch, so
no input pipeline in either row). Same 7840 GFLOP forward count in both: compile fuses
kernels, it does not change the math.

| config | img/s | ms/step | achieved TFLOPS | MFU (3× convention) | peak mem GB |
|---|---|---|---|---|---|
| plain | 9.2 | 653 | 36.0 | 51 % | 12.4 |
| `--compile --fused_opt` | 12.4 | 485 | 48.5 | 69 % | 12.5 |
| delta | +35 % | −26 % | +35 % | +18 pp | +0.1 |

Read: MFU and TFLOPS move by exactly the img/s ratio, because FLOPs per step are held
fixed and only the seconds change — MFU here is img/s rescaled to the GPU's peak, nothing
more. Both MFU values are inflated by the same ~1.5× from the frozen-backbone 3× convention,
so the +18 pp gap is real but the absolute level is not. The gain is compile; fused AdamW
over 6.8 M trainable parameters is a rounding error and rides along for free.

### Cycle-zero findings (2026-09-19, basis: measured)

1. **`--compile --fused_opt` is 35 % faster at the same memory — adopted.** Ceiling
   9.2 → 12.4 img/s; the gain is compile, fused AdamW is free over 6.8 M params.
2. **16 workers is enough.** Wait 1 % on g5.4xlarge (16 vCPU). No sweep. Caveat: the
   probe ran with albumentations 2.0.8 silently defaulting the recipe's degrade/noise/
   dropout arguments, so per-image CPU work was lighter than the real recipe — re-read
   `wait=` from the control arm's epoch 1; the verdict stands if it stays < 15 %.
3. **~10 GB of memory headroom** (peak 12.4–13.5 GB allocated of 24 GB; usable is closer
   to 7–8 GB once allocator fragmentation is counted). bs / res are family 2: run the
   ceiling `--bs` / `--res` sweep (minutes) to find the wall, then one score run per
   change with gen-val + CI, LR rule if bs moves. Headroom justifies the sweep, not a
   setting.
4. **The real run is already at the ceiling** (9.0 vs 9.2 img/s, 98 %), so the compile
   gain has nowhere to leak — expect ~12 img/s in the control arm from epoch 1 — and
   every further speedup must come from GPU-side knobs; plumbing has nothing left.
5. **The A10G is doing honest work.** Reported 51 % MFU ÷ the ~1.5× frozen-backbone
   inflation ≈ 35–40 % true, inside the verdict band. The L40S question (5× peak, 8 vCPU)
   is now cheap to answer with `ceiling` + `wait` on a g6e.2xlarge, but with 16 workers
   needed to hold wait at 1 % here, an 8-vCPU box will likely be input-bound; only worth
   it if the control arm's spot bill turns out to matter.
6. **No score information.** Every probe was one epoch on 48–500 images with AUC ≈ 0.5;
   the wait probe's 0.88 is noise.
7. **Eval is ≈ ¼ of an epoch and no readout saw it** (added 2026-09-22). The epoch
   line times training only; each epoch also scores val + gen-val (≈ 37.9k images).
   REF's single-scale `evaluate.py` pass ran ≈ 24 img/s on this GPU ⇒ ≈ 26 min/epoch
   against ≈ 85 min of compiled training. `train.py` now prints `eval=<s>` / logs
   `eval/sec`; `eval_bs` 8 → 32 (plumbing). Wall estimate for the 5-epoch control run:
   ≈ 9–10 h compiled (was 16–20 h, a prior).

## Cost per epoch by instance

Filled from MFU × instance price once cycle zero has run on more than one GPU.

| instance | GPU | $/h | img/s (measured) | $/epoch | note |
|---|---|---|---|---|---|
| local | A4500 | 0 (sunk) | | | |
| g5.4xlarge | A10G | 1.62 on-demand (us-east-1 list, unverified); spot ≈ 0.7 | 9.0 plain / ~12.4 compiled (ceiling) | pending step 4 (needs train-set size × steady-state img/s) | 16 vCPU matches `--workers 16`; wait 1 % measured |
| g6e.2xlarge | L40S | | | | 8 vCPU — watch data-wait |

## Decisions log

- **2026-09-22** — Follow-through on the 2026-09-21 entry: `--compile --fused_opt` and
  `eval_bs: 32` written into the control config; `eval/sec` readout added (finding 7);
  resume bug fixed (`save_state` after the best update); albumentations pinned to 2.0.8
  and `augment.py` ported to 2.x names at the INTENDED values (Kyungin's call — a stated
  deviation from what upstream actually trained on; `tests/test_augment.py` guards it).
  One more 3b smoke with the final config is required before pre-registration. Basis:
  measured for the numbers, decision for the augmentation values.

- **2026-09-21** — Cycle zero measured (basis: measured, 2026-09-19 on g5.4xlarge; raw
  rows in `results/gpu_ceiling.csv`, logs in `results/smoke_2026-09-19/`). Ceiling plain
  9.2 img/s vs compiled+fused 12.4 (+35 %, identical peak memory) ⇒ step 3c rule met,
  flags adopted into the control config. Wait probe 9.0 img/s at 1 % wait ⇒ step 3d: no
  workers sweep. Caveats: (a) reported MFU (51 % / 69 %) is inflated by the 3× convention,
  compare rows only; (b) the 35–45 % band in the verdict rule assumes 3× too, so 69 % is
  not "above the band" in any absolute sense; (c) the wait probe ran with albumentations
  2.0.8 silently ignoring the recipe's degrade/noise/dropout arguments (smoke logs show
  the warnings), so the CPU-side augmentation cost was lighter than the intended recipe —
  re-read `wait=` from the control arm's epoch 1 once the pin/port fix lands; (d) compile
  costs ~75 s once in epoch 0, so epoch-0 img/s and MFU under compile are not steady state.
  Smoke also exposed a resume bug (`save_state` runs before the best-epoch update in
  train.py, so a resumed run forgets epoch 0's best) — fix before step 4; the offline
  freeze is unaffected.

- **2026-09-18** — Recipe unfrozen (new-model goal). Doc restructured around the
  pairing rule and knob families; always-on readouts (data-wait, peak mem, TFLOPS/MFU)
  added to train.py; ceiling script gained sweeps + CSV; `--compile`, `--fused_opt`,
  `--precision`, `--grad_ckpt` flags added, all default-off so existing configs
  reproduce unchanged. Basis: prior. No measurement yet.
