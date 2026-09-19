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

MFU is what makes img/s comparable across GPUs (A4500 vs EC2 candidates), so it is
the number for the instance-choice question, not raw img/s. The 3× convention
slightly under-estimates MFU for frozen-backbone LoRA; comparisons are unaffected.

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
| DataLoader workers / pin_memory / disk | plumbing | `--workers` | data-wait frac | none (no math change) | not measured |
| torch.compile | 1 | `--compile` | ceiling script | parity run (bundled) | not measured |
| fused AdamW | 1 | `--fused_opt` | ceiling script | parity run (bundled) | not measured |
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
| | cycle 0, none (own-run plan step 3c/4) | A10G | ViT-L | 448x728 | 6 | | | | | | | | |
| | family 1: `--compile --fused_opt` (step 3c, ceiling only) | A10G | ViT-L | 448x728 | 6 | n/a | | n/a | | | n/a (adopted into control if ≥ 15 % faster) | | |
| | cycle 0 on the A4500, if reachable | A4500 | ViT-L | 448x728 | 6 | | | | | | | | |

## Cost per epoch by instance

Filled from MFU × instance price once cycle zero has run on more than one GPU.

| instance | GPU | $/h | img/s (measured) | $/epoch | note |
|---|---|---|---|---|---|
| local | A4500 | 0 (sunk) | | | |
| g5.4xlarge | A10G | | | | 16 vCPU matches `--workers 16` |
| g6e.2xlarge | L40S | | | | 8 vCPU — watch data-wait |

## Decisions log

- **2026-09-18** — Recipe unfrozen (new-model goal). Doc restructured around the
  pairing rule and knob families; always-on readouts (data-wait, peak mem, TFLOPS/MFU)
  added to train.py; ceiling script gained sweeps + CSV; `--compile`, `--fused_opt`,
  `--precision`, `--grad_ckpt` flags added, all default-off so existing configs
  reproduce unchanged. Basis: prior. No measurement yet.
