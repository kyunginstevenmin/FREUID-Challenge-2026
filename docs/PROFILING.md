# Training profiling study

Engineering companion to [BACKBONE_ABLATION.md](BACKBONE_ABLATION.md) (its
secondary latency readout) and [ABLATION.md](ABLATION.md). Not pre-registered —
this doc records measurements and the decisions they drove; edit freely.

## Questions

1. Is training GPU-bound or input-bound, per backbone size (L / B / S)?
2. What img/s ceiling does the GPU have vs what the full pipeline achieves?
3. Which instance type (local A4500 vs EC2 candidates) per hour of training is cheapest?

## Method

- Steady-state `img/s` from the train.py progress line (skip epoch-start).
- GPU ceiling: forward/backward loop on one cached batch, same res/bs — gap vs
  real img/s = input starvation.
- One `torch.profiler` trace per backbone (schedule wait=5 warmup=2 active=5,
  `tensorboard_trace_handler`), read in the TensorBoard trace viewer: GPU-stream
  gaps aligned with DataLoader = input-bound.
- `nvidia-smi dmon` + htop alongside, sanity check.
- Math-affecting knobs (bs/accum, precision, compile) are OFF-LIMITS — recipe is
  frozen by the ablation protocols. Plumbing only (workers, pin_memory, disk).

## Results

| run/backbone | GPU | img/s real | img/s ceiling | GPU util | verdict | action taken |
|---|---|---|---|---|---|---|
| ViT-L, A4500 | | | | | | |
| ViT-B, A4500 | | | | | | |
| ViT-S, A4500 | | | | | | |

## Decisions log

- (date) —
