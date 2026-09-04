# EC2 training runbook — ablation program

Operational doc for running the backbone + head ablations
([BACKBONE_ABLATION.md](BACKBONE_ABLATION.md), [ABLATION.md](ABLATION.md)) on EC2.
Companion: [PROFILING.md](PROFILING.md) (benchmark method). Edit freely.

Fill these once and use throughout:

```bash
export BUCKET=s3://<your-bucket>/freuid        # data + checkpoint store
export REGION=<region>                          # keep bucket and instance in the SAME region (else transfer fees + slowness)
export KEYPAIR=<your-ec2-keypair>
```

## 0. One-time local prep

```bash
cd reference/FREUID-Challenge-2026
du -sh the-freuid-challenge-dataset external weights          # know your sizes first
aws s3 sync the-freuid-challenge-dataset $BUCKET/dataset
aws s3 sync external $BUCKET/external                          # idnet_cropped_index.csv + IDNet images
aws s3 sync weights  $BUCKET/weights                           # cv5_full_ep2.pt etc. (REF + resume anchor)
aws s3 sync splits   $BUCKET/splits
git push fork feat/attnpool-head                               # instance clones from the fork
```

Paths matter: `data.py` resolves `DATA = <repo>/the-freuid-challenge-dataset` and
IDNet paths come from `external/idnet_cropped_index.csv` — check whether that CSV
stores absolute paths; if so, regenerate or sed them to the instance layout before
training.

## 1. Instance choice

Policy: **benchmark before committing** (PROFILING.md method — 30 min on the
candidate, read img/s, extrapolate). Shortlist:

| instance | GPU | vCPU | why |
|---|---|---|---|
| g5.4xlarge | A10G 24GB | 16 | ≈ A4500 speed; 16 vCPU matches `--workers 16` |
| g6e.2xlarge | L40S 48GB | 8 | ~2× GPU, but only 8 vCPU — benchmark for input starvation; drop `--workers` to ~6 |

Rules:
- vCPUs ≥ workers + 2, or the GPU starves (448×728 decode + `--attacks full` is real CPU work).
- **Spot is safe here**: train.py checkpoints every `--save_every 250` steps and has
  `--resume`; an interruption costs minutes. ~60–70% discount. Use on-demand only if
  spot capacity for the type is flaky in your region.
- Recipe flags are frozen by the protocols — the instance may change wall-clock,
  never the command line (workers is plumbing, allowed; bs/accum is not).

## 2. Launch

- AMI: **AWS Deep Learning AMI (PyTorch, Ubuntu)** — driver/CUDA preinstalled.
- Storage: instance NVMe if the type has it (g5.4xlarge does — mount it, put the
  dataset there); else gp3 EBS sized ≥ 2× dataset + 100 GB.
- IAM: attach an instance role with S3 read/write on `$BUCKET` — **no AWS keys on
  the box**.
- Security group: SSH (22) from your IP only. Nothing else inbound.
- Tag it (`project=freuid-ablation`) and set a billing alarm before the first long run.

## 3. Setup on the instance

```bash
tmux new -s abl                                # everything long-lived runs inside tmux
git clone -b feat/attnpool-head https://github.com/kyunginstevenmin/FREUID-Challenge-2026.git
cd FREUID-Challenge-2026
pip install -r docker/requirements.txt          # pinned: torch 2.6.0, timm 1.0.27, ...
pip install wandb mlflow awscli

aws s3 sync $BUCKET/dataset  the-freuid-challenge-dataset   # onto NVMe, not EBS root
aws s3 sync $BUCKET/external external
aws s3 sync $BUCKET/weights  weights
aws s3 sync $BUCKET/splits   splits

export WANDB_API_KEY=...                        # paste at runtime; never commit
python src/model.py                             # smoke: LoRA count, forward/backward
python src/train.py --full_data --limit 200 --epochs 1 --head_type patch --seed 42 --tag smoke ...  # full flags, tiny data
```

Smoke passes → run the PROFILING.md benchmark (15–30 min, fill its table) →
decide keep/switch instance → **bake an AMI** of the set-up box so reruns and
spot replacements boot in minutes instead of an hour.

## 4. Running the matrix

Window 1 — training (runs sequentially; the loop survives you detaching):

```bash
for RUN in "<arm/seed combos from the protocol doc>"; do
  python src/train.py <frozen recipe flags> $RUN 2>&1 | tee -a logs/$TAG.log
done
```

Window 2 — continuous checkpoint escrow (survives spot death; cheap):

```bash
while true; do
  aws s3 sync checkpoints $BUCKET/checkpoints
  aws s3 sync oof         $BUCKET/oof
  aws s3 sync logs        $BUCKET/logs
  sleep 600
done
```

W&B uploads metrics live; MLflow uses a local file store — it's inside the repo
dir, add `mlruns` to the sync loop so tracking survives the instance.

## 5. Spot interruption / resume

1. Launch replacement from the baked AMI (same type, or fall back to on-demand).
2. `aws s3 sync $BUCKET/checkpoints checkpoints` (+ external/dataset if not on AMI).
3. Re-run the SAME train.py command with `--resume`. Verify the log line shows the
   resumed step/epoch before walking away.

Interruption loses ≤ `--save_every` steps + up to one sync interval of escrow lag.

## 6. Retrieval and teardown

```bash
# on instance — final flush
aws s3 sync checkpoints $BUCKET/checkpoints && aws s3 sync mlruns $BUCKET/mlruns
# locally — pull only what analysis needs (lean ckpts + oof + logs), not every epoch ckpt
aws s3 sync $BUCKET/oof oof && aws s3 cp $BUCKET/logs logs --recursive
```

Then **terminate** (not stop — stopped instances keep billing EBS), and check:
no orphaned EBS volumes, no elastic IPs, billing dashboard shows the spend you
expected. The S3 copy is the durable record; the instance is disposable.

## 7. Cost envelope (sanity, from the ≈5 h/epoch A4500 anchor)

- Backbone study (~35 h, single-seed v2) + head study (~8–25 h) ≈ 45–60 h A4500-equivalent.
- g5.4xlarge spot ≈ $0.6–0.8/h → **≈ $30–50 total**; on-demand ≈ 2.5×.
- Storage/transfer: S3 ≈ $0.023/GB/mo, free within-region to EC2 — noise next to compute.
- If any single line item projects > $150, stop and re-benchmark before proceeding.
