#!/usr/bin/env bash
# Own-run plan steps 3-4 (docs/plans/2026-09-17-first-own-run-plan.md) ON the EC2 box, inside
# tmux, from the directory holding both repos ($WORK). One script, one mode per invocation:
#
#   bash own_run_ec2.sh setup     # clone fork, pinned venv, pull dataset_final + images (~33 GB)
#   bash own_run_ec2.sh smoke     # 3b: --limit 48 x1 plain, x1 --compile --fused_opt, + resume check
#   bash own_run_ec2.sh ceiling   # 3c: gpu_ceiling.py plain vs --compile --fused_opt -> results/gpu_ceiling.csv
#   bash own_run_ec2.sh wait      # 3d: --limit 500 --workers 16 pass; read the wait= readout
#   bash own_run_ec2.sh train     # 4:  the control arm (escrow loop alongside); needs PREREG=1
#   bash own_run_ec2.sh resume    # 5 of the runbook: same command + --resume after a spot death
#
# Env: WORK (default $PWD), WANDB_API_KEY (export before smoke/train), EXTRA="--compile --fused_opt"
# to pass extra flags to train/wait (the 3c decision goes into the YAML, EXTRA is for probes).
set -euo pipefail

DATA_BUCKET=s3://pscc-net-training                       # id-fraud-detection estate (its runbook step 7)
ABL_BUCKET=s3://freuid-ablation-839000214843/freuid      # this fork's results/checkpoints
BRANCH=feat/attnpool-head
WORK=${WORK:-$PWD}
FORK=$WORK/FREUID-Challenge-2026
IFD=$WORK/id-fraud-detection                             # private repo: not cloned, data only
VENV=$WORK/venv
PY=$VENV/bin/python
CFG=configs/own_v2_ctrl_s42.yaml
SPLIT=$IFD/dataset_final                                 # side-by-side layout, unlike the local ../../
MODE=${1:?mode: setup|smoke|ceiling|wait|train|resume}
EXTRA=${EXTRA:-}

setup() {
  cd "$WORK"
  [[ -d $FORK ]] || git clone -b $BRANCH https://github.com/kyunginstevenmin/FREUID-Challenge-2026.git
  # Pinned stack (CLAUDE.md CRITICAL #1): the PyPI torch wheel bundles its CUDA runtime, so the
  # DL AMI's driver is all we need from the image. Do NOT use /opt/pytorch (2.10) for training.
  # Ubuntu's python3 -m venv needs the python3-venv apt package to bootstrap pip; without it the
  # venv dir is created but has no pip. So test for pip, not the dir, and --clear a half-built one.
  python3 -c "import ensurepip" 2>/dev/null || { sudo apt-get update -q; sudo apt-get install -y -q python3-venv; }   # stale AMI index -> update first
  [[ -x $VENV/bin/pip ]] || python3 -m venv --clear "$VENV"
  $PY -m pip install -q --upgrade pip
  # docker/requirements.txt pins numpy 1.24.4 for the py3.11 submission image; it has no py3.12
  # wheel (AMI python is 3.12) and fails to build. Same pins otherwise; numpy = last 1.x with a wheel.
  grep -v '^numpy==' "$FORK/docker/requirements.txt" > /tmp/req.txt
  $PY -m pip install -q -r /tmp/req.txt numpy==1.26.4
  $PY -m pip install -q albumentations pyyaml scikit-learn wandb awscli pytest
  mkdir -p "$IFD"
  cd "$IFD"   # train/val/test.csv reference exactly these four prefixes (checked 2026-09-18)
  aws s3 sync $DATA_BUCKET/dataset_final/                    dataset_final/                        --only-show-errors
  aws s3 sync $DATA_BUCKET/dataset_cls/train/                dataset_cls/FREUID/FREUID_train/      --only-show-errors
  aws s3 sync $DATA_BUCKET/dataset_cls/FantasyID/            dataset_cls/FantasyID/                --only-show-errors
  aws s3 sync $DATA_BUCKET/dataset_cls/SIDT_clips_cropped/   dataset_cls/SIDT_clips_cropped/       --only-show-errors
  aws s3 sync $DATA_BUCKET/dataset_cls/IDNet_v2/             dataset_cls/IDNet_v2/                 --only-show-errors
  cd "$FORK"
  mkdir -p logs results checkpoints oof
  $PY -c "import torch, timm; import numpy; print('torch', torch.__version__, 'timm', timm.__version__, 'numpy', numpy.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name())"
  $PY -m pytest tests/ -q
  echo "setup done -> next: export WANDB_API_KEY=...; bash own_run_ec2.sh smoke"
}

# Every training invocation goes through here so the smoke, the wait probe and the real run
# share one command line (the only differences are the overrides listed after the config).
train_cmd() { $PY src/train.py --config $CFG --split_dir "$SPLIT" "$@"; }

smoke() {
  cd "$FORK"
  train_cmd --limit 48 --epochs 1 --workers 4 --wandb_project freuid-own-smoke --tag smoke_own_plain   $EXTRA 2>&1 | tee logs/smoke_own_plain.log
  train_cmd --limit 48 --epochs 1 --workers 4 --wandb_project freuid-own-smoke --tag smoke_own_compile --compile --fused_opt 2>&1 | tee logs/smoke_own_compile.log
  # resume check: one more epoch on the plain smoke must print "resumed ... -> start ep1"
  train_cmd --limit 48 --epochs 2 --workers 4 --wandb_project freuid-own-smoke --tag smoke_own_plain --resume 2>&1 | tee -a logs/smoke_own_plain.log
  grep -q "resumed .* start ep1" logs/smoke_own_plain.log && echo "resume OK"
  grep -H "compile: true" checkpoints/smoke_own_compile_resolved.yaml && echo "resolved dump shows the flags OK"
  grep -h "perf: gpu=" logs/smoke_own_*.log
  echo "smoke done -> next: bash own_run_ec2.sh ceiling"
}

ceiling() {
  cd "$FORK"
  $PY src/gpu_ceiling.py --backbone vit_large_patch14_reg4_dinov2 --res 448x728 --bs 6 2>&1 | tee -a logs/ceiling.log
  $PY src/gpu_ceiling.py --backbone vit_large_patch14_reg4_dinov2 --res 448x728 --bs 6 --compile --fused_opt 2>&1 | tee -a logs/ceiling.log
  aws s3 sync results $ABL_BUCKET/results --only-show-errors
  echo "ceiling done -> DECIDE (plan step 3c): compiled >= 15 % faster => add compile: true / fused_opt: true to $CFG, commit, git pull here"
}

wait_probe() {
  cd "$FORK"
  train_cmd --limit 500 --epochs 1 --workers 16 --wandb_project freuid-own-smoke --tag smoke_own_wait $EXTRA 2>&1 | tee logs/smoke_own_wait.log
  grep -h "img/s wait=\|peak_mem=" logs/smoke_own_wait.log | tail -3
  echo "wait probe done -> DECIDE (plan step 3d): wait >= 15 % => sweep --workers before launch"
}

train() {
  cd "$FORK"
  [[ "${PREREG:-}" == "1" ]] || { echo "refusing: pre-register in EVALUATION.md §6 first (config path + SHAs, seed, epochs, selection rule), then PREREG=1 bash own_run_ec2.sh train"; exit 2; }
  git rev-parse HEAD | tee logs/own_v2_ctrl_s42.sha
  bash scripts/escrow_loop.sh > logs/escrow.log 2>&1 &
  ESCROW=$!; trap 'kill $ESCROW 2>/dev/null || true' EXIT
  train_cmd $EXTRA ${RESUME:-} 2>&1 | tee -a logs/own_v2_ctrl_s42.log
  aws s3 sync checkpoints $ABL_BUCKET/checkpoints --only-show-errors
  aws s3 sync oof         $ABL_BUCKET/oof         --only-show-errors
  aws s3 sync logs        $ABL_BUCKET/logs        --only-show-errors
  echo "train done + synced -> local: aws s3 sync $ABL_BUCKET/oof oof; python src/freeze_epoch.py --tag own_v2_ctrl_s42"
}

case $MODE in
  setup)   setup ;;
  smoke)   smoke ;;
  ceiling) ceiling ;;
  wait)    wait_probe ;;
  train)   train ;;
  resume)  cd "$FORK"; aws s3 sync $ABL_BUCKET/checkpoints checkpoints --only-show-errors; RESUME=--resume PREREG=1 train ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
