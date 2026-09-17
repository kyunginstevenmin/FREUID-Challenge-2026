#!/usr/bin/env bash
# Step 0 of docs/plans/2026-09-17-first-own-run-plan.md: REF checkpoint on the gen-val gauge.
# Run ON the EC2 instance (Deep Learning AMI, instance role with S3 read on both buckets;
# only the public fork is cloned — id-fraud-detection is private, its data is synced from S3),
# inside tmux, from the directory you want both repos cloned into. Idempotent: re-running
# skips finished syncs and hits the prediction cache.
#
#   bash ref_genval_ec2.sh            # full: clone, pull ~33 GB, evaluate raw + deploy, push results
#   ONLY_EVAL=1 bash ref_genval_ec2.sh   # skip clone/pull (already on the box)
set -euo pipefail

DATA_BUCKET=s3://pscc-net-training                       # training estate (id-fraud-detection runbook step 7)
ABL_BUCKET=s3://freuid-ablation-839000214843/freuid      # weights + results (fork RUNBOOK-EC2)
BRANCH=feat/attnpool-head
WORK=${WORK:-$PWD}
# Deep Learning AMI: the CUDA torch lives in /opt/pytorch (not on a login shell PATH). We use
# its torch (2.10) rather than reinstalling the pinned 2.6 -- inference only; numerics may
# differ in the last digits vs the winner's report, which is fine for a baseline.
PY=${PY:-/opt/pytorch/bin/python}
FORK=$WORK/FREUID-Challenge-2026
IFD=$WORK/id-fraud-detection

if [[ -z "${ONLY_EVAL:-}" ]]; then
  cd "$WORK"
  [[ -d $FORK ]] || git clone -b $BRANCH https://github.com/kyunginstevenmin/FREUID-Challenge-2026.git
  mkdir -p "$IFD"        # private repo: not cloned; its CSVs + images come from S3 below
  $PY -m pip install -q timm==1.0.27 opencv-python-headless scikit-learn pandas pyyaml
  command -v aws >/dev/null || $PY -m pip install -q awscli

  # gen-val images: dataset_final/test.csv references these four prefixes (repo-root-relative)
  cd "$IFD"
  aws s3 sync $DATA_BUCKET/dataset_final/                    dataset_final/                        --only-show-errors
  aws s3 sync $DATA_BUCKET/dataset_cls/train/                dataset_cls/FREUID/FREUID_train/      --only-show-errors
  aws s3 sync $DATA_BUCKET/dataset_cls/FantasyID/            dataset_cls/FantasyID/                --only-show-errors
  aws s3 sync $DATA_BUCKET/dataset_cls/SIDT_clips_cropped/   dataset_cls/SIDT_clips_cropped/       --only-show-errors
  aws s3 sync $DATA_BUCKET/dataset_cls/IDNet_v2/             dataset_cls/IDNet_v2/                 --only-show-errors
  cd "$FORK"
  aws s3 sync $ABL_BUCKET/weights weights --only-show-errors
fi

cd "$FORK"
mkdir -p results preds
export GENVAL_CSV=$IFD/dataset_final/test.csv
$PY - <<'PY'
import os, pandas as pd
csv = os.environ["GENVAL_CSV"]; df = pd.read_csv(csv); root = os.path.dirname(os.path.dirname(csv))
missing = [p for p in df.image_path.sample(500, random_state=0) if not os.path.exists(os.path.join(root, p))]
assert not missing, f"{len(missing)}/500 sampled gen-val images missing, e.g. {missing[:3]}"
print(f"gen-val ok: {len(df)} rows, {df.type.nunique()} types, {df.source.nunique()} sources; sampled paths exist")
PY

for RO in raw deploy; do
  $PY src/evaluate.py --ckpt weights/cv5_full_ep2.pt --protocol genval --readout $RO --boot 1000 2>&1 | tee -a results/ref_genval.log
done

aws s3 sync preds   $ABL_BUCKET/preds   --only-show-errors
aws s3 sync results $ABL_BUCKET/results --only-show-errors
echo "done: results/eval_results.csv + results/ref_genval.log synced to $ABL_BUCKET/results"
