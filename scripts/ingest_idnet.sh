#!/usr/bin/env bash
# IDNet ingestion: Zenodo -> verified local zips -> immutable S3 raw zone.
# Idempotent + resumable: wget -c resumes partial files; MD5-verified files get a
# .ok marker and are skipped on rerun; aws s3 sync converges. One command, no args.
#
# Usage (on the ingest instance, from the repo root):
#   bash scripts/ingest_idnet.sh
#
# See docs/plans/2026-09-10-feat-idnet-ingestion-pipeline-plan.md (Phase 1).
set -euo pipefail

BUCKET="${FREUID_BUCKET:-s3://freuid-ablation-839000214843/freuid}"
WORK="${FREUID_INGEST_DIR:-$HOME/idnet_raw}"

# The 10 EU-country zips across 3 Zenodo records (US-state zips deliberately excluded).
# record_id:filename
FILES=(
  "10611634:alb.zip" "10611634:esp.zip" "10611634:est.zip"
  "10602369:aze.zip" "10602369:fin.zip" "10602369:grc.zip" "10602369:srb.zip"
  "10570622:lva.zip" "10570622:rus.zip" "10570622:svk.zip"
)

mkdir -p "$WORK"
cd "$WORK"

echo "== fetching expected MD5s from the Zenodo API =="
python3 - <<'EOF' > CHECKSUMS.expected
import json, urllib.request
for rec in ["10611634", "10602369", "10570622"]:
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}") as r:
        data = json.load(r)
    for f in data["files"]:
        md5 = f["checksum"].removeprefix("md5:")
        print(f"{md5}  {f['key']}  record={rec}")
EOF
cat CHECKSUMS.expected

echo "== downloading (resume-safe, 3 parallel) =="
printf '%s\n' "${FILES[@]}" | xargs -P 3 -I{} bash -c '
  spec="{}"; rec="${spec%%:*}"; f="${spec##*:}"
  if [ -f "$f.ok" ]; then echo "skip $f (verified)"; exit 0; fi
  wget -c -q --show-progress -O "$f" "https://zenodo.org/records/$rec/files/$f?download=1"
'

echo "== verifying MD5s =="
for spec in "${FILES[@]}"; do
  f="${spec##*:}"
  [ -f "$f.ok" ] && continue
  expected=$(awk -v f="$f" '$2==f{print $1}' CHECKSUMS.expected)
  actual=$(md5sum "$f" | awk '{print $1}')
  if [ "$expected" = "$actual" ]; then
    echo "$actual  $f  OK"
    touch "$f.ok"
  else
    echo "MD5 MISMATCH for $f: expected $expected got $actual -- deleting; rerun to redownload" >&2
    rm -f "$f"
    exit 1
  fi
done

date -u +"verified-at: %Y-%m-%dT%H:%M:%SZ" > CHECKSUMS
cat CHECKSUMS.expected >> CHECKSUMS

echo "== syncing to immutable raw zone =="
aws s3 sync "$WORK" "$BUCKET/raw/idnet/" --exclude "*" --include "*.zip" --include "CHECKSUMS*"
echo "== ingest complete: $(ls *.zip | wc -l)/10 zips verified and in $BUCKET/raw/idnet/ =="
