---
title: "feat: IDNet ingestion pipeline (external/ reconstruction)"
type: feat
date: 2026-09-10
---

# ✨ IDNet ingestion pipeline — reconstruct `external/` for the backbone ablation

## Overview

Rebuild the missing `external/` layer (IDNet images + `idnet_cropped_index.csv`) that
the cv5 recipe, tier-1 validation, and REF anchoring all depend on — as a scripted,
verifiable, one-command data pipeline. The original author's `external/` was never
published; the raw data is public (Zenodo, CC0). This plan turns an undocumented
manual step into a reproducible pipeline and resolves the one open construction
question (which 2 of 4 fraud folders) by reply, by measurement, or by documented
choice — in that order of preference.

**Blocks:** TASKS.md 3a (IDNet branch), 3b (REF anchor), all of step 4 (launches).

## Problem Statement

- `external/idnet_cropped_index.csv` is the single manifest every IDNet consumer
  reads (train.py:317 pool build → 80k/4k; evaluate.py tier1/external; legacy sweep
  scripts). It exists only on the original author's machine.
- Probe-verified facts (2026-09-10): the Zenodo release is perfectly uniform — 10 EU
  countries × 5 folders (`positive`, `fraud1_copy_and_move`, `fraud2_face_morphing`,
  `fraud3_face_replacement`, `fraud4_combined`) × 5,979 JPEGs (`.png` extension),
  uniform dims per country (929×580 EST … 2163×1488 SRB), ~100GB of zips total.
- His EST+SVK pool (35,874) = exactly positives + 2 fraud folders per country.
  **Open:** which 2 folders; whether the other 8 countries follow the same rule.
- Upstream ask filed: nadhirhasan/FREUID-Challenge-2026#1 (index CSV or construction).

## Proposed Solution

Three-layer pipeline embodying the agreed data-engineering practices:

```
Zenodo (DOI-pinned, CC0)
  └─ scripts/ingest_idnet.sh          idempotent download + MD5 verify (ONE command)
       └─ s3://<bucket>/freuid/raw/idnet/          immutable zips, never edited
            └─ selective extraction               staged behind the folder question
                 └─ external/idnet/<country>/<folder>/<img>       processed images
                      └─ src/make_idnet_index.py                  deterministic manifest
                           └─ external/idnet_cropped_index.csv    the interface
                                └─ tests/test_idnet_index.py      validation gate
                                └─ external/DATA.md               provenance record
```

### Key design decisions

1. **`id` = relative path** (`est/positive/generated.photos_v3_xxx.png`). Kills the
   basename-collision trap (fraud folders reuse positive filenames), is unique,
   deterministic, and human-readable. `path` column = absolute, generated per-machine
   by `--root` (default `<repo>/external/idnet`); the CSV is a *derived* artifact —
   regenerate per machine, never `sed`. Row identity across machines = construction
   determinism (mandated `sorted()` walk) — recorded as a SHA of the
   (id,label,type) columns in DATA.md. Disjointness granularity verified against
   the writeup (2026-09-10): his claim is uniformly image/row-level ("disjoint-image
   (not disjoint-country)", "zero row-level overlap" — references.bib, tex:223/359/365,
   train.py help), so unique per-image ids implement the documented semantics;
   same-document variants MAY straddle train/val, covered by his stated
   "partially optimistic" caveat.
2. **Two indexes from one script — emitted at different phases:**
   `idnet_full_index.csv` (every extracted image, all folders — analysis superset,
   feeds the fingerprint test; the ONLY index that exists in Phase 2) and the
   canonical `idnet_cropped_index.csv` (filtered to the chosen 2 folders — what
   train.py and evaluate.py consume, drop-in compatible, zero consumer patches;
   first emitted in Phase 3, once the folder pair is known — `--fraud-folders` is
   precisely the parameter the Phase-3 gate exists to determine). Sharing one
   script guarantees the canonical index is a row-subset of the superset by
   construction.
3. **Labels from folder membership** (`positive`→0, `fraud*`→1), corroborated by the
   release's `meta/` JSONs. `meta/` is extracted too (small; enables the protocol's
   failure-mode slicing by attack type later).
4. **Staged extraction behind the folder question:** extract now only what every
   hypothesis needs — EST+SVK all 4 fraud folders (fingerprint test) + `positive/`
   and `meta/` for all 10 countries. Remaining 8-country fraud folders extract from
   S3 raw zips only after the pair is identified (~40% storage deferred/saved).
5. **The 2-folder question has three exits, tried in order:**
   a. **Issue #1 reply** → adopt his construction (adapter mode `--from-csv` rewrites
      his path prefix; compare row sets against ours; keep both, document).
   b. **Fingerprint test** → score frozen `cv5_full_ep2.pt` ONCE (raw readout) over
      all 59,790 EST+SVK images via `evaluate.predict` + cache; compute freuid_score
      for each of the 6 two-folder candidate pools from the cache; match against his
      three published reference numbers (raw full-pool **0.0154**, tier-1-style val
      **0.0166**, deploy full-pool **0.0116**). Decision rule: rank candidates by
      distance to 0.0154; accept if winner–runner-up gap ≫ eval noise (bootstrap CI
      scale); tie-break with the deploy readout vs 0.0116 (GPU floats aren't
      bit-portable — match by separation, not exactness).
   c. **Documented choice + amendment** → `fraud1_copy_and_move` +
      `fraud3_face_replacement` (closest analogs to his synthetic attack suite;
      morphing plausibly undetectable at 448×728), recorded as a dated
      pre-registered amendment in BACKBONE_ABLATION.md.
   Exits (b)/(c) also require the already-agreed amendment noting REF trained on his
   80k rows vs B/S on ours (same procedure/distribution, different rows) + the
   trained-L contingency arm option.

## Technical Considerations

- **Infra:** us-east-1 (bucket + all instances — same-region = free transfer). New
  bucket `freuid-ablation-839000214843`, prefixes `raw/idnet/`, `processed/idnet/`,
  `external/`, plus the runbook step-0 prefixes (`dataset/`, `splits/`, `weights/`).
  Ingest box: c7i.large + 250GB gp3 EBS (no GPU needed). Fingerprint session:
  g5.xlarge (raw pass ~40min; deploy tie-break ~2.7h if needed).
- **Idempotency:** `wget -c` resume; MD5 gate before any file is used (expected
  checksums fetched from the Zenodo API *and* pinned into DATA.md); extraction with
  `unzip -n` + per-country marker files; `aws s3 sync` everywhere. Success test: a
  fresh instance + one command converges to the same S3 state.
- **Ordering determinism:** filesystem walk order is not guaranteed — `sorted()` on
  every listing, or the 80k/4k row identity silently differs between machines.
- **Format quirk:** files are JPEG bytes with `.png` names — harmless (`cv2.imread`
  sniffs content), documented in DATA.md.
- **Zenodo throttling:** downloads may run 10–30MB/s; parallelize per record
  (≤3 concurrent), budget half a day wall-clock, mostly unattended.
- **CI:** `tests/test_idnet_index.py` uses `pytest.mark.skipif(not data present)` —
  runs on the box, skips cleanly in GitHub Actions.
- **Cost:** ingest ≈ $1–2 compute + $20/mo EBS while alive (terminate same day);
  S3 ≈ $4–6/mo (zips + selected images); fingerprint ≈ $1–4 GPU.
- **License:** CC0 (verified: Zenodo records + report §Licenses).

## Implementation Phases

### Phase 1 — Ingest to immutable raw (`scripts/ingest_idnet.sh`)
- Create bucket + prefixes; launch c7i.large (250GB gp3, existing SG/keypair, tag
  `project=freuid-ablation`).
- Download 10 zips (records 10611634: alb/esp/est; 10602369: aze/fin/grc/srb;
  10570622: lva/rus/svk — skip `california.zip`), resume-safe, ≤3 parallel.
- Verify MD5s against Zenodo API; write `raw/idnet/CHECKSUMS` (file, md5, date, DOI).
- `aws s3 sync` zips → `raw/idnet/`. **Deliverable: 10/10 verified zips in S3.**

### Phase 2 — Selective extraction + superset index
- Extract: EST+SVK fully; `positive/` + `meta/` for all 10 countries. Layout:
  `external/idnet/<country>/<folder>/…`.
- `src/make_idnet_index.py`: sorted walk → `idnet_full_index.csv` (+ a `--fraud-folders
  f1,f3` mode emitting the canonical `idnet_cropped_index.csv`); `--from-csv` adapter
  held ready for exit (a).
- `tests/test_idnet_index.py` (phase-2 assertions): per-folder count 5,979; EST+SVK
  full = 59,790 images; positives 10×5,979; id uniqueness; `type` values ⊆ the 10
  frozen strings from the arm configs; label↔folder consistency; sample of 50 paths
  opens via cv2 with the country's expected dims.
- Sync `processed/` + `external/` to S3; draft DATA.md (DOIs, checksums, date,
  script commit, row-hash, open questions + issue link).
- **Deliverable: validated superset index; fingerprint prerequisites in place.**

### Phase 3 — Resolve the folder pair (gate)
- Check issue #1 → exit (a) if answered.
- Else fingerprint (GPU session, `src/fingerprint_folders.py` ~60 lines reusing
  `evaluate.predict` + cache): one raw pass over 59,790 → 6 candidate scores →
  decision rule above → exit (b); ambiguous → deploy tie-break → else exit (c).
- Extract the identified pair for the remaining 8 countries (from S3 raw zips);
  finalize canonical index. **Acceptance counts: 179,370 total; 35,874 EST+SVK**
  (uniformity assumption documented if unconfirmed by reply).
- Write the dated amendment to BACKBONE_ABLATION.md (exits b/c) — before any launch.

### Phase 4 — Wire into the program
- Finalize DATA.md; flip TASKS.md 3-pre to done; on-box lockstep check that
  train.py's 80k/4k build and `evaluate.tier1_df()` select identical rows from the
  new index; ride-along: sync the FREUID dataset (15GB, overnight from the Mac) to
  `dataset/` — completing runbook step 0 entirely. Step 3 (smoke + REF anchor)
  unblocked.

## Acceptance Criteria

- [ ] One command, fresh instance → identical S3 state (rerun converges; interrupted
      run resumes; no manual steps).
- [ ] 10/10 zips MD5-verified; checksums pinned in DATA.md and CHECKSUMS.
- [ ] Canonical index: 179,370 rows; EST+SVK slice 35,874; ids unique; `type` values
      exactly match the frozen config strings; label↔folder consistent.
- [ ] Row-hash of (id,label,type) recorded; regeneration on a second machine
      reproduces it.
- [ ] pytest index suite green on-box, skipped (not failed) in CI.
- [ ] Folder-pair resolution recorded with its evidence (reply / fingerprint numbers
      / documented choice), and the BACKBONE_ABLATION.md amendment committed when
      exits (b)/(c) apply.
- [ ] DATA.md answers "where did every byte come from" without this conversation.

## Dependencies & Risks

| Risk | Mitigation |
|---|---|
| Zenodo slow/throttled | resume + parallelism; raw zone means we pay once, ever |
| Fingerprint non-discriminating | 3 reference numbers + deploy tie-break; fallback exit (c) with amendment |
| Uniformity assumption wrong for other 8 countries | only exit (a) can confirm; documented as assumption in DATA.md + amendment; distributional sanity check vs 80k-balance arithmetic |
| GPU float drift vs published numbers | match by candidate separation, not absolute equality |
| Walk-order nondeterminism breaks row identity | `sorted()` mandated + row-hash acceptance check |
| Nadhir replies *after* rebuild | `--from-csv` adapter; indexes swappable one-file; evaluate.py caches keyed by content survive |

## References

- Protocol: [BACKBONE_ABLATION.md](../../BACKBONE_ABLATION.md) (row-identity guarantee, tier-1 definition); [RUNBOOK-EC2.md](../../RUNBOOK-EC2.md) step 0.
- Consumers: [src/train.py](../../src/train.py) (~line 317 pool build), [src/evaluate.py](../../src/evaluate.py) (`tier1_df`, `external_df`, prediction cache).
- Task tracking: [TASKS.md](../../TASKS.md) item 3-pre.
- Upstream question: https://github.com/nadhirhasan/FREUID-Challenge-2026/issues/1
- Raw data: Zenodo records 10611634, 10602369, 10570622 (CC0); paper arXiv:2408.01690.
- Reference numbers for the fingerprint: report .tex lines ~225 (0.0166), ~321 (0.0154→0.0116).
