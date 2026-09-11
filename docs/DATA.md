# IDNet external data — provenance record

Where every byte of `external/` came from and how the index was built. Companion to
[plans/2026-09-10-feat-idnet-ingestion-pipeline-plan.md](plans/2026-09-10-feat-idnet-ingestion-pipeline-plan.md);
pipeline code: `scripts/ingest_idnet.sh`, `src/make_idnet_index.py`,
`tests/test_idnet_index.py`.

## Source

IDNet (Guo et al., arXiv:2408.01690), **CC0**, official Zenodo release. EU-country
zips only (10 of 20 document types; all US-state zips excluded):

| record | files |
|---|---|
| [10611634](https://zenodo.org/records/10611634) | alb.zip, esp.zip, est.zip |
| [10602369](https://zenodo.org/records/10602369) | aze.zip, fin.zip, grc.zip, srb.zip |
| [10570622](https://zenodo.org/records/10570622) | lva.zip, rus.zip, svk.zip (california.zip excluded) |

Downloaded + MD5-verified **2026-09-11T05:27:05Z** against the Zenodo API's
published checksums (10/10 OK; full list in `raw/idnet/CHECKSUMS` in the bucket,
mirrored alongside the zips). Raw zone: `s3://freuid-ablation-839000214843/freuid/raw/idnet/`
(immutable — never edited; everything below is rebuildable from it).

## Release facts (verified by remote probe 2026-09-10 + on-box tests 2026-09-11)

- Perfectly uniform: per country, 5 folders (`positive`, `fraud1_copy_and_move`,
  `fraud2_face_morphing`, `fraud3_face_replacement`, `fraud4_combined`) × 5,979
  images. Images are JPEG bytes under `.png` names (harmless; cv2 sniffs content).
- Per-country pixel dims are uniform within a country, differ across countries
  (e.g. EST 929×580, SRB 2163×1488) — consistent with the solution report's
  template claim. Used **as shipped**: no re-cropping/re-encoding of any kind.
- Labels derive from folder membership (`positive`→0, `fraud*`→1); the release's
  `meta/` JSONs (extracted, in `processed/`) corroborate and enable per-attack-type
  slicing.

## Extraction state (Phase 2, 2026-09-11)

Staged behind the open folder-pair question (plan §Phase 3): EST+SVK extracted in
full (all 4 fraud folders, for the fingerprint test); `positive/` + `meta/` for all
10 countries. Processed zone: `s3://…/freuid/processed/idnet/` (25.4GB, 167,462
objects). Remaining 8-country fraud folders extract from the raw zips once the
pair is resolved.

## Index

`external/idnet_full_index.csv` — the analysis **superset** over everything
extracted (NOT the canonical training index):

- **107,622 rows** = 8 × 5,979 (positive-only countries) + 2 × 29,895 (EST/SVK full)
- Built by `src/make_idnet_index.py` (sorted-walk deterministic; `id` = relative
  path; `path` machine-local — regenerate per machine, never edit)
- **row-hash (id,label,type):** `95efae5b8af0b7473c8f11382ac48c2bccfbdbee19dc318a6aae8dc340649af1`
  — regeneration anywhere must reproduce this
- Acceptance tests green on the ingest box (7 passed: 5 synthetic + 2 real-data)

`external/idnet_cropped_index.csv` — the **canonical** cv5-compatible index
(positives + the chosen 2 fraud folders; target 179,370 rows, 35,874 EST+SVK):
**does not exist yet** — emitted at Phase 3 once the folder pair is resolved.

## Open questions

0. **⚠ MAJOR (2026-09-11, fingerprint session):** the as-shipped Zenodo images are
   **not what the winning model trained on**. Evidence: REF (`cv5_full_ep2.pt`)
   scores at CHANCE on every candidate EST+SVK pool (all 6 pairs ≈ FREUID 0.95–0.97,
   AuDET ≈ 0.50, vs his published 0.0154) — and scores the as-shipped *positives*
   fraud-ish (mean 0.70) despite training on 40k IDNet positives as genuine.
   Pipeline verified correct the same session: identical stack on FREUID train
   images gives perfect separation (genuine 0.082 / fraud 0.995, FREUID 0.0000).
   ⇒ An image-level transformation (plausibly what `_scanned` denotes — e.g. a
   print-scan simulation) sits between Zenodo and his index, unpublished.
   Checked: the Sept-2024 Zenodo records are the same release re-partitioned, not
   a scanned variant. Plan's exit (b) is dead as posed; exit (c) is compromised
   too (a from-Zenodo tier-1 val anchors REF at chance, trivializing
   non-inferiority). Prediction cache for the 59,790-image pass: S3
   `preds/913c44823e78_estsvk_all_raw.csv`; per-pair table `results/fingerprint_raw.csv`.

1. **Which 2 fraud folders** the original solution indexed — upstream ask:
   [nadhirhasan/FREUID-Challenge-2026#1](https://github.com/nadhirhasan/FREUID-Challenge-2026/issues/1)
   (no reply as of 2026-09-11); fallback = fingerprint test vs the report's
   published numbers (plan §exit b), else documented choice + protocol amendment.
2. **Uniformity of his selection across the other 8 countries** — only his reply
   can confirm; our construction assumes it and documents the assumption.
3. `_scanned`/`cropped` naming: concluded to be descriptive labels of the release's
   as-shipped form (paper defines no such variants; probe found one rendition,
   pre-cropped). Treated as opaque strings that must match the frozen configs.
