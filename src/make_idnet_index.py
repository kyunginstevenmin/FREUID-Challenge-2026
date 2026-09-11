"""Build the IDNet manifest(s) from the extracted tree (deterministic, machine-portable).

Layout expected under --root (default <repo>/external/idnet):
    <root>/<country>/<folder>/<image>
where <country> is the lowercase code (est, svk, ...) and <folder> is `positive`
or one of the four fraud folders. `meta/` and non-image files are ignored.

Outputs:
  idnet_full_index.csv     every image found (analysis superset; Phase 2)
  idnet_cropped_index.csv  positives + --fraud-folders only (canonical, cv5-compatible;
                           Phase 3, once the folder pair is known)

Columns (consumer contract -- train.py / evaluate.py read these as-is):
  id      relative path from root (unique; fraud variants share basenames, so the
          folder prefix is what disambiguates them)
  path    absolute path on THIS machine (derived; regenerate per machine, never sed)
  label   0 for positive/, 1 for fraud*/
  type    "<COUNTRY>_scanned" -- must match the frozen config strings exactly
  source  "idnet"

Determinism: every listing is sorted(), so row content AND order are a pure function
of the extracted data. The row hash (sha256 over ordered id,label,type) is printed
and must reproduce across machines -- pin it in external/DATA.md.

Usage:
  python src/make_idnet_index.py                       # superset only
  python src/make_idnet_index.py --fraud-folders fraud1_copy_and_move,fraud3_face_replacement
"""
from __future__ import annotations
import argparse, hashlib, os, sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_EXTS = (".png", ".jpg", ".jpeg")   # release ships JPEG bytes under .png names
POSITIVE = "positive"


def build_rows(root: str):
    root = os.path.abspath(root)
    rows = []
    for country in sorted(os.listdir(root)):
        cdir = os.path.join(root, country)
        if not os.path.isdir(cdir):
            continue
        for folder in sorted(os.listdir(cdir)):
            fdir = os.path.join(cdir, folder)
            if not os.path.isdir(fdir) or folder == "meta":
                continue
            for fname in sorted(os.listdir(fdir)):
                if not fname.lower().endswith(IMAGE_EXTS):
                    continue
                rows.append({
                    "id": f"{country}/{folder}/{fname}",
                    "path": os.path.join(fdir, fname),
                    "label": 0 if folder == POSITIVE else 1,
                    "type": f"{country.upper()}_scanned",
                    "source": "idnet",
                })
    return pd.DataFrame(rows, columns=["id", "path", "label", "type", "source"])


def row_hash(df: pd.DataFrame) -> str:
    """Machine-independent identity: ordered (id,label,type); `path` excluded."""
    h = hashlib.sha256()
    for t in df.itertuples(index=False):
        h.update(f"{t.id},{t.label},{t.type}\n".encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "external", "idnet"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "external"))
    ap.add_argument("--fraud-folders", default="",
                    help="comma list; when set, ALSO emit the canonical "
                         "idnet_cropped_index.csv = positives + these folders only")
    args = ap.parse_args()

    df = build_rows(args.root)
    if df.empty:
        sys.exit(f"no images found under {args.root}")
    assert df.id.is_unique, "id collision -- should be impossible with relpath ids"

    os.makedirs(args.out_dir, exist_ok=True)
    full_path = os.path.join(args.out_dir, "idnet_full_index.csv")
    df.to_csv(full_path, index=False)
    per_country = df.groupby(df.id.str.split("/").str[0]).size().to_dict()
    print(f"full index: {len(df)} rows -> {full_path}")
    print(f"  per country: {per_country}")
    print(f"  row-hash(full): {row_hash(df)}")

    if args.fraud_folders:
        chosen = {POSITIVE, *(s.strip() for s in args.fraud_folders.split(",") if s.strip())}
        folder_of = df.id.str.split("/").str[1]
        missing = chosen - set(folder_of.unique())
        assert not missing, f"--fraud-folders names folders not in the tree: {missing}"
        sub = df[folder_of.isin(chosen)].reset_index(drop=True)
        crop_path = os.path.join(args.out_dir, "idnet_cropped_index.csv")
        sub.to_csv(crop_path, index=False)
        print(f"canonical index: {len(sub)} rows ({sorted(chosen)}) -> {crop_path}")
        print(f"  row-hash(canonical): {row_hash(sub)}")


if __name__ == "__main__":
    main()
