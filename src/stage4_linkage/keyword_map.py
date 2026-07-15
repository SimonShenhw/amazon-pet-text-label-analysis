"""Stage 4b — map extracted on-package text to the SSF keyword dataset (146 keywords).

For every image x keyword pair, test whether the keyword is present in the OCR text at
FULL resolution and whether it still matches in the MOBILE-THUMBNAIL OCR text. This turns
"claims double as keyword candidates" (project plan) into a concrete deliverable:
  - which search keywords are already visually reinforced on-pack,
  - which of them silently vanish at mobile-search size,
  - which high-intent keywords competitors show on-pack but SSF does not.

Matching is OCR-noise tolerant:
  - texts and keywords are lowercased and stripped to alphanumerics;
  - "phrase" match = keyword with spaces removed occurs in the squashed text
    (catches OCR artifacts like "DRYSHAMPOO");
  - "tokens" match = every keyword token (len>=3) appears somewhere in the text
    (catches reordering / split regions).

Host python:  py -3.14 -m src.stage4_linkage.keyword_map
"""
from __future__ import annotations

import re

import pandas as pd

from src import config

KEYWORD_HITS_CSV = config.CSV_DIR / "stage4_keyword_hits.csv"
STOP_TOKENS = {"for", "the", "and", "with", "your"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def match(keyword: str, text_norm: str, text_squash: str, token_set: set) -> str | None:
    if squash(keyword) and squash(keyword) in text_squash:
        return "phrase"
    tokens = [t for t in norm(keyword).split() if len(t) >= 3 and t not in STOP_TOKENS]
    if tokens and all(t in token_set for t in tokens):
        return "tokens"
    return None


def main():
    config.ensure_dirs()
    s3 = pd.read_csv(config.STAGE3_CSV)
    if "full_text" not in s3.columns:
        raise SystemExit("stage3_readability.csv has no text columns — re-run Stage 3 first.")
    mf = pd.read_csv(config.MANIFEST_CSV)
    meta = pd.read_excel(config.PET_CV_XLSX, sheet_name="images_cv_features")
    kw = pd.read_excel(config.SSF_CV_XLSX, sheet_name="keywords_dataset")

    df = (s3.merge(mf[["image_id", "original_name"]], on="image_id")
            .merge(meta[["image_filename", "asin", "brand", "performance_tier"]],
                   left_on="original_name", right_on="image_filename", how="left"))

    rows = []
    for _, img in df.iterrows():
        full_n = norm(img.get("full_text", "") or "")
        full_sq = squash(img.get("full_text", "") or "")
        full_tok = set(full_n.split())
        th_n = norm(img.get("mobile_thumb_text", "") or "")
        th_sq = squash(img.get("mobile_thumb_text", "") or "")
        th_tok = set(th_n.split())
        for _, k in kw.iterrows():
            m_full = match(k["keyword"], full_n, full_sq, full_tok)
            if not m_full:
                continue
            m_thumb = match(k["keyword"], th_n, th_sq, th_tok)
            rows.append({
                "image_id": img["image_id"], "original_name": img["original_name"],
                "brand": img.get("brand"), "performance_tier": img.get("performance_tier"),
                "keyword": k["keyword"], "category": k["category"],
                "intent_level": k["intent_level"], "keyword_type": k["keyword_type"],
                "use_for": k.get("use_for"),
                "match_full": m_full,
                "survives_mobile_thumb": bool(m_thumb),
            })

    hits = pd.DataFrame(rows)
    hits.to_csv(KEYWORD_HITS_CSV, index=False)
    print(f"Wrote {KEYWORD_HITS_CSV} ({len(hits)} image-keyword hits)")
    if hits.empty:
        return

    print("\n=== keyword hits by tier (unique keywords) ===")
    print(hits.groupby("performance_tier")["keyword"].nunique().to_string())

    ssf = hits[hits["performance_tier"] == "sponsor"]
    print(f"\n=== SSF: {ssf['keyword'].nunique()} keywords visually present on-pack ===")
    lost = ssf[~ssf["survives_mobile_thumb"]]["keyword"].unique()
    kept = ssf[ssf["survives_mobile_thumb"]]["keyword"].unique()
    print(f"survive mobile thumbnail: {sorted(set(kept))}")
    print(f"LOST at mobile thumbnail: {sorted(set(lost) - set(kept))}")

    # high-intent keywords visible on high-tier competitor packs but absent from SSF
    high_kw = set(hits[(hits.performance_tier == "high")
                       & (hits.intent_level == "high")]["keyword"])
    ssf_kw = set(ssf["keyword"])
    missed = sorted(high_kw - ssf_kw)
    print(f"\n=== opportunity: high-intent keywords on high-tier packs, absent on SSF ===")
    print(missed if missed else "(none)")


if __name__ == "__main__":
    main()
