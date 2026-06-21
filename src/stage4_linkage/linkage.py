"""Stage 4 — light business linkage (HOST).

Joins our pipeline outputs (Stage 1 OCR aggregate, Stage 3 thumbnail readability,
Stage 2 Donut claims) with the sponsor dataset (pet_cv_dataset_full.xlsx ->
images_cv_features) on image filename, and reports how OCR / claim / readability
features relate to the product performance tier.

Descriptive only (small n). Every stage is OPTIONAL — whatever exists gets joined,
so this runs even before Stage 1/3 are available.

  py -3.14 -m src.stage4_linkage.linkage
"""
from __future__ import annotations

import json
import re

import pandas as pd

from src import config

CV_SHEET = "images_cv_features"
CV_KEEP = [
    "image_filename", "asin", "brand", "performance_tier", "known_rating",
    "known_reviews", "text_density_pct", "ocr_word_count", "white_bg_compliance",
    "background_type", "brightness_mean", "sharpness_laplacian",
]


def load_manifest():
    if not config.MANIFEST_CSV.exists():
        raise SystemExit(f"missing {config.MANIFEST_CSV}; run ingest first.")
    return pd.read_csv(config.MANIFEST_CSV)


def load_cv_metadata():
    if not config.PET_CV_XLSX.exists():
        print(f"[warn] {config.PET_CV_XLSX} missing; tier/metadata join skipped.")
        return None
    df = pd.read_excel(config.PET_CV_XLSX, sheet_name=CV_SHEET)
    return df[[c for c in CV_KEEP if c in df.columns]]


def maybe_csv(path):
    return pd.read_csv(path) if path.exists() else None


def claims_per_image():
    if not config.STAGE2_JSON.exists():
        return None
    data = json.loads(config.STAGE2_JSON.read_text(encoding="utf-8"))
    rows = []
    for proc_name, parsed in data.items():
        payload = json.dumps(parsed, ensure_ascii=False)
        has_text = bool(re.search(r"[A-Za-z]{3,}", payload))
        rows.append({
            "processed_name": proc_name,
            "donut_has_text": int(has_text),
            "donut_payload_len": len(payload),
        })
    return pd.DataFrame(rows)


def main():
    config.ensure_dirs()
    df = load_manifest().copy()

    cv = load_cv_metadata()
    if cv is not None:
        df = df.merge(cv, how="left", left_on="original_name", right_on="image_filename")
        matched = df["performance_tier"].notna().sum() if "performance_tier" in df else 0
        print(f"matched {matched}/{len(df)} images to pet_cv_dataset_full")

    s1 = maybe_csv(config.STAGE1_AGG_CSV)
    if s1 is not None:
        df = df.merge(s1[["image_id", "n_regions", "total_chars", "mean_score"]],
                      how="left", on="image_id")
        print("joined Stage 1 OCR aggregate")

    s3 = maybe_csv(config.STAGE3_CSV)
    if s3 is not None:
        cols = ["image_id"] + [c for c in s3.columns
                               if ("char_retention" in c) or (c == "full_chars")]
        df = df.merge(s3[cols], how="left", on="image_id")
        print("joined Stage 3 readability")

    cl = claims_per_image()
    if cl is not None:
        df = df.merge(cl, how="left", on="processed_name")
        print("joined Stage 2 Donut claims")

    df.to_csv(config.STAGE4_SUMMARY_CSV, index=False)
    print(f"\nWrote merged table -> {config.STAGE4_SUMMARY_CSV} "
          f"({df.shape[0]} rows x {df.shape[1]} cols)")

    # --- descriptive analyses on whatever columns are present ---
    if "performance_tier" in df.columns and df["performance_tier"].notna().any():
        agg_cols = [c for c in ["total_chars", "mobile_thumb_char_retention",
                                "text_density_pct", "ocr_word_count",
                                "donut_payload_len", "known_rating"]
                    if c in df.columns]
        if agg_cols:
            print("\n=== mean by performance_tier ===")
            print(df.groupby("performance_tier")[agg_cols].mean().round(2).to_string())

    if "total_chars" in df.columns and "ocr_word_count" in df.columns:
        sub = df[["total_chars", "ocr_word_count"]].dropna()
        if len(sub) > 1:
            print(f"\ncross-check corr(our total_chars, sponsor ocr_word_count) = "
                  f"{sub['total_chars'].corr(sub['ocr_word_count']):.3f}")


if __name__ == "__main__":
    main()
