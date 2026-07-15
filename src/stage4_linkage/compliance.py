"""Stage 4c — Amazon main-image text compliance screen (promised in the project plan).

Amazon's main-image policy: pure white background, the product only, and no text,
logos, badges, watermarks or graphics that are not part of the product itself.
Text printed on the physical packaging is allowed; overlay text/badges are not.

OCR alone cannot prove whether detected text sits on the package or is an overlay,
so this module produces a *screening report*, not verdicts:
  - joins our Stage 1 OCR results with the sponsor dataset's image-type labels and
    white-background compliance column,
  - flags main images whose text volume is anomalous vs. peer main images
    (likely badges/overlays) for manual review.

Host python:  py -3.14 -m src.stage4_linkage.compliance
"""
from __future__ import annotations

import pandas as pd

from src import config

COMPLIANCE_CSV = config.CSV_DIR / "stage4_compliance_flags.csv"


def main():
    config.ensure_dirs()
    s1 = pd.read_csv(config.STAGE1_AGG_CSV)
    mf = pd.read_csv(config.MANIFEST_CSV)
    meta = pd.read_excel(config.PET_CV_XLSX, sheet_name="images_cv_features")

    df = (s1.merge(mf[["image_id", "original_name"]], on="image_id")
            .merge(meta[["image_filename", "asin", "brand", "performance_tier",
                         "image_type_label", "white_bg_pct", "white_bg_compliance"]],
                   left_on="original_name", right_on="image_filename", how="left"))

    main_imgs = df[df["image_type_label"] == "product_main"].copy()
    if main_imgs.empty:
        raise SystemExit("no product_main images found in metadata join")

    med_chars = main_imgs["total_chars"].median()

    def risk(row):
        flags = []
        wbc = row.get("white_bg_compliance")
        try:
            not_white = float(wbc) == 0.0
        except (TypeError, ValueError):
            not_white = str(wbc).strip().lower() in {"false", "no", "non_compliant"}
        if not_white:
            flags.append("background_not_white")
        if row["total_chars"] > max(2 * med_chars, 60):
            flags.append("text_heavy_vs_peers")   # possible overlay text / badges
        elif row["total_chars"] > 0:
            flags.append("text_present_verify_on_pack")
        return ";".join(flags) if flags else "clean"

    main_imgs["compliance_flags"] = main_imgs.apply(risk, axis=1)
    cols = ["image_id", "original_name", "brand", "performance_tier",
            "n_regions", "total_chars", "white_bg_pct", "white_bg_compliance",
            "compliance_flags"]
    out = main_imgs[cols].sort_values("total_chars", ascending=False)
    out.to_csv(COMPLIANCE_CSV, index=False)
    print(f"Wrote {COMPLIANCE_CSV} ({len(out)} main images)")
    print(f"(peer median on-main text volume: {med_chars:.0f} chars)")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
