"""Stage 1 — run PaddleOCR over all processed images.

Writes two CSVs:
  - stage1_ocr.csv          : one row per detected text region (text + rec confidence + box)
  - stage1_ocr_per_image.csv: per-image aggregate (n_regions, total_chars, mean_score)

Run inside the Paddle container:
  docker compose run --rm paddle python -m src.stage1_paddle.run_ocr
"""
from __future__ import annotations

import cv2
import pandas as pd

from src import config
from src.stage1_paddle.ocr_engine import aggregate, get_ocr, run_ocr


def main():
    config.ensure_dirs()
    if not config.MANIFEST_CSV.exists():
        raise SystemExit(f"manifest not found: {config.MANIFEST_CSV}. Run ingest first.")

    manifest = pd.read_csv(config.MANIFEST_CSV)
    ocr = get_ocr(config.OCR_LANG)

    region_rows, agg_rows = [], []
    for _, row in manifest.iterrows():
        img_path = config.PROCESSED_DIR / row["processed_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[skip] cannot read {img_path}")
            continue

        regions = run_ocr(ocr, image)
        for i, r in enumerate(regions):
            region_rows.append({
                "image_id": row["image_id"],
                "processed_name": row["processed_name"],
                "region_idx": i,
                "text": r["text"],
                "rec_score": round(r["score"], 4),
                "box": r["box"],
            })
        agg = aggregate(regions)
        agg_rows.append({
            "image_id": row["image_id"],
            "processed_name": row["processed_name"],
            **agg,
        })
        print(f"{row['processed_name']}: {agg['n_regions']} regions, "
              f"{agg['total_chars']} chars, mean_score={agg['mean_score']}")

    pd.DataFrame(region_rows).to_csv(config.STAGE1_CSV, index=False)
    pd.DataFrame(agg_rows).to_csv(config.STAGE1_AGG_CSV, index=False)
    print(f"\nWrote {config.STAGE1_CSV} ({len(region_rows)} regions)")
    print(f"Wrote {config.STAGE1_AGG_CSV} ({len(agg_rows)} images)")


if __name__ == "__main__":
    main()
