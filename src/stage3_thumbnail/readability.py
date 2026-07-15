"""Stage 3 — thumbnail readability scoring.

Downsamples each processed image to the resolutions Amazon serves (mobile thumbnail
~160px, search grid ~320px), re-runs OCR, and measures how much detected text and
confidence survive vs full resolution. This is the core "do claims survive mobile
rendering?" metric for the sponsor.

Run inside the Paddle container:
  docker compose run --rm paddle python -m src.stage3_thumbnail.readability
"""
from __future__ import annotations

import cv2
import pandas as pd

from src import config
from src.stage1_paddle.ocr_engine import aggregate, get_ocr, run_ocr


def _fit(image, target_long_edge):
    """Aspect-preserving downscale so the long edge == target.

    Never upscales: if the source is already smaller than the target, Amazon
    serves it at (at most) its native resolution, so the readability delta is 0
    by construction — squashing to a fixed square would fabricate distortion.
    """
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= target_long_edge:
        return image
    scale = target_long_edge / long_edge
    new_wh = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image, new_wh, interpolation=cv2.INTER_AREA)


def main():
    config.ensure_dirs()
    if not config.MANIFEST_CSV.exists():
        raise SystemExit(f"manifest not found: {config.MANIFEST_CSV}. Run ingest first.")

    manifest = pd.read_csv(config.MANIFEST_CSV)
    ocr = get_ocr(config.OCR_LANG)

    rows = []
    for _, row in manifest.iterrows():
        img_path = config.PROCESSED_DIR / row["processed_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        full_regions = run_ocr(ocr, image)
        full = aggregate(full_regions)
        rec = {
            "image_id": row["image_id"],
            "processed_name": row["processed_name"],
            "full_n_regions": full["n_regions"],
            "full_chars": full["total_chars"],
            "full_mean_score": full["mean_score"],
            # concatenated region texts -> lets Stage 4 test which specific
            # claims/keywords survive at each resolution
            "full_text": " | ".join(r["text"] for r in full_regions),
        }
        for name, target in config.THUMBNAIL_SIZES.items():
            thumb_regions = run_ocr(ocr, _fit(image, target))
            thumb = aggregate(thumb_regions)
            rec[f"{name}_text"] = " | ".join(r["text"] for r in thumb_regions)
            rec[f"{name}_n_regions"] = thumb["n_regions"]
            rec[f"{name}_chars"] = thumb["total_chars"]
            rec[f"{name}_mean_score"] = thumb["mean_score"]
            # readability deltas: how much text/confidence is lost at thumbnail size
            rec[f"{name}_char_retention"] = (
                round(thumb["total_chars"] / full["total_chars"], 4)
                if full["total_chars"] else 0.0
            )
            rec[f"{name}_score_delta"] = round(thumb["mean_score"] - full["mean_score"], 4)
        rows.append(rec)
        print(f"{row['processed_name']}: full_chars={full['total_chars']} "
              f"mobile_retention={rec.get('mobile_thumb_char_retention')}")

    pd.DataFrame(rows).to_csv(config.STAGE3_CSV, index=False)
    print(f"\nWrote {config.STAGE3_CSV} ({len(rows)} images)")


if __name__ == "__main__":
    main()
