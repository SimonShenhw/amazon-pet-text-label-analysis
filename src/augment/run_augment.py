"""Data augmentation + OCR robustness evaluation (Stage 1.5).

Two purposes:
1. **Augmented training pool** (instructor requirement: grow 54 images to 200+ samples
   for a proper train/test split). With 5 text-safe transforms we get 54 x (1+5) = 324
   samples. Pass --save to write variants to data/synthetic/augmented/ (+ manifest).
2. **OCR robustness evaluation**: re-run PaddleOCR on every augmented variant and measure
   how much text survives each transform vs the untouched original. This quantifies how
   sensitive on-package text legibility is to realistic photo conditions (lighting,
   camera tilt, sensor noise, focus) — a finding in its own right.

Transform choice — deliberately TEXT-SAFE only:
  Horizontal/vertical flips are standard for CNN classification but are *wrong* for
  text pipelines: mirrored text is unreadable, destroys OCR ground truth, and never
  occurs in real product photography. We therefore use rotation, brightness, contrast,
  Gaussian noise and blur, and exclude flips by design.

Run inside the Paddle venv:
  .venv-paddle\\Scripts\\python.exe -m src.augment.run_augment [--save]
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from src import config
from src.stage1_paddle.ocr_engine import aggregate, get_ocr, run_ocr

AUGMENTED_DIR = config.SYNTHETIC_DIR / "augmented"
ROBUSTNESS_CSV = config.CSV_DIR / "augment_robustness.csv"
AUG_MANIFEST_CSV = config.SYNTHETIC_DIR / "augmented_manifest.csv"


def rotate(image, deg):
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def brightness(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def contrast(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def gaussian_noise(image, sigma):
    noise = np.random.default_rng(0).normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def blur(image, k):
    return cv2.GaussianBlur(image, (k, k), 0)


# name -> callable; keep deterministic for reproducibility
TRANSFORMS = {
    "rot_plus6": lambda im: rotate(im, 6),
    "rot_minus6": lambda im: rotate(im, -6),
    "bright_low": lambda im: brightness(im, 0.65),
    "bright_high_contrast": lambda im: contrast(im, 1.35, 10),
    "noise_blur": lambda im: blur(gaussian_noise(im, 12), 3),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="also write augmented images to data/synthetic/augmented/")
    args = ap.parse_args()

    config.ensure_dirs()
    if not config.MANIFEST_CSV.exists():
        raise SystemExit(f"manifest not found: {config.MANIFEST_CSV}. Run ingest first.")
    manifest = pd.read_csv(config.MANIFEST_CSV)
    ocr = get_ocr(config.OCR_LANG)
    if args.save:
        AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

    rows, saved = [], []
    for _, rec in manifest.iterrows():
        img_path = config.PROCESSED_DIR / rec["processed_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        base = aggregate(run_ocr(ocr, image))
        rows.append({"image_id": rec["image_id"], "transform": "original",
                     "n_regions": base["n_regions"], "total_chars": base["total_chars"],
                     "mean_score": base["mean_score"], "char_retention": 1.0})
        for name, fn in TRANSFORMS.items():
            variant = fn(image)
            agg = aggregate(run_ocr(ocr, variant))
            rows.append({
                "image_id": rec["image_id"], "transform": name,
                "n_regions": agg["n_regions"], "total_chars": agg["total_chars"],
                "mean_score": agg["mean_score"],
                "char_retention": (round(agg["total_chars"] / base["total_chars"], 4)
                                   if base["total_chars"] else 0.0),
            })
            if args.save:
                out_name = f"{rec['image_id']}__{name}.jpg"
                cv2.imwrite(str(AUGMENTED_DIR / out_name),
                            variant, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved.append({"file": out_name, "source": rec["processed_name"],
                              "transform": name})
        print(f"{rec['processed_name']}: base_chars={base['total_chars']}")

    df = pd.DataFrame(rows)
    df.to_csv(ROBUSTNESS_CSV, index=False)
    print(f"\nWrote {ROBUSTNESS_CSV} ({len(df)} rows, "
          f"{df['image_id'].nunique()} images x {df['transform'].nunique()} conditions)")
    n_samples = df["image_id"].nunique() * df["transform"].nunique()
    print(f"Total sample pool: {n_samples} (instructor target: 200+)")
    if args.save:
        pd.DataFrame(saved).to_csv(AUG_MANIFEST_CSV, index=False)
        print(f"Saved {len(saved)} augmented images -> {AUGMENTED_DIR}")

    print("\n=== mean char retention by transform ===")
    # NB: df["transform"], not df.transform — .transform is a DataFrame method
    print(df[df["transform"] != "original"]
          .groupby("transform")[["char_retention", "mean_score"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
