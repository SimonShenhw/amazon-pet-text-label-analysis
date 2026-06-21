"""Stage 0 — ingest: normalize raw product images into a clean, consistently-named set.

Scans data/raw/ for images, converts to RGB, optionally caps the long edge, saves
sequentially-numbered PNGs to data/processed/, and writes manifest.csv mapping each
processed image back to its original filename and dimensions.

Runs anywhere (host or Paddle container) — only needs Pillow + pandas.
"""
from __future__ import annotations

import pandas as pd
from PIL import Image

from src import config


def find_images():
    return sorted(
        p for p in config.RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMG_EXTS
    )


def cap_long_edge(img: Image.Image, max_edge):
    if not max_edge:
        return img
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    scale = max_edge / max(w, h)
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def main():
    config.ensure_dirs()
    images = find_images()
    if not images:
        raise SystemExit(
            f"No images found in {config.RAW_DIR}. "
            "Drop the 54 product images there and re-run."
        )
    rows = []
    for i, src in enumerate(images):
        img = Image.open(src).convert("RGB")
        orig_w, orig_h = img.size
        img = cap_long_edge(img, config.MAX_LONG_EDGE)
        image_id = f"product_{i:03d}"
        processed_name = f"{image_id}.png"
        img.save(config.PROCESSED_DIR / processed_name)
        rows.append({
            "image_id": image_id,
            "processed_name": processed_name,
            "original_name": src.name,
            "original_relpath": str(src.relative_to(config.RAW_DIR)),
            "orig_width": orig_w,
            "orig_height": orig_h,
            "proc_width": img.size[0],
            "proc_height": img.size[1],
        })
    pd.DataFrame(rows).to_csv(config.MANIFEST_CSV, index=False)
    print(f"Ingested {len(rows)} images -> {config.PROCESSED_DIR}")
    print(f"Wrote manifest: {config.MANIFEST_CSV}")


if __name__ == "__main__":
    main()
