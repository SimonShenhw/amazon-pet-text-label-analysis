"""Stage 0 — ingest: normalize raw product images into a clean, consistently-named set.

Scans data/raw/ for images, converts to RGB, optionally caps the long edge, saves
sequentially-numbered PNGs to data/processed/, and writes manifest.csv mapping each
processed image back to its original filename and dimensions.

Images are renamed to sequential product_NNN.png for a clean, predictable working set,
but the original ASIN-based filename is preserved in manifest.csv — that filename is
the JOIN KEY back to the sponsor's dataset (its image_filename column), which Stage 4
needs to link OCR/readability results to business data. Losing it would silently break
that join.

Runs anywhere (host or Paddle container) — only needs Pillow + pandas.

中文：Stage 0 把 data/raw/ 下命名混乱、格式不一的原始图片，规范成统一命名、统一格式的
干净数据集。图片被重命名为顺序编号的 product_NNN.png 方便后续处理，但原始文件名
（基于 ASIN）会完整保存到 manifest.csv 里——这一列是后续 Stage 4 与赞助方数据集关联的
JOIN KEY（对应赞助方表的 image_filename 列），一旦丢失，后面所有业务关联分析都会断链。
本模块只依赖 Pillow + pandas，可在宿主机或 Paddle 环境中原样运行。
"""
from __future__ import annotations

import pandas as pd
from PIL import Image

from src import config


def find_images():
    """Find every supported image under data/raw/, recursively, in stable sorted order.

    递归扫描 data/raw/ 下所有受支持格式的图片，按路径排序，保证遍历顺序稳定可复现
    （重新跑一遍 ingest 时 product_NNN 的编号不会因文件系统遍历顺序而漂移）。
    """
    return sorted(
        p for p in config.RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMG_EXTS
    )


def cap_long_edge(img: Image.Image, max_edge):
    """Downscale `img` so its long edge <= max_edge, preserving aspect ratio.

    No-op if max_edge is falsy or the image already fits. Uses LANCZOS resampling,
    Pillow's highest-quality downscale filter — worth the extra cost since this runs
    once per image at ingest time, not in a hot loop.

    按长边等比缩小图片（不超过 max_edge），若 max_edge 为空或图片已足够小则不处理。
    使用 LANCZOS 重采样——Pillow 中质量最高的缩小滤波器；因为每张图只跑一次，用它换
    更好的下游 OCR 精度是值得的。
    """
    if not max_edge:
        return img
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    scale = max_edge / max(w, h)
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def main():
    """Ingest all raw images into data/processed/ and write manifest.csv.

    将所有原始图片处理后写入 data/processed/，并生成 manifest.csv（下游各 stage 的
    输入契约）。
    """
    config.ensure_dirs()
    images = find_images()
    if not images:
        raise SystemExit(
            f"No images found in {config.RAW_DIR}. "
            "Drop the 54 product images there and re-run."
        )
    rows = []
    for i, src in enumerate(images):
        # Normalize mixed jpg/png/webp inputs to a single color mode before processing.
        # 统一混合格式输入的色彩模式（消除 CMYK/palette/RGBA 等差异），再做后续处理。
        img = Image.open(src).convert("RGB")
        orig_w, orig_h = img.size
        img = cap_long_edge(img, config.MAX_LONG_EDGE)
        image_id = f"product_{i:03d}"
        processed_name = f"{image_id}.png"
        img.save(config.PROCESSED_DIR / processed_name)
        rows.append({
            "image_id": image_id,
            "processed_name": processed_name,
            # original_name/original_relpath = the JOIN KEY to the sponsor dataset
            # (used by Stage 4). 原始文件名/相对路径 = 关联赞助方数据集的 JOIN KEY
            # （供 Stage 4 使用），不可丢失。
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
