"""Stage 1 — run PaddleOCR over all processed images.

Writes two CSVs:
  - stage1_ocr.csv          : one row per detected text region (text + rec confidence + box)
  - stage1_ocr_per_image.csv: per-image aggregate (n_regions, total_chars, mean_score)

CSV is the chosen interface between stages/environments: it's plain text, diffable,
and readable by pandas on either side of the host/Paddle-env split, with no shared
Python objects or pickle-format coupling required.

Run inside the Paddle container:
  docker compose run --rm paddle python -m src.stage1_paddle.run_ocr

中文：Stage1 对所有已处理图片跑 PaddleOCR，输出两份 CSV：逐区域明细
（stage1_ocr.csv）和逐图片汇总（stage1_ocr_per_image.csv）。选择 CSV 作为跨环境
（宿主机 / Paddle 环境）交接的接口格式：纯文本、可 diff、双端都能直接用 pandas 读取，
不依赖共享的 Python 对象或 pickle 格式，避免了环境耦合。
"""
from __future__ import annotations

import json

import cv2
import pandas as pd

from src import config
from src.stage1_paddle.ocr_engine import aggregate, get_ocr, run_ocr


def main():
    """Run OCR over every image in the manifest and write region + aggregate CSVs.

    对 manifest 中的每张图片跑 OCR，并写出逐区域明细 CSV 与逐图片汇总 CSV。
    """
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
            # Corrupt or unsupported file — skip this one image, don't crash the batch.
            # 文件损坏或格式不支持——跳过这一张，不让整个批处理因单张图片失败而中断。
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
                # Serialize as a JSON string, not a repr'd nested list, so downstream
                # readers (pandas, other stages, non-Python tools) can parse it safely.
                # 序列化为 JSON 字符串而非 Python repr 的嵌套列表，方便下游（pandas、
                # 其他 stage、非 Python 工具）安全地解析，而不必 eval 不受信的字符串。
                "box": json.dumps([[round(float(x), 1) for x in pt] for pt in r["box"]]),
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
