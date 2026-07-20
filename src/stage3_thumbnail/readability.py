"""Stage 3 — thumbnail readability scoring.

Downsamples each processed image to the resolutions Amazon serves (mobile thumbnail
~160px, search grid ~320px), re-runs OCR, and measures how much detected text and
confidence survive vs full resolution. This is the core "do claims survive mobile
rendering?" metric for the sponsor — arguably the single most business-relevant number
this project produces: it answers whether the marketing text/claims a seller put on
the package are actually legible to a real shopper scrolling on a phone.

Run inside the Paddle container:
  docker compose run --rm paddle python -m src.stage3_thumbnail.readability

中文：Stage3 是本项目最贴近业务价值的核心指标——"包装上的文字/卖点，在 Amazon 移动端
缩略图渲染后是否还能被看清？"。做法：把每张处理后的图片降采样到 Amazon 实际投放的
分辨率（约 160px 移动缩略图、320px 搜索网格图），重新跑一次 OCR，对比全分辨率下检测
到的文字量和置信度，衰减多少即代表可读性损失多少。这个数字直接回答赞助方最关心的
问题：卖家精心设计的包装文案，消费者在手机上刷到时到底还能读到多少。
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

    WHY this matters: an earlier version of this function did a fixed 160x160 square
    resize, which distorted non-square images and fabricated readability loss that
    wasn't real — a bug caught and fixed during development. Aspect-preserving resize
    (and never upscaling) is what makes this metric defensible: any measured loss is
    real OCR degradation, not an artifact of squashing the image into the wrong shape.
    Uses cv2.INTER_AREA, the correct interpolation for downscaling (area-based
    anti-aliasing avoids the artifacts nearest/bilinear would introduce here).

    按长边等比缩小图片到 target_long_edge，绝不放大：如果原图已经比目标分辨率还小，
    Amazon 最多也只会按原图分辨率展示，此时可读性衰减理应为 0——如果强行缩放成固定
    的正方形，会人为制造出并不存在的形变和文字损失。

    这一点很关键：早期版本曾用固定 160x160 的正方形 resize，导致非正方形图片被拉伸
    变形，凭空制造出虚假的可读性损失——这是开发过程中发现并修复的一个真实 bug。保持
    宽高比缩放、且绝不放大，才能让这个指标经得起推敲：测到的衰减一定是真实的 OCR
    退化，而不是缩放方式带来的伪影。使用 cv2.INTER_AREA 插值——基于区域平均的抗
    锯齿方法，是缩小图像时正确的插值方式（比最近邻/双线性插值更不容易产生锯齿伪影）。
    """
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= target_long_edge:
        return image
    scale = target_long_edge / long_edge
    new_wh = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image, new_wh, interpolation=cv2.INTER_AREA)


def main():
    """Score every image's thumbnail readability and write stage3_readability.csv.

    对每张图片计算缩略图可读性指标，写出 stage3_readability.csv。
    """
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
            # Concatenated region texts (not just counts) let Stage 4 test WHICH
            # specific claims/keywords survive at each resolution, not just how many.
            # 保存拼接后的区域文本（而非只是数量），这样 Stage4 能判断具体是哪些卖点/
            # 关键词在该分辨率下存活，而不只是知道"存活了多少个"。
            "full_text": " | ".join(r["text"] for r in full_regions),
        }
        # Re-run OCR at each Amazon-served resolution and diff against full-res — this
        # loop produces the actual readability metric for every thumbnail size.
        # 在每个 Amazon 实际投放的分辨率下重新跑 OCR，并与全分辨率结果做差——这个循环
        # 产出的就是每种缩略图尺寸下的可读性指标本身。
        for name, target in config.THUMBNAIL_SIZES.items():
            thumb_regions = run_ocr(ocr, _fit(image, target))
            thumb = aggregate(thumb_regions)
            rec[f"{name}_text"] = " | ".join(r["text"] for r in thumb_regions)
            rec[f"{name}_n_regions"] = thumb["n_regions"]
            rec[f"{name}_chars"] = thumb["total_chars"]
            rec[f"{name}_mean_score"] = thumb["mean_score"]
            # Readability deltas: how much text/confidence is lost at thumbnail size.
            # char_retention is a ratio (1.0 = no loss); guarded against div-by-zero.
            # 可读性衰减指标：缩略图相对全分辨率丢失了多少文字/置信度。char_retention
            # 是比例（1.0 = 无损失），对分母为 0 的情况做了保护。
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
