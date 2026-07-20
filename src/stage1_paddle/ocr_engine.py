"""Shared PaddleOCR engine + helpers, used by Stage 1 (run_ocr) and Stage 3 (readability).

Accepts either a file path or an in-memory BGR numpy array, so Stage 3 can OCR a
downsampled image without writing it to disk.

Deliberately built on PaddleOCR 2.7.x's high-level `.ocr()` API rather than 3.x:
2.7.x's `.ocr()` has a stable, well-documented result format, while PaddleOCR 3.x
replaced it with `.predict()` and a different result structure. Pinning to the older,
simpler API trades newer features for predictability, which matters more here than
chasing the latest release.

NOTE: the high-level PaddleOCR.ocr() API returns recognition confidence per text
region but not detection confidence. For detection confidence you need the lower-level
DBNet det model — left as a Phase-2 TODO.

中文：本模块封装 PaddleOCR 引擎，供 Stage1（run_ocr）和 Stage3（readability）共用。
故意选用 PaddleOCR 2.7.x 的高层 `.ocr()` 接口而非 3.x：2.7.x 返回格式稳定、文档
清晰；3.x 把它换成了 `.predict()` 且返回结构不同。锚定在较旧但简单稳定的接口上，
用新特性换取可预测性，对本项目而言这笔交易是值得的。同时支持传入文件路径或内存中的
BGR ndarray，这样 Stage3 可以直接对缩放后的图像做 OCR，无需先写盘再读盘。当前只暴露
识别（recognition）置信度，检测（detection）置信度需要更底层的 DBNet 模型，留作
Phase-2 的 TODO。
"""
from __future__ import annotations

import numpy as np

_OCR = None


def build_ocr(lang: str = "en"):
    """Construct a fresh PaddleOCR instance.

    Expensive: loads detection + recognition + angle-classification models, on the
    order of seconds. Callers should go through get_ocr() to reuse one instance
    rather than calling this per image.

    构建一个新的 PaddleOCR 实例。开销较大：需要加载检测、识别、方向分类三套模型，
    耗时可达数秒，因此调用方应通过 get_ocr() 复用同一个实例，而不要每张图片都新建。
    """
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)


def get_ocr(lang: str = "en"):
    """Lazily build a single shared PaddleOCR instance.

    Lazy singleton: engine construction costs seconds, so every caller across
    Stage 1 and Stage 3 shares one instance instead of paying that cost per image.

    延迟初始化的单例：引擎构建耗时数秒，Stage1 和 Stage3 的所有调用共用这一个实例，
    避免每张图片都重新付一次初始化开销。
    """
    global _OCR
    if _OCR is None:
        _OCR = build_ocr(lang)
    return _OCR


def run_ocr(ocr, image):
    """Return a list of {text, score, box}. `image` is a path str or BGR numpy array.

    Accepting either type lets Stage 3 pass an already-downsampled in-memory image
    straight through, instead of writing a temp file just to satisfy a path-only API.

    返回 [{text, score, box}, ...] 列表；`image` 可以是文件路径字符串，也可以是
    内存中的 BGR numpy 数组。两种输入都支持，Stage3 就能把缩放后的图像直接传入，
    不必为了满足"只接受路径"的接口而先写临时文件。
    """
    result = ocr.ocr(image, cls=True)
    regions = []
    if not result or result[0] is None:
        return regions
    for line in result[0]:
        box, (text, score) = line
        regions.append({"text": text, "score": float(score), "box": box})
    return regions


def aggregate(regions):
    """Summarize a list of regions into per-image readability features.

    把一张图片的所有文本区域，汇总成图片级的可读性特征（区域数、总字符数、平均置信
    度），供 Stage1 的汇总 CSV 和 Stage3 的分辨率对比复用。
    """
    n = len(regions)
    total_chars = sum(len(r["text"]) for r in regions)
    mean_score = float(np.mean([r["score"] for r in regions])) if regions else 0.0
    return {"n_regions": n, "total_chars": total_chars, "mean_score": round(mean_score, 4)}
