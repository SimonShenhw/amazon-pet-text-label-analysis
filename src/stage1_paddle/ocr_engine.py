"""Shared PaddleOCR engine + helpers, used by Stage 1 (run_ocr) and Stage 3 (readability).

Accepts either a file path or an in-memory BGR numpy array, so Stage 3 can OCR a
downsampled image without writing it to disk.

NOTE: the high-level PaddleOCR.ocr() API returns recognition confidence per text
region but not detection confidence. For detection confidence you need the lower-level
DBNet det model — left as a Phase-2 TODO.
"""
from __future__ import annotations

import numpy as np

_OCR = None


def build_ocr(lang: str = "en"):
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)


def get_ocr(lang: str = "en"):
    """Lazily build a single shared PaddleOCR instance."""
    global _OCR
    if _OCR is None:
        _OCR = build_ocr(lang)
    return _OCR


def run_ocr(ocr, image):
    """Return a list of {text, score, box}. `image` is a path str or BGR numpy array."""
    result = ocr.ocr(image, cls=True)
    regions = []
    if not result or result[0] is None:
        return regions
    for line in result[0]:
        box, (text, score) = line
        regions.append({"text": text, "score": float(score), "box": box})
    return regions


def aggregate(regions):
    """Summarize a list of regions into per-image readability features."""
    n = len(regions)
    total_chars = sum(len(r["text"]) for r in regions)
    mean_score = float(np.mean([r["score"] for r in regions])) if regions else 0.0
    return {"n_regions": n, "total_chars": total_chars, "mean_score": round(mean_score, 4)}
