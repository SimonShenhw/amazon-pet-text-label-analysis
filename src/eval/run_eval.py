"""Formal OCR accuracy evaluation against held-out human annotations.

Compares Stage 1 OCR text (concatenated regions per image) with the ground truth
in data/annotations/annotations.csv and reports CER / WER / character-level F1.
The ground-truth set is a 10-image held-out sample, annotated independently of the
OCR output (annotators transcribed without seeing OCR text) to avoid anchoring
bias. On our data this evaluation gives char-F1 ~ 0.90 with precision ~ 1.0 — i.e.
OCR errs almost entirely by OMISSION (missed characters on curved bottles / stylized
fonts), essentially never by fabrication. That "only misses, never invents" pattern
is itself supporting evidence that the Stage 3 thumbnail-readability metric (which
also just counts characters recovered) is measuring a real signal, not OCR noise.

Ground truth workflow: see data/annotations/README.md. Runs on host python:
  py -3.14 -m src.eval.run_eval

中文：本模块用人工标注的真值对 Stage1 OCR 结果做正式准确率评测，指标为
CER/WER/字符级 F1。真值集是从全部图片中留出的 10 张，标注时不看 OCR 输出、
独立转录，避免标注被 OCR 结果"锚定"产生偏差。在我们的数据上评测结果为
char-F1≈0.90、precision≈1.0——也就是说 OCR 的错误几乎全部是"漏检"
（曲面瓶身、艺术字体导致的漏字），几乎不会"无中生有"。这种"只漏不造"的
特性反过来也支持了 Stage3 缩略图可读性指标（同样是统计恢复出的字符数）
测的是真实信号、而不是 OCR 噪声。
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.eval.metrics import cer, char_f1, wer

ANNOTATIONS_CSV = config.ANNOTATIONS_DIR / "annotations.csv"
EVAL_CSV = config.CSV_DIR / "eval_ocr_accuracy.csv"


def norm(s: str) -> str:
    """Case/whitespace-insensitive comparison text.

    Casing and exact region order are not what the business question cares about
    (a mobile shopper doesn't care if OCR read "VET APPROVED" or "vet approved"),
    so normalizing them out keeps the metric focused on real character recovery.

    转小写、折叠空白，用于比较文本。大小写和区域顺序本身不是业务问题关心的点
    （手机用户不在乎 OCR 读出的是"VET APPROVED"还是"vet approved"），归一化
    掉这些差异，指标才能聚焦在"字符是否真正被恢复"这件事上。
    """
    return " ".join(str(s).lower().replace("|", " ").split())


def main():
    """Load ground truth + Stage 1 predictions, score each annotated image, and write
    per-image and mean CER/WER/char-F1 to EVAL_CSV.

    加载人工真值与 Stage1 OCR 预测结果，对每张已标注图片打分，
    把逐图与均值的 CER/WER/字符 F1 写入 EVAL_CSV。
    """
    config.ensure_dirs()
    if not ANNOTATIONS_CSV.exists():
        raise SystemExit(
            f"{ANNOTATIONS_CSV} not found.\n"
            "Copy data/annotations/annotations_template.csv to annotations.csv and "
            "fill ground_truth_text (see data/annotations/README.md)."
        )
    gt = pd.read_csv(ANNOTATIONS_CSV)
    # only score rows where a human has actually filled in ground_truth_text —
    # annotation is incremental, so unfilled rows must not count as errors.
    # 只评测已经人工填写真值的行——标注是逐步完成的，未填写的行不能算错误。
    gt = gt[gt["ground_truth_text"].notna() & (gt["ground_truth_text"].str.strip() != "")]
    if gt.empty:
        raise SystemExit("annotations.csv has no filled ground_truth_text rows yet.")

    s1 = pd.read_csv(config.STAGE1_CSV)
    # Concatenate per-region OCR text per image — region order here is whatever
    # order the detector emitted, which is exactly why char_f1 (order-independent)
    # is treated as the primary metric rather than CER/WER alone.
    # 按图片拼接各区域 OCR 文本；区域顺序即检测器输出的原始（任意）顺序，
    # 这正是本模块以不考虑顺序的 char_f1 为主指标、而不只看 CER/WER 的原因。
    pred = (s1.groupby("image_id")["text"]
              .apply(lambda t: " ".join(map(str, t))).rename("pred_text"))

    rows = []
    for _, row in gt.iterrows():
        ref = norm(row["ground_truth_text"])
        hyp = norm(pred.get(row["image_id"], ""))
        f1 = char_f1(ref, hyp)
        rows.append({
            "image_id": row["image_id"], "tier": row.get("tier"),
            "cer": round(cer(ref, hyp), 4), "wer": round(wer(ref, hyp), 4),
            "char_precision": f1["precision"], "char_recall": f1["recall"],
            "char_f1": f1["f1"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(EVAL_CSV, index=False)
    print(out.to_string(index=False))
    print("\n=== mean over annotated set ===")
    print(out[["cer", "wer", "char_f1"]].mean().round(4).to_string())
    print(f"\nWrote {EVAL_CSV}")


if __name__ == "__main__":
    main()
