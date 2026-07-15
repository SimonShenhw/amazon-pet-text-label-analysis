"""Formal OCR accuracy evaluation against held-out human annotations.

Compares Stage 1 OCR text (concatenated regions per image) with the ground truth
in data/annotations/annotations.csv and reports CER / WER / character-level F1.

Ground truth workflow: see data/annotations/README.md. Runs on host python:
  py -3.14 -m src.eval.run_eval
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.eval.metrics import cer, char_f1, wer

ANNOTATIONS_CSV = config.ANNOTATIONS_DIR / "annotations.csv"
EVAL_CSV = config.CSV_DIR / "eval_ocr_accuracy.csv"


def norm(s: str) -> str:
    """Case/whitespace-insensitive comparison text."""
    return " ".join(str(s).lower().replace("|", " ").split())


def main():
    config.ensure_dirs()
    if not ANNOTATIONS_CSV.exists():
        raise SystemExit(
            f"{ANNOTATIONS_CSV} not found.\n"
            "Copy data/annotations/annotations_template.csv to annotations.csv and "
            "fill ground_truth_text (see data/annotations/README.md)."
        )
    gt = pd.read_csv(ANNOTATIONS_CSV)
    gt = gt[gt["ground_truth_text"].notna() & (gt["ground_truth_text"].str.strip() != "")]
    if gt.empty:
        raise SystemExit("annotations.csv has no filled ground_truth_text rows yet.")

    s1 = pd.read_csv(config.STAGE1_CSV)
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
