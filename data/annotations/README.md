# Held-out OCR annotations (ground truth)

Fill `annotations.csv` (copy from `annotations_template.csv`) to enable the formal
accuracy evaluation (`py -3.14 -m src.eval.run_eval`).

## How to annotate (≈5 min per image)

1. Open the image from `data/processed/<image_id>.png` at full zoom.
2. In the `ground_truth_text` column, type **every legible piece of text** on the
   image (brand, claims, ingredients, net weight, slogans), separated by ` | `.
   - Keep original casing and punctuation where readable.
   - Skip text that is genuinely illegible at full resolution.
   - Decorative icons (paw prints, leaves) are not text — ignore.
3. Set `annotator` to your name and leave `notes` for anything ambiguous
   (curved text, cut-off words, stylized fonts).

The 10 template rows are a stratified held-out set: 3 sponsor (SSF), 5 high-tier,
2 medium-tier images, all text-bearing. Do not look at `outputs/csv/stage1_ocr.csv`
while annotating (it would bias the ground truth).

## What the evaluation reports

`src/eval/run_eval.py` compares Stage 1 OCR output against your annotations and
reports per-image and mean **CER**, **WER**, and **character-level F1**
(order-independent, robust to region ordering).
