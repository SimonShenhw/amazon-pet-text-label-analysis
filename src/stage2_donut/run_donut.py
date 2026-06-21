"""Stage 2 — Donut zero-shot structured extraction (HOST GPU).

Phase 1: run the pretrained CORD-v2 Donut to emit structured JSON per image and
validate the pipeline end-to-end. (CORD-v2 was trained for receipts, so the field
names will be receipt-like — Phase 1 just proves Donut produces structured output.)

Phase 2: fine-tune on a product-label schema {brand, primary_claim, claims[],
ingredients[]} using SynthDoG synthetic data + LoRA — see TODO at the bottom.

Runs on the host python (py -3.14) with torch 2.11+cu130 on the 5090.
torch.compile is intentionally NOT used (known Blackwell hang).

  py -3.14 -m src.stage2_donut.run_donut
"""
from __future__ import annotations

import json
import re

import pandas as pd

from src import config

MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
TASK_PROMPT = "<s_cord-v2>"


def load_model():
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


def run_one(processor, model, device, image):
    import torch

    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    seq = processor.batch_decode(outputs.sequences)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # strip the first task token
    return processor.token2json(seq)


def main():
    from PIL import Image

    config.ensure_dirs()
    if not config.MANIFEST_CSV.exists():
        raise SystemExit(f"manifest not found: {config.MANIFEST_CSV}. Run ingest first.")

    manifest = pd.read_csv(config.MANIFEST_CSV)
    processor, model, device = load_model()
    print(f"Loaded {MODEL_NAME} on {device}")

    results = {}
    for _, row in manifest.iterrows():
        img_path = config.PROCESSED_DIR / row["processed_name"]
        image = Image.open(img_path).convert("RGB")
        try:
            parsed = run_one(processor, model, device, image)
        except Exception as e:  # keep going; record the failure
            parsed = {"error": str(e)}
        results[row["processed_name"]] = parsed
        print(f"{row['processed_name']}: {parsed}")

    config.STAGE2_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {config.STAGE2_JSON}")


# TODO (Phase 2) — fine-tune for the product-label schema:
#   target JSON: {"brand": ..., "primary_claim": ..., "claims": [...], "ingredients": [...]}
#   1. synthesize labels with SynthDoG (clovaai/donut synthdog/), corpus = Amazon claim phrases
#   2. wrap encoder with peft.LoraConfig(target_modules=["query", "value"]) to limit overfit
#   3. training loop: adapt NielsRogge/Transformers-Tutorials Donut/CORD notebook
#   4. evaluate with src/eval/metrics.py (char-level F1 on field values)

if __name__ == "__main__":
    main()
