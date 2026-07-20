"""Stage 2 — Donut zero-shot structured extraction (HOST GPU).

Donut (Document Understanding Transformer) is OCR-free: a Swin encoder reads pixels
directly and a BART decoder emits structured text, so it can succeed even where
Stage 1 PaddleOCR's flat text output has no schema attached to it.

Phase 1: run the pretrained CORD-v2 Donut to emit structured JSON per image and
validate the pipeline end-to-end. (CORD-v2 was trained for receipts, so the field
names will be receipt-like — Phase 1 just proves Donut produces structured output.)
Zero-shot first is a deliberate risk-management choice: with only 54 real images,
training anything from scratch would overfit badly, so we validate the pretrained
checkpoint end-to-end before investing effort in fine-tuning. The receipt-schema
mismatch we expect here (menu/price/total field names) is exactly what motivates the
LoRA fine-tune in Phase 2.

Phase 2: fine-tune on a product-label schema {brand, primary_claim, claims[],
ingredients[]} using SynthDoG synthetic data + LoRA — see TODO at the bottom.

Runs on the host python (py -3.14) with torch 2.11+cu130 on the 5090.
torch.compile is intentionally NOT used (known hang on this machine's RTX 5090 /
Blackwell architecture — we accept eager-mode speed rather than risk a stuck process).

  py -3.14 -m src.stage2_donut.run_donut

中文说明：本模块是 Stage2 的零样本(zero-shot)验证阶段。Donut 是无需 OCR 的文档理解
Transformer（Swin 视觉编码器 + BART 文本解码器），直接从图像像素生成结构化文本，
弥补 Stage1 PaddleOCR 纯文本输出没有schema 的短板。先跑预训练 CORD-v2 checkpoint
而不直接微调，是风险控制的考量：仅有 54 张真实图片，从零训练极易过拟合，所以先端到
端验证预训练模型能否跑通，再决定是否投入微调成本。CORD 是收据数据集，字段名
(menu/price/total)预期会与我们的宠物产品标签schema 不匹配——这个预期中的错配正是
催生 Phase 2 LoRA 微调的原因。torch.compile 在本机 RTX 5090 (Blackwell 架构)上有
已知的卡死问题，因此有意不使用。
"""
from __future__ import annotations

import json
import re

import pandas as pd

from src import config

MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
TASK_PROMPT = "<s_cord-v2>"


def load_model():
    """Load the pretrained CORD-v2 Donut processor/model and pick a device.

    加载预训练 CORD-v2 Donut 的处理器与模型，并自动选择运行设备（GPU 优先）。
    """
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


def run_one(processor, model, device, image):
    """Run Donut's generate() on one image and decode the output into JSON.

    对单张图片执行 Donut 的生成式推理，并把解码结果转换为结构化 JSON。
    """
    import torch

    # seed the decoder with the CORD-v2 task-prompt token so generate() starts in the
    # right output mode — 用任务提示符 token 初始化 decoder,让生成从 CORD-v2 的输出格式开始
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
            bad_words_ids=[[processor.tokenizer.unk_token_id]],  # forbid <unk> — 禁止生成<unk>token
            return_dict_in_generate=True,
        )

    seq = processor.batch_decode(outputs.sequences)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # strip the first task token — 去掉开头的任务提示符token
    return processor.token2json(seq)


def main():
    """Batch-run zero-shot Donut over Stage 1's manifest and dump results to JSON.

    对 manifest 中的每张图片批量执行零样本 Donut 推理，并把结果写入 Stage2 JSON 文件。
    """
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
        except Exception as e:  # keep going; record the failure — 单张失败不中断整批,记录错误信息
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
# 中文：Phase 2 计划——用 SynthDoG 合成标签数据 + LoRA 微调，让模型输出我们自己的产品
# 标签schema而非 CORD 的收据schema；实际实现见 make_finetune_data.py / train_lora.py /
# infer_lora.py（此处 TODO 保留作为原始设计草案的记录）。

if __name__ == "__main__":
    main()
