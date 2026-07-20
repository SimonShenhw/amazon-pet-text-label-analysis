"""Phase 2 — LoRA fine-tune Donut for the product-label schema (SCAFFOLD — review before launching).

Trains naver-clova-ix/donut-base-finetuned-cord-v2 with LoRA adapters to emit our
domain schema {"brand", "product_type", "claims", "net_weight"} on the synthetic set
from make_finetune_data.py. Design notes:

  * LoRA rank 8 on the attention projections ("query"/"value" cover the Swin vision
    encoder; "q_proj"/"v_proj" cover the BART text decoder) trains <1% of the model's
    weights — deliberately right-sized against overfitting, since we only have 480
    synthetic samples to train on.
  * Targets are serialized as RAW JSON strings (no custom special tokens) — avoids
    resizing token embeddings, so LoRA-only training stays tiny and stable.
  * Hand-rolled training loop (not HF Trainer) — the transformers 5.x Trainer API has
    been drifting release to release, so a small loop we fully control is more
    reliable than chasing API changes; it also keeps checkpointing fully explicit.
  * STEP-LEVEL RESUMABLE: saves adapter + optimizer + step every --save-steps and
    auto-resumes from the latest checkpoint in --out. A crash never loses the run —
    this project also targeted HPC/Slurm execution, where preemption is a routine
    risk rather than an edge case.
  * transformers 5.x quirk: VisionEncoderDecoderConfig no longer proxies missing attrs
    such as pad_token_id / decoder_start_token_id from the sub-configs, and
    forward(labels=...) needs BOTH to correctly call shift_tokens_right internally —
    hence the explicit config assignment below (the same fix must be mirrored at
    inference time in infer_lora.py, or generation silently breaks).
  * bf16 autocast (native on modern GPUs, no fp16 loss-scaler complexity).
  * torch.compile is intentionally NOT used (known Blackwell hang on this machine).

Host GPU (py -3.14). Estimated 480 samples x 3 epochs on the 5090: ~15-30 min.
  py -3.14 -m src.stage2_donut.train_lora --epochs 3

中文说明：本模块用 LoRA 微调预训练 Donut，使其输出我们自己的产品标签schema而非
CORD 的收据schema。关键设计决策：(1) LoRA rank=8 只训练注意力投影层，参数量占比
<1%，是针对仅 480 张合成样本、极易过拟合而特意选定的"小体量"微调方式；(2) 训练
目标直接用原始 JSON 字符串而非自定义特殊 token，避免 resize token embedding，让
LoRA-only 训练保持轻量稳定；(3) 手写训练循环而非 HF Trainer——transformers 5.x 的
Trainer API 版本间变动频繁，自己掌控的小循环更可靠，也让断点保存逻辑完全透明；
(4) 按步保存断点(adapter+optimizer+step)且自动续跑，因为本项目的目标运行环境还
包含 HPC/Slurm，抢占式调度中断是常态而非例外；(5) transformers 5.x 起 config 不再
自动代理缺失属性，训练与推理两端必须显式设置且保持一致，否则生成结果会错位；
(6) bf16 混合精度，现代 GPU 原生支持、无需 fp16 那套 loss scaler 机制。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src import config

DATA_DIR = config.SYNTHETIC_DIR / "donut_train"
OUT_DIR = config.PROJECT_ROOT / "models" / "donut-lora-ssf"
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
MAX_TARGET_LEN = 256


def load_split(split: str):
    """Load one Donut-format split as (image path, ground-truth JSON string) pairs.

    读取 make_finetune_data.py 生成的某个数据分片(train/validation)，返回
    (图片路径, ground truth JSON 字符串) 组成的列表。
    """
    meta = DATA_DIR / split / "metadata.jsonl"
    if not meta.exists():
        raise SystemExit(f"{meta} not found — run make_finetune_data first.")
    rows = [json.loads(l) for l in meta.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [(DATA_DIR / split / r["file_name"], r["ground_truth"]) for r in rows]


def latest_checkpoint(out_dir: Path):
    """Return the highest-numbered checkpoint-* dir under out_dir, or None.

    返回 out_dir 下编号最大的 checkpoint-* 目录（供断点续跑使用），不存在则返回 None。
    """
    cks = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return cks[-1] if cks else None


def main():
    """CLI entry: LoRA-finetune Donut on the synthetic set, with step-level
    resumable checkpointing and per-epoch validation-loss logging.

    命令行入口：对 Donut 做 LoRA 微调，支持按步保存/自动续跑断点，并在每个
    epoch 结束后打印验证集 loss。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    from PIL import Image

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    tok = processor.tokenizer

    # transformers 5.x: VisionEncoderDecoderConfig no longer proxies missing attrs,
    # and forward(labels=...) reads BOTH of these for shift_tokens_right.
    # 中文：transformers 5.x 起 config 不再自动代理缺失属性，需在此手动显式赋值；
    # 训练和推理两端必须用同一套 pad/start token id，否则解码会错位。
    model.config.pad_token_id = tok.pad_token_id
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = (
            getattr(model.config.decoder, "decoder_start_token_id", None)
            or tok.bos_token_id
        )

    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        # covers Swin attention (query/value) and BART attention (q_proj/v_proj)
        # 中文：覆盖 Swin 视觉编码器(query/value)与 BART 文本解码器(q_proj/v_proj)的注意力投影
        target_modules=["query", "value", "q_proj", "v_proj"],
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resume = latest_checkpoint(OUT_DIR)
    if resume:
        # resume from the latest saved adapter — 从最近保存的 checkpoint 恢复 LoRA adapter 权重
        model = PeftModel.from_pretrained(model, resume / "adapter", is_trainable=True)
        print(f"RESUMED adapters from {resume}")
    else:
        model = get_peft_model(model, lora)  # fresh run — 全新训练,套用上面的 LoRA 配置
    model.print_trainable_parameters()
    model.to(device)

    train_rows = load_split("train")
    val_rows = load_split("validation")
    print(f"train={len(train_rows)}  val={len(val_rows)}")

    def encode(path, gt_json):
        """Encode one (image path, ground-truth JSON string) pair into tensors.

        将一条 (图片路径, ground truth JSON 字符串) 编码为模型输入张量：像素张量 +
        labels（padding 位置置为 -100，让 loss 计算时忽略这些位置）。
        """
        pixel = processor(Image.open(path).convert("RGB"),
                          return_tensors="pt").pixel_values[0]
        ids = tok(gt_json, max_length=MAX_TARGET_LEN, truncation=True,
                  padding="max_length", return_tensors="pt").input_ids[0]
        labels = ids.clone()
        labels[labels == tok.pad_token_id] = -100  # ignore padding in loss — 忽略padding位置的loss
        return pixel, labels

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    start_step = 0
    if resume and (resume / "trainer_state.json").exists():
        # restore step counter + optimizer momentum so resuming is seamless —
        # 恢复步数计数器与优化器动量状态,使续跑效果等同于从未中断
        state = json.loads((resume / "trainer_state.json").read_text())
        start_step = state["global_step"]
        opt.load_state_dict(torch.load(resume / "optimizer.pt", map_location=device))

    def save(step):
        """Write a step-level resumable checkpoint: adapter + optimizer + step count.

        保存断点续跑所需的最小状态：LoRA adapter 权重、优化器状态、当前全局步数。
        """
        ck = OUT_DIR / f"checkpoint-{step}"
        (ck / "adapter").mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ck / "adapter")
        torch.save(opt.state_dict(), ck / "optimizer.pt")
        (ck / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        print(f"  [checkpoint-{step} saved]")

    global_step = start_step
    steps_per_epoch = (len(train_rows) + args.batch - 1) // args.batch
    model.train()
    for epoch in range(args.epochs):
        # skip epochs already covered on resume
        # 中文：续跑时跳过已经训练过的 epoch，避免重复训练
        if (epoch + 1) * steps_per_epoch <= start_step:
            continue
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(order)
        running = 0.0
        for bi in range(0, len(order), args.batch):
            step_in_run = global_step - start_step
            batch = [train_rows[i] for i in order[bi:bi + args.batch]]
            pixels, labels = zip(*(encode(p, g) for p, g in batch))
            pixels = torch.stack(pixels).to(device)
            labels = torch.stack(labels).to(device)
            # bf16 autocast: native on modern GPUs, no fp16 loss-scaler complexity
            # bf16 自动混合精度:现代 GPU 原生支持,无需 fp16 那套 loss scaler 机制
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(pixel_values=pixels, labels=labels).loss / args.accum
            loss.backward()
            running += loss.item() * args.accum
            if (bi // args.batch + 1) % args.accum == 0:  # accumulate grads over --accum micro-batches — 梯度累积
                opt.step()
                opt.zero_grad()
            global_step += 1
            if global_step % 25 == 0:
                print(f"epoch {epoch} step {global_step}: loss {running / 25:.4f}")
                running = 0.0
            if global_step % args.save_steps == 0:
                save(global_step)

        # end-of-epoch validation loss (capped at 24 samples to keep this quick)
        # 每个 epoch 结束后在验证集的前24个样本上算 loss(限制数量以控制耗时)
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for p, g in val_rows[:24]:
                px, lb = encode(p, g)
                vloss += model(pixel_values=px[None].to(device),
                               labels=lb[None].to(device)).loss.item()
        print(f"== epoch {epoch}: val_loss {vloss / min(len(val_rows), 24):.4f} ==")
        model.train()

    save(global_step)
    print(f"DONE. Final adapters in {OUT_DIR}. "
          "Load with PeftModel.from_pretrained(base, <ckpt>/adapter) and parse the "
          "generated raw-JSON output with json.loads().")


if __name__ == "__main__":
    main()
