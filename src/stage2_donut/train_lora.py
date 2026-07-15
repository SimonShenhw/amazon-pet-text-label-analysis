"""Phase 2 — LoRA fine-tune Donut for the product-label schema (SCAFFOLD — review before launching).

Trains naver-clova-ix/donut-base-finetuned-cord-v2 with LoRA adapters to emit our
domain schema {"brand", "product_type", "claims", "net_weight"} on the synthetic set
from make_finetune_data.py. Design notes:

  * Targets are serialized as RAW JSON strings (no custom special tokens) — avoids
    resizing token embeddings, so LoRA-only training stays tiny and stable.
  * Hand-rolled training loop (not HF Trainer) — immune to Trainer API drift and
    keeps checkpointing explicit.
  * STEP-LEVEL RESUMABLE: saves adapter + optimizer + step every --save-steps and
    auto-resumes from the latest checkpoint in --out. A crash never loses the run.
  * torch.compile is intentionally NOT used (known Blackwell hang on this machine).

Host GPU (py -3.14). Estimated 480 samples x 3 epochs on the 5090: ~15-30 min.
  py -3.14 -m src.stage2_donut.train_lora --epochs 3
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
    meta = DATA_DIR / split / "metadata.jsonl"
    if not meta.exists():
        raise SystemExit(f"{meta} not found — run make_finetune_data first.")
    rows = [json.loads(l) for l in meta.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [(DATA_DIR / split / r["file_name"], r["ground_truth"]) for r in rows]


def latest_checkpoint(out_dir: Path):
    cks = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return cks[-1] if cks else None


def main():
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
    model.config.pad_token_id = tok.pad_token_id
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = (
            getattr(model.config.decoder, "decoder_start_token_id", None)
            or tok.bos_token_id
        )

    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        # covers Swin attention (query/value) and BART attention (q_proj/v_proj)
        target_modules=["query", "value", "q_proj", "v_proj"],
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resume = latest_checkpoint(OUT_DIR)
    if resume:
        model = PeftModel.from_pretrained(model, resume / "adapter", is_trainable=True)
        print(f"RESUMED adapters from {resume}")
    else:
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.to(device)

    train_rows = load_split("train")
    val_rows = load_split("validation")
    print(f"train={len(train_rows)}  val={len(val_rows)}")

    def encode(path, gt_json):
        pixel = processor(Image.open(path).convert("RGB"),
                          return_tensors="pt").pixel_values[0]
        ids = tok(gt_json, max_length=MAX_TARGET_LEN, truncation=True,
                  padding="max_length", return_tensors="pt").input_ids[0]
        labels = ids.clone()
        labels[labels == tok.pad_token_id] = -100
        return pixel, labels

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    start_step = 0
    if resume and (resume / "trainer_state.json").exists():
        state = json.loads((resume / "trainer_state.json").read_text())
        start_step = state["global_step"]
        opt.load_state_dict(torch.load(resume / "optimizer.pt", map_location=device))

    def save(step):
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
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(pixel_values=pixels, labels=labels).loss / args.accum
            loss.backward()
            running += loss.item() * args.accum
            if (bi // args.batch + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad()
            global_step += 1
            if global_step % 25 == 0:
                print(f"epoch {epoch} step {global_step}: loss {running / 25:.4f}")
                running = 0.0
            if global_step % args.save_steps == 0:
                save(global_step)

        # end-of-epoch validation loss
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
