"""Post-training evaluation + inference for the Donut LoRA adapter (runs on H200).

A) Synthetic held-out validation (48 samples, exact ground truth):
   field-level accuracy -> eval_lora_metrics.json
B) Real product images (54): fine-tuned structured extraction
   -> stage2_claims_finetuned.json

Run from /scratch/$USER/xn-donut-lora:
   python -m src.stage2_donut.infer_lora
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
CKPT_ROOT = ROOT / "models" / "donut-lora-ssf"
VAL_DIR = ROOT / "data" / "synthetic" / "donut_train" / "validation"
REAL_DIR = ROOT / "data" / "processed"
OUT_METRICS = ROOT / "outputs" / "json" / "eval_lora_metrics.json"
OUT_REAL = ROOT / "outputs" / "json" / "stage2_claims_finetuned.json"


def latest_adapter():
    cks = sorted(CKPT_ROOT.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not cks:
        raise SystemExit(f"no checkpoints under {CKPT_ROOT}")
    return cks[-1] / "adapter"


def extract_json(text: str):
    """Parse the first {...} block; tolerate trailing garbage / truncation."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    for candidate in (m.group(0), m.group(0) + "}", m.group(0).rsplit(",", 1)[0] + "}"):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


KEY_NAMES = {"gt_parse", "brand", "product", "product_type", "claims", "net_weight"}


def reconstruct(text: str):
    """Lenient schema-guided recovery for near-JSON output.

    The LoRA model reliably emits the right quoted strings but sometimes uses ':'
    instead of ',' between pairs (single-token syntax slips). Recover fields from
    the ordered sequence of quoted strings instead of failing outright.
    """
    quoted = re.findall(r'"([^"]+)"', text)
    if not quoted:
        return None
    out = {"brand": None, "product_type": None, "claims": [], "net_weight": None}
    consumed = set()
    for i, q in enumerate(quoted):
        key = q.strip().lower()
        if key in ("brand", "product", "product_type", "net_weight"):
            for j in range(i + 1, len(quoted)):
                if quoted[j].strip().lower() not in KEY_NAMES:
                    field = "product_type" if key in ("product", "product_type") else key
                    if out.get(field) in (None, []):
                        out[field] = quoted[j].strip()
                        consumed.add(j)
                    break
    weight_re = re.compile(r"\d+\s*(oz|OZ|g|ml)", re.IGNORECASE)
    for i, q in enumerate(quoted):
        qs = q.strip()
        if qs.lower() in KEY_NAMES or i in consumed:
            continue
        if out["net_weight"] is None and weight_re.search(qs):
            out["net_weight"] = qs
            continue
        # claims: caps-heavy strings that aren't the brand/type/weight values
        if (qs.upper() == qs and any(c.isalpha() for c in qs)
                and qs not in (out["brand"], out["product_type"], out["net_weight"])):
            if qs not in out["claims"]:
                out["claims"].append(qs)
    if not any([out["brand"], out["product_type"], out["claims"], out["net_weight"]]):
        return None
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    base = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    tok = processor.tokenizer

    # mirror the training-time config fix (transformers 5.x strict config attrs);
    # decoder_start_token_id MUST match what training used (tok.bos fallback).
    base.config.pad_token_id = tok.pad_token_id
    if getattr(base.config, "decoder_start_token_id", None) is None:
        base.config.decoder_start_token_id = (
            getattr(base.config.decoder, "decoder_start_token_id", None)
            or tok.bos_token_id
        )

    adapter = latest_adapter()
    model = PeftModel.from_pretrained(base, adapter)
    model.to(device).eval()
    print(f"loaded adapter: {adapter}")

    @torch.no_grad()
    def infer(img: Image.Image) -> str:
        pixel = processor(img.convert("RGB"), return_tensors="pt").pixel_values.to(device)
        out = model.generate(
            pixel_values=pixel, max_length=256, num_beams=1,
            decoder_start_token_id=base.config.decoder_start_token_id,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        )
        text = tok.decode(out[0], skip_special_tokens=True)
        return text

    # ---------- A) synthetic held-out validation ----------
    meta = [json.loads(l) for l in
            (VAL_DIR / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(meta)
    strict_ok = lenient_ok = 0
    hits = {"brand": 0, "product_type": 0, "net_weight": 0}
    claims_p, claims_r = [], []
    for i, row in enumerate(meta):
        gt = json.loads(row["ground_truth"])["gt_parse"]
        raw = infer(Image.open(VAL_DIR / row["file_name"]))
        pred = extract_json(raw)
        if pred is not None:
            strict_ok += 1
            if isinstance(pred.get("gt_parse"), dict):
                pred = pred["gt_parse"]
        else:
            pred = reconstruct(raw)
            if pred is not None:
                lenient_ok += 1
        if pred is None:
            continue
        for f in ("brand", "product_type", "net_weight"):
            if str(pred.get(f, "")).strip().lower() == str(gt[f]).strip().lower():
                hits[f] += 1
        gset = {c.strip().lower() for c in gt["claims"]}
        pcl = pred.get("claims")
        pset = ({str(c).strip().lower() for c in pcl} if isinstance(pcl, list) else set())
        inter = len(gset & pset)
        claims_p.append(inter / len(pset) if pset else 0.0)
        claims_r.append(inter / len(gset) if gset else 0.0)
        if i % 10 == 0:
            print(f"val {i}/{n}")

    recovered = strict_ok + lenient_ok
    p = sum(claims_p) / max(len(claims_p), 1)
    r = sum(claims_r) / max(len(claims_r), 1)
    metrics = {
        "adapter": str(adapter),
        "n_validation": n,
        "strict_json_rate": round(strict_ok / n, 4),
        "recovered_rate": round(recovered / n, 4),   # strict + schema-guided lenient
        "brand_acc": round(hits["brand"] / max(recovered, 1), 4),
        "product_type_acc": round(hits["product_type"] / max(recovered, 1), 4),
        "net_weight_acc": round(hits["net_weight"] / max(recovered, 1), 4),
        "claims_precision": round(p, 4),
        "claims_recall": round(r, 4),
        "claims_f1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("VAL METRICS:", json.dumps(metrics, indent=2))

    # ---------- B) real product images ----------
    results = {}
    reals = sorted(REAL_DIR.glob("product_*.png"))
    for i, pth in enumerate(reals):
        raw = infer(Image.open(pth))
        pred = extract_json(raw)
        mode = "strict"
        if pred is None:
            pred, mode = reconstruct(raw), "lenient"
        if pred is None:
            results[pth.name] = {"_parse": "failed", "_raw": raw[:400]}
        else:
            if isinstance(pred, dict) and isinstance(pred.get("gt_parse"), dict):
                pred = pred["gt_parse"]
            pred["_parse"] = mode
            results[pth.name] = pred
        if i % 10 == 0:
            print(f"real {i}/{len(reals)}")
    OUT_REAL.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"wrote {OUT_REAL} ({len(results)} images)")
    print("DONE_INFer_OK")


if __name__ == "__main__":
    main()
