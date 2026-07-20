"""Generate a SynthDoG-style synthetic training set for Donut fine-tuning (Phase 2).

Renders synthetic pet-product labels with KNOWN structure, so each image carries an
exact ground-truth JSON — no manual annotation needed. Because we draw the text with
PIL ourselves, the ground truth comes for free as a byproduct of rendering, instead of
costing manual labeling effort on top of the already-scarce 54 real images. Combined
with the text-safe augmentations from src/augment, ~120 base labels x (1 + 3 transforms)
≈ 480 samples — the augmented variants reuse (rather than duplicate) the base label's
ground truth, since the transforms are text-safe and never change what the text says.
A fixed RNG seed (default 7) makes the generated set reproducible across runs.

Output (Donut convention):
  data/synthetic/donut_train/
    train/*.jpg + metadata.jsonl     (90%)
    validation/*.jpg + metadata.jsonl (10%)
  each JSONL line: {"file_name": ..., "ground_truth": "{\"gt_parse\": {...}}"}

Target schema (replaces the receipt schema of the pretrained CORD checkpoint):
  {"brand": str, "product_type": str, "claims": [str, ...], "net_weight": str}

Run inside the Paddle venv (needs PIL + cv2 + numpy, no GPU):
  .venv-paddle\\Scripts\\python.exe -m src.stage2_donut.make_finetune_data --n 120

中文说明：本模块合成带已知结构的宠物产品标签图片，渲染时同步生成精确的 JSON
ground truth，完全省去人工标注成本——这对本就只有 54 张真实图片、标注成本敏感的
项目尤其重要。复用 src/augment 里已验证的文本安全增广变换：增广后的图片与原图
共享同一份 ground truth（因为这些变换不改变文字内容），不必为每个变换重新标注。
固定随机种子保证多次运行生成的数据集完全一致，便于复现实验。输出遵循 Donut 官方
JSONL 元数据格式，供 train_lora.py 直接读取。
"""
from __future__ import annotations

import argparse
import json
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src import config
from src.augment.run_augment import TRANSFORMS

OUT_DIR = config.SYNTHETIC_DIR / "donut_train"
FONT_DIR = r"C:\Windows\Fonts"

BRANDS = ["SIT STAY FOREVER", "HAPPY PAWS CO", "PURE PET NATURALS", "CANINE CARE LAB",
          "FURRY FRESH", "WAG & WASH", "GENTLE GROOM", "PAWFECT CLEAN"]
PRODUCT_TYPES = ["DRY SHAMPOO", "DRY POWDER SHAMPOO", "WATERLESS SHAMPOO",
                 "GROOMING POWDER", "DEODORIZING POWDER", "NO-RINSE FOAM"]
CLAIMS = ["VET APPROVED", "FRAGRANCE-FREE", "HYPOALLERGENIC", "WATERLESS FORMULA",
          "ALL NATURAL", "PARABEN-FREE", "MADE IN USA", "FOR DOGS & CATS",
          "TALC FREE", "ORGANIC INGREDIENTS", "SENSITIVE SKIN SAFE", "NO RINSE NEEDED",
          "PLANT BASED", "CRUELTY FREE", "PH BALANCED", "SOOTHES ITCHY SKIN"]
WEIGHTS = ["8 OZ (227 g)", "12 OZ (340 g)", "16 OZ (454 g)", "6 OZ (170 g)", "28 oz./828 ml."]
FONTS = ["arialbd.ttf", "arial.ttf", "georgia.ttf", "verdana.ttf", "seguisb.ttf", "calibri.ttf"]
PALETTES = [((245, 250, 246), (60, 120, 90)), ((250, 246, 240), (140, 90, 40)),
            ((242, 246, 252), (50, 80, 140)), ((252, 244, 246), (150, 60, 90)),
            ((255, 255, 255), (30, 30, 30))]


def font(name, size):
    """Load a Windows TrueType font by filename, falling back to PIL's default.

    按文件名加载 Windows 系统字体；找不到时回退到 PIL 默认字体，保证渲染不中断。
    """
    try:
        return ImageFont.truetype(f"{FONT_DIR}\\{name}", size)
    except OSError:
        return ImageFont.load_default()


def render_label(rng: random.Random):
    """Render one synthetic label; return (PIL image, gt dict).

    Randomly composes brand/product-type/claims/net-weight text onto a card-style
    background so the visual layout varies while the ground truth stays exact.

    随机组合品牌、产品类型、宣传语和净重文本绘制成一张标签图片；文本内容和绘制
    坐标全部由本函数控制，因此返回的 gt 字典就是这张图片精确无误的 ground truth。
    """
    brand = rng.choice(BRANDS)
    ptype = rng.choice(PRODUCT_TYPES)
    claims = rng.sample(CLAIMS, rng.randint(2, 5))
    weight = rng.choice(WEIGHTS)
    bg, accent = rng.choice(PALETTES)
    head_font, body_font = rng.choice(FONTS), rng.choice(FONTS)

    W, H = 640, 840
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([80, 50, W - 80, H - 50], radius=36, fill=bg,
                        outline=accent, width=rng.randint(3, 7))
    d.text((W // 2, 150), brand, font=font(head_font, rng.randint(34, 44)),
           fill=accent, anchor="mm")
    d.text((W // 2, 260), ptype, font=font(head_font, rng.randint(40, 52)),
           fill=(20, 20, 20), anchor="mm")
    y = 360
    for c in claims:
        d.text((W // 2, y), c, font=font(body_font, rng.randint(24, 32)),
               fill=(50, 50, 50), anchor="mm")
        y += rng.randint(52, 66)
    d.text((W // 2, H - 130), f"NET WT {weight}", font=font(body_font, 26),
           fill=(40, 40, 40), anchor="mm")
    gt = {"brand": brand, "product_type": ptype, "claims": claims, "net_weight": weight}
    # this dict IS the ground truth — no separate annotation step needed
    # 中文：这个字典本身就是 ground truth，无需额外的标注步骤
    return img, gt


def main():
    """CLI entry: render --n base labels + their augmented variants, split 90/10,
    and write Donut-format train/validation JSONL manifests.

    命令行入口：渲染 --n 张基础标签及其增广变体，按 90/10 划分训练/验证集，
    并写出符合 Donut 格式约定的 JSONL 标注文件。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="number of base labels")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    aug_names = list(TRANSFORMS)[:3]  # 3 transforms per base label — 每张基础图配3种增广变体
    splits = {"train": [], "validation": []}
    for split in splits:
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        img, gt = render_label(rng)
        split = "validation" if i % 10 == 0 else "train"  # every 10th sample -> val (~10%)
        gt_line = json.dumps({"gt_parse": gt}, ensure_ascii=False)
        base_name = f"label_{i:04d}.jpg"
        img.save(OUT_DIR / split / base_name, quality=92)
        splits[split].append({"file_name": base_name, "ground_truth": gt_line})
        # augmented variants share the same ground truth (text-safe transforms) —
        # 增广变体与原图共用同一份ground truth(变换是文本安全的,不改变文字内容)
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for t in aug_names:
            var_name = f"label_{i:04d}__{t}.jpg"
            cv2.imwrite(str(OUT_DIR / split / var_name),
                        TRANSFORMS[t](arr), [cv2.IMWRITE_JPEG_QUALITY, 92])
            splits[split].append({"file_name": var_name, "ground_truth": gt_line})

    for split, rows in splits.items():
        with open(OUT_DIR / split / "metadata.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} samples")
    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
