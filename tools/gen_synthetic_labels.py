"""Generate synthetic pet-product label images as STAND-IN test data.

The real dataset (54 images) lives on Canvas (auth-required) and cannot be fetched
automatically. These synthetic labels let us validate the full OCR + thumbnail
pipeline right now. When the real images are available, clear data/raw/ and drop them
in — the pipeline commands are identical. (Also a useful seed for Phase-2 SynthDoG.)

Files are written as SYNTH_label_*.png so they're obviously placeholders.
"""
from __future__ import annotations

import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FONT_DIR = Path(r"C:\Windows\Fonts")

BRAND = "SIT STAY FOREVER"
SUBTITLE = "Natural Dry Shampoo for Pets"
CLAIMS = [
    "VET APPROVED", "FRAGRANCE-FREE", "HYPOALLERGENIC", "WATERLESS FORMULA",
    "ALL NATURAL", "PARABEN-FREE", "MADE IN USA", "FOR DOGS & CATS",
]
INGREDIENTS = ("Ingredients: Purified Water, Aloe Vera, Oatmeal Extract, "
               "Coconut-derived Cleansers, Vitamin E, Chamomile.")


def font(name: str, size: int):
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except Exception:
        return ImageFont.load_default()


def make_label(idx: int, claims: list[str]) -> Path:
    W, H = 700, 900
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([120, 60, W - 120, H - 60], radius=40,
                        outline=(60, 120, 90), width=6, fill=(245, 250, 246))
    d.text((W // 2, 180), BRAND, font=font("arialbd.ttf", 46), fill=(30, 80, 55), anchor="mm")
    d.text((W // 2, 240), SUBTITLE, font=font("arial.ttf", 26), fill=(80, 80, 80), anchor="mm")
    d.text((W // 2, 360), claims[0], font=font("arialbd.ttf", 54), fill=(20, 20, 20), anchor="mm")
    y = 470
    for c in claims[1:]:
        d.text((W // 2, y), c, font=font("arial.ttf", 34), fill=(50, 50, 50), anchor="mm")
        y += 60
    d.text((W // 2, y + 30), "NET WT 8 OZ (227 g)", font=font("arial.ttf", 28),
           fill=(40, 40, 40), anchor="mm")
    # small ingredient text -> stresses thumbnail readability
    wrapped = "\n".join(textwrap.wrap(INGREDIENTS, 58))
    d.multiline_text((150, H - 190), wrapped, font=font("arial.ttf", 16),
                     fill=(70, 70, 70), spacing=6)
    out = RAW / f"SYNTH_label_{idx:02d}.png"
    img.save(out)
    return out


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    random.seed(7)
    n = 6
    for i in range(n):
        k = random.randint(2, 3)
        extra = random.sample([c for c in CLAIMS if c != CLAIMS[i % len(CLAIMS)]], k)
        claims = [CLAIMS[i % len(CLAIMS)]] + extra
        p = make_label(i, claims)
        print("wrote", p.name, "| claims:", claims)
    print(f"\n{n} synthetic labels -> {RAW}")


if __name__ == "__main__":
    main()
