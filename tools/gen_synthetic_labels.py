"""Generate synthetic pet-product label images as STAND-IN test data.

The real dataset (54 images) lives on Canvas (auth-required) and cannot be fetched
automatically. These synthetic labels let us validate the full OCR + thumbnail
pipeline right now, before the real dataset arrives. When the real images are
available, clear data/raw/ and drop them in — the pipeline commands are identical.
Quality is deliberately throwaway (simple shapes/fonts, no photo realism): the goal
is only to exercise the pipeline end to end, not to produce a realistic benchmark.
Claim text is rendered at several sizes, including small ingredient-list print, to
stress-test thumbnail readability the same way real packaging text would. (Later
also served as the seed/prototype for the Phase-2 Donut fine-tune data generator —
see src/stage2_donut/make_finetune_data.py.)

Files are written as SYNTH_label_*.png so they're obviously placeholders and can
never be mistaken for real sponsor data.

中文：生成合成宠物产品标签图片，作为真实数据集到位前的占位测试数据。真实数据集
（54 张图）放在 Canvas 上（需要登录认证），无法自动下载，所以先用这些合成标签
把 OCR + 缩略图整条管线跑通验证。等真实图片到手后，只需清空 data/raw/ 目录并
放入真实图片，管线命令完全不用改。合成图片画质刻意做得很简陋（简单形状/字体，
不追求照片级真实感）——目的只是把管线跑通，不是做真实的评测基准。Claim 文字
用多种字号渲染，包括很小的配料表印刷字，用于模拟真实包装上最容易在缩略图尺寸
下读不清的那种小字。文件名统一加 SYNTH_ 前缀，保证一看就知道是占位数据、
不会与真实赞助商数据混淆。这批生成代码后来也被用作 Phase-2 Donut 微调数据
生成器的原型/种子（见 src/stage2_donut/make_finetune_data.py）。
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
    """Load a Windows system TrueType font by filename; fall back to PIL's default
    bitmap font if it isn't found, so this script stays runnable on any machine.

    按文件名加载 Windows 系统字体；找不到时退回 PIL 自带的默认字体，
    保证脚本在没有对应字体文件的机器上也能跑通。
    """
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except Exception:
        return ImageFont.load_default()


def make_label(idx: int, claims: list[str]) -> Path:
    """Render one synthetic label PNG: brand, subtitle, a headline claim, secondary
    claims, net weight, and small ingredient-list text.

    渲染一张合成标签图片：品牌名、副标题、主标题 claim、若干次要 claim、
    净重信息，以及小字配料表文本。
    """
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
    # 小字配料表，专门用来考验缩略图可读性
    wrapped = "\n".join(textwrap.wrap(INGREDIENTS, 58))
    d.multiline_text((150, H - 190), wrapped, font=font("arial.ttf", 16),
                     fill=(70, 70, 70), spacing=6)
    out = RAW / f"SYNTH_label_{idx:02d}.png"
    img.save(out)
    return out


def main():
    """Generate a small batch (n=6) of synthetic labels with randomized claim
    combinations and write them to data/raw/.

    生成一小批（n=6）合成标签，每张随机搭配不同的 claim 组合，
    写入 data/raw/ 目录。
    """
    RAW.mkdir(parents=True, exist_ok=True)
    # fixed seed -> reproducible synthetic batch across reruns
    # 固定随机种子，保证每次重跑生成的批次一致，便于复现调试
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
