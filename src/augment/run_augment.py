"""Data augmentation + OCR robustness evaluation (Stage 1.5).

Two purposes:
1. **Augmented training pool** (instructor requirement: grow 54 images to 200+ samples
   for a proper train/test split). With 5 text-safe transforms we get 54 x (1+5) = 324
   samples. Pass --save to write variants to data/synthetic/augmented/ (+ manifest).
2. **OCR robustness evaluation**: re-run PaddleOCR on every augmented variant and measure
   how much text survives each transform vs the untouched original. This quantifies how
   sensitive on-package text legibility is to realistic photo conditions (lighting,
   camera tilt, sensor noise, focus) — a finding in its own right. This module therefore
   doubles as a controlled experiment: the result (these photo perturbations cost only
   15-22% of characters, vs 69% lost at Stage 3's thumbnail resolution) is evidence that
   image RESOLUTION, not photography conditions, is the dominant driver of legibility
   loss in this pipeline.

Transform choice — deliberately TEXT-SAFE only:
  Horizontal/vertical flips are standard for CNN classification but are *wrong* for
  text pipelines: mirrored text is unreadable, destroys OCR ground truth, and never
  occurs in real product photography. We therefore use rotation, brightness, contrast,
  Gaussian noise and blur, and exclude flips by design.

Determinism: the Gaussian-noise RNG is seeded and every other transform's parameters
are fixed constants (see TRANSFORMS below), so re-running this module reproduces the
same augmented images and the same robustness numbers every time.

--save controls disk usage: by default the module only needs images in memory long
enough to OCR them (purpose 2 above); pass --save when you actually want the 324-image
training pool written to disk (purpose 1), since most runs only care about the
robustness metrics and shouldn't pay the disk cost.

Run inside the Paddle venv:
  .venv-paddle\\Scripts\\python.exe -m src.augment.run_augment [--save]

中文说明：本模块承担两项任务。(1) 数据增广：按老师要求把 54 张真实图片扩充到
200+ 样本，用 5 种文本安全的变换得到 54×(1+5)=324 张；--save 时才把变体落盘到
data/synthetic/augmented/，默认只在内存中跑完 OCR 即丢弃，节省磁盘。(2) OCR 鲁棒性
实验：对每个增广变体重跑 PaddleOCR，量化真实拍摄条件(光照、倾斜、噪点、失焦)对
包装文字可读性的影响——这本身就是一项独立发现。实测结果是：这类拍摄扰动只损失
15-22% 的字符，而 Stage3 缩略图分辨率损失却高达 69%，说明本 pipeline 中文字可读性
的主要瓶颈是"分辨率"而非"拍摄条件"。变换选择上刻意排除左右/上下翻转：翻转对分类
任务是常规操作，但对文字任务是错误的选择——镜像文字不可读、会破坏 OCR ground
truth，且真实产品摄影中从不会出现镜像文字。为保证鲁棒性数值可复现，随机数生成
固定种子、其余变换参数均为常量。
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from src import config
from src.stage1_paddle.ocr_engine import aggregate, get_ocr, run_ocr

AUGMENTED_DIR = config.SYNTHETIC_DIR / "augmented"
ROBUSTNESS_CSV = config.CSV_DIR / "augment_robustness.csv"
AUG_MANIFEST_CSV = config.SYNTHETIC_DIR / "augmented_manifest.csv"


def rotate(image, deg):
    """Rotate image by deg degrees around its center, replicating border pixels.

    绕图像中心旋转 deg 度；边界用像素复制填充，避免出现黑边影响 OCR。
    """
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def brightness(image, factor):
    """Scale pixel intensity by factor to simulate a linear brightness change.

    按 factor 线性缩放像素亮度（不加偏移），模拟拍摄光照变化。
    """
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def contrast(image, alpha, beta):
    """Apply gain alpha and bias beta to simulate a contrast/brightness change.

    用增益 alpha 和偏移 beta 模拟对比度（及轻微亮度）变化。
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def gaussian_noise(image, sigma):
    """Add zero-mean Gaussian noise (std sigma) to simulate camera sensor noise.

    添加均值为 0、标准差为 sigma 的高斯噪声，模拟相机传感器噪点；RNG 固定种子(0)
    以保证结果可复现。
    """
    noise = np.random.default_rng(0).normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def blur(image, k):
    """Apply a k x k Gaussian blur to simulate camera focus loss.

    应用 k×k 高斯模糊，模拟对焦不准/手抖导致的失焦。
    """
    return cv2.GaussianBlur(image, (k, k), 0)


# name -> callable; keep deterministic for reproducibility
# 名称到可调用变换函数的映射；参数均为固定常量，保证多次运行结果一致、可复现
TRANSFORMS = {
    "rot_plus6": lambda im: rotate(im, 6),
    "rot_minus6": lambda im: rotate(im, -6),
    "bright_low": lambda im: brightness(im, 0.65),
    "bright_high_contrast": lambda im: contrast(im, 1.35, 10),
    "noise_blur": lambda im: blur(gaussian_noise(im, 12), 3),
}


def main():
    """CLI entry: OCR the original + every augmented variant of each Stage-1 image,
    write per-transform robustness metrics, and optionally save the augmented pool.

    命令行入口：对每张 Stage1 图片的原图及所有增广变体跑 OCR，写出按变换分组的
    鲁棒性指标 CSV；若指定 --save 则同时把增广图片落盘作为训练样本池。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="also write augmented images to data/synthetic/augmented/")
    args = ap.parse_args()

    config.ensure_dirs()
    if not config.MANIFEST_CSV.exists():
        raise SystemExit(f"manifest not found: {config.MANIFEST_CSV}. Run ingest first.")
    manifest = pd.read_csv(config.MANIFEST_CSV)
    ocr = get_ocr(config.OCR_LANG)
    if args.save:
        AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

    rows, saved = [], []
    for _, rec in manifest.iterrows():
        img_path = config.PROCESSED_DIR / rec["processed_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        base = aggregate(run_ocr(ocr, image))
        rows.append({"image_id": rec["image_id"], "transform": "original",
                     "n_regions": base["n_regions"], "total_chars": base["total_chars"],
                     "mean_score": base["mean_score"], "char_retention": 1.0})
        for name, fn in TRANSFORMS.items():
            variant = fn(image)
            agg = aggregate(run_ocr(ocr, variant))
            rows.append({
                "image_id": rec["image_id"], "transform": name,
                "n_regions": agg["n_regions"], "total_chars": agg["total_chars"],
                "mean_score": agg["mean_score"],
                "char_retention": (round(agg["total_chars"] / base["total_chars"], 4)
                                   if base["total_chars"] else 0.0),
            })
            if args.save:
                out_name = f"{rec['image_id']}__{name}.jpg"
                cv2.imwrite(str(AUGMENTED_DIR / out_name),
                            variant, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved.append({"file": out_name, "source": rec["processed_name"],
                              "transform": name})
        print(f"{rec['processed_name']}: base_chars={base['total_chars']}")

    df = pd.DataFrame(rows)
    df.to_csv(ROBUSTNESS_CSV, index=False)
    print(f"\nWrote {ROBUSTNESS_CSV} ({len(df)} rows, "
          f"{df['image_id'].nunique()} images x {df['transform'].nunique()} conditions)")
    n_samples = df["image_id"].nunique() * df["transform"].nunique()
    print(f"Total sample pool: {n_samples} (instructor target: 200+)")
    if args.save:
        pd.DataFrame(saved).to_csv(AUG_MANIFEST_CSV, index=False)
        print(f"Saved {len(saved)} augmented images -> {AUGMENTED_DIR}")

    print("\n=== mean char retention by transform ===")
    # NB: df["transform"], not df.transform — .transform is a DataFrame method
    print(df[df["transform"] != "original"]
          .groupby("transform")[["char_retention", "mean_score"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
