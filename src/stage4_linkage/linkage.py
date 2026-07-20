"""Stage 4 — light business linkage (HOST).

Joins our pipeline outputs (Stage 1 OCR aggregate, Stage 3 thumbnail readability,
Stage 2 Donut claims) with the sponsor dataset (pet_cv_dataset_full.xlsx ->
images_cv_features) on image filename, and reports how OCR / claim / readability
features relate to the product performance tier.

Descriptive/correlational only, deliberately NOT predictive modeling: n=54 images
would overfit any model, and turning CV features into a performance predictor is a
sibling group's scope, not ours. Every input stage is OPTIONAL — whatever artifacts
already exist get joined — so this script can run mid-pipeline, even before Stage
1/3 outputs exist. The join key is manifest `original_name` <-> sponsor dataset
`image_filename`; this only works because ingest preserved original filenames
end to end. A cross-check at the bottom of main() — corr(our total_chars, sponsor's
independent ocr_word_count) ~ 0.71 — is a sanity check that our OCR extraction
agrees with an independently computed measurement.

中文：本模块做 Stage4 业务关联分析，把我们管线各阶段产出（Stage1 OCR 汇总、
Stage3 缩略图可读性、Stage2 Donut claims）与赞助商数据集（按图片文件名）拼接，
统计 OCR/claim/可读性特征与产品销售层级(performance_tier)的关系。刻意只做
描述性/相关性分析、不做预测建模：n=54 太小容易过拟合，且"用 CV 特征预测销售
表现"属于另一小组的研究范围。每个输入阶段都是可选的——有什么产出就拼什么——
因此本脚本可以在 Stage1/3 尚未跑完时就先跑。拼接键是 manifest 里的
original_name 与赞助商数据里的 image_filename（这也是为何 ingest 阶段要全程
保留原始文件名）。main() 末尾的交叉校验 corr≈0.71，用于验证我们的 OCR 抽取
结果与赞助商独立统计的 word_count 是否一致。

  py -3.14 -m src.stage4_linkage.linkage
"""
from __future__ import annotations

import json
import re

import pandas as pd

from src import config

CV_SHEET = "images_cv_features"
CV_KEEP = [
    "image_filename", "asin", "brand", "performance_tier", "known_rating",
    "known_reviews", "text_density_pct", "ocr_word_count", "white_bg_compliance",
    "background_type", "brightness_mean", "sharpness_laplacian",
]


def load_manifest():
    """Load the ingest manifest (image_id <-> original_name mapping); fail fast if
    it does not exist, since every join below depends on it.

    加载 ingest 阶段生成的 manifest（image_id 与原始文件名的映射）；
    文件缺失时直接报错退出，因为后面所有拼接都依赖它。
    """
    if not config.MANIFEST_CSV.exists():
        raise SystemExit(f"missing {config.MANIFEST_CSV}; run ingest first.")
    return pd.read_csv(config.MANIFEST_CSV)


def load_cv_metadata():
    """Load sponsor performance-tier/metadata columns; return None (not raise) if the
    sponsor file is absent, so main() can skip this optional join gracefully.

    加载赞助商的 performance_tier 等元数据；赞助商文件缺失时返回 None 而不是
    报错，这样 main() 可以跳过这一步可选拼接，不影响其他数据源的分析。
    """
    if not config.PET_CV_XLSX.exists():
        print(f"[warn] {config.PET_CV_XLSX} missing; tier/metadata join skipped.")
        return None
    df = pd.read_excel(config.PET_CV_XLSX, sheet_name=CV_SHEET)
    return df[[c for c in CV_KEEP if c in df.columns]]


def maybe_csv(path):
    """Read a CSV if it exists, else return None.

    This is the building block behind "every stage is optional" below: whatever
    artifact exists gets joined, whatever doesn't is simply skipped.

    如果文件存在就读取，否则返回 None——这是下面"每个阶段均可选"拼接逻辑
    的基础函数：有产出就拼，没有就跳过。
    """
    return pd.read_csv(path) if path.exists() else None


def claims_per_image():
    """Summarize Stage 2 Donut output per processed image into two coarse proxy
    columns: whether the parsed payload contains text-like tokens, and its length.

    There is no compact "claims" field to join directly from raw Donut JSON, so
    these two derived proxies are a good-enough signal for a descriptive join.

    汇总 Stage2 Donut 输出，按处理后的图片名派生两个粗代理指标：payload 里
    是否含有类似文字的 token（正则判断），以及 payload 序列化后的长度。
    Donut 原始 JSON 没有现成的"claims"字段可直接拼接，这两个派生指标对
    描述性分析已经足够。
    """
    if not config.STAGE2_JSON.exists():
        return None
    data = json.loads(config.STAGE2_JSON.read_text(encoding="utf-8"))
    rows = []
    for proc_name, parsed in data.items():
        payload = json.dumps(parsed, ensure_ascii=False)
        has_text = bool(re.search(r"[A-Za-z]{3,}", payload))
        rows.append({
            "processed_name": proc_name,
            "donut_has_text": int(has_text),
            "donut_payload_len": len(payload),
        })
    return pd.DataFrame(rows)


def main():
    """Run all optional joins into one Stage 4 table, write it out, and print
    descriptive summaries (mean-by-tier, extraction cross-check).

    依次执行以上各个可选拼接、写出 Stage4 汇总表，并打印描述性统计
    （按 tier 分组均值、OCR 抽取交叉校验）。
    """
    config.ensure_dirs()
    df = load_manifest().copy()

    cv = load_cv_metadata()
    if cv is not None:
        df = df.merge(cv, how="left", left_on="original_name", right_on="image_filename")
        matched = df["performance_tier"].notna().sum() if "performance_tier" in df else 0
        print(f"matched {matched}/{len(df)} images to pet_cv_dataset_full")

    s1 = maybe_csv(config.STAGE1_AGG_CSV)
    if s1 is not None:
        df = df.merge(s1[["image_id", "n_regions", "total_chars", "mean_score"]],
                      how="left", on="image_id")
        print("joined Stage 1 OCR aggregate")

    s3 = maybe_csv(config.STAGE3_CSV)
    if s3 is not None:
        cols = ["image_id"] + [c for c in s3.columns
                               if ("char_retention" in c) or (c == "full_chars")]
        df = df.merge(s3[cols], how="left", on="image_id")
        print("joined Stage 3 readability")

    cl = claims_per_image()
    if cl is not None:
        df = df.merge(cl, how="left", on="processed_name")
        print("joined Stage 2 Donut claims")

    df.to_csv(config.STAGE4_SUMMARY_CSV, index=False)
    print(f"\nWrote merged table -> {config.STAGE4_SUMMARY_CSV} "
          f"({df.shape[0]} rows x {df.shape[1]} cols)")

    # --- descriptive analyses on whatever columns are present ---
    # 对现有列做描述性统计（不做预测），有什么列就分析什么列
    if "performance_tier" in df.columns and df["performance_tier"].notna().any():
        agg_cols = [c for c in ["total_chars", "mobile_thumb_char_retention",
                                "text_density_pct", "ocr_word_count",
                                "donut_payload_len", "known_rating"]
                    if c in df.columns]
        if agg_cols:
            print("\n=== mean by performance_tier ===")
            print(df.groupby("performance_tier")[agg_cols].mean().round(2).to_string())

    if "total_chars" in df.columns and "ocr_word_count" in df.columns:
        # Cross-check against an independent measurement (sponsor's own OCR word
        # count) — a high correlation validates that our extraction pipeline is
        # measuring the same underlying signal, not a pipeline artifact.
        # 用赞助商独立统计的 word_count 交叉校验：相关系数高，说明我们的抽取
        # 结果可信，不是管线本身产生的假象。
        sub = df[["total_chars", "ocr_word_count"]].dropna()
        if len(sub) > 1:
            print(f"\ncross-check corr(our total_chars, sponsor ocr_word_count) = "
                  f"{sub['total_chars'].corr(sub['ocr_word_count']):.3f}")


if __name__ == "__main__":
    main()
