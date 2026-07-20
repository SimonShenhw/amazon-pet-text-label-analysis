"""Stage 4c — Amazon main-image text compliance screen (promised in the project plan).

Amazon's main-image policy: pure white background, the product only, and no text,
logos, badges, watermarks or graphics that are not part of the product itself.
Text printed on the physical packaging is allowed; overlay text/badges are not.

OCR alone cannot prove whether detected text sits on the package or is an overlay,
so this module produces a *screening report*, not verdicts — every flag means
"manual review needed", not "violation confirmed". This is the honest framing given
what OCR can and cannot see:
  - joins our Stage 1 OCR results with the sponsor dataset's image-type labels and
    white-background compliance column,
  - flags main images whose text volume is anomalous vs. peer main images
    (likely badges/overlays) for manual review.
Text-volume flags use a threshold RELATIVE to the peer median (2x median or >60
chars, whichever is larger) rather than a fixed absolute count, because an
absolute threshold would be arbitrary across the differing product types in this
dataset.

中文：本模块实现项目计划中承诺的"Amazon 主图合规性检查"。Amazon 主图规范要求
纯白背景、画面只有产品本身、不能有产品之外的文字/logo/徽章/水印等叠加图层，
但印在实物包装上的文字是允许的。OCR 本身无法区分"文字印在包装实物上"（合规）
还是"文字是后期叠加的图层/徽章"（违规），所以本模块只产出*筛查报告*而非最终
判定——每条 flag 的含义是"需要人工复核"，不是"确认违规"，这是基于 OCR 能力
边界的诚实表述：
  - 按图片文件名把 Stage1 OCR 结果与赞助商数据集里的图片类型标签、
    白底合规列拼接起来；
  - 对文字量相对同类主图异常偏高的主图打标（可能是徽章/叠加文字），
    留给人工复核。
文字量异常的判定阈值是相对同类主图中位数的（中位数 2 倍或 60 字符，取较大值），
而不是固定绝对值，因为数据集里产品类型差异很大，绝对阈值缺乏统一意义。

Host python:  py -3.14 -m src.stage4_linkage.compliance
"""
from __future__ import annotations

import pandas as pd

from src import config

COMPLIANCE_CSV = config.CSV_DIR / "stage4_compliance_flags.csv"


def main():
    """Screen product_main images for compliance risk and write a manual-review report.

    对 product_main 类型的图片做合规风险筛查，写出供人工复核的报告表。
    """
    config.ensure_dirs()
    s1 = pd.read_csv(config.STAGE1_AGG_CSV)
    mf = pd.read_csv(config.MANIFEST_CSV)
    meta = pd.read_excel(config.PET_CV_XLSX, sheet_name="images_cv_features")

    df = (s1.merge(mf[["image_id", "original_name"]], on="image_id")
            .merge(meta[["image_filename", "asin", "brand", "performance_tier",
                         "image_type_label", "white_bg_pct", "white_bg_compliance"]],
                   left_on="original_name", right_on="image_filename", how="left"))

    main_imgs = df[df["image_type_label"] == "product_main"].copy()
    if main_imgs.empty:
        raise SystemExit("no product_main images found in metadata join")

    # peer median, not an absolute constant: text-heaviness only means something
    # relative to other main images in this same (product-type-varied) dataset.
    # 用同类图片的中位数而非固定常数：文字量是否"异常"只能相对同类产品判断。
    med_chars = main_imgs["total_chars"].median()

    def risk(row):
        flags = []
        wbc = row.get("white_bg_compliance")
        # white_bg_compliance arrives from the sponsor's Excel export as a float
        # 0.0/1.0, so try the numeric form first; fall back to string forms in
        # case a row was exported differently.
        # white_bg_compliance 大多是从赞助商 Excel 读出的 0.0/1.0 浮点数，
        # 所以先按数值判断；个别行若导出成字符串形式则用字符串兜底判断。
        try:
            not_white = float(wbc) == 0.0
        except (TypeError, ValueError):
            not_white = str(wbc).strip().lower() in {"false", "no", "non_compliant"}
        if not_white:
            flags.append("background_not_white")
        # threshold relative to peer median (see module docstring) — an absolute
        # cutoff would be arbitrary across differing product types.
        if row["total_chars"] > max(2 * med_chars, 60):
            # possible overlay text / badges — 可能是叠加文字/徽章，需人工复核
            flags.append("text_heavy_vs_peers")
        elif row["total_chars"] > 0:
            flags.append("text_present_verify_on_pack")
        return ";".join(flags) if flags else "clean"

    main_imgs["compliance_flags"] = main_imgs.apply(risk, axis=1)
    cols = ["image_id", "original_name", "brand", "performance_tier",
            "n_regions", "total_chars", "white_bg_pct", "white_bg_compliance",
            "compliance_flags"]
    out = main_imgs[cols].sort_values("total_chars", ascending=False)
    out.to_csv(COMPLIANCE_CSV, index=False)
    print(f"Wrote {COMPLIANCE_CSV} ({len(out)} main images)")
    print(f"(peer median on-main text volume: {med_chars:.0f} chars)")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
