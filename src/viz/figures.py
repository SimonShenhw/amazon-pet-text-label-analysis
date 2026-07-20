"""Presentation figures from pipeline outputs -> outputs/figures/*.png.

Reads only CSV/XLSX artifacts (no OCR dependency), so this module runs on host
python without needing the PaddleOCR venv or container:
  py -3.14 -m src.viz.figures

Style follows a validated, CVD-checked accessible palette with fixed categorical
slots — sponsor/high/medium always map to the same 3 colors, in the same order,
in every figure in this file ("color follows entity") — plus thin marks, direct
data labels instead of legends where possible, a single axis, and a recessive
hairline grid, so the 4 figures read as one consistent system rather than four
one-off charts. Label DODGING in fig1 (see the vertical-separation loop) exists
because two tiers' endpoint values land almost exactly on top of each other
(~35%) and would otherwise overlap illegibly. The Agg backend is set before
importing pyplot so the module renders headlessly (no display needed).

中文：本模块只读取管线产出的 CSV/XLSX 结果文件（不依赖 OCR），因此可以直接在
宿主机 Python 环境跑，不需要 PaddleOCR 的 venv 或容器。配色采用经过色觉障碍
(CVD)校验的无障碍配色方案，且 sponsor/high/medium 三个层级在本文件全部图里都
固定映射到同一组颜色、同一顺序（"颜色跟随实体"），保证多张图放在一起时风格
统一；线条细、尽量用直接数据标注代替图例、单轴、网格线用浅色细线降低视觉
权重，使 4 张图读起来像一套系统而不是各自为战。fig1 里的标签"避让"(dodge)
逻辑（见纵向错位的循环）是因为 sponsor 和 medium 两条线在末端的取值几乎重合
（都在 35% 附近），直接标注会互相遮挡，所以做了错位处理。Agg 后端在导入
pyplot 之前设置，保证脚本可以在无显示环境下渲染出图。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config

# --- validated reference palette (light mode) --- 已校验的无障碍配色方案（浅色模式）
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
# fixed slots 1-3, same tier -> color everywhere in this file ("color follows entity")
# 固定映射（同一层级永远同一颜色），全部图统一配色
CAT = {"sponsor": "#2a78d6", "high": "#1baf7a", "medium": "#eda100"}
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
SEQ_LIGHT, SEQ_DARK = "#86b6ef", "#0d366b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "Segoe UI", "text.color": INK,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": False, "figure.dpi": 150,
})


def _clean(ax, xgrid=False, ygrid=False):
    """Strip top/right spines and optionally add a recessive hairline grid on one axis.

    Shared helper so every figure gets the same "clean" look from one call site.

    去掉上/右边框，按需在指定轴上加一条浅色细网格线。所有图共用这一个辅助
    函数，保证视觉风格统一。
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def load_joined():
    """Join Stage 3 readability with sponsor performance_tier via the manifest
    filename — the shared data prep for fig1/fig2/fig4 (fig3 reads keyword hits
    separately).

    通过 manifest 文件名把 Stage3 可读性数据与赞助商的 performance_tier
    拼接起来；fig1/fig2/fig4 共用这份数据准备（fig3 单独读取关键词命中表）。
    """
    s3 = pd.read_csv(config.STAGE3_CSV)
    mf = pd.read_csv(config.MANIFEST_CSV)
    meta = pd.read_excel(config.PET_CV_XLSX, sheet_name="images_cv_features")
    return (s3.merge(mf[["image_id", "original_name"]], on="image_id")
              .merge(meta[["image_filename", "performance_tier"]],
                     left_on="original_name", right_on="image_filename", how="left"))


def fig1_cliff(df):
    """Slope chart: text retention collapses from full-res -> 320px -> 160px.

    折线图：文字保留率从全分辨率 -> 320px -> 160px 逐级崩塌。
    """
    tiers = ["sponsor", "high", "medium"]
    xs = [0, 1, 2]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ends = []
    for tier in tiers:
        sub = df[df.performance_tier == tier]
        ys = [1.0, sub["search_grid_char_retention"].mean(),
              sub["mobile_thumb_char_retention"].mean()]
        label = {"sponsor": "Sponsor (SSF)", "high": "High tier", "medium": "Medium tier"}[tier]
        ax.plot(xs, ys, color=CAT[tier], linewidth=2, marker="o", markersize=6, zorder=3)
        ends.append((tier, label, ys[2]))
    # dodge endpoint labels: keep >= 0.05 vertical separation — sponsor and medium
    # tiers land at nearly the same value (~35%) and would overlap without this.
    # 标签避让：保持>=0.05纵向间距——sponsor 和 medium 两条线末端取值接近(~35%)，
    # 不错位会重叠。
    ends.sort(key=lambda t: -t[2])
    label_y = []
    for _, _, v in ends:
        y = v
        while any(abs(y - prev) < 0.05 for prev in label_y):
            y -= 0.05
        label_y.append(y)
    for (tier, label, v), y in zip(ends, label_y):
        ax.annotate(f"{label}  {v:.0%}", (2, v), xytext=(10, (y - v) * 400),
                    textcoords="offset points", va="center", fontsize=9.5, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Full resolution", "Search grid (320 px)", "Mobile thumbnail (160 px)"])
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("OCR character retention")
    ax.set_xlim(-0.15, 2.9)
    _clean(ax, ygrid=True)
    ax.set_title("On-package text collapses at Amazon's mobile thumbnail size",
                 fontsize=12, fontweight="bold", loc="left", color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig1_thumbnail_cliff.png", bbox_inches="tight")
    plt.close(fig)


def fig2_distribution(df):
    """Per-image mobile retention, dot strip by tier — the distribution, honestly.

    按 tier 展示每张图片的移动端保留率散点条——如实呈现分布而非只给均值，
    避免掩盖组内差异。
    """
    tiers = ["sponsor", "high", "medium"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    rng = pd.Series(range(len(df)))
    for i, tier in enumerate(tiers):
        sub = df[df.performance_tier == tier]
        # slight de-overlap: jitter y so overlapping dots stay visually distinguishable
        # 轻微错位，避免重叠的散点相互遮挡
        y = [i + (j % 5 - 2) * 0.07 for j in range(len(sub))]
        ax.scatter(sub["mobile_thumb_char_retention"], y, s=42, color=CAT[tier],
                   edgecolors=SURFACE, linewidths=1.2, zorder=3)
        m = sub["mobile_thumb_char_retention"].mean()
        ax.plot([m, m], [i - 0.28, i + 0.28], color=INK, linewidth=1.4, zorder=4)
        ax.annotate(f"mean {m:.0%}", (m, i + 0.34), ha="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(tiers)))
    ax.set_yticklabels(["Sponsor (SSF)", "High tier", "Medium tier"])
    ax.set_xlim(-0.03, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Mobile-thumbnail character retention (per image)")
    ax.invert_yaxis()
    _clean(ax, xgrid=True)
    ax.set_title("Most images keep under half of their text on mobile",
                 fontsize=12, fontweight="bold", loc="left", color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig2_retention_distribution.png", bbox_inches="tight")
    plt.close(fig)


def fig3_keyword_survival():
    """SSF on-pack search keywords: which survive the 160px thumbnail.

    展示 SSF 印在包装上的搜索关键词里，哪些能在 160px 缩略图尺寸下存活——
    这是本项目关键词分析部分最核心的一张图。
    """
    hits = pd.read_csv(config.CSV_DIR / "stage4_keyword_hits.csv")
    ssf = (hits[hits.performance_tier == "sponsor"]
           .groupby("keyword")["survives_mobile_thumb"].any().sort_values())
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(ssf) + 1.6))
    for i, (kw, ok) in enumerate(ssf.items()):
        color = GOOD if ok else CRITICAL
        mark = "survives" if ok else "LOST at 160 px"
        ax.plot([0, 1], [i, i], color=GRID, linewidth=0.8, zorder=1)
        ax.scatter([1], [i], s=90, color=color, edgecolors=SURFACE, linewidths=1.2, zorder=3)
        ax.annotate(mark, (1, i), xytext=(12, 0), textcoords="offset points",
                    va="center", fontsize=9, color=color, fontweight="bold")
    ax.set_yticks(range(len(ssf)))
    ax.set_yticklabels(ssf.index, fontsize=9.5)
    ax.set_xticks([])
    ax.set_xlim(-0.02, 1.65)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("SSF search keywords printed on-pack: 1 of 8 survives the mobile thumbnail",
                 fontsize=11.5, fontweight="bold", loc="left", color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig3_keyword_survival.png", bbox_inches="tight")
    plt.close(fig)


def fig4_robustness(df):
    """Photo-condition robustness vs the resolution effect — one hue, magnitude.

    用同一色系对比"拍摄条件扰动"（亮度/旋转/噪声模糊）与"分辨率下降"两类
    因素对 OCR 文字保留率的影响幅度，强调这是量级对比而非分类对比——
    结论是分辨率才是决定性因素。
    """
    rob = pd.read_csv(config.CSV_DIR / "augment_robustness.csv")
    per = (rob[rob["transform"] != "original"]
           .groupby("transform")["char_retention"].mean())
    labels = {
        "bright_low": "Brightness −35%", "bright_high_contrast": "Brightness +35% / contrast",
        "rot_plus6": "Rotation +6°", "rot_minus6": "Rotation −6°",
        "noise_blur": "Sensor noise + blur",
    }
    thumb = df["mobile_thumb_char_retention"].mean()
    rows = sorted(per.items(), key=lambda kv: -kv[1])
    names = [labels.get(k, k) for k, _ in rows] + ["Mobile thumbnail (160 px)"]
    vals = [v for _, v in rows] + [thumb]
    colors = [SEQ_LIGHT] * len(rows) + [SEQ_DARK]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ys = range(len(vals))
    ax.barh(ys, vals, height=0.62, color=colors, zorder=3)
    for y, v in zip(ys, vals):
        ax.annotate(f"{v:.0%}", (v, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Mean OCR character retention vs. original")
    _clean(ax, xgrid=True)
    ax.set_title("Resolution — not photo conditions — is what kills text legibility",
                 fontsize=12, fontweight="bold", loc="left", color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig4_robustness_vs_resolution.png", bbox_inches="tight")
    plt.close(fig)


def main():
    """Generate all 4 presentation figures into outputs/figures/.

    生成全部 4 张展示用图表，输出到 outputs/figures/ 目录。
    """
    config.ensure_dirs()
    df = load_joined()
    fig1_cliff(df)
    fig2_distribution(df)
    fig3_keyword_survival()
    fig4_robustness(df)
    print("Wrote 4 figures ->", config.FIG_DIR)


if __name__ == "__main__":
    main()
