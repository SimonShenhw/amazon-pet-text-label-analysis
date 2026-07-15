"""Presentation figures from pipeline outputs -> outputs/figures/*.png.

Reads only CSV/XLSX artifacts (no OCR), so it runs on host python:
  py -3.14 -m src.viz.figures

Style: fixed-order categorical palette, thin marks, direct labels, single axis,
recessive hairline grid (validated reference palette).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config

# --- validated reference palette (light mode) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
CAT = {"sponsor": "#2a78d6", "high": "#1baf7a", "medium": "#eda100"}  # slots 1-3
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
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def load_joined():
    s3 = pd.read_csv(config.STAGE3_CSV)
    mf = pd.read_csv(config.MANIFEST_CSV)
    meta = pd.read_excel(config.PET_CV_XLSX, sheet_name="images_cv_features")
    return (s3.merge(mf[["image_id", "original_name"]], on="image_id")
              .merge(meta[["image_filename", "performance_tier"]],
                     left_on="original_name", right_on="image_filename", how="left"))


def fig1_cliff(df):
    """Slope chart: text retention collapses from full-res -> 320px -> 160px."""
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
    # dodge endpoint labels: keep >= 0.05 vertical separation
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
    """Per-image mobile retention, dot strip by tier — the distribution, honestly."""
    tiers = ["sponsor", "high", "medium"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    rng = pd.Series(range(len(df)))
    for i, tier in enumerate(tiers):
        sub = df[df.performance_tier == tier]
        y = [i + (j % 5 - 2) * 0.07 for j in range(len(sub))]  # slight de-overlap
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
    """SSF on-pack search keywords: which survive the 160px thumbnail."""
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
    """Photo-condition robustness vs the resolution effect — one hue, magnitude."""
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
    config.ensure_dirs()
    df = load_joined()
    fig1_cliff(df)
    fig2_distribution(df)
    fig3_keyword_survival()
    fig4_robustness(df)
    print("Wrote 4 figures ->", config.FIG_DIR)


if __name__ == "__main__":
    main()
