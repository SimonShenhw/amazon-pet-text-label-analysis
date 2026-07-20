"""Central paths and constants for the XN Text & Label pipeline.

Single source of truth for every path/constant used across the pipeline, so the exact
same code runs unmodified wherever it executes: the host (Stage 2/4, Python 3.14 GPU),
the Paddle uv venv (Stage 1/3, Python 3.11 CPU), or a container, wherever the repo is
mounted. Centralizing these avoids the classic bug where one stage's hard-coded path
silently diverges from another's.

Everything is anchored to the repo root so the same code runs identically on the
host (Stage 2/4) and inside the Paddle container (Stage 1/3), where the repo is
mounted at /app.

中文：本模块是全流程路径与常量的唯一权威来源 (single source of truth)。Stage1/3 的
PaddleOCR 运行在隔离的 uv venv（Python 3.11，CPU）中，Stage2/4 跑在宿主机（Python
3.14，GPU）——两个环境彼此隔离，仅通过文件（CSV/JSON/图片）通信，以此实现解耦与可
复现。把路径和常量集中到一处，使同一份代码不需修改即可在宿主机、venv 或容器中得到
完全一致的结果。
"""
from __future__ import annotations

from pathlib import Path

# Repo root = parent of src/ — 仓库根目录（src/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- data --- 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
MANIFEST_CSV = PROCESSED_DIR / "manifest.csv"

# --- outputs --- 输出目录：各 stage 产出的 CSV/JSON/图表，按类型分子目录存放
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CSV_DIR = OUTPUTS_DIR / "csv"
JSON_DIR = OUTPUTS_DIR / "json"
FIG_DIR = OUTPUTS_DIR / "figures"

# one row per detected text region — 每个检测到的文本区域一行
STAGE1_CSV = CSV_DIR / "stage1_ocr.csv"
STAGE1_AGG_CSV = CSV_DIR / "stage1_ocr_per_image.csv"
# structured claim extraction — 结构化的宣传语/卖点提取结果
STAGE2_JSON = JSON_DIR / "stage2_claims.json"
# thumbnail readability deltas — 缩略图可读性衰减指标
STAGE3_CSV = CSV_DIR / "stage3_readability.csv"
STAGE4_SUMMARY_CSV = CSV_DIR / "stage4_linkage_summary.csv"

# Sponsor dataset (drop into data/raw/) — 赞助方提供的数据集（手动放入 data/raw/）
PET_CV_XLSX = RAW_DIR / "pet_cv_dataset_full.xlsx"
SSF_CV_XLSX = RAW_DIR / "SSF_CV_Dataset.xlsx"

# --- constants --- 常量
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
OCR_LANG = "en"

# Resolutions Amazon actually serves (long-edge px, aspect ratio preserved) — these are
# the business-critical sizes: 160px mobile search thumbnail, 320px search grid. They
# decide whether on-pack text/claims are legible to a real shopper.
# Amazon 实际投放的分辨率（长边像素，保持宽高比）——这是业务关键分辨率：160px 对应
# 移动端搜索缩略图，320px 对应搜索结果网格图，直接决定包装文字能否被真实买家看清。
THUMBNAIL_SIZES = {
    "mobile_thumb": 160,
    "search_grid": 320,
}

# Cap the long edge of processed full-res images (keep aspect ratio). None = no cap.
# 1600px caps huge input images for OCR speed, at negligible accuracy cost — OCR
# detail saturates well below this resolution, so this trade buys speed, not accuracy.
# 限制处理后全分辨率图像的长边（保持宽高比），None 表示不设上限；1600px 用于控制
# OCR 速度，对精度影响可忽略（原图细节远早于此分辨率就已饱和）。
MAX_LONG_EDGE = 1600


def ensure_dirs() -> None:
    """Create output dirs if missing (idempotent).

    创建缺失的输出目录（幂等操作，重复调用不会报错或产生副作用）。
    """
    for d in (PROCESSED_DIR, CSV_DIR, JSON_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
