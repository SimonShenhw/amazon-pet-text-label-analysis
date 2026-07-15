"""Central paths and constants for the XN Text & Label pipeline.

Everything is anchored to the repo root so the same code runs identically on the
host (Stage 2/4) and inside the Paddle container (Stage 1/3), where the repo is
mounted at /app.
"""
from __future__ import annotations

from pathlib import Path

# Repo root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- data ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
MANIFEST_CSV = PROCESSED_DIR / "manifest.csv"

# --- outputs ---
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CSV_DIR = OUTPUTS_DIR / "csv"
JSON_DIR = OUTPUTS_DIR / "json"
FIG_DIR = OUTPUTS_DIR / "figures"

STAGE1_CSV = CSV_DIR / "stage1_ocr.csv"          # one row per detected text region
STAGE1_AGG_CSV = CSV_DIR / "stage1_ocr_per_image.csv"
STAGE2_JSON = JSON_DIR / "stage2_claims.json"     # structured claim extraction
STAGE3_CSV = CSV_DIR / "stage3_readability.csv"   # thumbnail readability deltas
STAGE4_SUMMARY_CSV = CSV_DIR / "stage4_linkage_summary.csv"

# Sponsor dataset (drop into data/raw/)
PET_CV_XLSX = RAW_DIR / "pet_cv_dataset_full.xlsx"
SSF_CV_XLSX = RAW_DIR / "SSF_CV_Dataset.xlsx"

# --- constants ---
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
OCR_LANG = "en"

# Resolutions Amazon serves (long-edge px, aspect ratio preserved).
# Mobile search thumbnail is the business-critical one.
THUMBNAIL_SIZES = {
    "mobile_thumb": 160,
    "search_grid": 320,
}

# Cap the long edge of processed full-res images (keep aspect ratio). None = no cap.
MAX_LONG_EDGE = 1600


def ensure_dirs() -> None:
    """Create output dirs if missing (idempotent)."""
    for d in (PROCESSED_DIR, CSV_DIR, JSON_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
