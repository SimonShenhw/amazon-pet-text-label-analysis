# Amazon Pet-Product Text & Label Analysis

A three-stage computer-vision pipeline that extracts **on-package text and label claims**
from Amazon pet-grooming product images, measures whether those claims stay **legible at
mobile thumbnail resolution**, and links the results to product performance and search
keywords — turning visual content into concrete listing recommendations.

Built for a Northeastern University industry-sponsored project with **Sit Stay Forever (SSF)**,
a natural pet-products brand. This repo is the **Text & Label Analysis** workstream
(sibling teams cover Image Quality, Color, and Semantic & Attention).

---

## Why this matters

For a small e-commerce brand, the Amazon product image is the primary purchase-decision
surface. On-image text drives both **discoverability** (image–text alignment with search
queries) and **trust** (legibility of claims like *“vet-approved”*, *“fragrance-free”*).
A claim that is sharp on the full-resolution image but unreadable at the ~160 px thumbnail
Amazon serves on mobile search is, in effect, invisible to most shoppers.

---

## Pipeline

| Stage | What it does | Runtime | Output |
|---|---|---|---|
| **0. Ingest** | normalize raw images → consistent RGB PNGs + `manifest.csv` | host / venv | `data/processed/` |
| **1. PaddleOCR** | DB-Net detection + SVTR/CRNN recognition → text + confidence per region | **uv venv (CPU)** | `outputs/csv/stage1_ocr.csv` |
| **2. Donut** | OCR-free document understanding → structured JSON | **host GPU** | `outputs/json/stage2_claims.json` |
| **3. Thumbnail readability** | downsample to 160/320 px, re-run OCR, measure Δ char-count / Δ confidence | **uv venv (CPU)** | `outputs/csv/stage3_readability.csv` |
| **4. Business linkage** | claim/readability ↔ performance tier; cross-check vs sponsor CV dataset; keyword mapping | host | `outputs/csv/stage4_linkage_summary.csv` |

Stages communicate **only through files** in `data/` and `outputs/` — there is no shared
process, so the PaddleOCR environment (Python 3.11) and the Donut/analysis environment
(host Python 3.14) stay fully decoupled.

---

## Project structure

```
.
├── README.md
├── docker-compose.yml            # optional Docker path for Stage 1/3 (see note below)
├── docker/
│   ├── Dockerfile.paddle         # PaddleOCR CPU image
│   └── Dockerfile.donut          # optional GPU image for Donut
├── requirements/
│   ├── paddle-cpu.txt            # Stage 1/3 deps (paddlepaddle 2.6.2 + paddleocr 2.7.3)
│   └── donut-host.txt            # Stage 2/4 deps (host; torch already installed)
├── docs/
│   └── XN_Project_Group3_Initial_Plan.docx
├── tools/
│   └── gen_synthetic_labels.py   # synthetic label generator (testing + SynthDoG seed)
├── notebooks/                    # optional notebook wrappers around src/
├── src/
│   ├── config.py                 # all paths & constants (thumbnail sizes, file names)
│   ├── ingest/
│   │   └── ingest.py             # raw images → processed/ + manifest.csv
│   ├── stage1_paddle/
│   │   ├── ocr_engine.py         # shared PaddleOCR engine + helpers
│   │   └── run_ocr.py            # batch OCR → stage1_ocr.csv (+ per-image aggregate)
│   ├── stage2_donut/
│   │   └── run_donut.py          # Donut zero-shot → stage2_claims.json (host GPU)
│   ├── stage3_thumbnail/
│   │   └── readability.py        # thumbnail re-OCR deltas → stage3_readability.csv
│   ├── stage4_linkage/
│   │   └── linkage.py            # join with sponsor dataset; tier analysis
│   └── eval/
│       └── metrics.py            # CER / WER / character-level F1
├── data/        # raw / processed / synthetic / annotations   (NOT tracked — see Data)
└── outputs/     # csv / json / figures                        (NOT tracked)
```

---

## Setup & usage

The pipeline uses **two isolated Python environments** because PaddleOCR (PaddlePaddle)
and Donut (PyTorch) have conflicting dependency trees and different Python/CUDA needs.

### Stage 1 & 3 — PaddleOCR in a uv-managed Python 3.11 venv (CPU)

```powershell
# one-time setup (uv keeps Python 3.11 in its own cache; does not touch system Python)
winget install astral-sh.uv
uv venv .venv-paddle --python 3.11
uv pip install --python .venv-paddle\Scripts\python.exe -r requirements/paddle-cpu.txt

# run from the repo root
.venv-paddle\Scripts\python.exe -m src.ingest.ingest
.venv-paddle\Scripts\python.exe -m src.stage1_paddle.run_ocr
.venv-paddle\Scripts\python.exe -m src.stage3_thumbnail.readability
```

The full 54-image batch runs on CPU in ~1–2 min; PP-OCR models download automatically on
first run. GPU is unnecessary at this scale.

### Stage 2 & 4 — host Python (Donut on GPU, analysis on CPU)

```powershell
py -3.14 -m pip install -r requirements/donut-host.txt   # most deps already present; do NOT reinstall torch
py -3.14 -m src.stage2_donut.run_donut       # Donut zero-shot (CUDA)
py -3.14 -m src.stage4_linkage.linkage       # join + tier analysis
```

> **Why not Docker?** The original plan ran Stage 1/3 in a PaddleOCR CPU container.
> Docker Desktop 4.66.1 on the dev machine crashes at startup — its AI *Inference manager*
> fails to create/remove a unix socket under `AppData\Local\Docker\run` (`ERROR_INVALID_NAME`,
> triggered by a space in the Windows username) and could not be disabled via settings.
> We pivoted to the uv venv, which is the supported path here. `docker-compose.yml` and
> `docker/` are kept for anyone with a working Docker.

---

## Data

The dataset (54 Amazon product images, `pet_cv_dataset_full.xlsx`, `SSF_CV_Dataset.xlsx`)
is the **sponsor’s / course’s proprietary data and is intentionally NOT included** in this
public repository (it contains real product imagery and competitor analysis). `data/` and
`outputs/` are git-ignored.

To run end-to-end, place the files into `data/raw/`:

```
data/raw/
├── <ASIN>_imgNN.jpg            # 54 product images
├── pet_cv_dataset_full.xlsx    # per-image CV features + performance tiers (join key: image_filename)
└── SSF_CV_Dataset.xlsx         # sponsor listings + 146-keyword dataset
```

`src/ingest/ingest.py` renames images to `product_NNN.png` while preserving the original
ASIN-based filename in `manifest.csv`, which Stage 4 uses to join back to the CV dataset.
No data? `tools/gen_synthetic_labels.py` generates synthetic product labels to exercise the
full pipeline.

---

## Example results (54-image sample)

- **Stage 1** produces clean OCR — e.g. a sponsor image yields `DRY SHAMPOO`, `DOG & CAT`,
  `MADE IN MAINE`, `made with natural, safe, organic … ingredients` at ~0.95–0.99 confidence.
- **Stage 3** is the headline metric: at the 160 px mobile thumbnail, only **~34–36 %** of
  the characters detected at full resolution are still recognized — i.e. **roughly two-thirds
  of on-pack text becomes illegible at mobile-search size.**
- **Stage 4** joins all 54 images to the CV dataset (100 % match). Our PaddleOCR character
  counts correlate **r ≈ 0.71** with the dataset’s independently computed `ocr_word_count`,
  validating the extraction.

> Actionable takeaway for the sponsor: the most claim-dense images lose the most text at
> thumbnail size — key claims should be enlarged / repositioned for mobile legibility.

---

## Evaluation

`src/eval/metrics.py` provides pure-Python **CER**, **WER**, and **character-level F1** for
scoring OCR against held-out annotations and Donut field values against ground truth.

---

## Roadmap

- **Phase 1** — repo + environments, ingest, baseline PaddleOCR, annotation schema, eval harness. ✅
- **Phase 2** — Donut zero-shot ✅ → SynthDoG synthetic data + LoRA fine-tune for a
  `{brand, claim, ingredient}` schema; full thumbnail scoring. ✅ (zero-shot) / ⏳ (fine-tune)
- **Phase 3** — Stage 4 analysis, sponsor recommendations, final presentation.

---

## Methods & references

PP-OCRv3 ([arXiv:2206.03001](https://arxiv.org/abs/2206.03001)) ·
DB-Net ([arXiv:1911.08947](https://arxiv.org/abs/1911.08947)) ·
SVTR ([arXiv:2205.00159](https://arxiv.org/abs/2205.00159)) ·
Donut / SynthDoG ([arXiv:2111.15664](https://arxiv.org/abs/2111.15664)) ·
LoRA ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)) ·
ASTER (TPAMI 2019) ·
SQID ([arXiv:2405.15190](https://arxiv.org/abs/2405.15190)) ·
ABO ([arXiv:2110.06199](https://arxiv.org/abs/2110.06199)).
Consumer-behavior framing: Cue Utilization Theory + Stimulus–Organism–Response (S-O-R).

## Team

Fengbo Lyu · Haowei Shen · Yuang Dai — Northeastern University.

- **Fengbo** — Stage 1 (PaddleOCR)
- **Haowei** — Stage 2 (Donut) + Stage 3 (thumbnail readability) + integration
- **Yuang** — claim verification, Stage 4 sponsor analysis, final report
