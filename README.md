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
| **4. Business linkage** | claim/readability ↔ performance tier; cross-check vs sponsor CV dataset | host | `outputs/csv/stage4_linkage_summary.csv` |
| **4b. Keyword mapping** | match on-pack text to the sponsor's 146-keyword dataset; test survival at thumbnail size | host | `outputs/csv/stage4_keyword_hits.csv` |
| **4c. Compliance screen** | Amazon main-image policy screen (white background, on-main text volume) | host | `outputs/csv/stage4_compliance_flags.csv` |

Supporting modules:

- **`src/augment`** — text-safe augmentation (rotation/brightness/contrast/noise/blur;
  flips excluded by design — mirrored text destroys OCR ground truth). Grows 54 images
  into a 324-sample pool (instructor target: 200+) and doubles as an **OCR robustness
  evaluation** under realistic photo conditions.
- **`src/viz`** — presentation figures → `outputs/figures/*.png`.
- **`src/eval`** — CER/WER/char-F1 metrics + `run_eval.py` scoring Stage 1 against
  held-out human annotations (workflow in `data/annotations/README.md`).
- **`src/stage2_donut/make_finetune_data.py` + `train_lora.py`** — Phase-2 scaffold:
  SynthDoG-style synthetic labels with exact JSON ground truth (480 samples) and a
  step-resumable LoRA fine-tune targeting `{brand, product_type, claims, net_weight}`.

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
│   └── gen_synthetic_labels.py   # quick synthetic label generator (pipeline testing)
├── notebooks/
│   └── 01_results_overview.ipynb # results walkthrough (reads output CSVs only)
├── src/
│   ├── config.py                 # all paths & constants (thumbnail sizes, file names)
│   ├── ingest/
│   │   └── ingest.py             # raw images → processed/ + manifest.csv
│   ├── stage1_paddle/
│   │   ├── ocr_engine.py         # shared PaddleOCR engine + helpers
│   │   └── run_ocr.py            # batch OCR → stage1_ocr.csv (+ per-image aggregate)
│   ├── stage2_donut/
│   │   ├── run_donut.py          # Donut zero-shot → stage2_claims.json (host GPU)
│   │   ├── make_finetune_data.py # synthetic labels + exact JSON ground truth (480 samples)
│   │   └── train_lora.py         # step-resumable LoRA fine-tune (Phase 2)
│   ├── stage3_thumbnail/
│   │   └── readability.py        # aspect-preserving thumbnail re-OCR deltas + texts
│   ├── stage4_linkage/
│   │   ├── linkage.py            # join with sponsor dataset; tier analysis
│   │   ├── keyword_map.py        # on-pack text ↔ 146-keyword dataset; thumbnail survival
│   │   └── compliance.py         # Amazon main-image text/background screen
│   ├── augment/
│   │   └── run_augment.py        # text-safe augmentation + OCR robustness eval
│   ├── viz/
│   │   └── figures.py            # presentation figures → outputs/figures
│   └── eval/
│       ├── metrics.py            # CER / WER / character-level F1
│       └── run_eval.py           # score Stage 1 vs held-out annotations
├── data/
│   ├── annotations/              # annotation guide + template (tracked)
│   └── raw|processed|synthetic   # sponsor data & derivatives (NOT tracked — see Data)
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
py -3.14 -m src.stage4_linkage.keyword_map   # keyword hits + thumbnail survival
py -3.14 -m src.stage4_linkage.compliance    # main-image compliance screen
py -3.14 -m src.viz.figures                  # presentation figures
py -3.14 -m src.eval.run_eval                # accuracy vs held-out annotations (once filled)
```

Augmentation / robustness (Paddle venv) and the Phase-2 fine-tune scaffold:

```powershell
.venv-paddle\Scripts\python.exe -m src.augment.run_augment --save          # 324-sample pool + robustness CSV
.venv-paddle\Scripts\python.exe -m src.stage2_donut.make_finetune_data     # 480 synthetic training samples
py -3.14 -m src.stage2_donut.train_lora --epochs 3                         # LoRA fine-tune (host GPU)
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
- **Stage 3** is the headline metric (aspect-preserving downscale, no upscaling): at the
  160 px mobile thumbnail only **~31 %** of full-resolution characters are still recognized —
  **roughly two-thirds of on-pack text is illegible at mobile-search size.** At the 320 px
  search grid the sponsor still retains ~75 %; the cliff is specifically mobile.
- **Robustness (augmentation)**: under text-safe photo perturbations (±6° rotation,
  ±35 % brightness, sensor noise + blur) OCR retains **78–85 %** — versus 31 % at
  thumbnail size. **Resolution, not photo conditions, is what kills legibility.**
- **Keyword mapping**: of the sponsor's search keywords visually present on-pack,
  **only 1 of 8 still matches in the mobile-thumbnail OCR** — including the loss of the
  brand's primary category keyword.
- **Compliance screen**: 3 of 9 main images in the sample fail the white-background
  check; all text-bearing main images are flagged for on-pack-vs-overlay review.
- **Validation**: 54/54 images join to the sponsor CV dataset; our character counts
  correlate **r ≈ 0.71** with its independently computed `ocr_word_count`. Against a
  10-image human-annotated held-out set, Stage 1 scores **char-F1 0.90 with
  precision ≈ 1.0** — the OCR errs by omission (curved bottles, stylized fonts),
  never by fabrication, which is exactly the failure mode the readability metric needs.
- **Phase-2 fine-tune**: rank-8 LoRA on 480 synthetic labels reaches **100 % field-level
  accuracy (strict JSON) on 48 held-out synthetic samples** — replacing the pretrained
  receipt schema with `{brand, product_type, claims, net_weight}`. On real photos the
  schema holds and claims/weights extract correctly; brand/type assignment shows the
  expected synthetic→real domain gap (next step: annotate a small real set).

> Actionable takeaway for the sponsor: the most claim-dense images lose the most text at
> thumbnail size — key claims should be enlarged / repositioned for mobile legibility.

Figures reproducing these findings are generated locally by `py -3.14 -m src.viz.figures`
(not committed — they derive from sponsor data).

---

## Evaluation

`src/eval/metrics.py` provides pure-Python **CER**, **WER**, and **character-level F1** for
scoring OCR against held-out annotations and Donut field values against ground truth.

---

## Roadmap

- **Phase 1** — repo + environments, ingest, baseline PaddleOCR, annotation workflow, eval harness. ✅
- **Phase 2** — Donut zero-shot ✅ · thumbnail scoring ✅ · augmentation + robustness ✅ ·
  synthetic fine-tune data (480 samples) ✅ · **LoRA fine-tune trained & evaluated
  (100 % on held-out synthetic; see `src/stage2_donut/train_lora.py` + `infer_lora.py`)** ✅
- **Phase 3** — keyword mapping ✅ · compliance screen ✅ · figures ✅ · held-out accuracy
  eval (annotations pending) ⏳ · sponsor recommendations & final presentation ⏳

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
