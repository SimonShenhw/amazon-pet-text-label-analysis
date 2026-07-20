"""Stage 4b — map extracted on-package text to the SSF keyword dataset (146 keywords).

For every image x keyword pair, test whether the keyword is present in the OCR text at
FULL resolution and whether it still matches in the MOBILE-THUMBNAIL OCR text. This turns
"claims double as keyword candidates" (project plan) into a concrete deliverable:
  - which search keywords are already visually reinforced on-pack,
  - which of them silently vanish at mobile-search size,
  - which high-intent keywords competitors show on-pack but SSF does not.

Matching is OCR-noise tolerant BY DESIGN:
  - texts and keywords are lowercased and stripped to alphanumerics;
  - "phrase" match = keyword with spaces removed occurs in the squashed text
    (catches OCR artifacts like "DRYSHAMPOO");
  - "tokens" match = every keyword token (len>=3) appears somewhere in the text
    (catches reordering / split regions);
  - short stop tokens ("for", "the", ...) are excluded from the tokens match so a
    keyword can't get credited purely for sharing filler words with the text.

The headline deliverable is thumbnail SURVIVAL: a keyword matched at full resolution
but absent from the mobile-thumbnail OCR text is a visually-reinforced keyword that
mobile shoppers effectively never see. On our data, only 1 of 8 SSF keywords survives.

中文：本模块把 Stage3 抽取出的包装文字与 SSF 关键词表（146 个）逐一匹配，分别在
"全分辨率"和"移动端缩略图"两种 OCR 文本上判断关键词是否命中，把项目计划里
"claims 也可当关键词候选"这句话落地成可量化的交付物。匹配算法刻意做了 OCR
容错设计：文本和关键词先转小写并去除非字母数字字符；"phrase"（去空格整体子串）
匹配可以兼容 OCR 把词粘连的情况（如"DRYSHAMPOO"）；"tokens"（关键词各词都出现
即可，忽略顺序）匹配可以兼容 OCR 分区域打乱顺序的情况；STOP_TOKENS 过滤掉
"for/the/and"等虚词，避免仅因共享虚词就产生假匹配。本模块最重要的结论是
"缩略图存活率"：某关键词在全分辨率图上命中、但在 160px 移动缩略图 OCR 文本里
消失，说明这是一个"包装上写了但手机购物者基本看不到"的关键词——在我们的数据
里，SSF 的 8 个关键词只有 1 个能在缩略图尺寸下存活。

Host python:  py -3.14 -m src.stage4_linkage.keyword_map
"""
from __future__ import annotations

import re

import pandas as pd

from src import config

KEYWORD_HITS_CSV = config.CSV_DIR / "stage4_keyword_hits.csv"
# excluded from the "tokens" match so shared filler words don't create false hits
# 排除虚词，避免仅因共享虚词造成假匹配
STOP_TOKENS = {"for", "the", "and", "with", "your"}


def norm(s: str) -> str:
    """Lowercase and collapse to space-separated alphanumeric tokens (for token match).

    转小写并把非字母数字字符替换为空格，用于按 token 做匹配。
    """
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def squash(s: str) -> str:
    """Lowercase and strip ALL non-alphanumeric chars, including spaces (for
    substring/"phrase" matching).

    Removing spaces too means OCR artifacts that glue words together
    (e.g. "DRYSHAMPOO" for "Dry Shampoo") still match as a plain substring.

    转小写并去掉所有非字母数字字符（含空格），用于整体子串匹配。去掉空格
    是关键：这样 OCR 把词粘连在一起的情况（如"DRYSHAMPOO"对应"Dry Shampoo"）
    依然能作为子串匹配上。
    """
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def match(keyword: str, text_norm: str, text_squash: str, token_set: set) -> str | None:
    """Return "phrase", "tokens", or None describing whether/how `keyword` matches.

    Tries the stricter squashed-substring match first (handles OCR-glued text),
    then falls back to the looser all-tokens-present match (handles split/reordered
    regions); the returned label tells callers how confident the hit is.

    判断 keyword 是否命中，返回 "phrase"（整体子串匹配，更严格）、
    "tokens"（关键词各词都出现即可，更宽松）或 None。先尝试严格的整体子串
    匹配（应对 OCR 粘连文字），再退化到宽松的分词匹配（应对区域拆分/顺序
    打乱），返回值本身就标示了这次命中的可信度。
    """
    if squash(keyword) and squash(keyword) in text_squash:
        # guard against an empty squashed keyword: "" is a substring of every
        # string in Python, which would otherwise false-match every image.
        # 防止 keyword squash 后为空字符串（Python 里空串是任何字符串的子串），
        # 否则会造成"全部命中"的假匹配。
        return "phrase"
    tokens = [t for t in norm(keyword).split() if len(t) >= 3 and t not in STOP_TOKENS]
    if tokens and all(t in token_set for t in tokens):
        return "tokens"
    return None


def main():
    """Join OCR text with sponsor metadata, match every image against every keyword
    at both resolutions, write the hit table, and print the survival/opportunity
    summaries.

    与 linkage.py 相同的方式（按文件名）拼接赞助商元数据，对每张图片 x 每个
    关键词做全分辨率与缩略图两次匹配，写出命中明细表，并打印"缩略图存活
    情况"与"高意图关键词缺口"两个业务结论摘要。
    """
    config.ensure_dirs()
    s3 = pd.read_csv(config.STAGE3_CSV)
    if "full_text" not in s3.columns:
        raise SystemExit("stage3_readability.csv has no text columns — re-run Stage 3 first.")
    mf = pd.read_csv(config.MANIFEST_CSV)
    meta = pd.read_excel(config.PET_CV_XLSX, sheet_name="images_cv_features")
    kw = pd.read_excel(config.SSF_CV_XLSX, sheet_name="keywords_dataset")

    df = (s3.merge(mf[["image_id", "original_name"]], on="image_id")
            .merge(meta[["image_filename", "asin", "brand", "performance_tier"]],
                   left_on="original_name", right_on="image_filename", how="left"))

    rows = []
    for _, img in df.iterrows():
        full_n = norm(img.get("full_text", "") or "")
        full_sq = squash(img.get("full_text", "") or "")
        full_tok = set(full_n.split())
        th_n = norm(img.get("mobile_thumb_text", "") or "")
        th_sq = squash(img.get("mobile_thumb_text", "") or "")
        th_tok = set(th_n.split())
        for _, k in kw.iterrows():
            m_full = match(k["keyword"], full_n, full_sq, full_tok)
            if not m_full:
                continue
            m_thumb = match(k["keyword"], th_n, th_sq, th_tok)
            rows.append({
                "image_id": img["image_id"], "original_name": img["original_name"],
                "brand": img.get("brand"), "performance_tier": img.get("performance_tier"),
                "keyword": k["keyword"], "category": k["category"],
                "intent_level": k["intent_level"], "keyword_type": k["keyword_type"],
                "use_for": k.get("use_for"),
                "match_full": m_full,
                "survives_mobile_thumb": bool(m_thumb),
            })

    hits = pd.DataFrame(rows)
    hits.to_csv(KEYWORD_HITS_CSV, index=False)
    print(f"Wrote {KEYWORD_HITS_CSV} ({len(hits)} image-keyword hits)")
    if hits.empty:
        return

    print("\n=== keyword hits by tier (unique keywords) ===")
    print(hits.groupby("performance_tier")["keyword"].nunique().to_string())

    ssf = hits[hits["performance_tier"] == "sponsor"]
    print(f"\n=== SSF: {ssf['keyword'].nunique()} keywords visually present on-pack ===")
    lost = ssf[~ssf["survives_mobile_thumb"]]["keyword"].unique()
    kept = ssf[ssf["survives_mobile_thumb"]]["keyword"].unique()
    print(f"survive mobile thumbnail: {sorted(set(kept))}")
    print(f"LOST at mobile thumbnail: {sorted(set(lost) - set(kept))}")

    # high-intent keywords visible on high-tier competitor packs but absent from SSF
    # 高意图关键词：竞品(high tier)包装上有、但 SSF 没有——潜在的关键词机会
    high_kw = set(hits[(hits.performance_tier == "high")
                       & (hits.intent_level == "high")]["keyword"])
    ssf_kw = set(ssf["keyword"])
    missed = sorted(high_kw - ssf_kw)
    print(f"\n=== opportunity: high-intent keywords on high-tier packs, absent on SSF ===")
    print(missed if missed else "(none)")


if __name__ == "__main__":
    main()
