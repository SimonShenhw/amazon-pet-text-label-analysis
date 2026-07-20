"""OCR evaluation metrics: CER, WER, and character-level F1.

Pure-python (no heavy deps) so it runs identically on host and in the container,
regardless of which packages happen to be installed in either environment. CER/WER
are computed via a classic Levenshtein dynamic program, O(n*m) — fine at the label
scale used here (short strings, tens of images). char_f1 treats text as a MULTISET
of characters (order-independent) because OCR region order is arbitrary — regions
come back in whatever order the detector emits them — so an order-sensitive metric
would incorrectly punish harmless reordering. Used to score Stage 1 OCR against
held-out annotations and Stage 2 field values against ground truth.

中文：OCR 评测指标模块，实现 CER（字符错误率）、WER（词错误率）和字符级 F1。
纯 Python 实现、不依赖任何重量级库，保证在宿主机和容器两种环境里都能原样跑通，
不受各环境已装包差异影响。CER/WER 用经典 Levenshtein 编辑距离动态规划实现，
复杂度 O(n*m)，在本项目这种短文本、几十张图片的规模下完全够用。char_f1 把
文本当作字符的多重集合（不考虑顺序）来比较，原因是 OCR 各文字区域的输出顺序
本身是随意的（检测器按自己的内部顺序返回区域），如果用对顺序敏感的指标，
会把这种无害的顺序差异误判为错误。本模块用于评测 Stage1 OCR 输出与人工标注、
以及 Stage2 字段值与真值之间的差异。
"""
from __future__ import annotations

from collections import Counter


def edit_distance(a, b) -> int:
    """Levenshtein distance over two sequences (lists or strings).

    两个序列（列表或字符串）之间的 Levenshtein 编辑距离。
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def cer(ref: str, hyp: str) -> float:
    """Character Error Rate = edit_distance / len(ref).

    字符错误率 = 编辑距离 / 参考文本长度。
    """
    ref = ref or ""
    if len(ref) == 0:
        return 0.0 if len(hyp or "") == 0 else 1.0
    return edit_distance(list(ref), list(hyp or "")) / len(ref)


def wer(ref: str, hyp: str) -> float:
    """Word Error Rate = word-level edit_distance / num words in ref.

    词错误率 = 按空格分词后的编辑距离 / 参考文本词数。
    """
    r = (ref or "").split()
    h = (hyp or "").split()
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return edit_distance(r, h) / len(r)


def char_f1(ref: str, hyp: str) -> dict:
    """Multiset character-level F1 (order-independent).

    Robust for comparing thumbnail vs full-resolution OCR where word order may
    differ but the set of recovered characters is what matters.

    多重集合意义下的字符级 F1（不考虑顺序）。适合比较缩略图与全分辨率 OCR
    结果——两者的文字顺序可能不同，但真正关心的是"恢复出了多少字符"，
    与顺序无关。
    """
    rc, hc = Counter(ref or ""), Counter(hyp or "")
    tp = sum((rc & hc).values())
    pred = sum(hc.values())
    gold = sum(rc.values())
    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


if __name__ == "__main__":
    # self-test with known examples — a lightweight regression test for the metric
    # implementations themselves, independent of any pipeline data.
    # 用已知示例做自检，相当于针对指标实现本身的轻量回归测试（不依赖管线数据）。
    assert edit_distance("kitten", "sitting") == 3
    print("CER :", cer("vet approved", "vet approvod"))
    print("WER :", wer("fragrance free formula", "fragrance free"))
    print("F1  :", char_f1("hypoallergenic", "hypoalergenic"))
