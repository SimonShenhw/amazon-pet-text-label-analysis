"""OCR evaluation metrics: CER, WER, and character-level F1.

Pure-python (no heavy deps) so it runs on host or in the container. Used to score
Stage 1 OCR against held-out annotations and Stage 2 field values against ground truth.
"""
from __future__ import annotations

from collections import Counter


def edit_distance(a, b) -> int:
    """Levenshtein distance over two sequences (lists or strings)."""
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
    """Character Error Rate = edit_distance / len(ref)."""
    ref = ref or ""
    if len(ref) == 0:
        return 0.0 if len(hyp or "") == 0 else 1.0
    return edit_distance(list(ref), list(hyp or "")) / len(ref)


def wer(ref: str, hyp: str) -> float:
    """Word Error Rate = word-level edit_distance / num words in ref."""
    r = (ref or "").split()
    h = (hyp or "").split()
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return edit_distance(r, h) / len(r)


def char_f1(ref: str, hyp: str) -> dict:
    """Multiset character-level F1 (order-independent).

    Robust for comparing thumbnail vs full-resolution OCR where word order may
    differ but the set of recovered characters is what matters.
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
    assert edit_distance("kitten", "sitting") == 3
    print("CER :", cer("vet approved", "vet approvod"))
    print("WER :", wer("fragrance free formula", "fragrance free"))
    print("F1  :", char_f1("hypoallergenic", "hypoalergenic"))
