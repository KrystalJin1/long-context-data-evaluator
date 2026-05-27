from __future__ import annotations

import math
import re
from collections import Counter

from ..tokenizer import TokenCounter

# Strong document boundary patterns — "第N篇" alone is NOT a strong boundary
# (it's a coreference expression used in MRCR contexts)
STRONG_DOC_BOUNDARY_PATTERN = re.compile(
    r"(Document\s+\d+|文档\s*\d+|Doc\s*\d+|<doc[^>]*>)",
    re.IGNORECASE,
)


def _safe_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val).strip()


def compute_multi_doc_summary_fit(
    sample: dict, text: str, total_tokens: int, tokenizer: TokenCounter
) -> dict:
    """Returns {"score": float|None, "missing_reason": str|None}."""
    pipeline_type = _safe_str(sample.get("pipeline_type")).lower()

    # If pipeline_type is mrcr, never compute MultiDocSummary-Fit
    if pipeline_type == "mrcr":
        return {"score": None, "missing_reason": "pipeline_type_is_mrcr"}

    has_doc_id = bool(_safe_str(sample.get("doc_id")))
    has_strong_boundary = bool(STRONG_DOC_BOUNDARY_PATTERN.search(text)) if text else False

    # Only compute if pipeline_type=multi_doc_summary OR strong doc boundary/doc_id
    if pipeline_type != "multi_doc_summary" and not has_strong_boundary and not has_doc_id:
        return {"score": None, "missing_reason": "no_doc_boundary"}

    prompt = _safe_str(sample.get("prompt")) or text or ""
    candidate = _safe_str(sample.get("candidate")) or _safe_str(sample.get("summary"))

    if not candidate:
        return {"score": None, "missing_reason": "no_summary"}

    doc_segments = _split_documents(prompt)
    doc_count = len(doc_segments)

    doc_count_score = _doc_count_score(doc_count)
    context_length_score = _context_length_score(total_tokens)
    topic_coherence_score = _topic_coherence_score(doc_segments)
    redundancy_control_score = _redundancy_control_score(doc_segments)
    summary_coverage_score = _summary_coverage_score(doc_segments, candidate)
    structure_boundary_score = _structure_boundary_score(sample, prompt)
    evidence_traceability_score = _evidence_traceability_score(sample)

    fit = (
        0.16 * doc_count_score
        + 0.14 * context_length_score
        + 0.16 * topic_coherence_score
        + 0.14 * redundancy_control_score
        + 0.16 * summary_coverage_score
        + 0.12 * structure_boundary_score
        + 0.12 * evidence_traceability_score
    )

    return {"score": round(fit, 2), "missing_reason": None}


def _split_documents(text: str) -> list[str]:
    parts = STRONG_DOC_BOUNDARY_PATTERN.split(text)
    segments = []
    for part in parts:
        stripped = part.strip()
        if stripped and not STRONG_DOC_BOUNDARY_PATTERN.fullmatch(stripped):
            segments.append(stripped)
    return segments if len(segments) >= 2 else [text]


def _doc_count_score(count: int) -> int:
    if count <= 1:
        return 20
    if count == 2:
        return 40
    if count <= 5:
        return 70
    if count <= 10:
        return 90
    return 100


def _context_length_score(total_tokens: int) -> int:
    if total_tokens < 8192:
        return 20
    if total_tokens < 32768:
        return 40
    if total_tokens < 65536:
        return 60
    if total_tokens < 131072:
        return 75
    if total_tokens < 262144:
        return 90
    return 100


def _topic_coherence_score(segments: list[str]) -> int:
    if len(segments) < 2:
        return 50

    keyword_sets = []
    for seg in segments:
        words = set(re.findall(r"[一-鿿]{2,}|[a-zA-Z]{3,}", seg.lower()))
        keyword_sets.append(words)

    total_overlap = 0
    pairs = 0
    for i in range(len(keyword_sets)):
        for j in range(i + 1, len(keyword_sets)):
            if keyword_sets[i] and keyword_sets[j]:
                union = keyword_sets[i] | keyword_sets[j]
                intersection = keyword_sets[i] & keyword_sets[j]
                if union:
                    total_overlap += len(intersection) / len(union)
                pairs += 1

    if pairs == 0:
        return 50

    avg_overlap = total_overlap / pairs
    return min(100, int(avg_overlap * 200))


def _redundancy_control_score(segments: list[str]) -> int:
    if len(segments) < 2:
        return 80

    all_words = []
    for seg in segments:
        words = re.findall(r"[一-鿿]|[a-zA-Z]+", seg.lower())
        all_words.append(set(words))

    total_overlap = 0
    pairs = 0
    for i in range(len(all_words)):
        for j in range(i + 1, len(all_words)):
            if all_words[i] and all_words[j]:
                union = all_words[i] | all_words[j]
                intersection = all_words[i] & all_words[j]
                if union:
                    total_overlap += len(intersection) / len(union)
                pairs += 1

    if pairs == 0:
        return 80

    avg_redundancy = total_overlap / pairs
    return max(0, min(100, int((1.0 - avg_redundancy) * 100)))


def _summary_coverage_score(segments: list[str], candidate: str) -> int:
    if not candidate or len(segments) < 2:
        return 30

    candidate_lower = candidate.lower()
    covered = 0
    for seg in segments:
        keywords = set(re.findall(r"[一-鿿]{2,}|[a-zA-Z]{3,}", seg.lower()))
        top_keywords = sorted(keywords, key=lambda w: len(w), reverse=True)[:5]
        if top_keywords:
            hits = sum(1 for kw in top_keywords if kw in candidate_lower)
            if hits / len(top_keywords) >= 0.3:
                covered += 1

    if len(segments) == 0:
        return 30
    coverage_ratio = covered / len(segments)
    return min(100, int(coverage_ratio * 100))


def _structure_boundary_score(sample: dict, text: str) -> int:
    score = 0
    if _safe_str(sample.get("doc_id")):
        score += 50
    if STRONG_DOC_BOUNDARY_PATTERN.search(text):
        score += 50
    return min(score, 100)


def _evidence_traceability_score(sample: dict) -> int:
    score = 0
    if _safe_str(sample.get("doc_id")):
        score += 50
    if _safe_str(sample.get("evidence")):
        score += 50
    return min(score, 100)
