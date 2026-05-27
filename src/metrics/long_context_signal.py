from __future__ import annotations

import math
import re


CROSS_REF_PATTERNS = re.compile(
    r"(前文|上述|该文档|之前提到|如前所述|前面提到|上文|previously\s+mentioned|"
    r"above-mentioned|aforementioned|as\s+noted\s+earlier|as\s+discussed)",
    re.IGNORECASE,
)

ENTITY_PATTERN = re.compile(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*|[一-鿿]{2,4}")


def _safe_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val).strip()


def compute_long_context_signal(sample: dict, text: str, total_tokens: int) -> int | None:
    if not text:
        return None

    score = 0
    has_any_signal = False

    target = _safe_str(sample.get("target")) or _safe_str(sample.get("candidate")) or ""
    if target and len(target) > 2 and target in text:
        pos = text.find(target)
        relative_pos = pos / max(len(text), 1)
        if relative_pos > 0.33:
            score += 25
            has_any_signal = True

    mid = len(text) // 2
    first_half_entities = set(ENTITY_PATTERN.findall(text[:mid]))
    second_half_entities = set(ENTITY_PATTERN.findall(text[mid:]))
    overlap = first_half_entities & second_half_entities
    if len(overlap) >= 2:
        score += 20
        has_any_signal = True

    if CROSS_REF_PATTERNS.search(text):
        score += 20
        has_any_signal = True

    if sample.get("doc_id") or re.search(r"Document\s+\d|文档\s*\d", text, re.IGNORECASE):
        score += 15
        has_any_signal = True

    target_index = sample.get("target_index")
    if target_index is not None:
        try:
            if int(target_index) > 1:
                score += 20
                has_any_signal = True
        except (ValueError, TypeError):
            pass

    if not has_any_signal:
        return None

    return min(score, 100)
