from __future__ import annotations

import math
import re

from ..tokenizer import TokenCounter

COREFERENCE_PATTERNS_ZH = [
    r"第[一二三四五六七八九十\d]+[篇个条]",
    r"刚才那个",
    r"前面那篇",
    r"上一个",
    r"之前写的",
    r"重新输出",
    r"原文给我",
    r"再发一遍",
]

COREFERENCE_PATTERNS_EN = [
    r"the\s+\d+(?:st|nd|rd|th)",
    r"the\s+previous\s+one",
    r"the\s+earlier\s+one",
    r"what\s+you\s+wrote\s+before",
    r"reproduce",
    r"verbatim",
    r"send\s+again",
]

COREFERENCE_RE = re.compile(
    "|".join(COREFERENCE_PATTERNS_ZH + COREFERENCE_PATTERNS_EN), re.IGNORECASE
)

TURN_PATTERN = re.compile(
    r"(用户[：:]|助手[：:]|user[：:]|assistant[：:]|Human[：:]|AI[：:])", re.IGNORECASE
)


def _safe_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val).strip()


def compute_mrcr_fit(sample: dict, text: str, total_tokens: int, tokenizer: TokenCounter) -> dict:
    """Returns {"score": float|None, "missing_reason": str|None}."""
    pipeline_type = _safe_str(sample.get("pipeline_type")).lower()
    target_index = sample.get("target_index")
    if isinstance(target_index, float) and math.isnan(target_index):
        target_index = None

    if pipeline_type != "mrcr" and target_index is None:
        return {"score": None, "missing_reason": "no_target_index"}

    # Compute all sub-scores
    scores = {
        "context_length": (0.15, _context_length_score(total_tokens)),
        "turn_structure": (0.15, _turn_structure_score(text)),
        "target_candidate": (0.18, _target_candidate_score(sample)),
        "distractor": (0.15, _distractor_score(text, _turn_structure_score(text))),
        "target_position": (0.15, _target_position_score(sample, text)),
        "coreference_marker": (0.12, _coreference_marker_score(text)),
        "answer_extractability": (0.10, _answer_extractability_score(sample)),
    }

    # Weight renormalization for N/A sub-scores
    available_weight = 0.0
    weighted_sum = 0.0
    for _, (weight, score) in scores.items():
        if score is not None:
            available_weight += weight
            weighted_sum += weight * score

    if available_weight == 0:
        return {"score": None, "missing_reason": "all_subscores_na"}

    fit = weighted_sum / available_weight
    return {"score": round(fit, 2), "missing_reason": None}


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


def _turn_structure_score(text: str) -> int:
    turns = len(TURN_PATTERN.findall(text))
    if turns == 0:
        return 0
    if turns <= 4:
        return 50
    if turns <= 10:
        return 75
    return 100


def _target_candidate_score(sample: dict) -> int | None:
    """Score based on target_candidate_count (preferred) or target_index (weak proxy).

    - target_candidate_count available: directly map count to score
    - only target_index available: validate it's a legal positive int, give weak rule score
    - neither available: return None (N/A, reduce score_coverage)
    """
    target_candidate_count = sample.get("target_candidate_count")
    target_index = sample.get("target_index")

    if target_candidate_count is not None:
        try:
            count = int(target_candidate_count)
        except (ValueError, TypeError):
            count = 0
        if count <= 1:
            return 30
        if count <= 2:
            return 60
        if count <= 4:
            return 80
        return 100

    if target_index is not None:
        try:
            idx = int(target_index)
            if idx < 1:
                return None
        except (ValueError, TypeError):
            return None
        # target_index is valid: we know there are at least idx targets
        if idx >= 2:
            return 50  # weak rule score: multiple targets implied but total unknown
        return 30  # idx=1: at least one target exists

    return None


def _distractor_score(text: str, turn_score: int) -> int:
    if turn_score >= 75:
        return 70
    if turn_score >= 50:
        return 50
    return 20


def _target_position_score(sample: dict, text: str) -> int:
    target = _safe_str(sample.get("target")) or _safe_str(sample.get("candidate"))
    if not target or not text:
        return 50

    pos = text.find(target)
    if pos == -1:
        return 50

    relative = pos / max(len(text), 1)
    if relative < 0.33:
        return 50
    if relative < 0.66:
        return 100
    return 85


def _coreference_marker_score(text: str) -> int:
    matches = COREFERENCE_RE.findall(text)
    count = len(matches)
    if count == 0:
        return 0
    if count == 1:
        return 60
    if count <= 3:
        return 80
    return 100


def _answer_extractability_score(sample: dict) -> int:
    candidate = _safe_str(sample.get("candidate"))
    prompt = _safe_str(sample.get("prompt"))
    if not candidate or not prompt:
        return 0

    if candidate in prompt:
        return 90

    candidate_words = set(candidate.split())
    prompt_words = set(prompt.split())
    if candidate_words and candidate_words.issubset(prompt_words):
        return 50

    overlap = len(candidate_words & prompt_words)
    if candidate_words and overlap / len(candidate_words) > 0.5:
        return 50

    return 0
