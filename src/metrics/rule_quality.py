from __future__ import annotations

import math

REQUIRED_METADATA = ["sample_id", "source", "language"]
CONDITIONAL_METADATA = {
    "mrcr": ["target_index", "pipeline_type"],
    "rag_qa": ["evidence", "pipeline_type"],
    "retrieval": ["evidence", "pipeline_type"],
    "multi_doc_summary": ["doc_id", "pipeline_type"],
    "default_task": ["doc_id", "pipeline_type", "evidence", "target"],
}


def compute_rule_quality_score(
    html_noise_ratio: float,
    weird_char_ratio: float,
    ngram_repetition_ratio: float,
    duplicate_line_ratio: float,
    structure_score: int,
    info_density_ratio: float,
    field_avail: dict,
    sample: dict,
) -> dict:
    cleanliness = max(0.0, min(100.0, (1.0 - html_noise_ratio - weird_char_ratio) * 100))
    dedup = max(0.0, min(100.0, (1.0 - ngram_repetition_ratio - 0.5 * duplicate_line_ratio) * 100))
    structure = float(structure_score)
    info_density = info_density_ratio * 100

    completeness = _compute_completeness_score(sample)
    metadata = _compute_metadata_score(sample)

    rqs = (
        0.25 * cleanliness
        + 0.20 * dedup
        + 0.15 * structure
        + 0.15 * completeness
        + 0.15 * info_density
        + 0.10 * metadata
    )

    return {
        "rule_quality_score": round(rqs, 2),
        "cleanliness_rule_score": round(cleanliness, 2),
        "dedup_repetition_score": round(dedup, 2),
        "structure_rule_score": round(structure, 2),
        "completeness_rule_score": round(completeness, 2),
        "information_density_score": round(info_density, 2),
        "metadata_score": round(metadata, 2),
    }


def compute_ess(rule_quality_score: float, lcu_general: float | None) -> float | None:
    if lcu_general is None:
        return rule_quality_score
    return round(0.55 * rule_quality_score + 0.45 * lcu_general, 2)


def _safe_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val).strip()


def _compute_completeness_score(sample: dict) -> float:
    pipeline_type = _safe_str(sample.get("pipeline_type")).lower()
    candidate = _safe_str(sample.get("candidate"))
    prompt = _safe_str(sample.get("prompt"))
    text = _safe_str(sample.get("text"))

    io_is_content_only = not candidate or len(str(candidate).strip()) < 10

    if io_is_content_only and not pipeline_type:
        # content_only: check text presence and reasonable length
        if not text:
            return 20.0
        if len(text) > 1000:
            return 100.0
        return 80.0

    if pipeline_type == "mrcr":
        score = 0.0
        has_turns = bool(prompt and ("\n" in prompt or "用户" in prompt or "user" in prompt.lower()))
        if has_turns:
            score += 33.3
        if sample.get("target_index") is not None:
            score += 33.3
        if candidate:
            score += 33.3
        return min(100.0, score)

    if pipeline_type in ("rag_qa", "retrieval"):
        score = 0.0
        if prompt:
            score += 33.3
        if candidate:
            score += 33.3
        if sample.get("evidence"):
            score += 33.3
        return min(100.0, score)

    if pipeline_type == "multi_doc_summary":
        score = 0.0
        has_doc_boundary = bool(sample.get("doc_id") or
                                (prompt and ("Document" in prompt or "文档" in prompt)))
        if has_doc_boundary:
            score += 50.0
        if candidate or sample.get("summary"):
            score += 50.0
        return min(100.0, score)

    # Generic: prompt + candidate presence
    if prompt and candidate:
        return 100.0
    if prompt or candidate:
        return 60.0
    return 20.0


def _compute_metadata_score(sample: dict) -> float:
    pipeline_type = _safe_str(sample.get("pipeline_type")).lower()
    is_task_sample = bool(pipeline_type)

    required_present = sum(
        1 for f in REQUIRED_METADATA
        if _safe_str(sample.get(f))
    )
    required_total = len(REQUIRED_METADATA)
    required_ratio = required_present / required_total

    if not is_task_sample:
        return required_ratio * 100.0

    # Task samples: 60% required + 40% conditional
    conditional_fields = CONDITIONAL_METADATA.get(pipeline_type, CONDITIONAL_METADATA["default_task"])
    conditional_present = sum(
        1 for f in conditional_fields
        if _safe_str(sample.get(f))
    )
    conditional_total = len(conditional_fields)
    conditional_ratio = conditional_present / conditional_total if conditional_total > 0 else 1.0

    return (0.6 * required_ratio + 0.4 * conditional_ratio) * 100.0
