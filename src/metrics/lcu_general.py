from __future__ import annotations

RISK_PENALTIES = {
    "high_repetition": 10,
    "template_like": 8,
    "html_noise": 8,
    "garbled_text": 12,
    "low_information_density": 10,
    "metadata_missing": 8,
    "source_missing": 5,
}

RISK_PENALTY_CAP = 30

LCU_WEIGHTS = {
    "length": 0.25,
    "effective_context": 0.20,
    "information_density": 0.25,
    "structure": 0.15,
    "long_context_signal": 0.15,
}


def compute_risk_penalty(risk_flags: dict) -> int:
    total = 0
    for flag_name, penalty in RISK_PENALTIES.items():
        if risk_flags.get(flag_name, False):
            total += penalty
    return min(total, RISK_PENALTY_CAP)


def compute_lcu_general(
    length_score: int,
    effective_context_score: int,
    info_density_score: float,
    structure_score: int,
    long_context_signal_score: int | None,
    risk_flags: dict,
) -> dict:
    scores = {
        "length": length_score,
        "effective_context": effective_context_score,
        "information_density": info_density_score,
        "structure": structure_score,
        "long_context_signal": long_context_signal_score,
    }

    available_weight = 0.0
    weighted_sum = 0.0
    for key, weight in LCU_WEIGHTS.items():
        s = scores[key]
        if s is not None:
            available_weight += weight
            weighted_sum += weight * s

    if available_weight == 0:
        return {"lcu_general": None, "score_coverage": 0.0}

    raw_score = weighted_sum / available_weight
    risk_penalty = compute_risk_penalty(risk_flags)
    final = max(0.0, min(100.0, raw_score - risk_penalty))

    total_weight = sum(LCU_WEIGHTS.values())
    coverage = available_weight / total_weight

    return {
        "lcu_general": round(final, 2),
        "score_coverage": round(coverage, 3),
        "risk_penalty": risk_penalty,
    }
