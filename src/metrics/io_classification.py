from __future__ import annotations

IO_TYPE_SCORES = {
    "content_only": 60,
    "long_input_short_output": 90,
    "both_long": 100,
    "short_input_long_output": 70,
    "normal_or_medium": 40,
}


def classify_io_type(prompt_tokens: int | None, candidate_tokens: int | None) -> str:
    if candidate_tokens is None or candidate_tokens < 50:
        return "content_only"
    if prompt_tokens is None:
        return "normal_or_medium"
    if prompt_tokens >= 32768 and candidate_tokens >= 8192:
        return "both_long"
    if prompt_tokens >= 32768 and candidate_tokens < 2048:
        return "long_input_short_output"
    if prompt_tokens < 8192 and candidate_tokens >= 8192:
        return "short_input_long_output"
    return "normal_or_medium"
