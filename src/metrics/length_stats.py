from ..tokenizer import TokenCounter

LENGTH_BUCKETS = [
    (4096, "0-4K"),
    (8192, "4-8K"),
    (16384, "8-16K"),
    (32768, "16-32K"),
    (65536, "32-64K"),
    (131072, "64-128K"),
    (262144, "128-256K"),
]


def assign_bucket(tokens: int) -> str:
    for limit, label in LENGTH_BUCKETS:
        if tokens < limit:
            return label
    return "256K+"


def compute_length_score(total_tokens: int) -> int:
    if total_tokens < 8192:
        return 0
    if total_tokens < 16384:
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


def compute_lengths(sample: dict, tokenizer: TokenCounter) -> dict:
    text = str(sample.get("text") or "")
    prompt = str(sample.get("prompt") or "")
    candidate = str(sample.get("candidate") or "")

    prompt_tokens = tokenizer.count(prompt) if prompt.strip() else None
    candidate_tokens = tokenizer.count(candidate) if candidate.strip() else None

    if text.strip() and not prompt.strip():
        total_tokens = tokenizer.count(text)
    else:
        total_tokens = (prompt_tokens or 0) + (candidate_tokens or 0)

    return {
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "candidate_tokens": candidate_tokens,
        "length_bucket": assign_bucket(total_tokens),
    }
