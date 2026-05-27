import re
from collections import Counter


HTML_PATTERN = re.compile(r"<[^>]+>|https?://\S+|www\.\S+", re.IGNORECASE)


def compute_ngram_repetition_ratio(text: str, n: int = 5) -> float:
    if not text:
        return 0.0
    tokens = _tokenize_for_ngram(text)
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return 1.0 - (unique / total)


def compute_duplicate_line_ratio(text: str) -> float:
    if not text:
        return 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    duplicated = sum(c - 1 for c in counts.values() if c > 1)
    return duplicated / len(lines)


def compute_html_noise_ratio(text: str) -> float:
    if not text:
        return 0.0
    matches = HTML_PATTERN.findall(text)
    noise_chars = sum(len(m) for m in matches)
    return noise_chars / len(text)


def compute_weird_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    weird_count = 0
    for ch in text:
        cp = ord(ch)
        if (cp < 32 and cp not in (9, 10, 13)) or \
           (0xE000 <= cp <= 0xF8FF) or \
           cp == 0xFFFD or \
           (0xFFF0 <= cp <= 0xFFFE):
            weird_count += 1
    return weird_count / len(text)


def compute_info_density(ngram_rep: float, dup_line: float,
                         html_noise: float, weird_char: float) -> float:
    """MVP formula: template_ratio not included (marked N/A, v2 will implement)."""
    raw = 1.0 - 0.8 * ngram_rep - 0.6 * dup_line - 0.5 * html_noise - 0.5 * weird_char
    return max(0.0, min(1.0, raw))


def compute_quality_risk(text: str, thresholds) -> dict:
    ngram_rep = compute_ngram_repetition_ratio(text)
    dup_line = compute_duplicate_line_ratio(text)
    html_noise = compute_html_noise_ratio(text)
    weird_char = compute_weird_char_ratio(text)
    info_density = compute_info_density(ngram_rep, dup_line, html_noise, weird_char)

    flags = {
        "high_repetition": ngram_rep > thresholds.high_repetition,
        "html_noise": html_noise > thresholds.html_noise,
        "garbled_text": weird_char > thresholds.weird_char,
        "low_information_density": info_density < thresholds.info_density_low,
        "template_like": False,
    }
    flags["high_risk"] = any(flags.values())

    return {
        "ngram_repetition_ratio": round(ngram_rep, 4),
        "duplicate_line_ratio": round(dup_line, 4),
        "html_noise_ratio": round(html_noise, 4),
        "weird_char_ratio": round(weird_char, 4),
        "template_ratio": None,  # N/A: not_implemented in MVP, v2 will add cross-sample detection
        "info_density_ratio": round(info_density, 4),
        "flags": flags,
    }


def _tokenize_for_ngram(text: str) -> list[str]:
    words = re.findall(r"[一-鿿]|[a-zA-Z]+|[0-9]+|[^\s一-鿿a-zA-Z0-9]", text)
    return words
