import re


def compute_structure_score(sample: dict, text: str) -> int:
    if not text:
        return 0

    score = 0

    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) >= 3:
        score += 20

    heading_pattern = r"(^|\n)(#{1,6}\s|第[一二三四五六七八九十百\d]+[章节篇部]|Chapter\s+\d|\d+\.\d*\s+\S)"
    if re.search(heading_pattern, text):
        score += 20

    if sample.get("doc_id") or re.search(r"(Document\s+\d|<doc|文档\s*\d|Doc\s*\d)", text, re.IGNORECASE):
        score += 20

    turn_pattern = r"(用户[：:]|助手[：:]|user[：:]|assistant[：:]|Human[：:]|AI[：:]|\n用户：|\n助手：)"
    if re.search(turn_pattern, text, re.IGNORECASE):
        score += 20

    if sample.get("prompt") and sample.get("candidate"):
        score += 20

    return min(score, 100)
