"""Preprocess 0601 multi-doc summary data into evaluator-compatible JSONL."""
import json
import glob
from pathlib import Path

INPUT_DIR = Path(__file__).parent.parent / "examples" / "0601数据"
OUTPUT_FILE = Path(__file__).parent.parent / "examples" / "0601_mapped.jsonl"

SUMMARY_MARKER = "[摘要]"


def transform_record(raw: dict) -> dict:
    content = raw.get("content", "")
    idx = content.find(SUMMARY_MARKER)
    if idx >= 0:
        prompt = content[:idx].strip()
        candidate = content[idx + len(SUMMARY_MARKER):].strip()
    else:
        prompt = content
        candidate = ""

    return {
        "sample_id": raw.get("group_id", ""),
        "prompt": prompt,
        "candidate": candidate,
        "text": "",
        "source": raw.get("domain", ""),
        "language": "zh",
        "pipeline_type": "multi_doc_summary",
        "doc_count": raw.get("doc_count", 0),
        "llm_model": raw.get("llm_model", ""),
        "total_tokens_original": raw.get("total_tokens", 0),
    }


def main():
    files = sorted(glob.glob(str(INPUT_DIR / "synthesized*.jsonl")))
    records = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw = json.loads(line)
                    records.append(transform_record(raw))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
