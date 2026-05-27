from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import load_config
from .reader import read_data
from .tokenizer import create_tokenizer
from .metrics import (
    check_field_availability,
    compute_lengths,
    assign_bucket,
    classify_io_type,
    compute_quality_risk,
    compute_structure_score,
    compute_long_context_signal,
    compute_lcu_general,
    compute_rule_quality_score,
    compute_ess,
    compute_mrcr_fit,
    compute_multi_doc_summary_fit,
)
from .aggregator import compute_dataset_summary
from .writer import write_sample_metrics, write_dataset_summary
from .reporter import generate_report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Long-context training data evaluation pipeline"
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSONL or CSV file")
    parser.add_argument("--config", "-c", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--output-dir", "-o", default="outputs", help="Output directory")
    parser.add_argument("--tokenizer", "-t", default="tiktoken", choices=["tiktoken", "char"],
                        help="Tokenizer to use")
    return parser.parse_args(argv)


def run_pipeline(args):
    config = load_config(args.config)
    tokenizer = create_tokenizer(args.tokenizer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_data(args.input, config)
    if df.empty:
        print("No data found in input file.")
        sys.exit(1)

    results = []
    for _, row in df.iterrows():
        sample = row.to_dict()
        metrics = evaluate_sample(sample, config, tokenizer)
        results.append(metrics)

    metrics_df = pd.DataFrame(results)

    summary = compute_dataset_summary(metrics_df)
    summary["tokenizer_name"] = tokenizer.name

    write_sample_metrics(metrics_df, output_dir / "sample_metrics.csv")
    write_dataset_summary(summary, output_dir / "dataset_summary.csv")
    generate_report(metrics_df, summary, output_dir / "report.html")

    print(f"Done. Outputs written to {output_dir}/")
    print(f"  - sample_metrics.csv ({len(metrics_df)} samples)")
    print(f"  - dataset_summary.csv")
    print(f"  - report.html")
    print(f"  Tokenizer: {tokenizer.name}")


def evaluate_sample(sample: dict, config, tokenizer) -> dict:
    import math

    def _s(val):
        if val is None:
            return ""
        if isinstance(val, float) and math.isnan(val):
            return ""
        return str(val)

    text = _s(sample.get("text"))
    prompt = _s(sample.get("prompt"))
    candidate = _s(sample.get("candidate"))
    full_text = text or (prompt + "\n" + candidate if prompt else candidate)

    field_avail = check_field_availability(sample)

    lengths = compute_lengths(sample, tokenizer)
    total_tokens = lengths["total_tokens"]
    prompt_tokens = lengths["prompt_tokens"]
    candidate_tokens = lengths["candidate_tokens"]
    length_bucket = lengths["length_bucket"]

    io_type = classify_io_type(prompt_tokens, candidate_tokens)

    risk = compute_quality_risk(full_text, config.thresholds)

    structure_score = compute_structure_score(sample, full_text)

    lcs_score = compute_long_context_signal(sample, full_text, total_tokens)

    info_density_score = risk["info_density_ratio"] * 100

    from .metrics.length_stats import compute_length_score
    length_score = compute_length_score(total_tokens)

    from .metrics.io_classification import IO_TYPE_SCORES
    effective_context_score = IO_TYPE_SCORES.get(io_type, 40)

    lcu_result = compute_lcu_general(
        length_score=length_score,
        effective_context_score=effective_context_score,
        info_density_score=info_density_score,
        structure_score=structure_score,
        long_context_signal_score=lcs_score,
        risk_flags=risk["flags"],
    )

    rqs_result = compute_rule_quality_score(
        html_noise_ratio=risk["html_noise_ratio"],
        weird_char_ratio=risk["weird_char_ratio"],
        ngram_repetition_ratio=risk["ngram_repetition_ratio"],
        duplicate_line_ratio=risk["duplicate_line_ratio"],
        structure_score=structure_score,
        info_density_ratio=risk["info_density_ratio"],
        field_avail=field_avail,
        sample=sample,
    )

    ess = compute_ess(rqs_result["rule_quality_score"], lcu_result["lcu_general"])

    mrcr_result = compute_mrcr_fit(sample, full_text, total_tokens, tokenizer)
    multi_doc_result = compute_multi_doc_summary_fit(sample, full_text, total_tokens, tokenizer)

    return {
        "sample_id": sample.get("id") or sample.get("sample_id", ""),
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "candidate_tokens": candidate_tokens,
        "length_bucket": length_bucket,
        "io_type": io_type,
        "ngram_repetition_ratio": risk["ngram_repetition_ratio"],
        "duplicate_line_ratio": risk["duplicate_line_ratio"],
        "html_noise_ratio": risk["html_noise_ratio"],
        "weird_char_ratio": risk["weird_char_ratio"],
        "template_ratio": risk["template_ratio"],
        "info_density_ratio": risk["info_density_ratio"],
        "high_risk_flag": risk["flags"].get("high_risk", False),
        "structure_score": structure_score,
        "long_context_signal_score": lcs_score,
        "lcu_general": lcu_result["lcu_general"],
        "score_coverage": lcu_result["score_coverage"],
        "rule_quality_score": rqs_result["rule_quality_score"],
        "ess": ess,
        "mrcr_fit": mrcr_result["score"],
        "mrcr_fit_missing_reason": mrcr_result["missing_reason"],
        "multi_doc_summary_fit": multi_doc_result["score"],
        "multi_doc_fit_missing_reason": multi_doc_result["missing_reason"],
        "source": sample.get("source", ""),
        "language": sample.get("language", ""),
        "pipeline_type": sample.get("pipeline_type", ""),
    }


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
