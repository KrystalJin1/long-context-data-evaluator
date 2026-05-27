from __future__ import annotations

import pandas as pd


def compute_dataset_summary(df: pd.DataFrame) -> dict:
    summary = {}

    summary["sample_count"] = len(df)

    # Length statistics
    if "total_tokens" in df.columns:
        total_tokens = df["total_tokens"]
        summary["total_tokens_mean"] = round(total_tokens.mean(), 0)
        summary["total_tokens_median"] = round(total_tokens.median(), 0)
        summary["total_tokens_p90"] = round(total_tokens.quantile(0.9), 0)
        summary["total_tokens_max"] = int(total_tokens.max())
        summary["32k_plus_ratio"] = round((total_tokens >= 32768).mean(), 4)
        summary["64k_plus_ratio"] = round((total_tokens >= 65536).mean(), 4)
        summary["128k_plus_ratio"] = round((total_tokens >= 131072).mean(), 4)
        summary["256k_plus_count"] = int((total_tokens >= 262144).sum())

    # Length bucket distribution
    if "length_bucket" in df.columns:
        bucket_dist = df["length_bucket"].value_counts(normalize=True).to_dict()
        for bucket, ratio in bucket_dist.items():
            summary[f"bucket_{bucket}_ratio"] = round(ratio, 4)

    # I/O type distribution
    if "io_type" in df.columns:
        io_dist = df["io_type"].value_counts(normalize=True).to_dict()
        for io_type, ratio in io_dist.items():
            summary[f"io_{io_type}_ratio"] = round(ratio, 4)

    # Source distribution
    if "source" in df.columns:
        source_series = df["source"].dropna().replace("", pd.NA).dropna()
        source_counts = source_series.value_counts()
        if len(source_counts) > 0:
            total = source_counts.sum()
            summary["source_count"] = len(source_counts)
            summary["source_top1_share"] = round(source_counts.iloc[0] / total, 4) if total > 0 else 0
            top3 = source_counts.head(3).sum()
            summary["source_top3_share"] = round(top3 / total, 4) if total > 0 else 0

    # Language distribution
    if "language" in df.columns:
        lang_series = df["language"].dropna().replace("", pd.NA).dropna()
        lang_dist = lang_series.value_counts(normalize=True).to_dict()
        for lang, ratio in lang_dist.items():
            summary[f"lang_{lang}_ratio"] = round(ratio, 4)

    # Pipeline type distribution
    if "pipeline_type" in df.columns:
        pt_series = df["pipeline_type"].dropna().replace("", pd.NA).dropna()
        pt_dist = pt_series.value_counts().to_dict()
        for pt, count in pt_dist.items():
            summary[f"pipeline_{pt}_count"] = int(count)

    # Risk flags
    if "high_risk_flag" in df.columns:
        summary["high_risk_count"] = int(df["high_risk_flag"].sum())
        summary["high_risk_ratio"] = round(df["high_risk_flag"].mean(), 4)

    # Field availability summary
    for col in ["source", "language", "pipeline_type"]:
        if col in df.columns:
            non_empty = df[col].dropna().replace("", pd.NA).dropna()
            summary[f"field_{col}_available_ratio"] = round(len(non_empty) / len(df), 4)

    # Score distributions (handles single-value case correctly)
    for score_col in ["lcu_general", "ess", "rule_quality_score"]:
        if score_col in df.columns:
            valid = df[score_col].dropna()
            if len(valid) > 0:
                summary[f"{score_col}_mean"] = round(valid.mean(), 2)
                summary[f"{score_col}_p50"] = round(valid.median(), 2)
                summary[f"{score_col}_p90"] = round(valid.quantile(0.9), 2)
                summary[f"{score_col}_min"] = round(valid.min(), 2)
                summary[f"{score_col}_max"] = round(valid.max(), 2)

    # Task fit scores
    for fit_col in ["mrcr_fit", "multi_doc_summary_fit"]:
        if fit_col in df.columns:
            valid = df[fit_col].dropna()
            summary[f"{fit_col}_count"] = int(len(valid))
            if len(valid) > 0:
                summary[f"{fit_col}_mean"] = round(valid.mean(), 2)
                summary[f"{fit_col}_p50"] = round(valid.median(), 2)
                summary[f"{fit_col}_p90"] = round(valid.quantile(0.9), 2)

    # Missing reason counts for task fits
    for reason_col in ["mrcr_fit_missing_reason", "multi_doc_fit_missing_reason"]:
        if reason_col in df.columns:
            reasons = df[reason_col].dropna().replace("", pd.NA).dropna()
            reason_dist = reasons.value_counts().to_dict()
            for reason, count in reason_dist.items():
                summary[f"{reason_col}_{reason}"] = int(count)

    return summary
