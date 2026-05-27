from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from jinja2 import Template


TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


def generate_report(metrics_df: pd.DataFrame, summary: dict, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    template_path = TEMPLATE_DIR / "report.html.j2"
    with open(template_path, "r", encoding="utf-8") as f:
        template = Template(f.read())

    context = _build_report_context(metrics_df, summary)

    html = template.render(**context)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _build_report_context(metrics_df: pd.DataFrame, summary: dict) -> dict:
    top_risk = metrics_df[metrics_df.get("high_risk_flag", pd.Series(dtype=bool)) == True].head(10)

    mrcr_samples = metrics_df[metrics_df["mrcr_fit"].notna()] if "mrcr_fit" in metrics_df.columns else pd.DataFrame()
    multi_doc_samples = metrics_df[metrics_df["multi_doc_summary_fit"].notna()] if "multi_doc_summary_fit" in metrics_df.columns else pd.DataFrame()

    top_lcu = metrics_df.nlargest(5, "lcu_general") if "lcu_general" in metrics_df.columns else pd.DataFrame()
    bottom_lcu = metrics_df.nsmallest(5, "lcu_general") if "lcu_general" in metrics_df.columns else pd.DataFrame()

    return {
        "summary": summary,
        "sample_count": len(metrics_df),
        "metrics_df": metrics_df,
        "top_risk_samples": _to_records(top_risk),
        "bucket_distribution": _get_distribution(metrics_df, "length_bucket"),
        "io_distribution": _get_distribution(metrics_df, "io_type"),
        "source_distribution": _get_distribution(metrics_df, "source"),
        "language_distribution": _get_distribution(metrics_df, "language"),
        "pipeline_distribution": _get_distribution(metrics_df, "pipeline_type"),
        "score_columns": ["lcu_general", "ess", "rule_quality_score", "mrcr_fit", "multi_doc_summary_fit"],
        "field_availability": _compute_field_availability(metrics_df),
        "mrcr_samples": _to_records(mrcr_samples.nlargest(5, "mrcr_fit")) if not mrcr_samples.empty else [],
        "mrcr_count": len(mrcr_samples),
        "multi_doc_samples": _to_records(multi_doc_samples.nlargest(5, "multi_doc_summary_fit")) if not multi_doc_samples.empty else [],
        "multi_doc_count": len(multi_doc_samples),
        "top_lcu_samples": _to_records(top_lcu),
        "bottom_lcu_samples": _to_records(bottom_lcu),
        "mrcr_missing_reasons": _get_distribution(metrics_df, "mrcr_fit_missing_reason"),
        "multi_doc_missing_reasons": _get_distribution(metrics_df, "multi_doc_fit_missing_reason"),
    }


def _to_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    records = df.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
    return records


def _get_distribution(df: pd.DataFrame, col: str) -> list[dict]:
    if col not in df.columns:
        return []
    series = df[col].dropna().replace("", pd.NA).dropna()
    if series.empty:
        return []
    counts = series.value_counts()
    total = len(df)
    result = []
    for value, count in counts.items():
        result.append({
            "label": str(value),
            "count": int(count),
            "ratio": round(count / total * 100, 1),
        })
    return result


def _compute_field_availability(df: pd.DataFrame) -> list[dict]:
    fields_to_check = ["source", "language", "pipeline_type", "prompt", "candidate"]
    result = []
    for field in fields_to_check:
        if field in df.columns:
            non_empty = df[field].dropna().replace("", pd.NA).dropna()
            available = len(non_empty)
            ratio = round(available / len(df) * 100, 1) if len(df) > 0 else 0
            result.append({"field": field, "available": available, "total": len(df), "ratio": ratio})
    return result
