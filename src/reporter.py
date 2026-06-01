from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from jinja2 import Template


TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


def generate_report(metrics_df: pd.DataFrame, summary: dict, output_path: str | Path, raw_df: pd.DataFrame | None = None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    template_path = TEMPLATE_DIR / "report.html.j2"
    with open(template_path, "r", encoding="utf-8") as f:
        template = Template(f.read())

    context = _build_report_context(metrics_df, summary, raw_df)

    html = template.render(**context)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _build_report_context(metrics_df: pd.DataFrame, summary: dict, raw_df: pd.DataFrame | None = None) -> dict:
    top_risk = metrics_df[metrics_df.get("high_risk_flag", pd.Series(dtype=bool)) == True].head(10)

    mrcr_samples = metrics_df[metrics_df["mrcr_fit"].notna()] if "mrcr_fit" in metrics_df.columns else pd.DataFrame()
    multi_doc_samples = metrics_df[metrics_df["multi_doc_summary_fit"].notna()] if "multi_doc_summary_fit" in metrics_df.columns else pd.DataFrame()

    top_lcu = metrics_df.nlargest(5, "lcu_general") if "lcu_general" in metrics_df.columns else pd.DataFrame()
    bottom_lcu = metrics_df.nsmallest(5, "lcu_general") if "lcu_general" in metrics_df.columns else pd.DataFrame()

    source_dist = _get_source_distribution(metrics_df)

    return {
        "summary": summary,
        "sample_count": len(metrics_df),
        "metrics_df": metrics_df,
        "top_risk_samples": _to_records(top_risk),
        "bucket_distribution": _get_distribution(metrics_df, "length_bucket"),
        "io_distribution": _get_distribution(metrics_df, "io_type"),
        "source_distribution": source_dist["items"],
        "source_total": source_dist["total"],
        "source_non_empty": source_dist["non_empty"],
        "language_distribution": _get_distribution(metrics_df, "language"),
        "pipeline_distribution": _get_distribution(metrics_df, "pipeline_type"),
        "score_columns": ["lcu_general", "ess", "rule_quality_score", "mrcr_fit", "multi_doc_summary_fit"],
        "field_availability": _compute_field_availability(raw_df if raw_df is not None else metrics_df),
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


def _get_source_distribution(df: pd.DataFrame) -> dict:
    """Source distribution with explicit denominator tracking."""
    if "source" not in df.columns:
        return {"items": [], "total": len(df), "non_empty": 0}
    series = df["source"].dropna().replace("", pd.NA).dropna()
    non_empty = len(series)
    if series.empty:
        return {"items": [], "total": len(df), "non_empty": 0}
    counts = series.value_counts()
    result = []
    for value, count in counts.items():
        result.append({
            "label": str(value),
            "count": int(count),
            "ratio_of_total": round(count / len(df) * 100, 1),
            "ratio_of_non_empty": round(count / non_empty * 100, 1),
        })
    return {"items": result, "total": len(df), "non_empty": non_empty}


def _compute_field_availability(df: pd.DataFrame) -> list[dict]:
    fields_to_check = [
        ("id", "sample_id", "样本ID"),
        ("sample_id", "sample_id", "样本ID"),
        ("text", "text", "正文"),
        ("prompt", "prompt", "输入/提示"),
        ("candidate", "candidate", "输出/候选"),
        ("source", "source", "数据来源"),
        ("language", "language", "语言"),
        ("pipeline_type", "pipeline_type", "流水线类型"),
    ]
    seen_labels = set()
    result = []
    for col, display_field, label in fields_to_check:
        if col in df.columns and label not in seen_labels:
            seen_labels.add(label)
            non_empty = df[col].dropna().replace("", pd.NA).dropna()
            available = len(non_empty)
            ratio = round(available / len(df) * 100, 1) if len(df) > 0 else 0
            result.append({"field": display_field, "label": label, "available": available, "total": len(df), "ratio": ratio})
    return result
