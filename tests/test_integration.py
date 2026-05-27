"""Integration tests for the evaluation pipeline using mock_data.jsonl."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from src.cli import evaluate_sample, parse_args, run_pipeline
from src.config import load_config
from src.tokenizer import create_tokenizer
from src.reader import read_data


ROOT = Path(__file__).parent.parent
MOCK_DATA = ROOT / "examples" / "mock_data.jsonl"
CONFIG_PATH = ROOT / "configs" / "default.yaml"
OUTPUT_DIR = ROOT / "outputs" / "test_run"


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def tokenizer():
    return create_tokenizer("tiktoken")


@pytest.fixture
def mock_df(config):
    return read_data(MOCK_DATA, config)


@pytest.fixture
def sample_results(mock_df, config, tokenizer):
    results = []
    for _, row in mock_df.iterrows():
        sample = row.to_dict()
        metrics = evaluate_sample(sample, config, tokenizer)
        results.append(metrics)
    return results


def _find_result(results, sample_id):
    for r in results:
        if r["sample_id"] == sample_id:
            return r
    raise ValueError(f"sample_id {sample_id} not found")


class TestReader:
    def test_reads_12_records(self, mock_df):
        assert len(mock_df) == 12

    def test_has_sample_id(self, mock_df):
        assert "id" in mock_df.columns


class TestSample001ContentOnly:
    def test_io_type(self, sample_results):
        r = _find_result(sample_results, "sample_001")
        assert r["io_type"] == "content_only"

    def test_mrcr_na(self, sample_results):
        r = _find_result(sample_results, "sample_001")
        assert r["mrcr_fit"] is None or (isinstance(r["mrcr_fit"], float) and math.isnan(r["mrcr_fit"]))

    def test_multi_doc_na(self, sample_results):
        r = _find_result(sample_results, "sample_001")
        assert r["multi_doc_summary_fit"] is None or (isinstance(r["multi_doc_summary_fit"], float) and math.isnan(r["multi_doc_summary_fit"]))

    def test_has_lcu_and_ess(self, sample_results):
        r = _find_result(sample_results, "sample_001")
        assert r["lcu_general"] is not None
        assert r["ess"] is not None


class TestSample002RagQA:
    def test_has_pipeline_type(self, sample_results):
        r = _find_result(sample_results, "sample_002")
        assert r["pipeline_type"] == "rag_qa"

    def test_has_prompt_and_candidate(self, mock_df):
        row = mock_df[mock_df["id"] == "sample_002"].iloc[0]
        assert pd.notna(row.get("prompt"))
        assert pd.notna(row.get("candidate"))

    def test_has_ess(self, sample_results):
        r = _find_result(sample_results, "sample_002")
        assert r["ess"] is not None and r["ess"] > 0


class TestSample003MRCR:
    def test_mrcr_fit_computed(self, sample_results):
        r = _find_result(sample_results, "sample_003")
        assert r["mrcr_fit"] is not None
        assert r["mrcr_fit"] > 0

    def test_pipeline_type(self, sample_results):
        r = _find_result(sample_results, "sample_003")
        assert r["pipeline_type"] == "mrcr"

    def test_multi_doc_excluded(self, sample_results):
        r = _find_result(sample_results, "sample_003")
        assert r["multi_doc_summary_fit"] is None
        assert r["multi_doc_fit_missing_reason"] == "pipeline_type_is_mrcr"


class TestSample004MultiDoc:
    def test_multi_doc_fit_computed(self, sample_results):
        r = _find_result(sample_results, "sample_004")
        assert r["multi_doc_summary_fit"] is not None
        assert r["multi_doc_summary_fit"] > 0

    def test_pipeline_type(self, sample_results):
        r = _find_result(sample_results, "sample_004")
        assert r["pipeline_type"] == "multi_doc_summary"

    def test_mrcr_na(self, sample_results):
        r = _find_result(sample_results, "sample_004")
        assert r["mrcr_fit"] is None
        assert r["mrcr_fit_missing_reason"] == "no_target_index"


class TestSample005HighRepetition:
    def test_high_risk_flag(self, sample_results):
        r = _find_result(sample_results, "sample_005")
        assert r["high_risk_flag"] is True

    def test_low_lcu(self, sample_results):
        r = _find_result(sample_results, "sample_005")
        assert r["lcu_general"] < 20

    def test_high_ngram_rep(self, sample_results):
        r = _find_result(sample_results, "sample_005")
        assert r["ngram_repetition_ratio"] > 0.25


class TestSample006HTMLNoise:
    def test_high_risk_flag(self, sample_results):
        r = _find_result(sample_results, "sample_006")
        assert r["high_risk_flag"] is True

    def test_html_noise_detected(self, sample_results):
        r = _find_result(sample_results, "sample_006")
        assert r["html_noise_ratio"] > 0.05


class TestSample007Garbled:
    def test_high_risk_flag(self, sample_results):
        r = _find_result(sample_results, "sample_007")
        assert r["high_risk_flag"] is True

    def test_weird_char_detected(self, sample_results):
        r = _find_result(sample_results, "sample_007")
        assert r["weird_char_ratio"] > 0.02


class TestSample008MissingMetadata:
    def test_no_source(self, sample_results):
        r = _find_result(sample_results, "sample_008")
        val = r["source"]
        assert val == "" or val is None or (isinstance(val, float) and math.isnan(val))

    def test_no_language(self, sample_results):
        r = _find_result(sample_results, "sample_008")
        val = r["language"]
        assert val == "" or val is None or (isinstance(val, float) and math.isnan(val))

    def test_still_has_ess(self, sample_results):
        r = _find_result(sample_results, "sample_008")
        assert r["ess"] is not None and r["ess"] > 0


class TestSample009MRCRWithoutTargetIndex:
    def test_mrcr_fit_computed(self, sample_results):
        r = _find_result(sample_results, "sample_009")
        assert r["mrcr_fit"] is not None
        assert r["mrcr_fit"] > 0

    def test_pipeline_type_mrcr(self, sample_results):
        r = _find_result(sample_results, "sample_009")
        assert r["pipeline_type"] == "mrcr"


class TestSample010MultiDoc3Docs:
    def test_multi_doc_fit_computed(self, sample_results):
        r = _find_result(sample_results, "sample_010")
        assert r["multi_doc_summary_fit"] is not None
        assert r["multi_doc_summary_fit"] > 0

    def test_pipeline_type(self, sample_results):
        r = _find_result(sample_results, "sample_010")
        assert r["pipeline_type"] == "multi_doc_summary"


class TestSample011StructuredEnglish:
    def test_has_evidence(self, mock_df):
        row = mock_df[mock_df["id"] == "sample_011"].iloc[0]
        assert pd.notna(row.get("evidence"))

    def test_structure_score_positive(self, sample_results):
        r = _find_result(sample_results, "sample_011")
        assert r["structure_score"] > 0


class TestEndToEnd:
    def test_full_pipeline(self, tmp_path):
        args = parse_args([
            "--input", str(MOCK_DATA),
            "--config", str(CONFIG_PATH),
            "--output-dir", str(tmp_path),
        ])
        run_pipeline(args)

        assert (tmp_path / "sample_metrics.csv").exists()
        assert (tmp_path / "dataset_summary.csv").exists()
        assert (tmp_path / "report.html").exists()

        metrics_df = pd.read_csv(tmp_path / "sample_metrics.csv")
        assert len(metrics_df) == 12

        summary_df = pd.read_csv(tmp_path / "dataset_summary.csv")
        assert summary_df["sample_count"].iloc[0] == 12
        assert summary_df["tokenizer_name"].iloc[0] == "tiktoken_cl100k_base"

    def test_report_self_contained(self, tmp_path):
        args = parse_args([
            "--input", str(MOCK_DATA),
            "--config", str(CONFIG_PATH),
            "--output-dir", str(tmp_path),
        ])
        run_pipeline(args)

        html = (tmp_path / "report.html").read_text()
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html
        assert "cdn" not in html.lower()
        assert "http://" not in html
        assert "https://" not in html

    def test_report_no_nan_display(self, tmp_path):
        args = parse_args([
            "--input", str(MOCK_DATA),
            "--config", str(CONFIG_PATH),
            "--output-dir", str(tmp_path),
        ])
        run_pipeline(args)

        html = (tmp_path / "report.html").read_text()
        import re
        nan_matches = re.findall(r"(?<!\w)nan(?!\w)", html, re.IGNORECASE)
        assert len(nan_matches) == 0, f"Found NaN in HTML: {nan_matches}"

    def test_report_has_all_sections(self, tmp_path):
        args = parse_args([
            "--input", str(MOCK_DATA),
            "--config", str(CONFIG_PATH),
            "--output-dir", str(tmp_path),
        ])
        run_pipeline(args)

        html = (tmp_path / "report.html").read_text()
        assert "Executive Summary" in html
        assert "Field Availability" in html
        assert "General Data Profile" in html
        assert "Quality Risk" in html
        assert "Task Fit" in html
        assert "Known Limitations" in html
        assert "Entry Recommendations" in html


class TestTemplateRatioNA:
    def test_template_ratio_is_none(self, sample_results):
        for r in sample_results:
            assert r.get("template_ratio") is None


class TestTokenizerName:
    def test_tokenizer_has_name(self, tokenizer):
        assert tokenizer.name == "tiktoken_cl100k_base"

    def test_char_fallback_name(self):
        t = create_tokenizer("char")
        assert t.name == "char_fallback"


class TestMissingReasons:
    def test_mrcr_missing_reason_content_only(self, sample_results):
        r = _find_result(sample_results, "sample_001")
        assert r["mrcr_fit_missing_reason"] == "no_target_index"

    def test_multi_doc_missing_reason_no_boundary(self, sample_results):
        r = _find_result(sample_results, "sample_001")
        assert r["multi_doc_fit_missing_reason"] == "no_doc_boundary"

    def test_mrcr_pipeline_excludes_multi_doc(self, sample_results):
        r = _find_result(sample_results, "sample_003")
        assert r["multi_doc_fit_missing_reason"] == "pipeline_type_is_mrcr"
