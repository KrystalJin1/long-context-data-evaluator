from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class FieldMapping:
    id: str = "sample_id"
    text: str = "text"
    prompt: str = "prompt"
    candidate: str = "candidate"
    source: str = "source"
    language: str = "language"
    pipeline_type: str = "pipeline_type"
    doc_id: str = "doc_id"
    target: str = "target"
    target_index: str = "target_index"
    evidence: str = "evidence"
    summary: str = "summary"


@dataclass
class Thresholds:
    long_context_tokens: int = 32768
    ultra_long_context_tokens: int = 131072
    info_density_low: float = 0.60
    high_repetition: float = 0.25
    html_noise: float = 0.05
    weird_char: float = 0.02


@dataclass
class Config:
    fields: FieldMapping = field(default_factory=FieldMapping)
    thresholds: Thresholds = field(default_factory=Thresholds)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    fields_data = raw.get("fields", {})
    thresholds_data = raw.get("thresholds", {})

    fields = FieldMapping(**{k: v for k, v in fields_data.items() if hasattr(FieldMapping, k)})
    thresholds = Thresholds(**{k: v for k, v in thresholds_data.items() if hasattr(Thresholds, k)})

    return Config(fields=fields, thresholds=thresholds)
