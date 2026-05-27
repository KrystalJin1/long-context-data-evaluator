from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import Config


def read_data(path: str | Path, config: Config) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".jsonl", ".json"):
        records = _read_jsonl(path)
    elif ext == ".csv":
        records = _read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .jsonl or .csv")

    df = pd.DataFrame(records)
    df = _apply_field_mapping(df, config)
    return df


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_csv(path: Path) -> list[dict]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df.to_dict(orient="records")


def _apply_field_mapping(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    mapping = vars(config.fields)
    reverse_map = {}
    for logical_name, raw_name in mapping.items():
        if raw_name in df.columns and logical_name != raw_name:
            reverse_map[raw_name] = logical_name

    if reverse_map:
        df = df.rename(columns=reverse_map)
    return df
