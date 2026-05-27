from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_sample_metrics(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_dataset_summary(summary: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([summary])
    df.to_csv(path, index=False, encoding="utf-8-sig")
