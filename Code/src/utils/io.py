"""Minimal IO helpers used across the project."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_meta(path: Path | str, payload: dict[str, Any]) -> Path:
    """Write a sibling ``*.meta.json`` next to a data artifact.

    Records the snapshot timestamp so we satisfy the §9 rule of never silently
    overwriting raw data without a paper trail.
    """
    path = Path(path)
    payload = {
        **payload,
        "snapshot_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def read_meta(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_parquet(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    return path


def from_parquet(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(Path(path))
