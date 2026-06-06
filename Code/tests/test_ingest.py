"""Tests for the Oxford-Man CSV loader.

Network-dependent download paths are intentionally not exercised here -- those
are covered by the manual Phase 1 ingestion run. This file pins the *parsing*
logic against the schema quirks of the real library (unnamed date column,
dotted index symbols).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.ingest import load_oxfordman_symbol


def _write_csv(path, text: str):
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def oxfordman_csv(tmp_path):
    """Synthetic CSV mirroring the 2022 snapshot schema: unnamed leading date
    column, a ``Symbol`` column, and an ``rv5`` realized-variance column.
    """
    csv = tmp_path / "oxfordmanrealizedvolatilityindices.csv"
    rows = [",Symbol,rv5,medrv"]
    for d in ("2000-01-03", "2000-01-04", "2000-01-05"):
        rows.append(f"{d} 00:00:00+00:00,.SPX,0.00012,0.00010")
        rows.append(f"{d} 00:00:00+00:00,.AEX,0.00020,0.00018")
    return _write_csv(csv, "\n".join(rows) + "\n")


def test_loads_requested_symbol(oxfordman_csv):
    df = load_oxfordman_symbol(oxfordman_csv, symbol=".SPX")
    assert len(df) == 3
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "date"
    assert "rv5" in df.columns


def test_symbol_filter_is_exact(oxfordman_csv):
    spx = load_oxfordman_symbol(oxfordman_csv, symbol=".SPX")
    aex = load_oxfordman_symbol(oxfordman_csv, symbol=".AEX")
    assert (spx["rv5"] == 0.00012).all()
    assert (aex["rv5"] == 0.00020).all()


def test_symbol_match_is_case_insensitive(oxfordman_csv):
    df = load_oxfordman_symbol(oxfordman_csv, symbol=".spx")
    assert len(df) == 3


def test_index_is_sorted(oxfordman_csv):
    df = load_oxfordman_symbol(oxfordman_csv, symbol=".SPX")
    assert df.index.is_monotonic_increasing


def test_missing_symbol_raises(oxfordman_csv):
    with pytest.raises(ValueError, match="not found"):
        load_oxfordman_symbol(oxfordman_csv, symbol=".NOPE")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_oxfordman_symbol(tmp_path / "absent.csv", symbol=".SPX")
