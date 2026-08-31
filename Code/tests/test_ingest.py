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


def test_bst_summer_dates_are_not_shifted(tmp_path):
    """Regression guard for the 2026-07 date bug.

    Oxford-Man timestamps carry a +01:00 (BST) offset in summer. A ``utc=True``
    parse converts them to 23:00 UTC on the previous day and then normalises to
    the WRONG calendar date (Monday 2020-06-01 -> Sunday 2020-05-31). The loader
    must keep the local session date instead. The other fixtures only use +00:00
    winter rows, so this BST case previously went untested.
    """
    csv = tmp_path / "oxfordmanrealizedvolatilityindices.csv"
    rows = [",Symbol,rv5,medrv"]
    for d in ("2020-06-01", "2020-06-02", "2020-06-03"):  # Mon/Tue/Wed, BST (+01:00)
        rows.append(f"{d} 00:00:00+01:00,.SPX,0.00012,0.00010")
    rows.append("2020-01-06 00:00:00+00:00,.SPX,0.00013,0.00010")  # winter (+00:00)
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    df = load_oxfordman_symbol(csv, symbol=".SPX")
    got = list(df.index.strftime("%Y-%m-%d"))
    assert got == ["2020-01-06", "2020-06-01", "2020-06-02", "2020-06-03"], got
    assert (df.index.dayofweek < 5).all()  # no BST row pushed onto a weekend


class TestAssetAliasCLI:
    """The ``--asset`` alias must come from the config, not a compile-time list.

    Regression guard for a real failure on 2026-08-30: the asset registry was
    wired into the download *dispatch* but ``--asset`` kept
    ``choices=sorted(YFINANCE_SYMBOLS)`` — a module constant evaluated when the
    parser is built, long before ``--config`` has been read. ``--asset RUT`` was
    rejected with argparse's "invalid choice: 'RUT' (choose from 'AAPL', 'GSPC',
    'TSLA', 'VIX')" even though the registry, the frame builder and the symbol
    map all resolved it fine. A static choice list over config-driven values can
    only ever be a stale copy of them.

    The lesson generalises past this flag: importing cleanly and compiling
    cleanly is not the same as running. These tests drive the parser and the
    resolver the way the command line does.
    """

    def test_parser_accepts_a_registry_alias(self):
        from src.data.ingest import _build_argparser

        args = _build_argparser().parse_args(["--asset", "RUT"])
        assert args.asset == "RUT"

    def test_parser_accepts_any_alias_and_defers_validation(self):
        """Parsing must not adjudicate aliases; ``main`` does, against the config."""
        from src.data.ingest import _build_argparser

        assert _build_argparser().parse_args(["--asset", "FTSE"]).asset == "FTSE"
        # An unknown alias parses too — and is rejected later, with a better message.
        assert _build_argparser().parse_args(["--asset", "NOPE"]).asset == "NOPE"

    def test_asset_flag_declares_no_static_choices(self):
        """Pin the cause, not just the symptom."""
        from src.data.ingest import _build_argparser

        action = next(a for a in _build_argparser()._actions
                      if "--asset" in getattr(a, "option_strings", []))
        assert action.choices is None

    def test_symbol_map_covers_every_registry_alias(self):
        from omegaconf import OmegaConf

        from src.data.ingest import yfinance_symbol_map

        cfg = OmegaConf.create({
            "assets": {"primary": "SPX", "registry": {
                "SPX": {"yfinance": "^GSPC", "cache_alias": "GSPC"},
                "RUT": {"yfinance": "^RUT", "cache_alias": "RUT"},
                "FTSE": {"yfinance": "^FTSE", "cache_alias": "FTSE"},
                "VIX": {"yfinance": "^VIX", "cache_alias": "VIX"},
            }},
        })
        m = yfinance_symbol_map(cfg)
        assert m["RUT"] == "^RUT" and m["FTSE"] == "^FTSE" and m["GSPC"] == "^GSPC"

    def test_primary_alias_comes_from_the_registry(self):
        from omegaconf import OmegaConf

        from src.data.ingest import resolve_primary_alias

        cfg = OmegaConf.create({
            "assets": {"primary": "SPX", "registry": {
                "SPX": {"yfinance": "^GSPC", "cache_alias": "GSPC"}}},
        })
        assert resolve_primary_alias(cfg) == "GSPC"
        assert resolve_primary_alias(OmegaConf.create({})) == "GSPC"
