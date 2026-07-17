"""Data ingestion CLI -- Phase 0 / Phase 1 entry point.

Two data sources:

1. **yfinance** -- daily OHLCV for ^GSPC, ^VIX, AAPL, TSLA over 2000-2024. Used
   for the regression / DL inputs and the Yang-Zhang OHLC RV sensitivity series.
2. **Oxford-Man Realized Library** -- the curated intraday 5-minute realized
   variance series. This is the dissertation's *primary* RV target.

Both write to ``data/raw/<source>/<symbol>.parquet`` with a sibling
``*.meta.json`` recording the snapshot UTC timestamp, query parameters, and the
source URL where applicable. **Existing artifacts are never silently
overwritten** -- use ``--force`` to refresh.

Usage
-----
    python -m src.data.ingest --asset GSPC                # one symbol
    python -m src.data.ingest --all                       # the four yfinance symbols
    python -m src.data.ingest --oxfordman                 # only the Oxford-Man pull
    python -m src.data.ingest --all --oxfordman --force   # refresh everything
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import load_config, repo_path
from src.utils.io import ensure_dir, write_meta
from src.utils.logging import get_logger

log = get_logger("ingest")

YFINANCE_SYMBOLS: dict[str, str] = {
    # alias -> yfinance ticker
    "GSPC": "^GSPC",
    "VIX": "^VIX",
    "AAPL": "AAPL",
    "TSLA": "TSLA",
}


# --------------------------------------------------------------------------- #
# yfinance                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class YFinanceIngestor:
    out_dir: Path
    start: str
    end: str

    def fetch(self, alias: str, symbol: str, *, force: bool = False) -> Path:
        out_path = self.out_dir / f"{alias}.parquet"
        if out_path.exists() and not force:
            log.info("[yfinance] %s already cached at %s (use --force to refresh)", alias, out_path)
            return out_path

        # Import inside fetch so test environments without network can still
        # import the module.
        import yfinance as yf

        log.info("[yfinance] downloading %s (%s) %s -> %s ...", alias, symbol, self.start, self.end)
        df = yf.download(
            symbol,
            start=self.start,
            end=self.end,
            auto_adjust=False,
            progress=False,
            actions=False,
        )
        if df is None or df.empty:
            raise RuntimeError(f"yfinance returned no rows for {symbol}")

        # Newer yfinance returns a MultiIndex when given a single ticker; flatten it.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        df.index.name = "date"

        ensure_dir(out_path.parent)
        df.to_parquet(out_path)
        write_meta(
            out_path.with_suffix(".meta.json"),
            payload={
                "source": "yfinance",
                "symbol": symbol,
                "alias": alias,
                "start": self.start,
                "end": self.end,
                "rows": int(len(df)),
                "first_date": str(df.index.min().date()) if len(df) else None,
                "last_date": str(df.index.max().date()) if len(df) else None,
            },
        )
        log.info("[yfinance] %s saved %d rows -> %s", alias, len(df), out_path)
        return out_path


# --------------------------------------------------------------------------- #
# Oxford-Man Realized Library                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class OxfordManIngestor:
    """Attempt to fetch the Oxford-Man realized-vol CSV.

    The Oxford-Man Institute closed in 2022 and the library is no longer
    actively maintained, but the canonical CSV is mirrored on the wayback
    machine and on several academic GitHub repos. We try each candidate URL in
    order; if all fail we print explicit manual-download instructions and exit
    non-zero so the user knows to drop the file in by hand.
    """

    out_dir: Path
    filename: str
    candidate_urls: list[str]

    def _try_download(self, url: str) -> bytes | None:
        try:
            log.info("[oxfordman] trying %s", url)
            resp = requests.get(url, timeout=60, allow_redirects=True)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("[oxfordman] failed: %s", exc)
            return None

        content = resp.content
        # If the response is a zip archive, extract the first CSV.
        if url.lower().endswith(".zip") or content[:2] == b"PK":
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                    if not members:
                        log.warning("[oxfordman] zip at %s contained no .csv", url)
                        return None
                    return zf.read(members[0])
            except zipfile.BadZipFile:
                log.warning("[oxfordman] response from %s was not a valid zip", url)
                return None
        return content

    def fetch(self, *, force: bool = False) -> Path | None:
        out_path = self.out_dir / self.filename
        if out_path.exists() and not force:
            log.info(
                "[oxfordman] already cached at %s (use --force to refresh)", out_path
            )
            return out_path

        ensure_dir(out_path.parent)

        for url in self.candidate_urls:
            blob = self._try_download(url)
            if blob is None:
                time.sleep(1)
                continue
            out_path.write_bytes(blob)
            write_meta(
                out_path.with_suffix(out_path.suffix + ".meta.json"),
                payload={
                    "source": "oxford-man-realized-library",
                    "url": url,
                    "filename": self.filename,
                    "bytes": len(blob),
                },
            )
            log.info("[oxfordman] saved %d bytes from %s -> %s", len(blob), url, out_path)
            return out_path

        self._print_manual_instructions(out_path)
        return None

    def _print_manual_instructions(self, expected_path: Path) -> None:
        msg = (
            "\n[oxfordman] All candidate URLs failed. The Oxford-Man Realized "
            "Library is no longer hosted at its original URL.\n"
            "Manual download options:\n"
            "  - Wayback Machine snapshots of https://realized.oxford-man.ox.ac.uk\n"
            "  - The {filename} file mirrored in several academic GitHub repos\n"
            f"Once obtained, drop the CSV at: {expected_path}\n"
            "Then re-run: python -m src.data.ingest --oxfordman\n"
        ).format(filename=self.filename)
        log.error(msg)


def load_oxfordman_symbol(csv_path: Path | str, symbol: str = ".SPX") -> pd.DataFrame:
    """Load the Oxford-Man CSV and slice out one index symbol.

    The library is a long-format CSV with columns:
        ``Symbol, ...date column..., rv5, rv10, bv, medrv, rk_parzen, ...``

    Returns a DataFrame indexed by date with realized-variance columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Oxford-Man CSV not found at {csv_path}; run `python -m src.data.ingest --oxfordman` first"
        )
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalize Symbol column name across schema variants.
    sym_col_candidates = [c for c in df.columns if c.lower() in ("symbol", "ticker", "asset")]
    if not sym_col_candidates:
        raise ValueError(f"Oxford-Man CSV has no Symbol column; columns={list(df.columns)}")
    sym_col = sym_col_candidates[0]

    # Identify the date column. Different vintages use different names; the
    # 2022 snapshot stores the date in an unnamed leading column ("Unnamed: 0").
    date_col_candidates = [
        c
        for c in df.columns
        if c.lower() in ("date", "dates", "datetime", "obs_date", "index")
        or str(c).lower().startswith("unnamed")
    ]
    if not date_col_candidates:
        # Last resort: pick the first non-symbol column whose values parse as
        # datetimes (sampled on the head for speed).
        for c in df.columns:
            if c == sym_col:
                continue
            parsed = pd.to_datetime(df[c].head(50), errors="coerce", utc=True)
            if parsed.notna().mean() > 0.8:
                date_col_candidates = [c]
                break
    if not date_col_candidates:
        raise ValueError(f"Could not locate a date column in Oxford-Man CSV; columns={list(df.columns)}")
    date_col = date_col_candidates[0]

    sub = df[df[sym_col].astype(str).str.upper() == symbol.upper()].copy()
    if sub.empty:
        available = df[sym_col].dropna().astype(str).unique().tolist()[:25]
        raise ValueError(
            f"Symbol {symbol!r} not found in Oxford-Man CSV; first 25 available: {available}"
        )

    # Oxford-Man timestamps carry the London-local session date with a +00:00
    # (GMT, winter) or +01:00 (BST, summer) offset. Converting to UTC first
    # (utc=True) drags every BST row back across midnight to the previous
    # calendar day -- mis-dating ~59% of rows and pushing ~600 onto non-trading
    # Sundays, which then break alignment with the daily yfinance return series.
    # The daily RV date is just the local calendar date, so take the leading
    # YYYY-MM-DD and ignore the intra-day time and offset entirely.
    date_str = sub[date_col].astype(str).str.slice(0, 10)
    sub[date_col] = pd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")
    sub = sub.dropna(subset=[date_col]).set_index(date_col).sort_index()
    sub = sub[~sub.index.duplicated(keep="first")]
    sub.index.name = "date"
    return sub


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.ingest",
        description=(
            "Download raw data into data/raw/. Caches per-symbol parquet files "
            "plus a *.meta.json snapshot record."
        ),
    )
    p.add_argument(
        "--asset",
        choices=sorted(YFINANCE_SYMBOLS.keys()),
        help="Pull a single yfinance asset by alias.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Pull all four yfinance symbols (^GSPC, ^VIX, AAPL, TSLA).",
    )
    p.add_argument(
        "--oxfordman",
        action="store_true",
        help="Also attempt to download the Oxford-Man Realized Library CSV.",
    )
    p.add_argument("--start", default=None, help="Override window.start from data.yaml.")
    p.add_argument("--end", default=None, help="Override window.end from data.yaml.")
    p.add_argument("--force", action="store_true", help="Refresh cached artifacts.")
    p.add_argument("--config", default="data", help="Config name under configs/ (default: data).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    cfg = load_config(args.config)

    start = args.start or cfg.window.start
    end = args.end or cfg.window.end

    yfinance_dir = ensure_dir(repo_path(cfg.paths.yfinance))
    oxfordman_dir = ensure_dir(repo_path(cfg.paths.oxfordman))

    did_anything = False

    # --- yfinance ---------------------------------------------------------- #
    yf_ing = YFinanceIngestor(out_dir=yfinance_dir, start=str(start), end=str(end))
    targets: list[tuple[str, str]] = []
    if args.all:
        targets = list(YFINANCE_SYMBOLS.items())
    elif args.asset:
        targets = [(args.asset, YFINANCE_SYMBOLS[args.asset])]
    elif not args.oxfordman:
        # Default behaviour: pull the primary asset, so the Phase 0 gate
        # ("python -m src.data.ingest downloads S&P 500 successfully") passes
        # without any flags.
        targets = [("GSPC", YFINANCE_SYMBOLS["GSPC"])]

    for alias, sym in targets:
        try:
            yf_ing.fetch(alias, sym, force=args.force)
            did_anything = True
        except Exception as exc:
            log.error("[yfinance] %s failed: %s", alias, exc)
            return 2

    # --- Oxford-Man -------------------------------------------------------- #
    if args.oxfordman:
        om = OxfordManIngestor(
            out_dir=oxfordman_dir,
            filename=cfg.oxfordman.filename,
            candidate_urls=list(cfg.oxfordman.candidate_urls),
        )
        path = om.fetch(force=args.force)
        if path is None:
            return 3
        did_anything = True

    if not did_anything:
        log.warning("nothing to do; pass --asset, --all, or --oxfordman")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
