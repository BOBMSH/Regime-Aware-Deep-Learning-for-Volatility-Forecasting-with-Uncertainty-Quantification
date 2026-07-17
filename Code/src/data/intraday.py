"""Extend the Oxford-Man realized-variance series past its 2022-02-25 cutoff.

Motivation
----------
The Oxford-Man Realized Library stopped being updated when the institute closed
in early 2022, so the recoverable ``.SPX`` snapshot ends 2022-02-25 -- but the
maths behind ``rv5`` is fully open (Andersen-Bollerslev-Diebold-Labys 2003: the
sum of squared 5-minute intraday returns). Rather than compromise the 2022-2024
test window (which carries the inflation/rates shock) or the literature review's
commitment to intraday RV, we can compute the missing 2022-2024 realized variance
ourselves from an intraday price feed and splice it onto the Oxford-Man series.

Why this is gated, not automatic
--------------------------------
Splicing two differently-produced series is only safe if they measure the *same
thing on the same scale*. If our reconstruction sits at a different level than
Oxford-Man's, the splice injects a discontinuity **exactly at the start of the
test set** -- the worst possible place. So this module never splices blind:

1. ``compute_daily_rv`` builds our RV from intraday bars using the same open-to-
   close, 5-minute convention as Oxford-Man.
2. ``validate_overlap`` compares our RV against Oxford-Man's on the period they
   both cover (e.g. 2015-2021) and reports correlation and level ratio.
3. ``splice`` refuses to run unless the overlap check passes a tolerance, and it
   can optionally *level-adjust* our segment by the median overlap ratio so the
   two join seamlessly.

Data source (a user action, not a code action)
----------------------------------------------
The one thing this module cannot do for you is obtain 2022-2024 intraday data.
``yfinance`` only serves ~60 days of intraday history, and the S&P 500 *index*
(^GSPC) is not traded, so there is no free multi-year 5-minute index feed. In
practice you supply one of:

* an intraday export for a matching instrument (SPY ETF is the usual proxy;
  note it is a different instrument, which is itself validated by the overlap
  check), or
* a licensed index/futures feed (Polygon, Databento, FirstRate Data, AlgoSeek,
  Refinitiv Tick History, ...).

Point ``--intraday-file`` at a CSV/parquet with a timestamp index and a price
column and this module does the rest.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ingest import load_oxfordman_symbol
from src.data.realized_vol import realized_variance_intraday
from src.utils.config import load_config, repo_path
from src.utils.io import ensure_dir, to_parquet, write_meta
from src.utils.logging import get_logger

log = get_logger("intraday")

# Oxford-Man .SPX snapshot ends here; anything after is what we reconstruct.
OXFORDMAN_CUTOFF = pd.Timestamp("2022-02-25")


# --------------------------------------------------------------------------- #
# Loading intraday bars                                                       #
# --------------------------------------------------------------------------- #


def load_intraday_bars(
    path: Path | str,
    *,
    timestamp_col: str | None = None,
    price_col: str | None = None,
) -> pd.Series:
    """Load a user-supplied intraday file into a price Series indexed by timestamp.

    Accepts ``.csv``/``.parquet``. Column names are auto-detected when not given:
    the timestamp column is the first datetime-like column (or the index), and the
    price column prefers ``close``/``price``/``last``/``adj_close``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"intraday file not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Timestamp.
    if timestamp_col is None:
        cand = [c for c in df.columns if c.lower() in
                ("timestamp", "datetime", "date", "time", "dt")]
        if cand:
            timestamp_col = cand[0]
    if timestamp_col is not None and timestamp_col in df.columns:
        idx = pd.to_datetime(df[timestamp_col], utc=False, errors="coerce")
        df = df.drop(columns=[timestamp_col])
    else:
        idx = pd.to_datetime(df.index, errors="coerce")
    df.index = pd.DatetimeIndex(idx)

    # Price.
    if price_col is None:
        for c in ("close", "price", "last", "adj_close", "Close", "PRICE"):
            if c in df.columns:
                price_col = c
                break
    if price_col is None or price_col not in df.columns:
        raise ValueError(
            f"could not find a price column in {list(df.columns)}; pass price_col="
        )
    s = df[price_col].astype(float)
    s = s[s.index.notna()].sort_index()
    s.name = "price"
    return s


def compute_daily_rv(prices: pd.Series, *, resample_freq: str = "5min") -> pd.Series:
    """Our open-to-close daily realized *variance* from intraday bars (rv5)."""
    return realized_variance_intraday(prices, resample_freq=resample_freq, squared=True)


# --------------------------------------------------------------------------- #
# Overlap validation -- the splice gate                                       #
# --------------------------------------------------------------------------- #


@dataclass
class OverlapReport:
    n: int
    pearson: float
    spearman: float
    median_ratio: float   # our_rv / oxfordman_rv, median over the overlap
    mean_abs_log_ratio: float
    start: pd.Timestamp | None
    end: pd.Timestamp | None

    def passes(self, *, min_pearson: float = 0.9, ratio_tol: float = 0.15) -> bool:
        """A splice is trustworthy if the two series are highly correlated and
        sit at the same level. Defaults: Pearson >= 0.90 and median level ratio
        within +/-15% of 1.0."""
        if self.n < 60:
            return False
        return (self.pearson >= min_pearson) and (abs(self.median_ratio - 1.0) <= ratio_tol)


def validate_overlap(our_rv: pd.Series, oxfordman_rv: pd.Series) -> OverlapReport:
    """Compare our reconstruction against Oxford-Man on their common dates."""
    joined = pd.concat(
        [our_rv.rename("ours"), oxfordman_rv.rename("om")], axis=1, join="inner"
    ).dropna()
    joined = joined[(joined > 0).all(axis=1)]
    if joined.empty:
        return OverlapReport(0, float("nan"), float("nan"), float("nan"),
                             float("nan"), None, None)
    ratio = joined["ours"] / joined["om"]
    return OverlapReport(
        n=int(len(joined)),
        pearson=float(joined["ours"].corr(joined["om"])),
        spearman=float(joined["ours"].corr(joined["om"], method="spearman")),
        median_ratio=float(ratio.median()),
        mean_abs_log_ratio=float(np.abs(np.log(ratio)).mean()),
        start=joined.index.min(),
        end=joined.index.max(),
    )


def splice(
    oxfordman_rv: pd.Series,
    our_rv: pd.Series,
    *,
    cutoff: pd.Timestamp = OXFORDMAN_CUTOFF,
    level_adjust: bool = True,
    report: OverlapReport | None = None,
    force: bool = False,
) -> pd.Series:
    """Append our post-cutoff RV to the Oxford-Man series.

    Parameters
    ----------
    oxfordman_rv : Oxford-Man ``rv5`` (variance) up to the cutoff.
    our_rv : our reconstructed ``rv5`` (variance); only its post-cutoff portion
        is used.
    cutoff : last Oxford-Man date to keep; our data takes over strictly after.
    level_adjust : if True, multiply our segment by ``1 / median_ratio`` from the
        overlap report so the two series share a level (removes any residual
        instrument/methodology offset at the join).
    report : an :class:`OverlapReport`; required unless ``force``. The splice
        raises if the report does not pass its tolerance.
    force : bypass the validation gate (use only with a documented reason).
    """
    if report is None and not force:
        raise ValueError("splice requires an OverlapReport (or force=True)")
    if report is not None and not report.passes() and not force:
        raise RuntimeError(
            f"overlap validation failed (pearson={report.pearson:.3f}, "
            f"median_ratio={report.median_ratio:.3f}); refusing to splice. "
            "Fix the intraday source or pass force=True with a documented rationale."
        )

    head = oxfordman_rv.loc[:cutoff]
    tail = our_rv.loc[our_rv.index > cutoff].copy()
    if level_adjust and report is not None and np.isfinite(report.median_ratio):
        tail = tail / report.median_ratio
    spliced = pd.concat([head, tail]).sort_index()
    spliced = spliced[~spliced.index.duplicated(keep="first")]
    spliced.index.name = "date"
    return spliced.rename("rv5")


# --------------------------------------------------------------------------- #
# CLI driver                                                                  #
# --------------------------------------------------------------------------- #


def _print_no_data_instructions(expected_dir: Path) -> None:
    log.error(
        "\n[intraday] No --intraday-file supplied and none cached.\n"
        "To extend Oxford-Man RV to 2022-2024, provide an intraday price feed for\n"
        "a matching instrument (SPY ETF is the usual free-ish proxy; a licensed\n"
        "index/futures feed is cleaner):\n"
        "  - CSV or parquet, with a timestamp column and a price/close column\n"
        "  - ideally covering an overlap with Oxford-Man (>= 2015) so the splice\n"
        "    can be validated, plus 2022-01-01 -> 2024-12-31\n"
        f"  - drop it anywhere and pass --intraday-file <path>\n"
        f"  - the spliced series is written to {expected_dir}\n"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m src.data.intraday",
        description="Compute 2022-2024 intraday RV and splice onto Oxford-Man .SPX.",
    )
    p.add_argument("--intraday-file", default=None, help="CSV/parquet of intraday prices.")
    p.add_argument("--price-col", default=None, help="Price column name (auto if omitted).")
    p.add_argument("--timestamp-col", default=None, help="Timestamp column (auto if omitted).")
    p.add_argument("--resample-freq", default="5min", help="Sampling grid (default 5min).")
    p.add_argument("--symbol", default=".SPX", help="Oxford-Man symbol (default .SPX).")
    p.add_argument("--no-level-adjust", action="store_true", help="Disable level matching.")
    p.add_argument("--force", action="store_true", help="Splice even if validation fails.")
    p.add_argument("--config", default="data")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    om_csv = repo_path(cfg.paths.oxfordman, cfg.oxfordman.filename)
    interim_dir = ensure_dir(repo_path(cfg.paths.interim))
    out_path = interim_dir / "spx_rv5_spliced.parquet"

    om = load_oxfordman_symbol(om_csv, args.symbol)["rv5"].astype(float)
    om = om[om > 0]
    log.info("[intraday] Oxford-Man %s: %s -> %s (%d days)", args.symbol,
             om.index.min().date(), om.index.max().date(), len(om))

    if not args.intraday_file:
        _print_no_data_instructions(interim_dir)
        return 3

    prices = load_intraday_bars(
        args.intraday_file, timestamp_col=args.timestamp_col, price_col=args.price_col
    )
    log.info("[intraday] loaded %d intraday rows %s -> %s", len(prices),
             prices.index.min(), prices.index.max())

    our_rv = compute_daily_rv(prices, resample_freq=args.resample_freq)
    log.info("[intraday] reconstructed rv5 for %d days %s -> %s", len(our_rv),
             our_rv.index.min().date(), our_rv.index.max().date())

    rep = validate_overlap(our_rv, om)
    log.info(
        "[intraday] overlap %s..%s n=%d | pearson=%.3f spearman=%.3f median_ratio=%.3f",
        rep.start.date() if rep.start is not None else None,
        rep.end.date() if rep.end is not None else None,
        rep.n, rep.pearson, rep.spearman, rep.median_ratio,
    )
    if not rep.passes() and not args.force:
        log.error("[intraday] overlap validation FAILED; not splicing. See --force.")
        return 4

    spliced = splice(
        om, our_rv, level_adjust=not args.no_level_adjust, report=rep, force=args.force
    )
    to_parquet(spliced.to_frame(), out_path)
    write_meta(
        out_path.with_suffix(".meta.json"),
        payload={
            "source": "oxfordman + reconstructed intraday rv5",
            "symbol": args.symbol,
            "intraday_file": str(args.intraday_file),
            "resample_freq": args.resample_freq,
            "cutoff": str(OXFORDMAN_CUTOFF.date()),
            "overlap_n": rep.n,
            "overlap_pearson": rep.pearson,
            "overlap_median_ratio": rep.median_ratio,
            "level_adjusted": not args.no_level_adjust,
            "rows": int(len(spliced)),
            "first_date": str(spliced.index.min().date()),
            "last_date": str(spliced.index.max().date()),
        },
    )
    log.info("[intraday] spliced series (%d days) -> %s", len(spliced), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
