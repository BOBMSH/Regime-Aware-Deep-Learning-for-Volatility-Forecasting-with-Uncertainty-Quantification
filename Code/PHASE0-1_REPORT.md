# Phase 0 & Phase 1 — Implementation Report

**Project:** Regime-Aware Deep Learning for Financial Volatility Forecasting with Uncertainty Quantification
**Scope of this report:** ROADMAP.md Phases 0 (Setup) and 1 (Data pipeline & EDA)
**Date:** 2026-05-15
**Status:** Both phase gates pass. Three items need a decision before Phase 2 — see [§5](#5-issues-needing-a-decision).

---

## 1. Summary

Phases 0 and 1 of the roadmap are implemented in the `Code/` directory. The code
ingests the dissertation's data, computes realized-volatility estimators,
generates leakage-free walk-forward splits, is covered by 41 unit tests, and
produces the six EDA figures plus the RV summary table required by the Phase 1
milestone.

Both gates defined in the roadmap pass:

| Gate | Roadmap criterion | Result |
|---|---|---|
| Phase 0 | `pytest -q` runs | **41 passed** |
| Phase 0 | `python -m src.data.ingest` downloads S&P 500 | ^GSPC cached (6,288 rows) |
| Phase 1 | Milestone `m01_data.md` with 6 figures + RV table + sanity check | All produced |

One real data-availability constraint surfaced during Phase 1 (the Oxford-Man
library snapshot ends Feb 2022). It is **not a code bug** — it needs a scope
decision. Details in [§5.1](#51-oxford-man-coverage-ends-2022-02-25-blocking-for-phase-2-scope).

---

## 2. What was built — Phase 0 (Setup)

### 2.1 Project structure

The `Code/` tree follows ROADMAP §4:

```
Code/
├── pyproject.toml          pinned dependencies, Ruff/Black/pytest config
├── requirements-lock.txt   full `pip freeze` (142 packages) for reproducibility
├── .gitignore              data/, results artifacts, venv, checkpoints
├── README.md               setup + run instructions
├── configs/
│   └── data.yaml           canonical Phase 1 configuration
├── data/                   raw / interim / processed   (gitignored)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── data/               ingest, realized_vol, splits, features
│   └── utils/              seeding, logging, config, io, plotting
├── tests/                  41 unit tests
├── experiments/            per-experiment run dirs (populated from Phase 2)
└── results/
    ├── figures/m01/        6 EDA figures
    ├── tables/             m01_rv_summary.csv
    └── milestones/         m01_data.md
```

### 2.2 Reproducibility utilities (`src/utils/`)

| Module | Purpose |
|---|---|
| `seeding.py` | `set_seed()` seeds Python / NumPy / PyTorch and flips cuDNN into deterministic mode (ROADMAP §9). |
| `logging.py` | `get_logger()` — idempotent stdout + optional file logging. |
| `config.py` | `load_config()` loads YAML via OmegaConf with dotlist overrides; `snapshot_config()` dumps the resolved config + UTC timestamp per run. |
| `io.py` | parquet read/write; `write_meta()` writes a sibling `*.meta.json` snapshot record so raw data is never silently overwritten. |
| `plotting.py` | single source of figure styling + `shade_crises()` (dot-com, 2008 GFC, COVID-19, 2022 inflation bands). |

### 2.3 Dependency management

`pyproject.toml` pins all libraries with lower bounds; `requirements-lock.txt`
records the exact resolved versions. The package installs editable as `src`
(`pip install -e .[dev]`), with an optional `[deep]` extra for PyTorch (deferred
until Phase 3).

---

## 3. What was built — Phase 1 (Data pipeline & EDA)

### 3.1 Data ingestion (`src/data/ingest.py`)

A CLI with two data sources:

- **yfinance** — daily OHLCV for `^GSPC`, `^VIX`, `AAPL`, `TSLA` over 2000–2024.
  Each pull writes `data/raw/yfinance/<alias>.parquet` plus a `*.meta.json`
  recording the snapshot UTC time, row count and date range. Existing files are
  never overwritten without `--force`.
- **Oxford-Man Realized Library** — the curated intraday 5-minute realized
  variance series (the dissertation's *primary* RV target). The original host
  was discontinued when the Oxford-Man Institute closed in 2022; the ingestor
  tries a list of candidate URLs and, on total failure, prints explicit
  manual-download instructions instead of crashing.

`load_oxfordman_symbol()` parses the long-format library CSV and slices out one
index symbol (`.SPX` for the S&P 500).

CLI usage:

```powershell
python -m src.data.ingest --asset GSPC      # one symbol
python -m src.data.ingest --all             # all four yfinance symbols
python -m src.data.ingest --oxfordman       # Oxford-Man RV download
python -m src.data.ingest --all --oxfordman --force
```

### 3.2 Realized-volatility estimators (`src/data/realized_vol.py`)

| Function | Role |
|---|---|
| `yang_zhang_rv()` | Yang-Zhang (2000) OHLC realized-variance estimator (rolling window, default 21 days). The Phase 1 **sensitivity** estimator and documented fallback target. |
| `squared_returns_rv()` | Squared-return RV. **Diagnostic only** — noisy, never the headline target. |
| `parkinson_rv()` | Parkinson (1980) high-low estimator — internal helper. |
| `realized_vol_from_intraday_rv()` | Converts Oxford-Man daily realized *variance* (`rv5`) to realized *volatility* (its square root — the Andersen-Bollerslev-Diebold-Labys target). |
| `log_returns()` | Daily log returns. |

All estimators return daily-scale volatility by default; `squared=True` returns
variance, `annualize=True` scales by 252.

### 3.3 Walk-forward splits (`src/data/splits.py`)

- `fixed_three_way_split()` — the high-level train / validation / test partition
  (train ≤2019, val 2020–2021, test 2022–2024).
- `walk_forward_folds()` — anchored or rolling walk-forward fold generator.
  **Leakage discipline (ROADMAP §1.3) is enforced structurally**: each fold's
  training window ends strictly before its prediction window begins; refit
  cadence is configurable (monthly / 21 trading days by default).
- `SplitConfig` / `Fold` — frozen dataclasses; `SplitConfig` validates that the
  split bounds are monotone and the scheme is recognised.

### 3.4 Features (`src/data/features.py`)

`har_lagged_rv()` builds the daily / weekly / monthly lagged-RV features for the
HAR-RV baseline (Corsi 2009), each `shift(1)`-ed so they are leakage-safe.
`build_phase1_features()` assembles the minimal Phase 1 feature frame. Richer
feature engineering (regime posteriors etc.) is deferred to Phase 5.

### 3.5 Tests (`tests/`, 41 tests)

| File | Coverage |
|---|---|
| `test_realized_vol.py` | RV estimator validation, non-negativity, annualisation scaling, variance/volatility consistency, recovery of true σ on synthetic GBM. |
| `test_splits.py` | leakage-free fold boundaries, anchored vs rolling training windows, contiguous non-overlapping prediction coverage, min-train-size guard, duplicate-date rejection. |
| `test_ingest.py` | Oxford-Man CSV parsing against the real schema quirks (unnamed date column, dotted index symbols, case-insensitive matching). |
| `test_smoke.py` | package imports, `set_seed` runs without PyTorch. |

### 3.6 EDA notebook + milestone

`notebooks/01_data_exploration.ipynb` was executed end-to-end and produces:

- `results/figures/m01/fig01..fig06.png` — log returns with crisis shading; the
  three-estimator RV series; return distribution vs Gaussian; ACF of RV
  (long-memory check); RV vs |returns|; Oxford-Man vs Yang-Zhang sanity overlay.
- `results/tables/m01_rv_summary.csv` — mean/median/std/p99/n_obs per estimator.
- `results/milestones/m01_data.md` — the Phase 1 milestone note, with all gate
  boxes checked and a written narrative.

**Key Phase 1 findings:**
- The three RV estimators agree on level to within ~7% on the mean.
- RV shows the expected long-memory signature (ACF positive past 100 lags) —
  this empirically motivates the HAR-RV baseline.
- Returns are markedly fat-tailed vs Gaussian.
- Oxford-Man intraday RV vs 21-day Yang-Zhang: Pearson **r ≈ 0.76**, median
  level ratio **0.94** over 4,857 common days. Level agreement is strong; the
  moderate correlation reflects YZ's 21-day smoothing vs the daily intraday
  estimate (expected, not an error).

---

## 4. Gate verification (commands run)

```
pytest -q                                  -> 41 passed
python -m src.data.ingest --asset GSPC      -> 6,288 rows cached
python -m src.data.ingest --all --oxfordman -> 4 assets + Oxford-Man .SPX (5,552 rows)
jupyter nbconvert --execute 01_data_exploration.ipynb -> 6 figures + table
```

---

## 5. Issues needing a decision

### 5.1 Oxford-Man coverage ends 2022-02-25  *(blocking for Phase 2 scope)*

**The issue.** The Oxford-Man Realized Library was discontinued when the
institute closed in 2022. The only recoverable copy is a Wayback Machine
snapshot captured 2022-03-01; the `.SPX` series in it runs **2000-01-03 →
2022-02-25**. The roadmap's headline test window is **2022–2024**, so the
*intraday* RV target cannot cover the test segment.

**Why it matters.** RQ1–RQ3 are all evaluated on the 2022–2024 test set. If the
primary target only exists to Feb 2022, either the target or the test window has
to change.

**Options (ROADMAP risk #9 anticipates this):**

1. **Use Yang-Zhang OHLC RV as the test-window target** (available 2000–2024 via
   yfinance); keep Oxford-Man intraday RV as the train/validation target and the
   in-sample cross-estimator check. *Preserves the 2022 inflation stressor;
   consistent with §1.1 naming Yang-Zhang the documented fallback.* **— Recommended.**
2. **Shift the test window earlier** (e.g. test 2020 → Feb 2022) so intraday RV
   covers it — but this loses the 2022 inflation/rates shock as a test stressor,
   which §1.4 / Phase 8 specifically value.
3. **Source a newer realized-variance vendor** for 2022 onward — out of scope
   per ROADMAP §6 ("no bespoke intraday pipeline").

**Decision needed:** confirm option 1 (or pick another) before Phase 2 starts.
Recorded as an open issue in `results/milestones/m01_data.md`.

### 5.2 Test split currently spans a target that two estimators cover differently

If option 5.1.1 is chosen, the dissertation will report **train/val on intraday
RV, test on Yang-Zhang RV**. That is defensible (Patton 2011 proxy-robust losses
justify a noisy proxy) but it must be stated explicitly in the Methods chapter,
and the QLIKE/MSE comparison must be run estimator-consistently within each
segment. **Decision needed:** confirm this framing is acceptable to the supervisor.

### 5.3 Open roadmap questions still unanswered

ROADMAP §10 lists four open questions; none block Phase 1 but two should be
settled soon:

- **Time period** — currently 2000-01-01 → 2024-12-31 (default kept).
- **Test split** — currently train ≤2019 / val 2020–2021 / test 2022–2024 (kept).
- **Supervisor cadence** — needed to slot milestone checkpoints.
- **Nystrup implementation route** — only relevant at Phase 4; can wait.

---

## 6. Issues to fix / known limitations

None are blocking; listed for transparency.

| # | Item | Severity | Notes |
|---|---|---|---|
| 1 | Oxford-Man download depends on a Wayback Machine URL | Low | The archived URL is pinned in `configs/data.yaml`. If it ever 404s, the CSV is already cached in `data/raw/oxfordman/` and the ingestor skips re-download. Raw data should be committed to a private backup or kept safe — it is gitignored. |
| 2 | yfinance is a moving API | Low | Pulls are cached to parquet with `*.meta.json`; per ROADMAP §9 / risk #10, do **not** re-download once gathered. `TSLA` only has 3,651 rows (IPO'd 2010) — expected, not a bug. |
| 3 | Yang-Zhang window is fixed at 21 in the EDA | Low | The sanity-check correlation (r≈0.76) is partly a smoothing artefact of the 21-day window. A daily-vs-daily comparison (shorter window, or rolling intraday RV) could be added as a Phase 1 robustness extra if the supervisor wants a tighter cross-check. |
| 4 | `torch` not yet installed | None | Intentional — the `[deep]` extra is deferred to Phase 3. `set_seed()` already handles torch's absence gracefully. |
| 5 | EDA notebook needs the registered Jupyter kernel | Low | A kernel named `regime-vol` was registered for the project venv. The notebook also bootstraps `sys.path` so it runs under a bare kernel too. |

---

## 7. Environment notes

- A dedicated virtualenv was created at `Code/.venv` (Python 3.12.1). The
  interpreter that was active in the workspace belonged to an unrelated project
  (`DeepfakeDetectionApp`) — it was **not** used.
- To reproduce:
  ```powershell
  cd Code
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -e .[dev]
  pytest -q
  ```
- `requirements-lock.txt` should be refreshed (`pip freeze > requirements-lock.txt`)
  and committed before each milestone git tag.

---

## 8. Suggested next steps

1. **Decide §5.1** (target vs test window) — this is the only thing blocking a
   clean Phase 2 start.
2. Commit and tag the Phase 1 milestone (`m01-data`) per ROADMAP §9 git hygiene.
3. Begin Phase 2: GARCH(1,1) / EGARCH via `arch`, HAR-RV, and the rolling-OOS
   evaluation harness — the `walk_forward_folds` generator and the metrics
   module are the natural first build.
