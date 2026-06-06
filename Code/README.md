# Regime-Aware Deep Learning for Volatility Forecasting

Code for the MSc dissertation **"Regime-Aware Deep Learning for Financial Volatility Forecasting
with Uncertainty Quantification"** (University of Warwick, Applied AI). The full plan-of-record lives
in [../ROADMAP.md](../ROADMAP.md); this README only covers how to run the code.

## Layout

```
Code/
├── configs/          experiment YAML configs
├── data/             raw / interim / processed datasets (gitignored)
├── notebooks/        exploratory notebooks (not used for final results)
├── results/          milestone notes, figures, tables, predictions
├── src/              production code (data, models, training, eval, utils)
├── tests/            pytest unit tests
└── pyproject.toml
```

## Setup

```powershell
# from the Code/ directory
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

If you want the PyTorch-backed models (later phases) install the `deep` extra as well:

```powershell
pip install -e .[dev,deep]
```

## Phase 0 — gate checks

```powershell
pytest -q
python -m src.data.ingest --asset GSPC --start 2000-01-01 --end 2024-12-31
```

Both must succeed before moving to Phase 1 modelling work.

## Phase 1 — data pipeline

The ingestion CLI pulls daily OHLC for the four assets used in the study and (optionally) the
Oxford-Man Realized Library snapshot.

```powershell
# yfinance: pulls ^GSPC, ^VIX, AAPL, TSLA into data/raw/yfinance/
python -m src.data.ingest --all

# Oxford-Man: attempts the curated RV download.
# If the upstream host is unreachable, the CLI prints manual-download instructions
# and points to data/raw/oxfordman/oxfordmanrealizedvolatilityindices.csv as the
# expected location.
python -m src.data.ingest --oxfordman
```

Every raw pull writes a sibling `*.meta.json` recording the snapshot date — never overwritten on
subsequent runs (per §9 of the roadmap).

## Reproducibility contract

- Pinned deps in `pyproject.toml`; run `pip freeze > requirements-lock.txt` before each milestone tag.
- Seed all experiments via `src.utils.seeding.set_seed`.
- Configs live in `configs/`; resolved configs are dumped per run to
  `experiments/<name>/run_<timestamp>/config.yaml`.
- Tests in `tests/` cover RV computations, splits, and metrics.

## License

MIT (see `../LICENSE`).
