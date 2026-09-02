# Regime-Aware Deep Learning for Financial Volatility Forecasting with Uncertainty Quantification

Code for the MSc dissertation of the same name (University of Warwick, WMG, MSc Applied
Artificial Intelligence). The plan-of-record — every methodological decision, its reasoning, and
the dated changelog — is [../ROADMAP.md](../ROADMAP.md). This file is the **reproduction guide**:
what the study needs, how to run it, what it writes, and how to check that what it wrote is what
the dissertation reports.

---

## 1. What the study does

Nine phases, each producing a milestone note under `results/milestones/`:

| Phase | What it produces | Entry point |
|---|---|---|
| 1 | Data ingestion and the realised-variance target | `src.data.ingest` |
| 2 | Econometric baselines: GARCH(1,1), EGARCH, HAR-RV, RW-RV | `src.experiments.run_econometric` |
| 3 | LSTM baselines + the RV-only and VIX ablations | `src.experiments.run_lstm` |
| 4 | Regime identification: Baum-Welch HMM and the jump-penalised model | `src.experiments.run_regimes` |
| 5 | Regime-conditioned networks (feature and mixture-of-experts) | `src.experiments.run_regime_lstm` |
| 6 | Uncertainty quantification: MC-dropout and a quantile head | `src.experiments.run_uq` |
| 7 | The combined regime × uncertainty model, and the master table | `src.experiments.run_combined` |
| 8 | RQ4 cross-asset transfer, plus two sensitivities | `src.experiments.run_rq4` |
| 9 | The reproducibility gate | `src.experiments.verify_claims` |

Supporting reports: `report_gw` (Giacomini-White), `report_dm_diagnostics` (DM regularity),
`report_coverage` (interval backtests). These read parquets only and are fast.

---

## 2. Data the method requires

The methodology is specified against a **data interface**, not a source (Chapter 3 §3.2), so any
series meeting the requirements below drives the pipeline unchanged.

| Field | Type | Required | Constraint | Purpose |
|---|---|---|---|---|
| `date` | date | yes | unique, ascending, trading days | join key across sources |
| intraday prices or a realised measure | float | yes | fixed documented grid; strictly positive | the forecast target |
| `close` | float | yes | positive | close-to-close return |
| `open` | float | optional | positive | open-to-close return, matched to a session-only target |
| implied volatility | float | optional | positive; **same market as the target** | the auxiliary-feature ablation |

Minimum history is `22 + L` trading days of warm-up (the monthly HAR aggregate plus the network
lookback) before the training window. Rows missing a required field are dropped, never imputed.

**What this study actually used.** Oxford-Man Realized Library `rv5` (5-minute realised variance)
for `.SPX`, `.RUT` and `.FTSE`, 2000-01-03 → 2022-02-25, with daily OHLC and the CBOE VIX from
Yahoo Finance. `configs/data.yaml`'s `assets.registry` is the single source of truth mapping each
asset to its yfinance ticker, its local cache alias, its Oxford-Man symbol, its session, and
whether the VIX is a legitimate input for it (it is not, for the FTSE — the CBOE VIX prices S&P 500
options, and the data path refuses the join rather than leaving it to be noticed in a table).

---

## 3. Setup

```powershell
# from the Code/ directory
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,deep]"          # `deep` brings PyTorch
```

Exact versions behind the committed results are in `requirements-lock.txt` — Python 3.12.1,
torch 2.13.0+cpu, numpy 2.4.4, pandas 2.3.3, scipy 1.17.1, arch 8.0.0, hmmlearn 0.3.3,
statsmodels 0.14.6, scikit-learn 1.8.0. **No GPU is required**; every model in the study trains on
CPU in minutes.

Fetch the data (needs network):

```powershell
python -m src.data.ingest --all          # daily OHLC for every registry asset + the VIX
python -m src.data.ingest --oxfordman    # the realised-measure library
```

If the Oxford-Man host is unreachable the CLI prints manual-download instructions and the path it
expects. Every raw pull writes a sibling `*.meta.json` recording the snapshot date, never
overwritten.

---

## 4. Reproducing the reported results

One command regenerates everything in a single interpreter, in dependency order, and runs both
gates:

```powershell
.\rerun_all.ps1
```

Budget over an hour on CPU. Useful switches: `-NoSweep` (skip the Phase-3 hyperparameter search),
`-NoPhase8` (primary asset only), `-NoSensitivities` (skip log-HAR and the refit run, the longest
single step), `-SkipTests`. Per-step logs land in `results/logs/rerun-<timestamp>/`.

To run one phase at a time instead, follow the table in §1 top to bottom. Phase 5 consumes Phase 4's
parquet and Phase 7 consumes Phase 5's, so order matters; each runner states its inputs in its module
docstring.

### The two gates

Both run at the end of `rerun_all.ps1` and both must pass.

1. **Provenance.** One interpreter produced every artefact, and the Phase-7 combined model really is
   the Phase-5 model with dropout switched on at inference (checked to machine precision).
2. **Values** — `python -m src.experiments.verify_claims`. Every published point metric, interval
   metric, per-regime loss and sample size is **recomputed from the prediction parquets** by code
   that imports nothing from `src.evaluation`, then compared with the table that published it. The
   duplication of the formulas is deliberate: calling the project's own functions would check that a
   CSV matches what those functions return today, and agree enthusiastically with a regression in
   the functions themselves. Exit code 0 means every value re-derives. The detail lands in
   `results/tables/m09_verification.csv` and `results/milestones/m09_verification.md`.

Test statistics (DM, GW, MCS, Kupiec, Christoffersen, DQ) are not recomputed by the gate — their
values depend on HAC bandwidth rules, finite-sample corrections and bootstrap draws that are this
project's own conventions, so a second implementation would test agreement with itself. They are
covered by `tests/test_significance.py` against closed-form cases.

---

## 5. What gets written

```
results/
├── predictions/   one parquet per phase per profile — the primary artefacts;
│                  every published number is derived from these
├── tables/        every CSV quoted in the dissertation
├── figures/       PNG (embedded in the document) + a PDF vector companion
├── milestones/    the generated note for each phase
└── logs/          per-step logs from rerun_all.ps1
experiments/<phase>/run_<utc-timestamp>/config.yaml
                   the fully-resolved config for that run, stamped with the
                   interpreter and library versions that produced it
```

Prediction parquets and result tables are committed; figures and logs are not.

---

## 6. Configuration

`configs/` holds one YAML per phase, plus variants. Each carries the reasoning for its settings in
comments — they are documentation, not just values.

| Config | What it runs |
|---|---|
| `data.yaml` | paths, sample window, and the **asset registry** |
| `econometric.yaml` | the four baselines on the primary asset |
| `lstm_baseline.yaml` | the LSTM, its ablations, and the hyperparameter sweep grid |
| `hmm.yaml` | both regime estimators, the pre-registered λ grid and selection rule |
| `regime_lstm.yaml` / `_hmm.yaml` | regime-conditioned networks; `_hmm` is the estimator-robustness comparator |
| `uq.yaml` / `uq_hmm.yaml` | MC-dropout and the quantile head |
| `combined.yaml` / `combined_hmm.yaml` | the combined model |
| `*_rq4.yaml` | the Phase-8 cross-asset chain (`.RUT`, `.FTSE`) |
| `lstm_baseline_refit.yaml` | the refit-cadence sensitivity |
| `econometric_loghar.yaml` | HAR estimated in logs |

A profile inside a config names its asset, target and split dates. A config with more than one
profile must give `paths.milestone_file` a `{profile}` placeholder — the code refuses the omission
rather than letting one note overwrite another.

---

## 7. Reproducibility contract

- Every stochastic component is seeded through `src.utils.seeding.set_seed`, including the dropout
  masks that generate the intervals.
- Every dependency is pinned in `requirements-lock.txt`.
- One script regenerates every artefact in dependency order.
- Each run dumps its resolved config with an `_environment` stamp; rebuilding an artefact under a
  different interpreter was found to disagree materially with the committed one, which is why this
  is enforced rather than assumed.
- `results/` is the source of truth. No number in the dissertation comes from a notebook.

## 8. Tests

```powershell
pytest -q
```

Over 400 tests covering the realised-variance construction, the splits and walk-forward engine,
every metric, the regime estimators, the significance machinery, the timing conventions that decide
the per-regime results, and the milestone generators' prose.

## License

MIT — see `../LICENSE`.
