# MSc Dissertation Roadmap
## Regime-Aware Deep Learning for Financial Volatility Forecasting with Uncertainty Quantification

**Author:** Bob (University of Warwick, MSc Applied AI)
**Last updated:** 2026-05-09
**Target submission window:** ~12 weeks (≈ early August 2026)
**Compute:** Local machine with GPU
**Stack:** Python, PyTorch, pandas, numpy, scikit-learn, statsmodels, `arch`, `hmmlearn`

---

## 0. Executive summary

You are building a comparative study, not a product. The dissertation's contribution is methodological clarity: showing — under a fair, leakage-free, rolling-window protocol — whether (a) deep learning beats classical econometrics, (b) regime-awareness adds signal on top of that, and (c) uncertainty quantification produces predictions whose intervals are actually well-calibrated. The deliverable is a defensible answer to those three questions plus a writeup that an examiner can replicate.

In 12 weeks with one researcher you cannot do everything. The plan below is built around an MVP-first build order: every phase produces a *publishable-grade* artifact (numbers, plots, a milestone note) before moving on. Optional items are clearly flagged and only attempted if earlier phases finish on time.

---

## 1. Critical methodological calls (read first)

These are decisions I am recommending up front because they will materially affect both the work and the defense. None are final — push back if you disagree.

### 1.1 Daily data → "realized volatility proxy", not true RV
True realized volatility (RV) is computed from *intraday* returns (e.g. 5-minute, summed over the day). With daily data only, you are computing one of:

- **Rolling-window standard deviation of log returns** (most common as proxy)
- **Squared daily log returns** as a single-day RV estimator (very noisy)
- **Range-based estimators** (Parkinson, Garman-Klass, Yang-Zhang) using OHLC

**Recommendation:** Use Yang-Zhang or Garman-Klass as your primary RV proxy (they exploit OHLC and are dramatically less noisy than squared returns), and a 21-day rolling σ̂ as a smoothed reference series. Be explicit in your writeup that your target is a *proxy* for RV, cite the canonical justification (Andersen & Bollerslev 1998 acknowledged squared returns are unbiased but noisy; Patton 2011 shows MSE/QLIKE remain consistent under noisy proxies), and note the limitation honestly.

### 1.2 Add HAR-RV as an econometric baseline
GARCH(1,1) is the right baseline for *return* volatility, but for *realized* volatility forecasting the de-facto literature baseline is **HAR-RV** (Corsi 2009). It is essentially a linear regression on daily, weekly, and monthly lagged RV — trivially cheap, and it reliably beats GARCH on RV targets. Without HAR you will be asked in the viva why you didn't include it.

**Recommendation:** Add HAR-RV alongside GARCH(1,1). Drop EGARCH unless time permits (it adds asymmetry but is rarely the winning baseline on RV).

### 1.3 Walk-forward, anchored, with strict leakage discipline
The single biggest threat to dissertation credibility in this area is data leakage. Three rules to enforce in code:

1. **Anchored walk-forward** evaluation: train on `[t0, t]`, predict `t+1`, then expand the window. (Or rolling-fixed-window — pick one and document.) Refit every N days (e.g. monthly) for cost.
2. **HMM regimes must be re-fit on training data only** at each step. A full-sample HMM fit looks at the test set and is leakage. This is one of the easiest mistakes to make.
3. **All scaling/normalization parameters fit on training only.** Use sklearn-style fit/transform discipline.

### 1.4 Single primary asset, two robustness assets
Trying to do regime + UQ + 3 deep models × 4 assets in 12 weeks is not realistic. Make S&P 500 (^GSPC) the primary asset where you run *every* experiment. Then run a slimmed robustness check (baselines + best DL model only) on AAPL and TSLA to show your conclusions generalize. VIX is a *feature*, not an additional target.

### 1.5 Pick one UQ method as primary, one as comparison
MC Dropout *and* Quantile Regression *and* Deep Ensembles is too many. Recommend:

- **Primary:** MC Dropout LSTM (cleanest narrative; one architecture, two outputs)
- **Comparison:** Quantile Regression LSTM with pinball loss (gives you a different philosophical stance on UQ — frequentist/distribution-free vs Bayesian-flavored)

Drop deep ensembles unless you finish phases 1–4 ahead of schedule.

---

## 2. Formal research questions

Pin these in the dissertation; they're what every experiment must answer.

**RQ1 — Performance.** Do deep learning models (LSTM) outperform econometric baselines (GARCH, HAR-RV) for daily realized-volatility-proxy forecasting on S&P 500, under leakage-free walk-forward evaluation, as measured by MSE / MAE / QLIKE?

**RQ2 — Regime-awareness.** Does conditioning forecasts on latent market regimes (HMM-detected) measurably improve forecast accuracy versus regime-agnostic equivalents, and is the improvement concentrated in transitional or crisis periods?

**RQ3 — Uncertainty calibration.** Do uncertainty-aware models (MC Dropout, quantile regression) produce prediction intervals with empirical coverage close to nominal, and how does coverage degrade across regimes?

**Stretch RQ4 — Generalization.** Do conclusions from RQ1–3 hold on AAPL and TSLA, or are they index-specific?

---

## 3. Twelve-week phase plan

Each phase ends with a **milestone artifact**: a markdown note in `results/`, a tagged git commit, and at least one figure or table. Treat milestones as gates — don't start the next phase until the gate is passed.

### Phase 0 — Setup (Days 1–3)
- Initialize git repo, virtualenv, project structure (see §4)
- `requirements.txt` / `pyproject.toml` with pinned versions
- Logging + seeding utilities; reproducibility skeleton
- README stub
- **Gate:** `pytest -q` runs (even with one passing dummy test); `python -m src.data.ingest` downloads S&P 500 successfully

### Phase 1 — Data pipeline & EDA (Week 1, days 4–7)
- yfinance ingestion for ^GSPC, ^VIX, AAPL, TSLA (2000-01-01 → 2025-12-31)
- Implement realized-volatility proxies: Yang-Zhang, Garman-Klass, 21-day rolling σ̂, squared log returns
- Train/val/test split with explicit dates; walk-forward index generator
- EDA notebook: log returns, RV time series, autocorrelation, distribution, COVID/2008 markers
- Unit tests for RV computations and split generators
- **Gate:** Milestone note `results/m01_data.md` with 6 figures and a sanity-check table comparing RV proxies

### Phase 2 — Econometric baselines (Week 2)
- GARCH(1,1) via `arch`, fit-and-forecast wrapper with rolling refit
- HAR-RV implementation (linear regression on daily/weekly/monthly RV lags)
- Optional: EGARCH (only if Phase 2 finishes Friday)
- Evaluation harness: MSE, MAE, QLIKE losses, all coded with unit tests
- Rolling out-of-sample loop with monthly refit
- **Gate:** `results/m02_econometric.md` with rolling-OOS error tables and a forecast-vs-actual figure for the test period

### Phase 3 — LSTM baseline (Weeks 3–4)
- PyTorch `Dataset`/`DataLoader` for sliding windows
- Vanilla LSTM regressor (1–2 layers, dropout, MSE loss)
- Hyperparameter sweep on validation: hidden size {32, 64, 128}, lookback {10, 20, 40}, lr {1e-3, 1e-4}
- Walk-forward training with early stopping; weekly rather than daily refit (for compute)
- Direct comparison vs Phase 2 baselines on identical splits
- **Gate:** `results/m03_lstm.md` with hyperparameter heatmap, learning curves, and head-to-head vs GARCH/HAR

### Phase 4 — Regime detection (Week 5)
- HMM (`hmmlearn`) with Gaussian emissions
  - Inputs: log returns + RV proxy (try both 1-feature and 2-feature)
  - K = 2, 3, 4 states tested via BIC + interpretability
- Regime visualization: states overlaid on price/RV with COVID-19, 2008 GFC, dot-com bust labels
- Validation: do detected regimes correlate with VIX percentiles? (sanity check — should)
- **Optional comparison:** K-means on rolling stats; Markov-switching GARCH (overkill — skip)
- **Critical:** all HMM fits must be on the training window only at each walk-forward step
- **Gate:** `results/m04_regimes.md` with regime maps and a confusion table vs VIX-quantile labels

### Phase 5 — Regime-aware models (Weeks 6–7)
Two architectures, run head-to-head:
- **Approach A — Regime as feature:** append one-hot regime (or posterior probabilities) to LSTM input vector
- **Approach B — Regime-specific experts:** train one LSTM per regime, route via current regime posterior at inference (mixture-of-experts gating)
- Compare both vs Phase 3 LSTM and vs Phase 2 econometric baselines
- **Calm vs crisis subsetting:** segment the test set by regime label and report per-regime errors — this is core for RQ2
- **Gate:** `results/m05_regime_dl.md` with the four-way comparison table (GARCH | HAR | LSTM | Regime-LSTM-A | Regime-LSTM-B), per-regime breakdown, and a discussion paragraph

### Phase 6 — Uncertainty quantification (Weeks 8–9)
- **MC Dropout LSTM:** keep dropout active at inference; T=100 stochastic forward passes; report mean and predictive std → 95% intervals
- **Quantile Regression LSTM:** three output heads (q=0.05, 0.5, 0.95) trained with pinball loss
- Calibration metrics: PICP (prediction interval coverage probability), MPIW (mean prediction interval width), Winkler score
- Calibration *per regime* — this is where UQ usually breaks and where the contribution lies
- **Gate:** `results/m06_uq.md` with reliability diagrams, PICP/MPIW tables, per-regime calibration table

### Phase 7 — Combined model + significance + writeup start (Week 10)
- Combined Regime + MC Dropout model (best of phase 5 architecture × MC Dropout)
- Diebold-Mariano test on headline pairs: best DL vs HAR; regime-aware vs regime-agnostic; UQ-LSTM mean vs LSTM
- Begin Methods and Experiments chapters of dissertation
- **Gate:** `results/m07_combined.md` with the master results table that will appear in the dissertation

### Phase 8 — Robustness + Writing (Week 11)
- Run baselines + best model on AAPL and TSLA (RQ4)
- Sensitivity analyses: lookback length, refit frequency, RV proxy choice (one figure each)
- Writing: Background, Methods, Experiments, Results, Discussion
- **Gate:** Full dissertation draft v0.9 (everything except Conclusion + Abstract)

### Phase 9 — Polish, reproducibility check, submission (Week 12)
- Re-run all experiments end-to-end from a fresh checkout to validate reproducibility
- Final figures (publication-grade: matplotlib styled, vector formats)
- Conclusion + Abstract
- Pruning: cut weak experiments rather than padding
- **Gate:** Submitted dissertation + tagged release commit `v1.0-submitted`

---

## 4. Project structure

```
MSc Dissertation/
├── README.md
├── ROADMAP.md                  ← this file
├── pyproject.toml              ← pinned deps, Black/Ruff config
├── .gitignore                  ← data/, *.pkl, .venv/, __pycache__/, runs/
├── .python-version
│
├── configs/                    ← YAML configs per experiment
│   ├── data.yaml
│   ├── garch.yaml
│   ├── har.yaml
│   ├── lstm_baseline.yaml
│   ├── hmm.yaml
│   ├── regime_lstm.yaml
│   ├── mc_dropout_lstm.yaml
│   └── quantile_lstm.yaml
│
├── data/                       ← all gitignored
│   ├── raw/                    ← yfinance dumps, never modified
│   ├── interim/                ← cleaned series
│   └── processed/              ← feature matrices, splits
│
├── notebooks/                  ← exploration only, NOT for final results
│   ├── 01_data_exploration.ipynb
│   ├── 02_rv_proxies.ipynb
│   ├── 03_garch_diagnostics.ipynb
│   ├── 04_hmm_regimes.ipynb
│   ├── 05_lstm_baseline.ipynb
│   ├── 06_regime_dl.ipynb
│   ├── 07_uq_calibration.ipynb
│   └── 08_results_synthesis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── ingest.py           ← yfinance download + caching
│   │   ├── features.py         ← lags, technicals, regime features
│   │   ├── realized_vol.py     ← YZ, GK, rolling σ̂, squared returns
│   │   └── splits.py           ← walk-forward / anchored generators
│   ├── models/
│   │   ├── econometric/
│   │   │   ├── garch.py
│   │   │   ├── egarch.py       ← optional
│   │   │   └── har.py
│   │   ├── deep/
│   │   │   ├── lstm.py
│   │   │   ├── regime_lstm.py
│   │   │   ├── mc_dropout_lstm.py
│   │   │   └── quantile_lstm.py
│   │   └── regime/
│   │       ├── hmm.py
│   │       └── kmeans.py       ← optional comparison
│   ├── training/
│   │   ├── trainer.py          ← training loop, early stopping, checkpoints
│   │   ├── losses.py           ← MSE, QLIKE, pinball
│   │   └── callbacks.py
│   ├── evaluation/
│   │   ├── metrics.py          ← MSE, MAE, QLIKE, PICP, MPIW, Winkler
│   │   ├── rolling.py          ← walk-forward backtest engine
│   │   ├── significance.py     ← Diebold-Mariano
│   │   └── regime_analysis.py  ← per-regime breakdowns
│   └── utils/
│       ├── config.py           ← OmegaConf / Hydra-lite loading
│       ├── seeding.py
│       ├── io.py
│       └── plotting.py         ← single source of figure styling
│
├── experiments/                ← one folder per experiment, with config + results
│   ├── 01_garch/
│   ├── 02_har/
│   ├── 03_lstm/
│   ├── 04_hmm/
│   ├── 05_regime_lstm/
│   ├── 06_mc_dropout/
│   ├── 07_quantile/
│   ├── 08_combined/
│   └── 09_robustness_other_assets/
│
├── results/
│   ├── figures/
│   ├── tables/
│   ├── predictions/            ← every model's test-set forecasts as parquet
│   ├── logs/
│   └── milestones/             ← m01_data.md … m07_combined.md
│
├── tests/
│   ├── test_realized_vol.py
│   ├── test_splits.py
│   ├── test_metrics.py
│   ├── test_har.py
│   └── test_dataloader.py
│
└── dissertation/
    ├── chapters/
    ├── figures/                ← symlinks/copies from results/figures
    └── refs.bib
```

**Why this structure:** notebooks are for exploration, all production code lives in `src/` and is unit-testable; `experiments/` has reproducible YAML configs; `results/` is your single source of truth for what gets cited in the dissertation. Don't paste numbers from notebooks into the writeup — only cite numbers from `results/tables/`.

---

## 5. Experimental pipeline (the "engine")

Every model in the project — econometric or DL — passes through the same evaluation engine. Build the engine *once*, in Phase 1/2.

```
┌──────────┐   ┌────────────┐   ┌───────────────┐   ┌────────────┐
│ Raw OHLC │ → │ RV proxies │ → │ Walk-forward  │ → │ Per-fold   │ → metrics + predictions
│ (yfin)   │   │ + features │   │ split iter    │   │ fit+predict│
└──────────┘   └────────────┘   └───────────────┘   └────────────┘
                                                         │
                                                         ▼
                                                  results/predictions/<model>.parquet
                                                         │
                                                         ▼
                                              evaluation/metrics + regime_analysis
```

Once predictions are in parquet form (one column per model), every comparison — point error, DM test, regime breakdown, calibration — is a pandas operation. This decouples modelling from evaluation entirely and is what lets you swap in regime-aware and UQ models in later phases without rewriting the harness.

---

## 6. Build order — what first vs later

**Build first (non-negotiable):**
1. Data ingestion + RV proxies + splits
2. Evaluation harness (metrics + walk-forward loop) — even before any model
3. GARCH baseline (validates the harness against a known-good model)
4. HAR-RV baseline
5. LSTM baseline

**Build second:**
6. HMM regime detection (with leakage discipline)
7. Regime-as-feature LSTM
8. Regime-specific LSTM ensemble

**Build third:**
9. MC Dropout LSTM
10. Quantile Regression LSTM
11. Calibration metrics

**Build only if time permits:**
- EGARCH
- Markov-switching GARCH
- K-means regime comparison
- Deep ensembles
- Transformer / TFT
- Multi-asset robustness on additional assets

**Cut from scope (push back if you really want any of these):**
- Intraday RV (requires data sourcing not in scope)
- Reinforcement learning, options pricing, portfolio optimization downstream tasks
- Real-time / streaming infrastructure

---

## 7. Reading roadmap (prioritized)

Read papers as you build, not all upfront. The list below is ordered by when it becomes relevant.

### Tier 1 — Read in Week 1 (foundations)
- **Andersen & Bollerslev (1998)** — "Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts." *International Economic Review.* Foundational on using realized variance as evaluation target.
- **Andersen, Bollerslev, Diebold, Labys (2003)** — "Modeling and Forecasting Realized Volatility." *Econometrica.* Where RV-as-target methodology is canonized.
- **Bollerslev (1986)** — "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics.* GARCH original.
- **Corsi (2009)** — "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics.* HAR-RV original — must cite, must implement.
- **Patton (2011)** — "Volatility forecast comparison using imperfect volatility proxies." *Journal of Econometrics.* Justifies why MSE and QLIKE are robust under noisy RV proxies — your *direct* methodological cover for using daily-frequency proxies.

### Tier 2 — Weeks 2–3 (LSTM and DL volatility)
- **Hochreiter & Schmidhuber (1997)** — LSTM original.
- **Liu (2019)** — "Novel volatility forecasting using deep learning." Solid LSTM-vs-GARCH comparison.
- **Bucci (2020)** — "Realized Volatility Forecasting with Neural Networks." *Journal of Financial Econometrics.* Most directly relevant comparison paper.
- **Christensen, Siggaard, Veliyev (2022)** — "A machine learning approach to volatility forecasting." Modern ML benchmark.

### Tier 3 — Weeks 4–5 (regimes)
- **Hamilton (1989)** — "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica.* Regime-switching foundational.
- **Rabiner (1989)** — "A tutorial on hidden Markov models." Implementation reference.
- **Ang & Bekaert (2002)** — "Regime switches in interest rates." Methodologically canonical.
- **Guidolin (2011)** — "Markov Switching Models in Empirical Finance." Survey.

### Tier 4 — Weeks 6–8 (uncertainty)
- **Gal & Ghahramani (2016)** — "Dropout as a Bayesian Approximation." MC Dropout original.
- **Lakshminarayanan, Pritzel, Blundell (2017)** — "Simple and scalable predictive uncertainty estimation using deep ensembles." Useful even if you don't implement ensembles.
- **Koenker & Bassett (1978)** — Quantile regression original.
- **Wen et al. (2017)** — "A Multi-Horizon Quantile Recurrent Forecaster." Quantile RNN reference.
- **Pearce et al. (2018)** — "High-Quality Prediction Intervals for Deep Learning." On PI calibration.

### Tier 5 — Week 10 (significance, writeup)
- **Diebold & Mariano (1995)** — Comparing predictive accuracy.
- **Harvey, Leybourne, Newbold (1997)** — small-sample DM correction.

---

## 8. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Data leakage in walk-forward | Medium | Catastrophic | Code review checklist for every model; HMM/scaler refits must be inside the walk-forward loop; unit tests on split generator |
| 2 | LSTM does not beat HAR-RV | High | Low (it's a finding) | Frame honestly; literature supports this outcome; emphasize *why* (HAR captures long memory cheaply) |
| 3 | HMM regimes are not interpretable / collapse | Medium | Medium | Try multiple K, multiple emission features (returns, RV); validate against VIX quantiles; fallback to k-means |
| 4 | MC Dropout intervals are uncalibrated | Medium | Medium | Report PICP/MPIW honestly; quantile regression as comparison usually fares better; this is itself a finding |
| 5 | Compute bottleneck on rolling refits | Low (you have local GPU) | Medium | Refit weekly/monthly not daily; cache predictions to parquet; use multiprocessing for HMM |
| 6 | Scope creep (transformers, ensembles) | High | High | Cut list in §6 is binding until phase 7 done |
| 7 | Reproducibility breaks before submission | Medium | High | Phase 9 fresh-checkout end-to-end run; pinned deps; saved seeds; saved configs per experiment |
| 8 | Daily-frequency RV proxy critique in viva | High | Medium | Cite Patton (2011); justify Yang-Zhang choice; acknowledge limitation in Discussion |
| 9 | yfinance API change / data gaps | Low | Medium | Cache raw downloads to `data/raw/`; never re-download once gathered; record snapshot date |
| 10 | Personal: illness / time loss | Medium | High | Phase 7 ends end of week 10 so weeks 11–12 are buffer + writing — preserve them |

---

## 9. Reproducibility & rigor standards

Treat the following as non-negotiable from Phase 0 onwards:

- **Seeding:** every experiment sets `random`, `numpy`, `torch` seeds; `torch.use_deterministic_algorithms(True)` where feasible. Seed is part of the YAML config.
- **Config snapshots:** every experiment run dumps its resolved config to `experiments/<name>/run_<timestamp>/config.yaml`.
- **Data snapshot:** raw yfinance pulls saved with download date in filename; never overwritten.
- **Pinned dependencies:** `pyproject.toml` with exact versions; `pip freeze > requirements-lock.txt` checked in.
- **Tests:** at minimum, RV computations, splits, and metrics have unit tests. CI not required, but `pytest` must pass before tagging milestones.
- **Predictions persisted:** every model's full test-set predictions saved to parquet — this lets you re-run any comparison without re-training.
- **Single source of figures:** all figures cited in the dissertation are generated by scripts in `src/utils/plotting.py` from `results/predictions/*.parquet` — no notebook screenshots.
- **Git hygiene:** tag each milestone (`m01-data`, `m02-econometric`, …); branch for any speculative work; never force-push main.

---

## 10. Suggestions and open questions

A few things I'd like your call on before we start coding:

1. **Time period.** I'd default to 2000-01-01 → 2024-12-31 (covers dot-com bust, GFC, Euro crisis, COVID, 2022 inflation regime — enough crisis variety for RQ2). Acceptable, or do you want a different window?

2. **Test split.** Suggest training pre-2020, validation 2020–2021, test 2022–2024. The test set thus contains the 2022 inflation/rates shock — a non-COVID stressor distinct from the training window. Anchored walk-forward then expands across the test period. Good?

3. **HAR-RV.** Confirming you're happy to add it as a baseline (it's important; see §1.2).

4. **UQ scope.** Confirming MC Dropout (primary) + Quantile Regression (comparison), dropping deep ensembles unless we have time. OK?

5. **Robustness assets.** AAPL + TSLA, baselines + best DL only, in Week 11. OK or different choice?

6. **Supervisor cadence.** Are you meeting your supervisor weekly / fortnightly? I want to slot supervisor checkpoints into the milestones — usually after Phases 2, 5, and 7.

Answer those when convenient and I'll lock the plan, write Phase 0 setup files (`pyproject.toml`, `.gitignore`, folder skeleton, base configs, seed/util modules, harness stubs, the first unit tests), and we move into Phase 1.

---

*End of roadmap. This file is the canonical source of plan-of-record. Update it (with dated changelog entries at the bottom) whenever scope changes.*
