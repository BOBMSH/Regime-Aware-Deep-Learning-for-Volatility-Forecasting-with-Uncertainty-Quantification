# MSc Dissertation Roadmap
## Regime-Aware Deep Learning for Financial Volatility Forecasting with Uncertainty Quantification

**Author:** Bob (University of Warwick, MSc Applied AI)
**Last updated:** 2026-07-22
**Target submission window:** ~12 weeks (≈ early August 2026)
**Compute:** Local machine with GPU
**Stack:** Python, PyTorch, pandas, numpy, scikit-learn, statsmodels, `arch`, `hmmlearn`
**Status:** Phases 0–6 complete — milestones `m01`–`m06` on disk, 173/173 unit tests green, and the committed prediction parquets reproduce the milestone numbers. **Phase 7 (combined regime × uncertainty model + master results table) is next.** Phase 6 (uncertainty quantification) added an MC-Dropout LSTM (T=100 stochastic passes → log-normal intervals) and a distribution-free quantile-regression LSTM (monotone 0.05/0.5/0.95 pinball head): the point forecasts are statistically indistinguishable from the Phase-3 LSTM (DM p=0.19), the intervals mildly under-cover at 90% (MC-Dropout PICP 0.865, quantile 0.821) and — the RQ3 contribution — **coverage degrades in the crisis regime** (MC-Dropout 0.843, quantile 0.800). The formal Diebold–Mariano test (HLN-corrected, Student-t) and the Model Confidence Set (Hansen–Lunde–Nason 2011) are implemented in `src/evaluation/significance.py` and reported from Phase 5 (`m05`): the 90% MCS keeps the six deep-learning/HAR models and excludes the GARCH family and the random-walk floor. Final data scope is **2000-01-03 → 2022-02-25** (the Oxford-Man `.SPX` coverage boundary); the originally-planned 2022–2024 intraday extension was abandoned (no free intraday source). See the 2026-07-22 changelog entry at the foot of this file.

---

## 0. Executive summary

You are building a comparative study, not a product. The dissertation's contribution is methodological clarity: showing — under a fair, leakage-free, rolling-window protocol — whether (a) deep learning beats classical econometrics, (b) regime-awareness adds signal on top of that, and (c) uncertainty quantification produces predictions whose intervals are actually well-calibrated. The deliverable is a defensible answer to those three questions plus a writeup that an examiner can replicate.

In 12 weeks with one researcher you cannot do everything. The plan below is built around an MVP-first build order: every phase produces a *publishable-grade* artifact (numbers, plots, a milestone note) before moving on. Optional items are clearly flagged and only attempted if earlier phases finish on time.

---

## 1. Critical methodological calls (read first)

These are decisions I am recommending up front because they will materially affect both the work and the defense. None are final — push back if you disagree.

### 1.1 True realized volatility from intraday returns (Oxford-Man Realized Library)
The dependent variable is daily realized volatility constructed from intraday returns in the sense of Andersen, Bollerslev, Diebold and Labys (2003) — the square-root of the sum of squared 5-minute log returns within each trading day. This is the canonical target in the modern volatility literature and the one Chapter 2 (Literature Review) commits to in §2.3.

**Data source:** Oxford-Man Institute Realized Library — pre-cleaned 5-minute-aggregated realized variance series for the S&P 500 (and other major indices) from 2000 onwards, freely downloadable as CSV. Using a curated source removes the intraday data-cleaning overhead (microstructure noise, opening-auction handling, jump filtering) that would otherwise dominate Phase 1 and is not the methodological contribution of this dissertation.

**Robustness checks:** report a Yang-Zhang OHLC-based RV estimator from yfinance daily data as a sensitivity exercise, demonstrating that headline conclusions are not artefacts of the realised-variance estimator. Cite Patton (2011) for proxy-robust loss functions (MSE, QLIKE), which justify the use of a noisy RV estimate without biasing the forecast comparison.

*Changelog (2026-05-15):* this section originally specified OHLC-based proxies (Yang-Zhang, Garman-Klass, 21-day rolling σ̂) as the primary target, with intraday RV excluded from scope. The literature review committed to intraday RV in the Andersen-Bollerslev-Diebold-Labys sense, and the Oxford-Man Realized Library makes that target reachable within Phase 1 at modest cost; the scope was revised accordingly.

### 1.2 Econometric baselines: GARCH(1,1), EGARCH, HAR-RV
Three econometric baselines, all confirmed:

- **GARCH(1,1)** (Bollerslev 1986) — the canonical short-memory conditional-variance specification; required for comparison with any alternative.
- **EGARCH** (Nelson 1991) — captures the leverage effect through an asymmetric variance equation. Included as a falsification check: a regime-aware deep model that beats GARCH(1,1) but not EGARCH may be capturing asymmetric shock responses rather than genuine regime-conditional non-linearity. This separation is methodologically central to RQ2.
- **HAR-RV** (Corsi 2009) — the de-facto literature baseline on realized-volatility targets; a linear regression on daily, weekly, and monthly lagged RV that routinely matches or exceeds more elaborate specifications on equity RV. Without HAR-RV the viva will ask why.

*Changelog (2026-05-15):* EGARCH was previously listed as optional / "only if time permits." It is now a confirmed baseline because the literature review's falsification argument is methodologically stronger than the time-budget reason originally given for dropping it; the marginal implementation cost on top of GARCH(1,1) via `arch` is negligible.

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

**RQ1 — Performance.** Do deep learning models (LSTM) outperform econometric baselines (GARCH(1,1), EGARCH, HAR-RV) for daily realized-volatility forecasting on S&P 500, under leakage-free walk-forward evaluation, as measured by QLIKE / MSE / MAE?

**RQ2 — Regime-awareness.** Does conditioning forecasts on latent market regimes (HMM-detected) measurably improve forecast accuracy versus regime-agnostic equivalents, and is the improvement concentrated in transitional or crisis periods?

**RQ3 — Uncertainty calibration.** Do uncertainty-aware models (MC Dropout, quantile regression) produce prediction intervals with empirical coverage close to nominal, and how does coverage degrade across regimes?

**Stretch RQ4 — Generalization.** Do conclusions from RQ1–3 hold on AAPL and TSLA, or are they index-specific?

---

## 3. Twelve-week phase plan

Each phase ends with a **milestone artifact**: a markdown note in `results/`, a tagged git commit, and at least one figure or table. Treat milestones as gates — don't start the next phase until the gate is passed.

### Phase 0 — Setup (Days 1–3)  ·  ✅ **DONE**
- Initialize git repo, virtualenv, project structure (see §4)
- `requirements.txt` / `pyproject.toml` with pinned versions
- Logging + seeding utilities; reproducibility skeleton
- README stub
- **Gate:** `pytest -q` runs (even with one passing dummy test); `python -m src.data.ingest` downloads S&P 500 successfully

### Phase 1 — Data pipeline & EDA (Week 1, days 4–7)  ·  ✅ **DONE** (`results/milestones/m01_data.md`)
- **Oxford-Man Realized Library** ingestion for ^GSPC (`.SPX` intraday rv5, **2000-01-03 → 2022-02-25** — the library's coverage boundary and the final sample end); cached to `data/raw/oxfordman/` with snapshot date
- yfinance ingestion for ^GSPC, ^VIX, AAPL, TSLA daily OHLC over 2000–2024 (for returns features, the Yang-Zhang RV sensitivity, and robustness assets)
- Implement Yang-Zhang OHLC RV as a sensitivity check; squared daily returns retained only as a diagnostic baseline
- Train/val/test split with explicit dates; anchored walk-forward index generator — **final splits: train ≤ 2015-12-31, validation 2016–2018, test 2019-01-01 → 2022-02-25** (the COVID-19 crash is the headline out-of-sample stressor)
- EDA notebook: log returns, intraday-RV time series, autocorrelation (ACF of RV — expect long memory), distribution, dot-com bust / 2008 GFC / COVID-19 / 2022 inflation markers
- Unit tests for RV computations, OHLC RV sensitivity estimator, and split generators
- **Gate:** Milestone note `results/m01_data.md` with 6 figures, an RV summary table, and a sanity-check comparison between Oxford-Man intraday RV and the Yang-Zhang OHLC sensitivity series

### Phase 2 — Econometric baselines (Week 2)  ·  ✅ **DONE** — HAR-RV wins (QLIKE 0.2745) < GARCH 0.3302 < EGARCH 0.3328 < RW-RV 0.3584 (`m02_econometric.md`)
- GARCH(1,1) via `arch`, fit-and-forecast wrapper with rolling refit
- EGARCH via `arch` on the same harness (confirmed baseline; see §1.2)
- HAR-RV implementation (linear regression on daily/weekly/monthly RV lags)
- Evaluation harness: MSE, MAE, QLIKE losses, all coded with unit tests; QLIKE adopted as primary metric per Patton (2011)
- Rolling out-of-sample loop with monthly refit
- **Gate:** `results/m02_econometric.md` with rolling-OOS error tables and a forecast-vs-actual figure for the test period covering all three baselines

### Phase 3 — LSTM baseline (Weeks 3–4)  ·  ✅ **DONE** — best LSTM (RV-only) QLIKE 0.2596 < HAR-RV 0.2745; DM p≈0.081 (not yet significant) (`m03_lstm.md`)
- PyTorch `Dataset`/`DataLoader` for sliding windows
- Vanilla LSTM regressor (1–2 layers, dropout, MSE loss)
- Hyperparameter sweep on validation: hidden size {32, 64, 128}, lookback {10, 20, 40}, lr {1e-3, 1e-4}
- Walk-forward training with early stopping; weekly rather than daily refit (for compute)
- Direct comparison vs Phase 2 baselines on identical splits
- **Gate:** `results/m03_lstm.md` with hyperparameter heatmap, learning curves, and head-to-head vs GARCH/HAR

### Phase 4 — Regime detection (Week 5)  ·  ✅ **DONE** — HMM + Nystrup jump model, K=3; VIX Spearman 0.789; leakage-free causal filtered posterior built as the Phase-5 input (`m04_regimes.md`)
- Gaussian-emission HMM (`hmmlearn`) on standardised daily log returns (and RV as a 2-feature variant for ablation)
- **Primary specification:** K=3 states (calm / transitional / crisis), aligned with the literature review's regime-narrative commitment in §2.4
- **Sensitivity:** K=2 and K=4 reported with BIC and interpretability diagnostics
- **State-persistence remedy — Nystrup, Lindstrom & Madsen (2020) jump penalty.** `hmmlearn` does not ship this; budget ~2–3 days within Phase 4 to either (a) port the Nystrup et al. reference implementation to Python or (b) implement the jump penalty as a regularised EM step on top of `hmmlearn`. The penalty parameter is selected by cross-validation on a held-out segment of the training window. *Fallback if implementation slips past Friday of Week 5:* report standard Baum-Welch HMM with state-persistence diagnostics, and soften the corresponding lit-review §2.4 commitment to "motivated by Nystrup et al. (2020)" rather than "adopts."
- Regime visualisation: states overlaid on price/RV with dot-com bust, 2008 GFC, COVID-19, and 2022 inflation markers
- Validation: detected regimes correlate with VIX percentiles (sanity check)
- **Critical:** all HMM fits must be on the training window only at each walk-forward step; the regularisation parameter is also CV-selected on training-only data
- **Gate:** `results/m04_regimes.md` with regime maps, BIC sensitivity table for K∈{2,3,4}, state-persistence statistics, and a confusion table vs VIX-quantile labels

### Phase 5 — Regime-aware models (Weeks 6–7)  ·  ✅ **DONE** — Regime-LSTM-B (MoE) best QLIKE 0.2583 < LSTM 0.2650 (formal DM p=0.034); regime value concentrated in the COVID crisis (Regime-LSTM-A 0.2976, ~7% < best baseline); 90% Model Confidence Set = {HAR-RV, LSTM, LSTM-RVonly, Regime-LSTM-A, Regime-LSTM-A-RVonly, Regime-LSTM-B}, GARCH-family + RW excluded (`m05_regime_dl.md`)
Two architectures, run head-to-head:
- **Approach A — Regime as feature:** append one-hot regime (or posterior probabilities) to LSTM input vector
- **Approach B — Regime-specific experts:** train one LSTM per regime, route via current regime posterior at inference (mixture-of-experts gating)
- Compare both vs Phase 3 LSTM and vs Phase 2 econometric baselines
- **Calm vs crisis subsetting:** segment the test set by regime label and report per-regime errors — this is core for RQ2
- **Gate:** `results/m05_regime_dl.md` with the comparison table (GARCH | EGARCH | HAR | RW | LSTM | LSTM-RVonly | Regime-LSTM-A | Regime-LSTM-A-RVonly | Regime-LSTM-B), per-regime breakdown, formal Diebold–Mariano + Model Confidence Set, and a discussion paragraph

### Phase 6 — Uncertainty quantification (Weeks 8–9)  ·  ✅ **DONE** — MC-Dropout PICP 0.865 & quantile 0.821 at 90% (both mildly over-confident); coverage degrades in the crisis regime (0.843 / 0.800); point accuracy unchanged vs the Phase-3 LSTM (DM p=0.19) (`m06_uq.md`)
- *Infrastructure ready:* the calibration metrics PICP / MPIW / Winkler score are already implemented and unit-tested in `src/evaluation/metrics.py`; the MC-Dropout variant reuses the existing LSTM head dropout (kept active at inference).
- **MC Dropout LSTM:** keep dropout active at inference; T=100 stochastic forward passes; report mean and predictive std → 95% intervals — *done* as `src/models/deep/uncertainty.py::MCDropoutLSTMForecaster`; predictive log-variance law = epistemic (MC spread) + aleatoric (Duan residual variance) → log-normal intervals on the variance scale (Gal & Ghahramani 2016). Headline nominal level **90%** (common to both methods) plus 95%.
- **Quantile Regression LSTM:** three output heads (q=0.05, 0.5, 0.95) trained with pinball loss — *done* as `QuantileLSTMForecaster` with a **monotone (non-crossing) head** (cumulative soft-plus); distribution-free.
- Calibration metrics: PICP (prediction interval coverage probability), MPIW (mean prediction interval width), Winkler score — plus a reliability diagram over a dense τ grid and a coverage-error column.
- Calibration *per regime* — this is where UQ usually breaks and where the contribution lies — *done*: both methods under-cover most in the COVID-dominated **crisis** state; the MC-Dropout log-normal band is near-homoskedastic on the log scale so it cannot widen with the regime, while the quantile head can — the substantive RQ3 contrast.
- **Gate:** `results/m06_uq.md` with reliability diagrams, PICP/MPIW tables, per-regime calibration table — **met** (`results/milestones/m06_uq.md`, tables `m06_uq_*`, figures `results/figures/m06/*`, predictions `results/predictions/m06_uq_*.parquet`, 30 new unit tests). *Reproduce:* `python -m src.experiments.run_uq`.
- *Note:* combining regime-conditioning with UQ (best Phase-5 architecture × MC-Dropout) is deliberately left to **Phase 7**, per the plan; Phase 6 evaluates UQ on the base LSTM with per-regime calibration.

### Phase 7 — Combined model + significance + writeup start (Week 10)
- Combined Regime + MC Dropout model (best of phase 5 architecture × MC Dropout)
- Diebold-Mariano test on headline pairs: best DL vs HAR; regime-aware vs regime-agnostic; UQ-LSTM mean vs LSTM. *(The formal DM + Model Confidence Set apparatus was implemented early, in Phase 5 — `src/evaluation/significance.py` — so Phase 7 extends it to the UQ pairs rather than building it from scratch.)*
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
1. Data ingestion (Oxford-Man RV + yfinance OHLC) + Yang-Zhang sensitivity RV + splits
2. Evaluation harness (metrics + walk-forward loop) — even before any model
3. GARCH(1,1) baseline (validates the harness against a known-good model)
4. EGARCH baseline (falsification check for asymmetric-response absorption)
5. HAR-RV baseline
6. LSTM baseline

**Build second:**
7. HMM regime detection with Nystrup et al. (2020) jump penalty, leakage discipline
8. Regime-as-feature LSTM
9. Regime-specific LSTM ensemble (mixture-of-experts gating)

**Build third:**
10. MC Dropout LSTM
11. Quantile Regression LSTM with pinball loss
12. Calibration metrics (PICP, MPIW, Winkler, CRPS)

**Build only if time permits:**
- Markov-switching GARCH
- K-means regime comparison
- Deep ensembles
- Transformer / TFT
- Multi-asset robustness on additional assets beyond AAPL+TSLA

**Cut from scope (push back if you really want any of these):**
- Reinforcement learning, options pricing, portfolio optimization downstream tasks
- Real-time / streaming infrastructure
- Bespoke intraday cleaning pipeline (Oxford-Man provides curated RV; we do not re-derive it from tick data)

---

## 7. Reading roadmap (prioritized)

Read papers as you build, not all upfront. The list below is ordered by when it becomes relevant.

### Tier 1 — Read in Week 1 (foundations)
- **Cont (2001)** — "Empirical properties of asset returns: stylized facts and statistical issues." *Quantitative Finance.* The empirical contract any volatility model must satisfy.
- **Engle (1982)** — ARCH original. Frames the conditional-heteroskedasticity tradition that GARCH extends.
- **Bollerslev (1986)** — "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics.* GARCH original — primary baseline.
- **Nelson (1991)** — "Conditional Heteroskedasticity in Asset Returns: A New Approach." *Econometrica.* EGARCH original — second econometric baseline.
- **Andersen & Bollerslev (1998)** — "Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts." *International Economic Review.* Foundational on using realized variance as evaluation target.
- **Andersen, Bollerslev, Diebold, Labys (2003)** — "Modeling and Forecasting Realized Volatility." *Econometrica.* Where intraday-RV-as-target methodology is canonized — defines the dissertation's dependent variable.
- **Corsi (2009)** — "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics.* HAR-RV original — must cite, must implement.
- **Hansen & Lunde (2005)** — "A Forecast Comparison of Volatility Models: Does Anything Beat a GARCH(1,1)?" *Journal of Applied Econometrics.* Sets the empirical bar any alternative must clear.
- **Patton (2011)** — "Volatility forecast comparison using imperfect volatility proxies." *Journal of Econometrics.* QLIKE and MSE as proxy-robust losses.

### Tier 2 — Weeks 2–3 (LSTM and DL volatility)
- **Hochreiter & Schmidhuber (1997)** — LSTM original.
- **Lim & Zohren (2021)** — "Time-series forecasting with deep learning: a survey." *Phil. Trans. R. Soc. A.* Design-vocabulary reference for the dissertation; motivates the hybrid statistical-neural decomposition.
- **Liu (2019)** — "Novel volatility forecasting using deep learning." Solid LSTM-vs-GARCH comparison.
- **Kim & Won (2018)** — "Forecasting the volatility of stock price index: a hybrid model integrating LSTM with multiple GARCH-type models." *Expert Syst. Appl.* Direct empirical precedent for LSTM+GARCH hybrids.
- **Bucci (2020)** — "Realized Volatility Forecasting with Neural Networks." *Journal of Financial Econometrics.* Most directly relevant comparison paper on US equity RV.
- **Christensen, Siggaard, Veliyev (2022)** — "A machine learning approach to volatility forecasting." Modern ML benchmark.

### Tier 3 — Weeks 4–5 (regimes)
- **Hamilton (1989)** — "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica.* Regime-switching foundational.
- **Rabiner (1989)** — "A tutorial on hidden Markov models." Implementation reference.
- **Nystrup, Lindstrom & Madsen (2020)** — "Learning hidden Markov models with persistent states by penalizing jumps." *Expert Syst. Appl.* Jump-penalised estimator adopted in Phase 4.
- **Ang & Bekaert (2002)** — "Regime switches in interest rates." Methodologically canonical.
- **Guidolin (2011)** — "Markov Switching Models in Empirical Finance." Survey.

### Tier 4 — Weeks 6–8 (uncertainty)
- **Gal & Ghahramani (2016)** — "Dropout as a Bayesian Approximation." MC Dropout original.
- **Engle & Manganelli (2004)** — "CAViaR: Conditional Autoregressive Value at Risk by Regression Quantiles." *J. Bus. Econ. Stat.* Methodological justification for pinball-loss quantile regression on financial sequences.
- **Lakshminarayanan, Pritzel, Blundell (2017)** — "Simple and scalable predictive uncertainty estimation using deep ensembles." Useful even if you don't implement ensembles.
- **Koenker & Bassett (1978)** — Quantile regression original.
- **Wen et al. (2017)** — "A Multi-Horizon Quantile Recurrent Forecaster." Quantile RNN reference.
- **Pearce et al. (2018)** — "High-Quality Prediction Intervals for Deep Learning." On PI calibration.
- **Gneiting & Raftery (2007)** — "Strictly Proper Scoring Rules, Prediction, and Estimation." *JASA.* CRPS and Winkler score; canonical reference for evaluating probabilistic forecasts.

### Tier 5 — Week 10 (significance, writeup)
- **Diebold & Mariano (1995)** — Comparing predictive accuracy.
- **Giacomini & White (2006)** — "Tests of Conditional Predictive Ability." *Econometrica.* Refines DM for settings with persistent loss differentials — relevant given RV long-memory.
- **Harvey, Leybourne, Newbold (1997)** — small-sample DM correction.
- **Glosten, Jagannathan & Runkle (1993)** — GJR-GARCH; cited via Hansen-Lunde for asymmetric specifications.

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
| 8 | Nystrup jump-penalty implementation slips past Phase 4 budget | Medium | Medium | Fallback: standard Baum-Welch HMM with state-persistence diagnostics; soften lit-review §2.4 commitment to "motivated by Nystrup et al." rather than "adopts." Decide by end of Week 5 Friday. |
| 9 | Oxford-Man Realized Library schema change / availability | Low | High | Cache raw downloads to `data/raw/oxfordman/` with snapshot date; never re-download once gathered; keep Yang-Zhang OHLC RV implemented as a fallback target if the curated source becomes unavailable |
| 10 | yfinance API change / data gaps for return features and robustness assets | Low | Medium | Cache raw downloads to `data/raw/`; never re-download once gathered; record snapshot date |
| 11 | Personal: illness / time loss | Medium | High | Phase 7 ends end of week 10 so weeks 11–12 are buffer + writing — preserve them |

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

**Resolved (carried over from prior versions of this roadmap and the literature review):**
- ~~HAR-RV as a baseline~~ — confirmed (§1.2).
- ~~UQ scope: MC Dropout + Quantile Regression, no deep ensembles~~ — confirmed (§1.5).
- ~~Robustness assets AAPL + TSLA, baselines + best DL only, Week 11~~ — confirmed (§1.4, Phase 8).
- ~~Intraday RV via Oxford-Man Realized Library~~ — adopted 2026-05-15 (§1.1).
- ~~EGARCH as confirmed second econometric baseline~~ — adopted 2026-05-15 (§1.2).
- ~~HMM K=3 primary with K=2,4 BIC sensitivity, Nystrup et al. (2020) jump penalty~~ — adopted 2026-05-15 (§Phase 4).
- ~~Time period / sample window~~ — **finalised 2026-07-19:** 2000-01-03 → 2022-02-25 (Oxford-Man `.SPX` coverage boundary; the 2022–2024 intraday extension was abandoned for lack of a free source).
- ~~Test split~~ — **finalised 2026-07-19:** train ≤ 2015-12-31, validation 2016–2018, test 2019-01-01 → 2022-02-25 (anchored walk-forward, ~monthly refit). Supersedes the earlier "train pre-2020 / val 2020–21 / test 2022–24" suggestion.
- ~~Nystrup implementation route~~ — **resolved 2026-07-19:** implemented as a jump-penalised estimator in `src/models/regime/jump.py` (λ tuned by a persistence sweep); the Phase-4 fallback (Risk #8) was not needed, so Ch2 §2.4 "adopts" the estimator as written.

**Still open:**

1. **Supervisor cadence.** Are you meeting your supervisor weekly / fortnightly? Supervisor checkpoints slot naturally into the milestones — usually after Phases 2, 5, and 7.

*(The earlier open questions — time period, test split, and the Nystrup implementation route — were all resolved during the Phase 1–4 build; see the Resolved list above and the 2026-07-19 changelog entry.)*

---

## Changelog

**2026-05-15 — Alignment with Chapter 2 (Literature Review v2).**
Following completion of the literature review, the roadmap was updated to absorb the chapter's stronger methodological commitments. Concretely:
- §1.1 — Target variable switched from OHLC-based proxies (Yang-Zhang, Garman-Klass, 21-day rolling σ̂) to intraday-derived realized volatility in the Andersen-Bollerslev-Diebold-Labys (2003) sense. Data source: Oxford-Man Institute Realized Library. Yang-Zhang OHLC RV retained as a sensitivity check.
- §1.2 — EGARCH (Nelson 1991) promoted from "optional, only if time permits" to confirmed second econometric baseline, on the literature review's falsification argument (separating regime-conditional non-linearity from absorbed asymmetric responses).
- §2 — RQ1 wording updated to name GARCH(1,1), EGARCH, HAR-RV explicitly.
- Phase 1 — Data pipeline rewritten around Oxford-Man + yfinance OHLC; Yang-Zhang becomes a sensitivity, not the primary target.
- Phase 2 — EGARCH promoted into the confirmed-baseline list.
- Phase 4 — HMM specification fixed at K=3 primary (calm/transitional/crisis) with K=2,4 BIC sensitivity; Nystrup, Lindstrom & Madsen (2020) jump-penalised estimator adopted with a documented fallback if implementation slips.
- §6 — Build order updated; "Intraday RV" removed from the cut list; "bespoke intraday cleaning pipeline" added in its place.
- §7 — Reading roadmap expanded to include Cont (2001), Hansen-Lunde (2005), Lim-Zohren (2021), Nystrup et al. (2020), Engle-Manganelli (2004), Gneiting-Raftery (2007), Giacomini-White (2006), Kim-Won (2018), and other lit-review citations.
- §8 — Risk #8 retired (no longer applicable now that RV is intraday); replaced with implementation risk on the Nystrup jump penalty and a data-availability risk on Oxford-Man.
- §10 — Resolved questions struck through; one new open question on Nystrup implementation route.

**2026-07-19 — Phases 0–4 complete; scope and splits finalised.**
Roadmap brought into line with the implemented pipeline after the Phase 1–4 build (verified: 126/126 unit tests green; committed prediction parquets reproduce the milestone numbers):
- **Status header added.** Phases 0–4 are done (milestones `m01`–`m04`); Phase 5 (regime-aware models) is next.
- **Scope finalised to 2000-01-03 → 2022-02-25.** The Oxford-Man `.SPX` intraday rv5 series ends at the library's 2022-02-25 coverage boundary; the originally-planned 2022–2024 extension (splicing reconstructed intraday RV) was abandoned because no free 2022–2024 intraday source was available. yfinance daily data still spans 2000–2024 for the Yang-Zhang sensitivity and the robustness assets. (Phase 1; §10 Q1.)
- **Splits finalised** to train ≤ 2015-12-31, validation 2016–2018, test 2019-01-01 → 2022-02-25 (anchored walk-forward, ~monthly refit); the COVID-19 crash is the headline out-of-sample stressor. Supersedes the earlier "train pre-2020 / val 2020–21 / test 2022–24" suggestion. (Phase 1; §10 Q2.)
- **Nystrup jump penalty implemented** in `src/models/regime/jump.py` (λ tuned by a persistence sweep); the Phase-4 fallback (Risk #8) was not triggered. (§10 Q4.)
- **Phase-2 return input refined** to open-to-close log returns for the GARCH family, to fair-match the open-to-close Oxford-Man rv5 target (close-to-close embeds the overnight gap, which the ratio-based QLIKE penalises).
- **Phase-3 architecture-vs-information control added:** an RV-only LSTM matched to HAR-RV's exact inputs runs alongside the full LSTM, so any win over HAR-RV is attributable to the architecture rather than the extra return feature.
- **Two Phase-1 data bugs fixed** (an Oxford-Man BST date mis-parse that shifted ~59% of rows and produced weekend dates; a `realized_vol.py` syntax error), each with a regression test.
- **Phase-4 hand-off note corrected:** the persisted causal posterior comes from a single HMM fit on the training window (≤ 2015-12-31) then filtered forward and frozen — *not* refit per walk-forward step — and is consumed leakage-safely via the LSTM's t−1 input window.

**2026-07-21 — Phase 5 complete; formal DM + Model Confidence Set implemented (pulled forward from Phase 7).**
Roadmap brought into line with the implemented pipeline after the Phase 5 build and a full pre-Phase-6 audit (verified: 143/143 unit tests green; the committed `m05` prediction parquet reproduces every QLIKE, per-regime and significance number):
- **Status header updated.** Phases 0–5 are done (milestones `m01`–`m05`); Phase 6 (uncertainty quantification) is next.
- **Phase 5 delivered.** Regime-aware LSTMs built two ways — regime-as-feature (Regime-LSTM-A, plus an RV-only variant) and a mixture-of-experts (Regime-LSTM-B) — on the Phase-4 causal filtered HMM posteriors (leakage-safe: window ends t−1, MoE gate lagged to t−1). Headline: Regime-LSTM-B QLIKE 0.2583 < LSTM 0.2650 (formal DM p=0.034); the regime signal's value concentrates in the COVID crisis state (Regime-LSTM-A 0.2976, ≈7% below the best baseline), while the calm-dominated pooled mean understates it — the RQ2 hypothesis.
- **Formal significance apparatus implemented early.** `src/evaluation/significance.py` now provides the formal Diebold–Mariano test (Newey–West HAC + Harvey–Leybourne–Newbold small-sample correction + Student-t) and the Model Confidence Set (Hansen–Lunde–Nason 2011, via the `arch` stationary-bootstrap implementation), both on the QLIKE loss. These were originally scheduled for Phase 7; they are pulled forward so the Phase-5 model comparison reports a multiple-comparison-controlled verdict, matching the Chapter 2 §2.7 commitment. The 90% MCS retains the six deep-learning/HAR models and excludes GARCH, EGARCH and the random-walk floor (`m05_regime_dl.md`; `results/tables/m05_regime_dl_intraday_2019_2022_mcs.csv`). Unit-tested in `tests/test_significance.py`.
- **The earlier indicative DM readout in `run_lstm.py` was replaced** by an import of the formal test; at n=784, h=1 the corrected statistics are numerically unchanged (Regime-LSTM-B vs LSTM stays −2.13, p=0.034).

**2026-07-22 — Phase 6 complete; uncertainty quantification delivered (RQ3).**
- **Two UQ methods built on the Phase-3 LSTM** in `src/models/deep/uncertainty.py`, run on the identical leakage-free anchored walk-forward and fixed at the Phase-3 hyperparameters so the point forecasts stay comparable: (1) **MC-Dropout LSTM** — head dropout kept active at inference, T=100 stochastic passes; the predictive law on log-variance combines the epistemic (MC) spread with the aleatoric (Duan residual) variance (Gal & Ghahramani 2016) → a log-normal prediction interval; (2) **Quantile-regression LSTM** — one LSTM with a **monotone non-crossing** 3-quantile head (0.05/0.5/0.95) trained with the pinball loss (distribution-free). A third **LSTM-Gaussian** reference (aleatoric-only, no MC) isolates what the stochastic passes add, and reproduces the Phase-3 LSTM point to 8e-10.
- **Result (RQ3):** point accuracy is preserved (MC-Dropout mean QLIKE 0.2652 vs LSTM 0.2650; formal DM p=0.19). At 90% nominal, both interval methods mildly **under-cover** (MC-Dropout PICP 0.865, coverage error −0.035; quantile 0.821, −0.079); at 95% both parametric methods reach 0.920. The distinctive finding is **per-regime**: coverage is worst in the COVID-dominated **crisis** state (MC-Dropout 0.843, quantile 0.800) — the pooled number flatters both. Epistemic uncertainty is tiny at dropout 0.1 (≈0.6% of predictive variance), so MC-Dropout ≈ the Gaussian reference; the genuine contrast is the distribution-free quantile head (which can learn a state-dependent width) vs the near-homoskedastic log-normal band (which cannot). Honest caveats (median-vs-mean QLIKE functional; Gaussian/homoskedastic-residual assumption) are stated in `m06_uq.md`.
- **New calibration infrastructure:** `src/evaluation/calibration.py` (log-normal intervals, reliability curve, per-regime interval metrics) and appended `pinball_loss` / `mean_pinball_loss` / `coverage_error` to `src/evaluation/metrics.py`. Config `configs/uq.yaml`, driver `src/experiments/run_uq.py`. **30 new unit tests** (`tests/test_uncertainty.py`, `tests/test_calibration.py`, additions to `tests/test_metrics.py`); full suite **173/173 green**. An independent correctness/leakage/methodology audit returned SOUND (no leakage, no correctness bugs; formulas numerically verified).
- **Gate met:** `results/milestones/m06_uq.md`, tables `results/tables/m06_uq_*`, figures `results/figures/m06/{reliability,interval_band,per_regime}.png`, predictions `results/predictions/m06_uq_intraday_2019_2022.parquet`.
- **Status header updated.** Phases 0–6 are done (milestones `m01`–`m06`); Phase 7 (combined regime × uncertainty model + master results table) is next.

---

*End of roadmap. This file is the canonical source of plan-of-record. Update it (with dated changelog entries above) whenever scope changes.*
