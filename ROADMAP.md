# MSc Dissertation Roadmap
## Regime-Aware Deep Learning for Financial Volatility Forecasting with Uncertainty Quantification

**Author:** Bob (University of Warwick, MSc Applied AI)
**Last updated:** 2026-08-30
**Target submission window:** ~12 weeks (≈ early August 2026)
**Compute:** Local machine with GPU
**Stack:** Python, PyTorch, pandas, numpy, scikit-learn, statsmodels, `arch`, `hmmlearn`
**Status:** Phases 0–7 complete — milestones `m01`–`m07` on disk, ~334 unit tests, all phases re-run 2026-08-19 after the report/code alignment work, and the evaluation-side defects found in the three pre-Phase-7 audits closed (see the changelog: (ii) fixed the per-regime bucket **label**, (iii) fixed the transition **selector** and retracted the RQ3 transition headline, (iv) fixed the label **source** so the descriptive and Giacomini–White tables are computed on the same 784 days). **Phase 7 completed 2026-08-23** — both profiles trained at MC=100 (`m07_combined.md`, `m07_combined_hmm.md`) and every published statistic independently re-derived from the parquets in audit (vii). The one remediation that audit found — the committed artefacts spanning two Python interpreters — was **carried out 2026-08-24**: Phases 2–7 were regenerated in a single environment, every phase's run snapshot now reads the same interpreter and library set, and the Phase-5 inheritance guard passes at machine epsilon. Exactly one published number moved (the combined model's), and no conclusion changed; see the 2026-08-24 (viii) entry. **Phase 8's code was built 2026-09-02 (x) and has not been run** — the RQ4 chain, both sensitivities and 42 tests are in place; the fits must be produced in the Windows 3.12 environment. See Phase 8 for the run order. Audit (ix) on 2026-08-30 re-verified every published number against the regenerated artefacts (all reproduce), cleared the Phase-8 blockers, and settled the phase's three open scope decisions — see Phase 8 below and the (ix) changelog entry. Chapter 1's three factual errors about the study's own design were corrected in the .docx; **Chapter 2 was deliberately left untouched**, and the fifteen parked methodology declarations now live in §11 rather than in audit reports.

**The three headline findings, as they now stand:**

1. **RQ1 — deep learning beats HAR-RV, but only conditionally.** Pooled, no pairwise DM gap reaches significance (LSTM 0.2650 vs HAR-RV 0.2745, p=0.55). Conditional on the lagged regime indicator, the Giacomini–White test gives LSTM vs HAR-RV **p=4.0e-06**, concentrated in the transitional state (QLIKE 18.8% lower). The pooled test misses it because the loss differential *flips sign across states* — HAR-RV wins in calm, the LSTM wins decisively in transitional — which inflates the variance and cancels the mean. This is the dissertation's clearest RQ1 result and the reason Ch2 §2.7's commitment to Giacomini & White (2006) matters.
2. **RQ2 — regime conditioning adds little on top of the architecture.** Of Regime-LSTM-B's −0.0516 transitional edge over HAR-RV, 86% is the deep architecture and 14% the regime signal (B vs LSTM p=0.064, n.s.). Interpretation: the LSTM's 10-day input window already encodes most of what a 3-state HMM posterior summarises, so explicit regime conditioning is largely redundant *given a sequence model* — an empirical qualification of Ch2 §2.5's division-of-labour argument.
3. **RQ3 — the intervals are over-confident, and the miscalibration is a *level* problem, not a *timing* problem.** MC-Dropout PICP 0.865 and quantile 0.821 at 90% nominal — both significantly narrow. Beyond that, nothing knowable at t−1 predicts where the misses fall. Per-regime coverage under the correctly-timed t−1 label is non-monotone (MC-Dropout calm 0.844 / transitional 0.894 / crisis 0.842), and on the **ex-ante** transition selector `1{s(t−1) ≠ s(t−2)}` the two groups do not separate at all (MC-Dropout 0.885 flagged vs 0.863 otherwise; quantile 0.808 vs 0.822). *Two earlier headlines have now been retracted here: "coverage degrades monotonically with regime severity" (2026-08-19 (ii)) and "coverage collapses at regime transitions" (2026-08-19 (iii)). Both were selection on the outcome — the first in the bucket **label**, the second in the transition **selector**.* The surviving claim is cleaner and more useful: the band is the wrong width more or less uniformly, which is why a single global σ rescale (×1.14 for coverage, ×1.24 for CRPS) is the indicated fix and a state-conditional one is not.

**Robustness worth stating up front:** every conclusion above was checked against an alternative regime estimator (Baum-Welch vs jump-penalised HMM, 95.8% agreement on test days) and, since 2026-08-19, against the jump-penalty grid as well (λ ∈ [2, 3] all agree with the headline on ≥96.7% of test days). The transitional finding replicates and strengthens; the previously-claimed crisis accuracy advantage *reverses*, and is retracted. Relabelling 4% of days flipped it — which is why it is now reported as a null with a caution about the 76-day crisis bucket.

**One design asymmetry to keep in view when reading any of the above:** the deep models are trained **once** on data ≤ 2018-12-31 and frozen across the whole out-of-sample window (`refit_every_folds: 0`), while GARCH/EGARCH/HAR-RV refit every 21 trading days on an expanding window. Both are leakage-free, but the asymmetry runs *against* the deep models, so their wins are conservative and their weakness in high-volatility states cannot be separated from training-window staleness on this evidence alone. A `refit_every_folds: 1` sensitivity is Phase-8 work.

The formal Diebold–Mariano test (HLN-corrected, Student-t), the Model Confidence Set (Hansen–Lunde–Nason 2011), the Giacomini–White conditional predictive-ability test and the DM regularity diagnostics all live in `src/evaluation/significance.py`. The 90% MCS keeps the six deep-learning/HAR models and excludes the GARCH family and the random-walk floor. Final data scope is **2000-01-03 → 2022-02-25** (the Oxford-Man `.SPX` coverage boundary); the originally-planned 2022–2024 intraday extension was abandoned (no free intraday source). See the 2026-08-19 changelog entry at the foot of this file.

---

## 0. Executive summary

You are building a comparative study, not a product. The dissertation's contribution is methodological clarity: showing — under a fair, leakage-free, rolling-window protocol — whether (a) deep learning beats classical econometrics, (b) regime-awareness adds signal on top of that, and (c) uncertainty quantification produces predictions whose intervals are actually well-calibrated. The deliverable is a defensible answer to those three questions plus a writeup that an examiner can replicate.

In 12 weeks with one researcher you cannot do everything. The plan below is built around an MVP-first build order: every phase produces a *publishable-grade* artifact (numbers, plots, a milestone note) before moving on. Optional items are clearly flagged and only attempted if earlier phases finish on time.

---

## 1. Critical methodological calls (read first)

These are decisions I am recommending up front because they will materially affect both the work and the defense. None are final — push back if you disagree.

### 1.1 True realized volatility from intraday returns (Oxford-Man Realized Library)
The dependent variable is the daily realized **variance** constructed from intraday returns in the sense of Andersen, Bollerslev, Diebold and Labys (2003) — the sum of squared 5-minute log returns within each trading day (Oxford-Man's `rv5`). Every model forecasts, and every loss scores, on the *variance* scale: that is the scale on which Patton's (2011) proxy-robustness result for MSE and QLIKE is derived, and the scale GARCH forecasts natively, so it keeps the comparison apples-to-apples. ("Realized volatility" is used loosely throughout for the series; where the distinction matters the code is unambiguous — `src/evaluation/metrics.py` scores variance against variance.) This is the canonical target in the modern volatility literature and the one Chapter 2 (Literature Review) commits to in §2.3.

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
Trying to do regime + UQ + 3 deep models × 4 assets in 12 weeks is not realistic. Make S&P 500 (^GSPC) the primary asset where you run *every* experiment. Then run a slimmed robustness check (baselines + best DL model only) on **two further Oxford-Man indices — `.RUT` (Russell 2000) and `.FTSE` (FTSE 100)** — to show your conclusions generalize. VIX is a *feature*, not an additional target. *(2026-08-30 (ix): both assets, and every identifier they need, are now declared in `configs/data.yaml` under `assets.registry` and resolved by `src.data.datasets.resolve_asset`.)*

*Changed 2026-08-23 (vii):* the original choice was AAPL + TSLA. The Oxford-Man Realized Library publishes realized measures for **indices only** — 31 of them, no single stocks — so those two assets could only have been given a Yang-Zhang OHLC proxy, and Ch1 §1.8 reserves Yang-Zhang as a target-validity *diagnostic* rather than a forecasting target. Using it for RQ4 would have contradicted the chapter that scopes the study, and would have made RQ4's results incomparable with RQ1–3 because the dependent variable changed. `.RUT` and `.FTSE` keep the target identical in construction and are the harder test as well: a US small-cap index, and a different market with a different trading session, rather than two US large-caps that co-move with the S&P 500.

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

**Stretch RQ4 — Generalization.** Do conclusions from RQ1–3 hold on the Russell 2000 (`.RUT`) and the FTSE 100 (`.FTSE`), or are they specific to the S&P 500?

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

### Phase 3 — LSTM baseline (Weeks 3–4)  ·  ✅ **DONE** — best equal-information LSTM (RV-only) QLIKE 0.2596 < HAR-RV 0.2745, DM p≈0.081 (pooled; but see the *conditional* result below); **auxiliary-feature ablation LSTM-VIX QLIKE 0.2301, DM vs LSTM p=0.024** — the largest and only clearly significant model improvement in the study, walled out of the Phase-5 comparison as an unequal information set (`m03_lstm.md`)
- PyTorch `Dataset`/`DataLoader` for sliding windows
- Vanilla LSTM regressor (1–2 layers, dropout, MSE loss)
- Hyperparameter sweep on validation: hidden size {32, 64, 128}, lookback {10, 20, 40}, lr {1e-3, 1e-4}
- Walk-forward training with early stopping; weekly rather than daily refit (for compute)
- Direct comparison vs Phase 2 baselines on identical splits
- **Gate:** `results/m03_lstm.md` with hyperparameter heatmap, learning curves, and head-to-head vs GARCH/HAR

### Phase 4 — Regime detection (Week 5)  ·  ✅ **DONE** — HMM + Nystrup jump model, K=3; VIX Spearman 0.789; **jump-penalised HMM (λ=3, selected on training rows only) is the headline causal signal Phases 5–7 condition on**, with the Baum-Welch posterior retained as the regime-estimator comparator — the two agree on 95.8% of test days (`m04_regimes.md`)
- Gaussian-emission HMM (`hmmlearn`) on standardised daily log returns (and RV as a 2-feature variant for ablation)
- **Primary specification:** K=3 states (calm / transitional / crisis), aligned with the literature review's regime-narrative commitment in §2.4
- **Sensitivity:** K=2 and K=4 reported with BIC and interpretability diagnostics
- **State-persistence remedy — Nystrup, Lindstrom & Madsen (2020) jump penalty.** `hmmlearn` does not ship this; budget ~2–3 days within Phase 4 to either (a) port the Nystrup et al. reference implementation to Python or (b) implement the jump penalty as a regularised EM step on top of `hmmlearn`. The penalty parameter is selected by cross-validation on a held-out segment of the training window. *Fallback if implementation slips past Friday of Week 5:* report standard Baum-Welch HMM with state-persistence diagnostics, and soften the corresponding lit-review §2.4 commitment to "motivated by Nystrup et al. (2020)" rather than "adopts."
- Regime visualisation: states overlaid on price/RV with dot-com bust, 2008 GFC, COVID-19, and 2022 inflation markers
- Validation: detected regimes correlate with VIX percentiles (sanity check)
- **Critical:** all HMM fits must be on the training window only at each walk-forward step; the regularisation parameter is also CV-selected on training-only data
- **Gate:** `results/m04_regimes.md` with regime maps, BIC sensitivity table for K∈{2,3,4}, state-persistence statistics, and a confusion table vs VIX-quantile labels

### Phase 5 — Regime-aware models (Weeks 6–7)  ·  ✅ **DONE** — **pooled: no significant gain** from regime conditioning (all DM p ≥ 0.19; best pooled model is the regime-*agnostic* LSTM-RVonly 0.2596, Regime-LSTM-B 0.2621, LSTM 0.2650, HAR-RV 0.2745). **Conditionally the picture reverses**: Giacomini–White on the lagged regime indicator gives Regime-LSTM-B vs HAR-RV χ²(3)=40.1, **p=1.0e-08** (A vs HAR-RV p=4.3e-05), with the edge localised in the **transitional** state (per-regime DM p=4.3e-09, QLIKE 0.1851 vs 0.2367 — 21.8% lower on n=329). Attribution decomposes additively: of B's −0.0516 transitional edge over HAR-RV, **−0.0446 (86%) is the deep architecture and −0.0070 (14%) is regime conditioning** (B vs LSTM p=0.064, n.s.). **The earlier "value concentrated in the COVID crisis" claim is retracted**: crisis DM p≈0.8 on n≈75 under both regime estimators, and the crisis ranking *reverses* between them. Since 2026-08-19 (ii) the descriptive per-regime table is bucketed by the **t−1** label, matching the GW indicators; on that timing HAR-RV leads calm (0.3008 vs LSTM 0.3187) and crisis is a near-tie (0.3070 vs 0.3131), while transitional strengthens. 90% MCS = {HAR-RV, LSTM, LSTM-RVonly, Regime-LSTM-A, Regime-LSTM-A-RVonly, Regime-LSTM-B}, GARCH-family + RW excluded (`m05_regime_dl.md`)
Two architectures, run head-to-head:
- **Approach A — Regime as feature:** append one-hot regime (or posterior probabilities) to LSTM input vector
- **Approach B — Regime-specific experts:** train one LSTM per regime, route via current regime posterior at inference (mixture-of-experts gating)
- Compare both vs Phase 3 LSTM and vs Phase 2 econometric baselines
- **Calm vs crisis subsetting:** segment the test set by regime label and report per-regime errors — this is core for RQ2
- **Gate:** `results/m05_regime_dl.md` with the comparison table (GARCH | EGARCH | HAR | RW | LSTM | LSTM-RVonly | Regime-LSTM-A | Regime-LSTM-A-RVonly | Regime-LSTM-B), per-regime breakdown, formal Diebold–Mariano + Model Confidence Set, and a discussion paragraph

### Phase 6 — Uncertainty quantification (Weeks 8–9)  ·  ✅ **DONE** — MC-Dropout PICP 0.865 & quantile 0.821 at 90% (both mildly over-confident). **Neither the regime *level* nor an implementable regime *change* predicts where the misses fall.** Per-regime coverage under the correctly-timed t−1 label is *non-monotone* (MC calm 0.844 / transitional 0.894 / crisis 0.842) — the "degrades monotonically to 0.813/0.773 in crisis" reading was an artefact of the same-day label and is retracted (2026-08-19 (ii)). On the **ex-ante** transition selector `1{s(t−1) ≠ s(t−2)}` there is no separation either (MC 0.885 vs 0.863; quantile 0.808 vs 0.822) — the "0.712 vs 0.876 at transitions" reading used the **ex-post** selector `1{s(t) ≠ s(t−1)}`, which is chosen with the day-t outcome, and is retracted (2026-08-19 (iii)). Both selector timings are now written to disk, stamped `implementable`. Point accuracy unchanged vs the Phase-3 LSTM (DM p=0.19) (`m06_uq.md`)
- *Infrastructure ready:* the calibration metrics PICP / MPIW / Winkler score are already implemented and unit-tested in `src/evaluation/metrics.py`; the MC-Dropout variant reuses the existing LSTM head dropout (kept active at inference).
- **MC Dropout LSTM:** keep dropout active at inference; T=100 stochastic forward passes; report mean and predictive std → 95% intervals — *done* as `src/models/deep/uncertainty.py::MCDropoutLSTMForecaster`; predictive log-variance law = epistemic (MC spread) + aleatoric (Duan residual variance) → log-normal intervals on the variance scale (Gal & Ghahramani 2016). Headline nominal level **90%** (common to both methods) plus 95%.
- **Quantile Regression LSTM:** three output heads (q=0.05, 0.5, 0.95) trained with pinball loss — *done* as `QuantileLSTMForecaster` with a **monotone (non-crossing) head** (cumulative soft-plus); distribution-free.
- Calibration metrics: PICP (prediction interval coverage probability), MPIW (mean prediction interval width), Winkler score — plus a reliability diagram over a dense τ grid and a coverage-error column.
- Calibration *per regime* — this is where UQ usually breaks and where the contribution was expected to lie — *done*, and the result is not the one originally expected, twice over. Bucketed by the regime known at **t−1** (the only timing that supports a conditional claim — see 2026-08-19 (ii)), coverage is **not** monotone in regime severity. Nor does it separate on an implementable regime *change*: the ex-ante selector `1{s(t−1) ≠ s(t−2)}` shows no gap, and the large gap on the ex-post selector `1{s(t) ≠ s(t−1)}` is selection on the outcome, because a transition is detected by the same one-day surprise that breaks the interval (2026-08-19 (iii)). The MC-Dropout log-normal band is near-homoskedastic on the log scale — a fixed multiplicative factor on the point forecast — so it has no mechanism by which to be wrong *selectively*; it can only be wrong *uniformly*, which is exactly what the evidence shows. That is the substantive RQ3 contrast, and it is what makes a global width correction the indicated remedy.
- **Gate:** `results/m06_uq.md` with reliability diagrams, PICP/MPIW tables, per-regime calibration table — **met** (`results/milestones/m06_uq.md`, tables `m06_uq_*`, figures `results/figures/m06/*`, predictions `results/predictions/m06_uq_*.parquet`, 30 new unit tests). *Reproduce:* `python -m src.experiments.run_uq`.
- *Note:* combining regime-conditioning with UQ (best Phase-5 architecture × MC-Dropout) is deliberately left to **Phase 7**, per the plan; Phase 6 evaluates UQ on the base LSTM with per-regime calibration.

### Phase 7 — Combined model + significance + writeup start (Week 10)  ·  ✅ **DONE 2026-08-23** — both profiles trained (MC=100, no smoke banner) and every published statistic re-derived from the parquets in audit (vii). Code built and tested 2026-08-22 (vi): — `src/evaluation/coverage_tests.py` (Kupiec / Christoffersen / Engle–Manganelli DQ / difference-in-coverage / Holm, 46 tests), `src/models/deep/regime_uncertainty.py` (`MCDropoutRegimeExpertForecaster`, 16 tests), `src/experiments/report_coverage.py` (parquet-only) and `src/experiments/run_combined.py` + `configs/combined{,_hmm}.yaml`. Suite 249 → **311**. **The RQ3 tests are already run on both profiles** — Kupiec rejects for every method (MC-Dropout p=0.0017; quantile p=2.3e-11) and Christoffersen independence does not (p 0.29–0.97), both as pre-registered; the third clause is *partially falsified* on the upper tail, where calm-state under-warning survives Holm. Only `python -m src.experiments.run_combined` (the combined model's training) remains, and it must run on the same machine as every other deep model. See the 2026-08-22 (vi) changelog entry.
- Combined Regime + MC Dropout model (best of phase 5 architecture × MC Dropout). **Decision RECORDED 2026-08-22 (vi) — "best of Phase 5" is `Regime-LSTM-B`, the mixture-of-experts, selected on the transitional result.** The choice needed a rule because the candidates win different buckets: Regime-LSTM-B leads the regime models pooled (QLIKE 0.2621) and in the transitional state (0.1851), while Regime-LSTM-A-RVonly leads calm (0.2920) and Regime-LSTM-A leads crisis (0.2952). The rule is: **select on the only bucket where the regime effect is both large and formally tested.** That is the transitional state — per-regime DM p = 4.3e-09 against HAR-RV, and the state carrying the Giacomini–White moment that makes χ²(3) = 40.1 significant. Calm and crisis are selection on point estimates over 379 and 76 days with no significant test behind them, and crisis reverses sign between the two regime estimators. Selecting on an untested bucket would be choosing the architecture on noise. This rule is fixed *before* Phase 7 runs and belongs in Chapter 3 alongside the model descriptions.
- Diebold-Mariano test on headline pairs: best DL vs HAR; regime-aware vs regime-agnostic; UQ-LSTM mean vs LSTM. *(The formal DM + Model Confidence Set apparatus was implemented early, in Phase 5 — `src/evaluation/significance.py` — so Phase 7 extends it to the UQ pairs rather than building it from scratch. Giacomini–White conditional predictive ability and the DM regularity diagnostics landed with the 2026-08-18 alignment work; extend both to the UQ pairs too.)*
- **NEW — make RQ3 inferential, not descriptive.** The calibration claim is still point estimates of PICP with no hypothesis test, exactly the gap Giacomini–White closed for RQ2. Add an unconditional-coverage test (Kupiec 1995 LR), Christoffersen's (1998) conditional-coverage extension, and the Engle–Manganelli dynamic quantile test *already cited in Ch2 §2.6*. **Run all three on the FULL 784-day sample with regressors lagged into t−1 — not on a transition or per-regime subsample.** The 2026-08-19 (iii) entry explains why: the ex-post transition split is chosen using the day-t outcome, so a Kupiec test on it returns a small p-value by construction (p≈1e-04) while the same test on the implementable selector returns nothing (p≈0.7). A subsample is only a legitimate place to test coverage if its selector is measurable at t−1. `src/evaluation/regime_timing.selector_is_implementable` is the guard; read it before writing the test harness.
- **Expected shape of the answer, recorded before it is run** (so the result reads as a prediction, not a rationalisation): Kupiec should reject decisively — the intervals are 3.5–7.9 pp narrow — while Christoffersen's independence component and any t−1 conditioning variable should *not*, because the failures are unclustered and unpredictable. If that holds, the honest RQ3 conclusion is **"miscalibration is a level problem, not a timing problem"**, the combined regime × UQ model is a **pre-registered null**, and the deliverable is a global width correction (σ×1.14 for coverage, ×1.24 for CRPS — `m06_uq_*_sigma_scale.csv`) reported against it.
  - **OUTCOME, 2026-08-22 (vi) — kept below the prediction, unedited, so the two can be read against each other.** The first two clauses hold exactly: Kupiec rejects for every method (MC-Dropout p=0.0017, quantile p=2.3e-11) and Christoffersen independence does not (p 0.29–0.97). The third is **partially falsified**. Pooled two-sided, nothing knowable at t−1 predicts a miss once Holm-adjusted — as predicted. On the **upper tail** it does: MC-Dropout upper-violation rates on the t−1 label are calm 0.1055 / transitional 0.0426 / crisis 0.0921 against a 5% nominal, with calm-vs-rest and transitional-vs-rest both surviving family-wise control. The lower tail runs the other way, which is why the pooled view — the only one Phase 6 computed — saw nothing. So miscalibration is *mostly* a level problem, and there is in addition a real, implementable, one-sided regime effect: the model under-warns when it thinks conditions are calm. The deliverable is therefore a global scale **plus** a calm-state widening, not a single global scale; the per-regime scales sit beside the global one in `m07_*_sigma_scale.csv`.
- Begin Methods and Experiments chapters of dissertation
- **Gate:** `results/m07_combined.md` with the master results table that will appear in the dissertation

### Phase 8 — Robustness + Writing (Week 11)

**RQ4 — cross-asset generalisation on `.RUT` and `.FTSE`.** Decided 2026-08-23 (vii); see §1.4 for why not AAPL/TSLA. Both series are already in the cached Oxford-Man CSV and need no new intraday download: `.RUT` 5550 rows, `.FTSE` 5586 rows, both spanning 2000 → 2022-02-25 with **zero missing `rv5`** — the same window as `.SPX`, all three re-verified directly in the CSV on 2026-08-30.

**Three scope decisions, settled 2026-08-30 (ix). Do not re-open them mid-phase.**

1. **Regimes are re-estimated per asset — not transferred from the S&P.** RQ4 asks whether the *conclusions* generalise, and a conclusion is about the pipeline; the pipeline includes Phase 4, so substituting the S&P's regime path would test the pipeline-minus-Phase-4 and quietly answer a different question. The cost argument does not rescue the alternative either: the expensive component is the LSTMs, which must be refit per asset regardless, while the regime stage is a Gaussian HMM on ~5.5k standardised daily returns — minutes, not hours. §1.4's "slimmed" refers to the **model set**, not to skipping estimation stages, and the best deep model is regime-conditioned, so it needs a regime signal by construction. Each asset therefore gets its own feature standardisation, its own Baum-Welch and jump-penalised fits, and its own λ selected by the **same pre-registered rule on the same pre-registered grid**, applied to that asset's training rows. Report the per-asset λ, K and regime persistence beside the S&P's: if `.FTSE`'s λ lands somewhere very different, that is itself a finding about the method's portability.
   *Optional secondary run, if time allows:* condition `.FTSE` on the **S&P's** regime path as an explicitly-labelled transfer experiment. "Does US market state predict UK volatility regimes" is a genuinely interesting question and costs one extra run — but it is a **secondary** result and must never be presented as the RQ4 answer.
2. **The LSTM-VIX ablation is not run on `.FTSE`.** The CBOE VIX prices S&P 500 options; feeding it to a FTSE model is a cross-market spillover experiment, not the auxiliary-feature ablation Ch1 §1.8 describes, and the UK analogue (VFTSE) is not in the data. It would also silently intersect the LSE and NYSE calendars in the frame's inner join. This is now enforced in data rather than by memory: `assets.registry.FTSE.vix_feature: false` in `configs/data.yaml`, and `build_econometric_frame` refuses to join the column and logs why.
3. **RQ4 is declared in Chapter 1 §1.6 as a pre-registered *secondary* question**, with its reduced design (baselines + best deep model, two indices) stated in the same sentence. Ch1 previously said the study was organised around *three* research questions while this file had carried a stretch RQ4 since May; a results chapter answering four beside an introduction declaring three is the discrepancy an examiner notices first. Stating the reduced scope up front is also what inoculates the answer against the "under-powered" objection. Corrected in the .docx 2026-08-30 (ix).

**Blockers cleared 2026-08-30 (ix)** — the phase is startable except for the data pull:
- `configs/data.yaml` gained an **asset registry** (`assets.registry`): the single source of truth mapping each asset key to its yfinance ticker, OHLC cache alias, Oxford-Man symbol, trading session and VIX eligibility. `assets:` previously had *no consumer at all*, which is exactly why it still named AAPL and TSLA a week after (vii) replaced them — a stale value in a block nothing reads is invisible.
- `realized_variance_target` / `build_econometric_frame` now take an `asset=` key resolved through that registry. Before this the Oxford-Man symbol was read from `oxfordman.primary_symbol` unconditionally, so a Phase-8 profile would have regressed Russell 2000 returns on **S&P 500** realised variance and printed an entirely plausible table. `tests/test_datasets.py` guards it.
- `src/data/ingest.py` reads the registry too, so `--all` covers the Phase-8 assets and the ticker downloaded is by construction the ticker the modelling frame later looks for.
- `build_econometric_frame` now records calendar attrition in `frame.attrs["join_attrition"]` and warns above 2%. The frame is an inner join across sources; on `.SPX` that is invisible because everything shares the NYSE calendar, and on any other asset it is a live risk.
- **Still outstanding, and only doable on the Windows machine (needs network):** `python -m src.data.ingest --asset RUT` then `--asset FTSE`.

**Sensitivity analyses**, in descending value:
- **`refit_every_folds: 1`** — the design asymmetry disclosed in §1.7 runs *against* the deep models, so this is the sensitivity most likely to strengthen a headline result rather than qualify it. `--refit-every` already exists on `run_lstm`.
- **Log-HAR** (`har.transform: log`) through the harness. Implemented and config-exposed but never run through it; a one-off check gives log-HAR QLIKE 0.4882 against 0.2745 in levels, which kills the "the LSTM only wins because it models logs" objection outright. Worth having properly on the record.
- **Lookback length**, which doubles as the home for the HAR-regressor control below.
- **HAR-regressor RV-only LSTM** — set `robustness.rv_only_features: [log_rv_d, log_rv_w, log_rv_m]` in `configs/lstm_baseline.yaml` (the feature builders already exist) to give the RV-only LSTM Corsi's exact regressors rather than raw daily lags. Ch1 §1.5(ii) has been corrected to describe what is actually fed; this run would make the stronger claim testable rather than argued. Note the current design is the more **conservative** control — a 10-day window of raw daily log-RV reaches back less far than HAR's 22 days — so the existing result is not flattered by it.
- **RV proxy choice** (the Yang-Zhang diagnostic, already built).

**Already delivered by (ix), no run required beyond a `--from-predictions` rebuild:** the quantile model's point forecast is now also reported retransformed onto the mean scale (`Quantile-LSTM-mean`, QLIKE **0.2742** against 0.3216 for the median point). Every other deep model in the study is retransformed before scoring — the Phase-3 LSTM by `exp(log_rv + smear_var/2)`, MC-Dropout by the log-normal mean of its predictive law — and the pinball head was the only one that was not, so it was being ranked on an estimand it was never trained to produce. The row is **unranked** in the master table: it is the same forecast under a different retransformation, not a rival, and ranking it would have shifted every published rank below it.

**BUILT 2026-09-02 (x) — the phase is code-complete and nothing has been run.** The
chain below exists, imports, and is covered by 42 new tests; no artefact has been
produced, because the deep model has to be fitted in the Windows 3.12 environment
where the Phase-9 gate will re-run it. What was added:

- **The asset registry finally has a consumer.** `resolve_asset` and
  `build_econometric_frame(asset=)` were added on 2026-08-30 and **no runner
  passed either**, so a Phase-8 profile naming `RUT` would have been run on
  `.SPX` and printed a plausible table. All six runners now build their frame
  through `datasets.frame_for_profile(data_cfg, profile)`, which reads the
  profile's optional `asset` key; every pre-Phase-8 config omits it and is
  therefore unchanged. `run_econometric` also stopped titling every chart
  "S&P 500" unconditionally.
- **`configs/{econometric,hmm,regime_lstm}_rq4.yaml`** — two profiles each
  (`rq4_rut`, `rq4_ftse`). The regime config copies the headline's pre-registered
  jump-penalty grid, sensitivity grid, K grid, persistence floor and emission
  feature verbatim, and `tests/test_rq4.py` asserts the copy is exact, so
  widening a grid to make an asset behave breaks the suite.
- **`src/experiments/run_rq4.py`** — the cross-asset assembler. Reads the finished
  per-asset artefacts, computes DM + regime-conditional GW of `Regime-LSTM-B`
  against each baseline, and writes `m08_rq4_{metrics,ranks,tests,per_regime,
  regimes,sample,verdict,rank_matrix}.csv` plus `m08_rq4.md`. **The S&P reference
  is read from the committed Phase-5 artefacts at run time, never hardcoded.**
  The verdict comes from one tested function, `transfer_verdict`, whose branches
  are exhaustive and whose `indeterminate` case is real.
- **The multi-profile milestone trap, closed.** Every runner writes its milestone
  *inside* the profile loop, so a two-profile config would have written one note
  and kept only the last — invisible until Phase 8, the first config with two.
  `src.utils.config.milestone_path` now formats a `{profile}` placeholder and
  **raises** when a multi-profile config lacks one.
- **`configs/lstm_baseline_refit.yaml`** — the `refit_every_folds: 1` sensitivity
  Ch3 §3.6 promises Chapter 4 will report. Sweep off, hyperparameters pinned at
  the Phase-3 selection, so the contrast isolates the cadence.
- **`configs/econometric_loghar.yaml`** — log-HAR through the harness at last, so
  the "the LSTM only wins because it models logs" objection is answered by a
  regenerable artefact rather than a one-off check.
- **`tests/test_rq4.py`** — 42 tests: every verdict branch, rank agreement,
  `milestone_path`, the pre-registered-grid assertions, the asset threading, and
  an end-to-end pass over synthetic Phase-8 artefacts in a tmp tree.

**Read-only dry check of the data path (2026-09-02, no artefacts written).** Both
robustness frames build cleanly and their samples differ, which is the thing
Ch3 §3.9 commits to reporting rather than absorbing:

| asset | rows | target rows | lost to the calendar join | test days | VIX |
|---|---|---|---|---|---|
| `.SPX` | — | 5552 | — | 784 | joined |
| `.RUT` | 5528 | 5550 | 0 (0.00%) | 788 | joined |
| `.FTSE` | 5562 | 5586 | 2 (0.04%) | 795 | **refused, with the reason logged** |

`rv` is strictly positive on every test window, as Ch3 §3.2 requires. The FTSE
frame carries no `vix_close` column at all — the registry refuses it in the data
path rather than leaving it to be noticed in a results table.

**RUN ORDER (Windows, from `Code\`).** Regimes before the deep run, because
Phase 5 consumes Phase 4's parquet; `run_rq4` last, because it only reads.

```
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m src.experiments.run_econometric --config econometric_rq4
.\.venv\Scripts\python.exe -m src.experiments.run_regimes     --config hmm_rq4
.\.venv\Scripts\python.exe -m src.experiments.run_regime_lstm --config regime_lstm_rq4
.\.venv\Scripts\python.exe -m src.experiments.run_rq4
```

Then the two sensitivities, independent of RQ4 and of each other:

```
.\.venv\Scripts\python.exe -m src.experiments.run_econometric --config econometric_loghar
.\.venv\Scripts\python.exe -m src.experiments.run_lstm --config lstm_baseline_refit
```

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
- Multi-asset robustness on additional assets beyond `.RUT` + `.FTSE`

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
- ~~Robustness assets AAPL + TSLA, baselines + best DL only, Week 11~~ — **superseded 2026-08-23 (vii): `.RUT` + `.FTSE`**, baselines + best DL only (§1.4, Phase 8). The library carries no single-stock realized measures, so AAPL/TSLA had no target matching the study's.
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

## 11. Chapter 3 — the declarations checklist

Chapter 3 (Methodology) **does not exist yet**, and it is the largest single risk
in the project. It is a *writing* risk, not a technical one, which is precisely
why it keeps being deferred: every phase closes cleanly without it.

Fifteen decisions have been resolved during the build with the note "state this
in Chapter 3", and until 2026-08-30 they lived only in audit reports and session
memory — neither of which survives into a submitted dissertation. They are listed
here, in the plan-of-record, so the chapter has a specification rather than a
recollection to write from. Each is a decision that was **made deliberately** and
would look like an oversight if a reader met it undeclared.

**Estimation and protocol**

1. **The refit asymmetry.** Deep models are trained once on data ≤ 2018-12-31 and
   frozen (`refit_every_folds: 0`); GARCH/EGARCH/HAR-RV refit every 21 trading
   days on an expanding window. Both are leakage-free, the asymmetry runs
   *against* the deep models, and Ch1 §1.7 explicitly promises Chapter 3 examines
   it. The `refit_every_folds: 1` Phase-8 sensitivity is the evidence.
2. **The HMM's own training window is shorter still** — the regime model is fit on
   ≤ 2015-12-31 (the training split proper) and filtered forward without refit,
   while the models that consume its posterior are fit on ≤ 2018-12-31. This is
   deliberate and conservative: K and λ are *selected* on training rows, so
   refitting after selection would be the symmetric alternative and was not taken.
   `m04` discloses it; m05/m06/m07 do not, so Chapter 3 must.
3. **The λ *grid* was pre-registered, not just the selection rule.** Fixed
   2026-05-15, before any Phase-5 result existed. A finer grid would make the
   literal rule pick 1.5 — one grid point from the λ=1.0 collapse — so the
   sensitivity is published rather than the rule quietly redefined. Say that the
   grid, not only the rule, was fixed in advance; otherwise the defence sounds
   post-hoc even though it is not.
4. **K = 3 is defended on pre-registration and interpretability, never on BIC.**
   The margin over K = 4 is 4.8 points and the jump model's own BIC prefers K = 4.
   Guidolin (2011), already cited in Ch2 §2.4, is the interpretability argument.
5. **The "best of Phase 5" selection rule** — Regime-LSTM-B was chosen on the
   *transitional* result, the only bucket with a formal test behind it. Calm and
   crisis are point estimates on 379 and 76 days, and crisis reverses sign between
   regime estimators; selecting on either would be choosing an architecture on
   noise. The rule was fixed before Phase 7 ran.

**Scoring and inference**

6. **MAE sits outside Patton's (2011) proxy-robust class** and is therefore
   descriptive only — it never ranks models. Ch1 §1.7 already says so; Chapter 3
   should give the reason rather than the rule.
7. **The Giacomini–White estimation-window condition is violated on one side of
   every deep-vs-econometric pair.** GW assumes a finite estimation window; the
   frozen deep models satisfy it, the expanding-window econometric baselines do
   not. The test is applied in the form standard in the applied volatility
   literature, and the qualification is declared rather than discovered at the
   viva. Ch2 §2.7 records it; Chapter 3 owns it.
8. **The GW long-run covariance is mean-centred, which makes the test mildly
   over-sized.** Direction and magnitude are both measured, not hand-waved:
   Regime-LSTM-B vs HAR-RV is 40.132 centred against 38.178 uncentred (+5.1%,
   p 9.99e-09 → 2.59e-08); Regime-LSTM-A vs HAR-RV 22.884 against 22.235 (+2.9%).
   No conclusion changes at any conventional level. Centring is kept because it is
   what `diebold_mariano` uses, so the documented q = 1 DM/GW equivalence holds
   exactly.
9. **Subsample coverage is tested by difference-in-coverage and incremental DQ,
   never by a nominal Kupiec test on the subsample.** These intervals are too
   narrow globally, so a nominal test rejects on *any* subsample with power — it
   would report a level failure as a timing failure. See
   [[feedback-level-vs-timing]].
10. **The Holm family is (method, level, side), seven members**, mixing the four
    difference-in-coverage tests with the three incremental-DQ tests. This must
    reach Chapter 4 or the headline "5 of 16" is unreadable against a family of 7:
    the 16 is a count of model × conditioner rows, the 7 is the correction unit.
11. **The quantile model's point forecast is a median, not a QLIKE-optimal
    functional**, and is now reported both ways — see Phase 8 above and
    `run_uq.add_quantile_mean_column`. Chapter 3 states the estimand; Chapter 4
    reports both rows.

**Implementation choices that diverge from what Chapter 2 describes**

12. **The quantile head is monotone by construction** (`base + cumsum(softplus)`),
    so quantile crossing is impossible for any weights. Ch2 §2.6 correctly reports
    non-monotonicity as a caveat *in the literature*; Chapter 3 is where our
    implementation's removal of it belongs. **Do not edit Ch2** — the chapter is
    accurate about the literature, and rewriting a review to match our own code is
    the retrofitting this project has avoided throughout (see
    [[feedback-chapter-role-separation]]).
13. **`LSTM-Gaussian` is a declared reference ablation**, not a third UQ method:
    the MC-Dropout network's deterministic forward pass wrapped in an
    aleatoric-only interval, isolating what the T stochastic passes add. It
    appears in the master table but in neither chapter.
14. **`RW-RV` is a declared reference floor**, not a baseline. Ch1 §1.5(i)
    enumerates three econometric baselines; the random walk is a fourth column in
    every table.
15. **The RV-only LSTM's actual input set** — `[log_rv]` over a 10-day window,
    against HAR-RV's daily/weekly/monthly aggregates reaching 22 days. Same
    information *set*, different regressor construction, and less history reach,
    which makes it the more conservative control. Ch1 §1.5(ii) was corrected on
    2026-08-30 to say so; Chapter 3 gives the numbers.

**Provenance**

16. **One environment per artefact set** — every phase's run snapshot stamps the
    interpreter and library versions (`_environment` in
    `experiments/*/run_*/config.yaml`), and Phases 2–7 were regenerated in a single
    environment on 2026-08-24 for exactly this reason. `torch` is pinned in
    `requirements-lock.txt` as of 2026-08-30.

---

## Changelog

### 2026-09-02 (x.2) — RQ4 has an answer; three defects found by running it

**RQ4 answered.** `.RUT` **replicates** (Regime-LSTM-B QLIKE 0.2143 vs HAR-RV
0.2260; pooled DM p=0.153 n.s.; regime-conditional GW p=0.0087) — the same
conjunction as the S&P: ahead in level, invisible pooled, significant
conditional on the lagged state. `.FTSE` is **direction only** (0.2905 vs 0.2997;
DM p=0.605; **GW p=0.254**) — the deep model is still ahead, but the
state-dependence the S&P finding rests on does not reproduce across the market
and session boundary. All three assets selected the same pre-registered
**lambda = 3.0**, and `.FTSE`'s BIC prefers **K=4** for the Baum-Welch HMM where
`.SPX` and `.RUT` prefer 3.

**Both sensitivities answered.** Log-HAR QLIKE **0.4886** against 0.2745 in
levels, through the real harness, with RW-RV reproducing the headline exactly
(0.358432, identical to 16 digits) as the same-folds anchor — the "the LSTM only
wins because it models logs" objection is dead on the record rather than in a
one-off check. And `refit_every_folds: 1` gives LSTM **0.2607** against the
frozen **0.2650**: refitting at the baselines' cadence *does* help by ~1.6%, so
the headline protocol was costing the network something and its result is a
conservative lower bound — which is what Ch3 §3.6 predicted and could not
resolve.

**Three defects, all found by running, none by the suite.**

1. **`run_econometric` was the one runner not routed through `milestone_path`.**
   Both Phase-8 econometric configs therefore wrote to `m02_econometric.md` and
   **overwrote the headline Phase-2 note twice**. The tables and parquet are
   profile-named and untouched (still 2026-08-24); only the note was lost, and
   it regenerates. The unit test covering this tested `milestone_path` *itself* —
   the exact failure the project's own rule warns about, one layer up: a test
   that exercises a function in isolation does not certify that anything calls
   it. There is now a source-level AST guard asserting every runner's
   `write_milestone` call takes its destination from `milestone_path`.
2. **The selected penalty came back blank in `m08_rq4_regimes.csv`.**
   `_regime_summary` looked for a `selected` flag on the lambda sweep table,
   which Phase 4 does not write. It now reads the causal-signals table — the
   lambda the run actually used for the signal Phase 5 consumed — rather than
   re-implementing the selection rule, and warns loudly when it finds nothing.
   This is the single most important portability number in the phase.
3. The synthetic fixture behind the end-to-end test had **invented column names**
   Phase 4 does not use, which is why (2) passed its test. The fixture now
   matches the real schema.

Suite **398**.

### 2026-09-02 (x.1) — the first RQ4 run: `.RUT` clean, `.FTSE` crashed, and the third layer of the same defect

**`run_regimes` read `frame["vix_close"]` unconditionally.** The `.RUT` profile
completed; `.FTSE` died with a `KeyError` at the first line that touches the
frame, because `assets.registry.FTSE.vix_feature: false` correctly refuses the
column. Third instance in three days of one shape: **the data layer was right
and its consumer had not been told.** (i) the registry had no consumer; (ii) the
runners did not pass `asset=`; (iii) a runner assumed a column the registry
deliberately withholds.

The fix is not a guard. `vix_confusion` is generalised to
`state_concordance(state, series, K, key=)`, and an asset with no VIX is
validated against **its own realised variance** — which is still external to the
*estimator*, since the regime model is fitted on standardised daily log returns
and never sees `rv`. The two are different diagnostics, so nothing lets them be
compared as one number: `_vix_confusion.csv` vs `_rv_concordance.csv`,
`_vix_by_state.png` vs `_rv_by_state.png`, `spearman_state_vix` vs
`spearman_state_rv`, and a `validator` column on the agreement table. The VIX
branch is byte-identical — verified by recomputing the committed
`m04_regimes_rq4_rut_vix_confusion.csv` from its own parquet: the confusion table
matches exactly and both scalars reproduce to 16 digits.

**Two hardcoded literals found in the same pass**, both of the "verify by
running" family: the two regime-map figure titles said "S&P 500 realized
volatility" unconditionally, and the milestone's Scope paragraph said "on S&P 500
daily data" — so the `.RUT` note and charts on disk name the wrong index. The
milestone also *asserted* "Mean VIX rises monotonically with the regime index",
which held on `.SPX` and is exactly the claim a generator should not make before
looking; it is now computed, with a written branch for the non-monotone case.

**Consequence: re-run the whole `hmm_rq4` config, not just `.FTSE`.** `.RUT`'s
numbers are deterministic and will not move (seed 17, VIX path unchanged), but
its milestone and figures were produced by the code with the literals in them.
Suite **394**.

### 2026-09-02 (x) — Phase 8 built: RQ4 wired end to end, two latent traps closed

**The phase is code-complete and unrun.** Full detail in the Phase 8 section
above; the two findings worth carrying separately are both of the family this
project keeps producing — a declaration with no consumer.

**The asset registry had no consumer in the experiment layer.** `resolve_asset`
and `build_econometric_frame(asset=)` landed on 2026-08-30 with tests, and (ix)
recorded the phase as "startable except for the data pull". It was not: **not one
runner passed the asset**, all six calling `build_econometric_frame(data_cfg,
target=profile.target)`. A `rq4_rut` profile would have regressed Russell 2000
returns on S&P 500 realised variance — the exact defect the registry was built to
prevent, surviving *inside* the fix for it because the fix stopped at the data
layer. All six now go through `frame_for_profile`, and a test asserts the
profile's asset reaches `build_econometric_frame`. Related: `run_econometric`
titled every forecast chart "S&P 500" from a literal.

**Every runner would have silently discarded a milestone.** Each writes its note
inside the profile loop to a single filename, so a config with two profiles keeps
only the last. Harmless for eight months because every config had one profile;
Phase 8 is the first with two. `milestone_path` now raises rather than
overwrites.

Also: `join_attrition` had been recorded in `frame.attrs` since (ix) and read by
nothing, though Ch3 §3.9 commits to reporting it — `run_econometric` now writes
`m02_<profile>_sample.csv` per profile and `run_rq4` collates it.

Suite **389 passing** on the bridge (the four torch-dependent modules excluded
there, unchanged by this work).

### 2026-08-30 (ix) — audit, then the fixes: Phase-8 blockers cleared, Ch1 corrected, a scoring inconsistency closed

**Verification first.** Every headline statistic was re-derived from the committed parquets by independent code importing nothing from `src/`: the 14-model pooled QLIKE table, the GW χ² for B/A/LSTM against HAR-RV, seven DM pairs, the t−1 buckets 379/329/76, per-regime QLIKE, transitional DM p=4.294e-09, PICP and Kupiec at both levels, per-regime coverage 0.844/0.894/0.842, and both Holm counts (0 of 16 pooled, 5 of 16 upper-tail, the same five rows). All reproduce to printed precision, so the "result numbers pending re-check after the (viii) regeneration" item is **closed**. The Phase-5 inheritance guard re-measured at exactly **0.0** — bit-identical, better than the 2.22e-16 recorded. The suite was also run in a clean container on Python 3.11.15 / torch 2.13.0+cu130 / statsmodels 0.15.0, deliberately different from the stack that produced the artefacts: 345/345 passed, so the tests certify behaviour rather than one machine's floating point.

**Then the fixes.**

- **Asset registry** (`configs/data.yaml` → `assets.registry`). One block resolving each asset's yfinance ticker, OHLC cache alias, Oxford-Man symbol, session and VIX eligibility. The `assets:` block had **no consumer anywhere in the codebase**, which is why it still named AAPL and TSLA a week after (vii) replaced them: a stale value nothing reads cannot disagree with anything out loud. `src/data/ingest.py` now reads it too, so its hardcoded `YFINANCE_SYMBOLS` is a legacy fallback rather than a second, divergent list.
- **The Oxford-Man symbol is threaded through the data path.** `realized_variance_target` and `build_econometric_frame` take `asset=` / `symbol=`, defaulting to the previous behaviour exactly. Before this the target symbol was read from config unconditionally, so a Phase-8 profile would have paired Russell 2000 returns and OHLC with **S&P 500** realised variance — a silent, entirely plausible-looking error. `tests/test_datasets.py` (13 tests) is new: this file is the single entry point every phase uses and had no tests at all, which is the reason both defects survived.
- **Calendar attrition is reported.** The modelling frame is an inner join across sources; on `.SPX` that is invisible because every source shares the NYSE calendar. `frame.attrs["join_attrition"]` now records target rows, rows lost to the join and rows lost to warm-up, with a warning above 2%.
- **The VIX is refused for non-US assets** rather than silently joined — enforced by `assets.registry.<key>.vix_feature`, with the reasoning in the config beside the flag.
- **Interval metrics now share the point losses' NaN contract.** `picp` compared `NaN >= lower`, which is `False`, so a missing endpoint or realisation was scored as an interval **miss** — coverage understated by a plausible amount with no warning. It never fired on the complete `.SPX` panel, but RQ3 *is* the coverage question and Phase 8 adds assets on other calendars. `mpiw` and `winkler_score` returned `NaN` (loud, harmless); all three now drop non-finite rows and raise when nothing survives.
- **The quantile model's retransformation** — the substantive one. Every other deep forecaster is mapped to the mean scale before scoring (the Phase-3 LSTM by `exp(log_rv + smear_var/2)`, MC-Dropout by the log-normal mean of its predictive law) because the networks regress log variance while MSE and QLIKE are minimised at the conditional **mean** (Patton 2011). The pinball head was the sole exception, and its median sits below the MC-Dropout mean on **every one of the 784 evaluation days**, so it was being ranked on an estimand it was never trained to produce. `Quantile-LSTM-mean` applies the same correction using the model's **own** fitted spread — a conditional σ, and therefore a better scale estimate than the Phase-3 LSTM's constant `smear_var` — giving QLIKE **0.274237** against 0.321647 for the median point, level with HAR-RV's 0.274490 rather than well behind it. The row is **unranked**: it is one forecast under two retransformations, not two models, and ranking it would have shifted every published rank below it. Verified: all 13 ranked models keep their published rank and QLIKE to 5.6e-17. The m06 note previously carried this caveat in prose with no number attached — a caveat with no measurement behind it is the same failure as a claim with no test behind it.
- **Chapter 1, three factual corrections** (and Chapter 2 deliberately untouched, per [[feedback-chapter-role-separation]]): §1.5(ii)'s "inputs match those of HAR-RV **exactly**" was false — the RV-only LSTM is `[log_rv]` over 10 days against HAR's three aggregates reaching 22 — and now says "the same information set … supplied as raw daily lags rather than as Corsi's daily, weekly and monthly aggregates"; §1.6 declared *three* research questions while this file has carried a stretch RQ4 since May, and now declares four with RQ4 explicitly secondary and its reduced design stated in the same sentence; §1.8's "supplemented … by individual large-cap equities" now names `.RUT` and `.FTSE`. All three were errors about the study's own design, not results — correcting them is not retrofitting.
- **Sixth generated-prose overstatement fixed.** `run_uq.py` asserted the LSTM-Gaussian point reproduces the Phase-3 LSTM "exactly"; agreement is ~1e-6 relative, and the master table prints 4.484934e-08 beside 4.484935e-08 for the two rows. Now "to floating-point precision", with the reason.
- **`torch==2.13.0+cpu` pinned** in `requirements-lock.txt` — §9 requires pinned dependencies and Phase 9 re-runs from a fresh checkout; the deep models had no reproducible version outside the per-run `_environment` stamp.
- **§11 added**: the fifteen parked Chapter-3 declarations, previously resident only in audit reports and session memory.

**Phase 8's three scope decisions settled** — regimes re-estimated per asset (not transferred from the S&P), no LSTM-VIX ablation on `.FTSE`, RQ4 declared as secondary in Ch1. Reasoning in the Phase 8 section above.

**Suite 345 → 375.** The only step still outstanding before Phase 8 can run is the yfinance pull for `^RUT` and `^FTSE`, which needs the Windows machine.


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

**2026-08-18 — Report/code alignment audit; three mismatches closed in code.**
An audit of Chapters 1–2 against the implemented pipeline found six places where the reports and the code disagreed. Three were fixed now (the rest are listed below as open):
- **Nystrup jump penalty — code brought up to the chapter, not the reverse.** Ch2 §2.4 states the dissertation "adopts the Nystrup et al. (2020) jump-penalised estimator directly" and §2.8 conditions the regime-aware LSTMs on "a Gaussian-emission HMM **estimated with the Nystrup jump penalty**", but Phases 5–6 were consuming the plain Baum-Welch posterior (`hmm_filt_p*`); the jump model was only a Phase-4 comparator. New estimator `src/models/regime/jump_hmm.py::JumpPenalisedHMM`: the jump model's penalised state path supplies the state sequence, the Gaussian emissions and transition matrix are estimated from that path (Dirichlet pseudo-count keeps the chain irreducible), and the inherited forward recursion yields a genuine causal posterior. It subclasses `GaussianHMMRegime`, so every downstream consumer is unchanged. The jump model's own `predict_proba` was **not** usable for this: it is a softmax over the emission loss and ignores the jump penalty, i.e. exactly the persistence the chapter argues for. λ is re-selected on **training rows only** (`select_jump_penalty_causal`), satisfying the Phase-4 rule that the regularisation parameter is CV-selected on training-only data — the full-sample λ remains descriptive. Phase 4 now persists `jphmm_filt_p*`/`jphmm_filt_state` (**headline**) alongside `hmm_filt_*` (**robustness comparator**), plus a `causal_signals` table comparing the two. `configs/regime_lstm.yaml` and `configs/uq.yaml` point at the jump-penalised signal; `configs/regime_lstm_hmm.yaml` and `configs/uq_hmm.yaml` reproduce the Baum-Welch variant under the `intraday_2019_2022_hmm` profile. **The existing HMM-conditioned m05/m06 tables, parquets, figures and milestone notes were renamed to that `_hmm` profile so nothing is lost; Phases 4 → 5 → 6 must be re-run to regenerate the headline numbers, which may move.**
- **Giacomini–White (2006) implemented.** Ch2 §2.7 names it alongside DM as the inferential apparatus; it did not exist in code. `src/evaluation/significance.py::giacomini_white` adds the conditional predictive ability test (Wald/χ², Newey–West long-run covariance, PSD-safe by default and exactly DM-equivalent under `hac_divisor="n-k"`), with `regime_test_function` building the **lagged** one-hot regime indicators the test requires. `src/experiments/report_gw.py` runs it on any predictions parquet in seconds (no retraining) and writes `<stem>_gw_joint.csv` + `<stem>_gw_per_regime.csv`. This directly addresses the weakest claim in `m05`: that the regime edge "is concentrated in the crisis state" was asserted from 70 days of point estimates with no test attached.
- **Ch1 objective (i) and RQ1 now name HAR-RV** (with Corsi 2009 added to the Ch1 reference list). Both previously said "GARCH(1,1) and EGARCH" / "GARCH-family baselines", which omitted the benchmark the results actually hinge on.
- **Tests:** +22 (12 in `tests/test_jump_hmm.py`, 10 in `tests/test_significance.py`), including the causality invariant for the new posterior and the DM-equivalence identity for GW → suite goes 173 → 195.

**2026-08-19 — all phases re-run; three further mismatches closed; headline results restated.**
- **Phases 4 → 5 → 6 re-run on the jump-penalised signal.** λ selected on training rows only came out at 3.0 (identical to the full-sample value); the jump-penalised chain is more persistent (train-window expected durations 61/49/29 days vs Baum-Welch 41/29/26) and the two causal paths agree on **95.8%** of test days. The λ sweep also demonstrates *why* the persistence rule is used rather than likelihood: BIC rises monotonically with λ, i.e. it always prefers λ→0, the over-switching pathology.
- **The Phase-5 headline moved, and the plan-of-record now says so.** Regime-LSTM-B vs LSTM fell from DM p=0.034 (Baum-Welch) to p=0.544 (jump-penalised); the best pooled model is the regime-agnostic LSTM-RVonly. In exchange the Giacomini–White test found what the pooled tests were hiding — a large, replicable, transitional-state advantage over HAR-RV (p=1.0e-08). The crisis claim is retracted (see the Phase-5 line above).
- **A milestone-generator bug was fixed.** The Phase-5 note hardcoded its conclusions: it asserted "every regime-aware model beats every baseline" in the crisis state and printed "−6% lower" for a figure that was 5.7% *higher*. Every claim in those paragraphs is now computed from the table, including its direction. Same class of bug fixed in `run_uq.py` (a hardcoded "70 COVID-era days").
- **VIX auxiliary feature (Ch1 §1.8) implemented** as `LSTM-VIX` — identical architecture and hyperparameters plus log-VIX, so the row prices the feature alone. **QLIKE 0.2301 vs LSTM 0.2650, DM p=0.024**: the largest and only clearly significant model improvement in the study, and an independent replication of Poon & Granger's (2003) conclusion that option-implied information dominates historical-volatility models — i.e. an external validity check on the pipeline. It is **excluded from the Phase-5 comparison, DM pairs and MCS** via `comparison.exclude_models`, because it consumes information no other model on the board can see; including it would compare unequal information sets and pull the MCS for a non-architectural reason. Verified: with the exclusion in place the entire Phase-5 board is numerically identical to the pre-VIX run.
- **DM regularity diagnostics implemented** (`src/experiments/report_dm_diagnostics.py`; ACF with Bartlett bands, Ljung–Box, HAC-lag sensitivity), closing Ch2 §2.7's promise. Findings: the headline pairs are clean (Regime-LSTM-B vs HAR-RV: LB(20) p=0.48, lrv/γ₀=1.07), but **`LSTM-RVonly vs HAR-RV` is lag-sensitive** — p ranges [0.035, 0.081] across HAC lags 0–30. Its differential is significantly autocorrelated (LB(20) p=0.0002), so the naive iid variance manufactures significance and the reported HAC p=0.081 is the correct one. `Regime-LSTM-A vs LSTM` shows heavy persistence (LB(20)=203, lrv/γ₀=2.35) — the loss differential inherits the regime signal's own persistence, which is why a HAC variance is not optional in this design.
- **Ch1 prose fixed:** Yang–Zhang is now described as a target-validity diagnostic rather than a robustness check on the target.
- **Reproducibility note:** re-runs reproduce to ~9 significant figures, not bit-exactly (CPU thread-level float reduction ordering). Set `torch.set_num_threads(1)` if the Phase-9 gate needs exactness.

**2026-08-19 (ii) — pre-Phase-7 audit; evaluation-side defects closed, one headline finding corrected.**
An independent audit of Chapters 1–2 against the implemented pipeline verified that every headline number reproduces exactly from the persisted parquets and that the leakage discipline holds, and found one methodological defect plus a set of smaller ones. All are now closed. *(This audit's own write-up was a working file at the repo root and has since been deleted; this changelog entry is the surviving record of it.)*

- **Per-regime tables were conditioned on the outcome — fixed, and it changes RQ3.** `per_regime_qlike` and `per_regime_interval_metrics` bucketed day *t* by `reg_state[t]`, the causal filtered state — which is inferred **using the day-t return**. Bucketing a day's forecast error (or interval coverage) by a label that saw that day's outcome is selection on the outcome: the forecasts stayed clean, the *evaluation* did not. Both helpers now default to the **t−1 label** (`src/evaluation/regime_timing.py`, `DEFAULT_REGIME_SHIFT = 1`) — the label the regime models actually condition on, and the same one `regime_test_function` already fed to Giacomini–White, so the descriptive tables and the formal test finally answer the same question. Consequences, all now reflected in `m05`/`m06`:
  - **RQ3's headline is corrected.** The "coverage degrades monotonically with regime severity" claim does not survive: under the t−1 label the ordering is non-monotone (MC-Dropout calm 0.844 / transitional 0.894 / crisis 0.842) and calm-vs-crisis is 0.2 pp, inside noise on a 76-day bucket. **Replaced by a stronger and convention-free finding: coverage collapses at regime *transitions*** — MC-Dropout 0.712 vs 0.876 stable, quantile 0.673 vs 0.832, on the 52 days (6.6%) when the state changes. New table `m06_uq_*_transition_vs_stable.csv`. This is the sharper claim for a risk manager and it points Phase 7 at the right conditioning variable: the *change* in the regime posterior, not its level. — ⚠️ **This replacement was itself retracted on 2026-08-19 (iii): see that entry.** The transition selector `1{s(t) ≠ s(t−1)}` reads the day-t return, so the split repeats this same defect one level down; "convention-free" was the wrong test, "implementable" is. The effect vanishes on the ex-ante selector. Kept here unedited as the record of what was believed at the time.
  - **Two RQ2 per-regime rows flip.** Calm reverses (HAR-RV 0.3008 now beats LSTM 0.3187, the opposite of the ex-post table) and crisis goes from "deep fails by 33%" to a near-tie (HAR-RV 0.3070 vs LSTM 0.3131). Only the transitional finding survives — and it strengthens, which is why the GW-based headline was the trustworthy one all along.
  - Both timings are now written to disk and printed in the milestones, each stamped with `regime_shift`/`timing`, with the ex-post version explicitly marked *descriptive only*. The milestone prose is computed in both directions, so a future re-run cannot silently reassert a claim the numbers no longer support.
- **`--from-predictions` added** to `run_regime_lstm` and `run_uq`. Rebuilds every table, figure and milestone section from the persisted predictions without retraining — exact for any artefact that is a function of the forecasts alone, which is what an evaluation-convention fix needs, and it avoids the ~9-significant-figure drift a retrain introduces. Used to regenerate `m05`/`m06` here; the prediction parquets are byte-identical (md5-verified).
- **`walk_forward_folds` off-by-one fixed.** When `len(oos_dates) % refit_frequency_days == 1` the final OOS day was silently never predicted. It did not bite at 784/21 (r 7), so no result moves, but Phase 8's assets have different calendars. The closing boundary is now unconditional; regression test sweeps 3 cadences × ~130 segment lengths and asserts exact coverage.
- **CRPS implemented** (`crps_lognormal`, closed form, verified against a Monte-Carlo evaluation of the CRPS definition to <1%; `crps_from_quantiles`, trapezoidal, shown to converge to it). Closes Ch2 §2.6/§2.7's commitment to Gneiting–Raftery proper scoring rules. Note `mean_pinball_loss` was **not** the CRPS and no longer claims to be. New table `m06_uq_*_crps.csv`.
- **λ grid sensitivity published.** The pre-registered selection grid is coarse between 1 and 3; on a finer grid the literal rule ("smallest λ clearing the 15-day persistence floor") would pick λ=1.5 rather than 3.0. λ=1.5 sits one grid point from the λ=1 collapse (3.3-day regimes — the exact over-switching pathology the penalty exists to cure), while λ ∈ [2, 3] is a stable plateau agreeing with the headline on ≥96.7% of test days. **Decision: keep λ=3.0 on pre-registration grounds** — the grid was fixed 2026-05-15, before any Phase-5/6 result existed, and re-selecting from a wider grid after seeing the downstream numbers is a forking-paths problem. The full sweep and a λ-agreement table are now reported (`m04_regimes_*_jump_lambda_sensitivity.csv`, `configs/hmm.yaml::jump_penalty_sensitivity_grid`) so the choice is auditable rather than asserted. Verified: re-running Phase 4 reproduces the headline `jphmm_filt_*` columns **bit-for-bit**, so nothing downstream moves.
- **Refit-cadence asymmetry now disclosed** in `m05`/`m06` scope blocks and in Ch1 §1.7 (deep models trained once and frozen; econometric baselines refit every 21 days). For Phase 6 this is the mechanism behind the near-constant interval width, which is stated where the intervals are reported.
- **Giacomini–White estimation-window caveat documented** in `significance.py`: GW assumes a finite/rolling estimation window; the anchored econometric baselines formally violate it (the frozen deep models do not). Standard in applied work, but now declared rather than discovered at the viva.
- **"Duan (1983) smearing" renamed throughout** to the parametric log-normal retransformation `exp(μ̂ + σ̂²/2)` it actually is. Duan's estimator is the non-parametric `mean(exp(ε̂))`; the maths was right, the citation was not.
- **Repo hygiene closed** (previous open item 5): `results/{tables,predictions}` and `data/raw/oxfordman` are now tracked, and milestone tags `m01`–`m06` exist. A fresh checkout can now reproduce.
- **Tests:** 195 → 230. New `tests/test_regime_timing.py` (18), CRPS tests, the walk-forward coverage regression, and updated per-regime calibration tests including one asserting the two timings are *not* interchangeable.

*(Previous open items 1–4 from the 2026-08-18 audit were already closed by the 2026-08-19 (i) work — VIX implemented as `LSTM-VIX`, Yang–Zhang reworded, MCS added to Ch1 §1.7, DM regularity diagnostics implemented — and the list has been pruned accordingly.)*

**2026-08-19 (iii) — the RQ3 transition headline is retracted; the *selector* now obeys the same timing rule as the label.**
A second independent audit found that the 2026-08-19 (ii) fix was necessary but not sufficient. It moved the per-regime **bucket label** into the t−1 information set, which was right. But the replacement RQ3 headline — *"coverage collapses at regime transitions"*, MC-Dropout 0.712 vs 0.876 — split on `1{s(t) ≠ s(t−1)}`, and `s(t)` is inferred from the day-t return. A regime transition is **detected by** the same one-day surprise that breaks a prediction interval, so the subsample was chosen using the outcome being measured: the identical error, one level down. It was defended as "convention-free" (both label timings agree on which days those are), and convention-invariance turned out to be the wrong test — implementability is.

- **Measured, and the selection is the entire effect** (MC-Dropout, 90% nominal, 784 test days, 52 flagged either way):
  - ex-post `1{s(t) ≠ s(t−1)}`: 0.712 flagged vs 0.876 rest — a −16.4 pp gap; Kupiec LR 14.4, p = 1.5e-04
  - ex-ante `1{s(t−1) ≠ s(t−2)}`: **0.885 vs 0.863** — +2.1 pp, i.e. nothing; Kupiec p = 0.72
  - top-decile ex-ante regime uncertainty `1 − max_k P(s(t−1)=k)`: 0.886 vs 0.862 — also nothing
  - mechanism: realized variance on ex-post transition days runs ≈2× the median of stable days
- **Code.** `src/evaluation/regime_timing.py` gains `DEFAULT_SELECTOR_SHIFT = 1`, `transition_mask`, `selector_timing_label` and `selector_is_implementable`. `transition_day_summary` now takes `selector_shift` (default 1) and returns `selector` / `implementable` alongside the mask; the old `shift=` kwarg raises `TypeError` with a pointer rather than silently answering the ex-post question. `run_uq.transition_vs_stable_table` takes the same argument and stamps every row, and the milestone writes **both** timings — ex-ante as the headline, ex-post behind a `<details>` marked descriptive-only — with the retraction computed from the tables in both directions.
- **Phase 7 is re-pointed.** Kupiec / Christoffersen / DQ move from the transition subsample to the full sample with lagged regressors (see the Phase-7 section). Running them on the ex-post split would have attached a p-value to an artefact — and that table was slated for Chapter 4.
- **Also fixed in the same pass:** (a) a variable-shadowing bug in `run_regimes.py` — the λ-sensitivity loop used `lam` as its loop variable, overwriting the selected full-sample penalty, so the agreement table, the milestone prose and the jump regime-map figure caption all reported **λ=300** (a degenerate 0-jump solution) instead of the λ=3.0 actually fitted; models were never affected, three reported artefacts were. (b) `m05`'s Findings prose compared the best regime model against an *unmatched* regime-agnostic baseline and declared per-bucket winners by min-over-models; both now use matched pairs (same features, same hyperparameters), and the best-of-family line is explicitly labelled a selection statistic. (c) the m04 persistence table now says in the note that it describes the Baum–Welch chain, not the jump-penalised one.
- **Tests:** 230 → 238. New `TestSelectorTiming` in `tests/test_regime_timing.py`, including one that perturbs day *t* and asserts the ex-ante selector is invariant while the ex-post selector is fully determined by it, and one that rejects the legacy `shift=` kwarg.
- **Regenerate:** `python -m src.experiments.run_regimes` (λ reporting), then `run_regime_lstm --from-predictions` and `run_uq --from-predictions`. No retraining; the prediction parquets do not move.

**2026-08-22 (iv) — the label *source*; the descriptive and Giacomini–White tables now score the same 784 days.**
A third audit found the same defect family one step further out. *(Its write-up, like the first audit's, was a working file at the repo root and has since been deleted; this entry is the surviving record.)* The (ii) and (iii) fixes got the bucket *label* and the transition *selector* into F_(t−1); the *source* of the label was still wrong in one place. `report_gw.py` read `reg_state` out of the **predictions** parquet — already sliced to the test window — and only then lagged it, so the first out-of-sample day (2019-01-02) had nothing to lag into and was dropped.

- **Consequence: two published values for one cell.** `run_regime_lstm.per_regime_qlike` goes through `align_regime_label` on the full Phase-4 series and scored n=784 with a 76-day crisis bucket; `report_gw` scored n=783 with 75. The same quantity therefore appeared twice in `results/tables/` with different values — HAR-RV crisis QLIKE **0.3107** in `gw_per_regime.csv` against **0.3070** in `per_regime.csv` (LSTM 0.3159 vs 0.3131; Regime-LSTM-B 0.3251 vs 0.3211). Both files feed Chapter 4. The transitional bucket agreed (n=329 either way), so the 21.8% headline was never affected.
- **Fix.** `report_gw` now resolves the Phase-4 predictions parquet alongside its input and takes the causal hard state from there (`jphmm_filt_state`, or `hmm_filt_state` for a `_hmm` profile), overridable with `--regime-source` / `--state-col`; if the file is missing it falls back to the old column and says loudly that the numbers will not reconcile. `regime_test_function` gains an `index=` argument and now delegates its lag to `align_regime_label`, so **one shift implementation** remains in the project instead of three. `report_gw` also asserts the coverage invariant explicitly — every evaluated day must land in a bucket — and logs the counts.
- **Structural guard, not a patch.** `regime_timing.label_reach` reports whether a state series extends far enough back to satisfy a requested lag, and `align_regime_label(..., warn_unreachable=True)` raises `RegimeLabelReachWarning` when it does not. The three per-regime table builders (`per_regime_qlike`, `per_regime_interval_metrics`, `report_gw`) opt in; the default is off because index arithmetic alone cannot distinguish "you sliced it" from "that is the whole series" — only the caller knows.
- **Nothing moved.** Regenerated GW tables: Regime-LSTM-B vs HAR-RV χ²(3) 40.135 → **40.132** (p 9.978e-09 → 9.990e-09); A vs HAR-RV 22.907 → 22.884 (p 4.22e-05 → **4.27e-05**, so the Phase-5 line above now reads 4.3e-05); the transitional row is bit-identical. All 30 per-bucket QLIKE cells in `gw_per_regime.csv` now match `per_regime.csv` exactly. `m05`'s tables and milestone are unchanged — it never embedded a GW number — so no `--from-predictions` re-run was needed.
- **Tests:** 238 → 249. New `TestLabelReach` in `tests/test_regime_timing.py` (the pre-sliced source costs exactly one labelled day; the warning fires on the wrong source and not the right one; buckets sum to the evaluated days) plus two cross-path tests asserting the Giacomini–White one-hot selects exactly the rows the per-regime table averages over — the assertion that would have caught this.
- **Regenerate:** `python -m src.experiments.report_gw` and the same with `--predictions results/predictions/m05_regime_dl_intraday_2019_2022_hmm.parquet`. Seconds, no retraining.

**2026-08-22 (v) — the Baum-Welch comparator profile was never re-run after (ii)/(iii); regenerated, and the robustness claim gets stronger.**
Refreshing `results/` surfaced that the `_hmm` regime-estimator comparator had been left behind by the two earlier evaluation fixes. Its `m05_*_hmm_per_regime.csv` and `m06_*_hmm_per_regime_{mc,q}.csv` carried no `regime_shift`/`timing` columns and reproduced the **ex-post** bucketing exactly — i.e. the comparator was still on the contemporaneous label the headline retracted in (ii), and it was missing every artefact (ii)/(iii) added (`_per_regime_expost`, `_crps`, `_sigma_scale`, `_transition_vs_stable`). The robustness claim was therefore comparing a t−1-bucketed headline against an ex-post-bucketed comparator.

- **Regenerated** with `run_regime_lstm --config regime_lstm_hmm --from-predictions` and `run_uq --config uq_hmm --from-predictions`. No retraining; the pooled tables (`_metrics`, `_mcs`, `_dm`, `_calibration`, `_point`, `_reliability`, `_gw_*`) are byte-identical — only the per-regime tables were stale, which is exactly the expected footprint.
- **RQ2 replicates better than the stale table showed.** On the t−1 label the comparator now gives Regime-LSTM-B vs HAR-RV **−19.0%** in the transitional state (0.1951 vs 0.2409, n=332) against the headline's −21.8% (0.1851 vs 0.2367, n=329). The stale ex-post table had shown only −2.0%, badly understating the replication. Calm agrees in both (HAR-RV ahead: +5.4% headline, +2.7% comparator). Crisis still flips sign between estimators (+4.6% vs −2.8%), which is the documented reason the crisis claim stays retracted — now demonstrated on matched conventions rather than across two.
- **RQ3's retractions are independently confirmed.** The comparator's per-regime coverage on the t−1 label shows **no degradation with regime severity** (MC-Dropout calm 0.848 / transitional 0.880 / crisis 0.887 — if anything it *rises*), matching the headline's non-monotone 0.844 / 0.894 / 0.842 in ruling out the anticipated pattern, whereas its stale ex-post table (0.866 / 0.868 / 0.843) still showed the monotone degradation retracted in (ii). *(Wording corrected 2026-08-22 (vi): this entry originally described the comparator's 0.848 / 0.880 / 0.887 as "non-monotone", which it is not — it is monotone increasing. The claim being tested is the absence of degradation with severity, and that holds under both estimators. The error came from the milestone generator, which asserted "non-monotone" in the branch that fires whenever coverage is not monotone decreasing; the generator is fixed.)* On the ex-ante transition selector there is again no separation (MC 0.848 flagged vs 0.866 stable; quantile 0.780 vs 0.825), confirming (iii) on the second estimator.
- **Headline profile verified unchanged.** `run_regime_lstm --from-predictions` and `run_uq --from-predictions` were re-run end-to-end and reproduce every committed headline table exactly, apart from line endings and one 16th-significant-figure p-value (`Regime-LSTM-A-RVonly vs HAR-RV` 0.18813555586249617 → 0.1881355558624962, a scipy-version ulp). The regenerated headline artefacts were therefore **discarded rather than committed**, to keep the pre-tag diff free of line-ending noise.
- **A convention sweep over the whole results tree** then found one remaining gap: `<stem>_gw_per_regime.csv` was the only conditional table not carrying its bucketing timing on the row. `report_gw.per_regime_table` now stamps `regime_shift` / `timing` like every other per-regime table, so the (ii) rule — never quote a per-regime number without its timing — holds without exception. Both profiles regenerated; `gw_joint` is unaffected. Remaining by-design asymmetry: the DM regularity diagnostics (`_dm_acf`, `_dm_lag_sensitivity`, `_dm_ljungbox`, `_dm_summary`) exist for the headline profile only, since Ch2 §2.7 asks for them on the headline pairs.
- **Everything else re-derived from the persisted parquets and confirmed current:** `m02`/`m03` pooled metrics (both profiles incl. the Yang–Zhang diagnostic), `m04`'s causal-signal and agreement tables (jump-vs-Baum-Welch test agreement 0.9579081633, switch counts 51/58, state counts 379/330/75 and 381/333/70), and all five rows of `m05_*_dm_summary` (statistics and `lrv/γ₀` to 1e-9).

**2026-08-22 (vi) — fourth audit; Phase-7 code built, and RQ3 is now inferential.**
A fourth independent audit re-read Chapters 1–2 against the pipeline, re-derived every headline statistic from the persisted parquets with from-scratch reimplementations that import nothing from `src/`, and ran the suite in a clean container on numpy 2.4 / pandas 3.0 / scipy 1.17 / torch 2.13. Everything reproduced; the leakage discipline holds. Then Phase 7 was built.

**Audit findings, all now closed in code or documented:**
- **A generated sentence was false.** `run_uq.py`'s milestone generator tested only `_monotone_down(...)` and then asserted *"the ordering is non-monotone"* in the else-branch. For the Baum-Welch comparator the per-regime coverage is 0.848 / 0.880 / 0.887 — **monotone increasing** — so `m06_uq_hmm.md` published a clause its own table contradicted, and it propagated verbatim into the (v) entry above (now corrected in place). The generator now classifies three ways (monotone down / monotone up / non-monotone) and prints a **binomial standard error** beside any "inside sampling noise" claim, which the branch previously asserted with no SE computed anywhere. Third bug of this family; the rule is now written down — *when a **word** in a generated note looks wrong, suspect the generator, not the model*.
- **Two dangling citations.** The changelog cited `AUDIT_2026-08-19_pre_phase7.md` and `AUDIT_2026-08-19_iv_alignment_and_phase7_readiness.md`; both were working files at the repo root and have been deleted. The entries now say so and stand as the record themselves.
- **A config explained itself by a retracted finding.** `configs/uq_hmm.yaml`'s "WHY THIS EXISTS" block still framed the comparator as testing whether *"coverage degrades in the crisis regime"* is an estimator artefact — retracted in (ii). Rewritten around what the comparator actually establishes: that the **null** (no per-regime coverage structure, no ex-ante transition signal) is a property of the intervals rather than of the jump-penalised estimator's partition.
- **The GW centring caveat is now sized and documented** in `_hac_matrix`. GW (2006) at τ=1 imposes the null, so the uncentred second moment is correct and centring makes the test mildly liberal. Measured: Regime-LSTM-B vs HAR-RV 40.132 → 38.178 (+5.1%, p 9.99e-09 → 2.59e-08); A vs HAR-RV 22.884 → 22.235 (+2.9%, p 4.27e-05 → 5.83e-05). No conclusion moves. Still belongs in Chapter 3.
- **`.gitattributes` added** (`* text=auto eol=lf` plus explicit binaries), fixing the mixed CRLF/LF split between the headline artefacts and the `_hmm` ones that was making every regeneration produce a whole-file diff. **One-time migration:** `git add --renormalize .` as its own commit before re-tagging.
- **`GSPC.parquet` and `VIX.parquet` are now tracked.** Confirmed from the call order, not just empirically: `run_regime_lstm.py` builds the modelling frame *before* the `--from-predictions` branch, and `datasets.load_ohlc` reads `data/raw/yfinance/GSPC.parquet`. A results-only checkout could not rebuild a single table, so the Phase-9 gate could not pass. ~1 MB; the rest of `yfinance/` stays ignored.
- **`m01` provenance noted:** its summary is in σ (mean √rv ≈ 0.008446) while every downstream table is σ², and its `n_obs` 5552 differs from the modelling frame's 5530. Both now have a reconciling note in `m01_data.md`. It is also notebook-generated, against §9's own "no notebook" standard — left as-is for now, flagged for Phase 9.
- **Ch1 ↔ Ch2 metric set and the Ch2 §2.6 conformal cross-reference** were internally inconsistent (Ch1 promises MSE/MAE/QLIKE, Ch2 §2.3/§2.8 promise QLIKE/MSE; §2.8 claimed conformal was set aside "for stated reasons" that §2.6 never gave). Both were minimal, results-independent edits and are done — see the note at the foot of this entry.

**Phase 7 — BUILT, tested, and ready to run.**
- **`src/evaluation/coverage_tests.py`** — Kupiec (1995) LR_uc, Christoffersen (1998) LR_ind / LR_cc, the Engle–Manganelli (2004) DQ test, a **difference-in-coverage** two-sample test (iid and Newey–West HAC), Holm–Bonferroni adjustment, and `dq_incremental`. **46 unit tests**, including size simulations under the null for every statistic and a test class that reproduces the 2026-08-22 (iv) level-vs-timing trap directly.
- **`dq_incremental` is the methodological point of the phase.** The DQ design contains a constant, so a globally mis-levelled model rejects DQ through that constant alone whatever the conditioners do — the same level-for-timing substitution as (iv), restated in χ² form. Since `DQ = Hit'P_X Hit / (α(1−α))` and `P_X − P_{X₀}` is a projection, `DQ_full − DQ_restricted ~ χ²(k−k₀)` isolates the added regressors. Every conditional row in the Phase-7 tables is that incremental statistic.
- **`src/models/deep/regime_uncertainty.py`** — `MCDropoutRegimeExpertForecaster`: Regime-LSTM-B with MC-Dropout, training inherited unchanged. Each of the T passes is recombined **through the gate before** the moments are taken, so the epistemic term is the variance of the mixture (which lets gate disagreement widen the band) rather than a mixture of per-expert variances. **16 unit tests**, one of which asserts the dropout-off mixture reproduces the Phase-5 point forecast exactly — that is what makes the Phase-6 → Phase-7 contrast attributable to regime conditioning and nothing else.
- **`src/experiments/report_coverage.py`** — parquet-only, like `report_gw.py`: no torch, no `data/raw`, runs in seconds against any UQ predictions file.
- **`src/experiments/run_combined.py`** + `configs/combined.yaml` / `combined_hmm.yaml` — the Phase-7 driver: master results table, calibration, per-regime, DM and GW on the UQ pairs, the global **and per-regime** σ-rescale table, a width/tail breakdown, two figures, and a generated `m07_combined.md` whose verdict is computed in both directions so it can report its own falsification.
- **Tests: 249 → 311.**

**RQ3 results already in hand** (these need no training — they are deterministic functions of the committed Phase-6 parquets, and both profiles are done):
- **Kupiec rejects decisively for every method, exactly as pre-registered.** MC-Dropout p=0.0017 (PICP 0.8648), LSTM-Gaussian p=0.0012, Quantile p=2.3e-11 (PICP 0.8214). The level is wrong.
- **Christoffersen independence does not reject, exactly as pre-registered.** p = 0.29–0.97 across methods and sides. The misses are unclustered.
- **The third clause is partially falsified, and this is the interesting part.** On the *pooled* two-sided hit the pre-registration holds — no t−1 conditioner survives Holm at 90% (regime incremental DQ p=0.071, Holm 0.42). But on the **upper tail** — realized variance *above* the band, the under-warning miss — there is real, implementable structure: MC-Dropout upper-violation rates on the t−1 label are **calm 0.1055 / transitional 0.0426 / crisis 0.0921** against a 5% nominal, and the calm-vs-rest (z_HAC 2.89, Holm p=0.019) and transitional-vs-rest (z_HAC −3.39, Holm p=0.005) differences both survive family-wise control. The lower tail runs the *other* way (calm 0.0501 / transitional 0.0638), which is exactly why the pooled view nets it out and Phase 6 saw nothing.
- **Robustness:** the direction replicates under the Baum-Welch comparator (calm 0.0997 / transitional 0.0542 / crisis 0.0704) but the significance is weaker — only the transitional row survives Holm there, at 95%. **State it as directionally consistent under both estimators, significant under the headline one, partially so under the comparator.** Do not overclaim it.
- **Reading:** the model's weakness is *calm-state complacency*. That coheres with Phase 5, where calm is also the one bucket HAR-RV beats the deep models in. The remedy is no longer purely a single global σ scale — it is a global scale plus a calm-state widening, and `m07_*_sigma_scale.csv` reports the per-regime scales beside the global one so the choice is made on evidence.

**What remains for Phase 7 (one command, and it must run on the user's machine):**
```
python -m src.experiments.run_combined
python -m src.experiments.run_combined --config combined_hmm
```
The combined model has to be **trained**, and every other deep model in this study was trained on the user's Windows environment. Training it anywhere else would put one phase's weights on a different platform and BLAS from the other four, and the committed `m07` numbers would then not reproduce at the Phase-9 gate. The code is smoke-tested end-to-end (`--fast`, synthetic frame) and every table, figure and milestone section renders; only the training run is outstanding.

**Still open:**
1. **Commit + re-tag.** Tags `m01`–`m06` and `phases-0-6-reproducible` still point *before* the (iv)/(v) fixes, so a fresh checkout at `m06-uq` reproduces the retracted headline. Do the `.gitattributes` renormalise commit first.
2. `refit_every_folds: 1` sensitivity for the best LSTM, to separate architecture from training-window staleness (Phase 8).
3. A log-scale HAR-RV sensitivity (`har.transform: log`, already implemented) to close the obvious RQ1 objection that the LSTM's edge is an artefact of modelling logs while HAR is fit in levels. A one-off check gives log-HAR QLIKE 0.4882 pooled vs 0.2745 in levels — the objection does not survive — but it should be run through the harness and reported rather than left as a claim.
4. **Chapter 3 does not exist**, and roughly a dozen resolved audit items are parked as "state this in Chapter 3" with nowhere to live: the refit asymmetry, the GW finite-window and centring caveats, the λ grid pre-registration, the MAE-outside-Patton's-class point, the "best of Phase 5" selection rule, and now the incremental-DQ rationale. That backlog is a risk in its own right and grows with every audit.

*(Chapter edits made in (vi), both minimal and results-independent: Ch1 §1.7 now states that MAE sits outside Patton's proxy-robust class and is reported descriptively, so it agrees with Ch2 §2.3/§2.8; Ch2 §2.6 now gives the reason conformal methods are set aside, which §2.8 had been referring to without one existing. Nothing else in either chapter was touched — Ch4 reports our results and Ch5 reconciles them.)*

**2026-08-23 (vii) — fifth audit; Phase 7 closed, one reproducibility defect found, RQ4's asset set decided.**
A fifth independent audit re-read Chapters 1–2 against the pipeline and re-derived the whole Phase-7 headline from the persisted parquets with from-scratch implementations of QLIKE, the closed-form log-normal CRPS, Kupiec, Christoffersen, the Engle–Manganelli DQ and a Newey–West difference-in-coverage test, importing nothing from `src/`.

**Phase 7 is complete.** Both profiles are genuine runs — headline 00:59 UTC and comparator 01:00 UTC on 2026-08-23, `MC samples: 100`, neither milestone carrying a `SMOKE RUN` banner — so the training run listed as outstanding in (vi) is done. Everything reproduces to printed precision: the 13-model master table (QLIKE/MSE/MAE and rank order), PICP 0.860969 / MPIW 1.8133e-4 / Winkler 4.0059e-4 / CRPS 4.0608e-5, Kupiec 11.982874 (p=5.369e-4), Christoffersen LR_ind 0.435527 (p=0.509289), DQ 16.880805 (p=0.004731), the t−1 buckets 379/329/76 = 784, the upper-tail rates under both regime estimators, and the "5 of 16 / 0 of 16" Holm counts. Report↔code alignment is tight: every commitment in Ch1 §1.5–§1.8 and Ch2 §2.3–§2.8 is implemented as written, and the two chapter edits made in (vi) have landed.

**The one material defect — the committed artefacts span two Python interpreters.**
Phase 7 claims the combined model *is* the Phase-5 mixture-of-experts with only inference changed. On the artefacts it is not. Rebuilding the Phase-5 point forecast from Phase 7's dropout-off column and removing the constant smearing offset leaves a log deviation with **sd 1.44e-2, reaching +0.182 on 2020-03-03**, concentrated in the crisis bucket where the crisis expert has the fewest effective training samples.
- **The control that makes this decisive is already in the repo.** Phase 6 re-trains the Phase-3 LSTM from the same config and reproduces it to **9.6e-7** — so a same-environment refit of this codebase is exact to float precision, and the Phase-5 → Phase-7 gap is roughly 19,000× that.
- **Cause.** `__pycache__` holds both `cpython-310` and `cpython-312` bytecode. `regime_lstm.cpython-310.pyc` was written 2026-08-18 20:40 and the Phase-5 predictions parquet at 23:21 the same day; `run_combined.cpython-312.pyc` was written 2026-08-23 00:58, minutes before the Phase-7 parquet. The checked-in `.venv` is Python 3.12.1 with the locked library set, so a re-run today reproduces Phase 7 and not Phase 5.
- **Second, smaller break.** The Phase-4 parquet was regenerated 2026-08-19 16:16, seventeen hours *after* Phases 5 and 6 trained against the previous copy. The hard states agree on every test day, so nothing downstream is wrong, but the committed deep-model weights were fit against a file that no longer exists.
- **Why nothing caught it.** The unit test that certifies the equality fits both objects inside a single process, so it can only catch a coding error, never cross-run drift. And `experiments/<phase>/run_<ts>/config.yaml` dumps the resolved config with **no interpreter, torch or numpy version** — §9 asks for config snapshots and pinned dependencies, and `requirements-lock.txt` does not pin `torch` at all.
- **Impact.** *(Numbers in this bullet are pre-re-run and are superseded by the (viii) entry below.)* No conclusion moves — the combined model is QLIKE 0.2599 against `Regime-LSTM-B`'s 0.2621, DM p=0.31, and every coverage result is a deterministic function of the internally-consistent Phase-7 parquet. What it costs is the Phase-9 gate ("re-run all experiments end-to-end from a fresh checkout"), which will not reproduce the committed `m05`/`m06` numbers, and the accuracy of the `m07` sentence attributing the Phase-6 → Phase-7 contrast to regime conditioning "and nothing else".
- **Remediation, before Phase 8.** Re-run Phases 2 → 7 in one session under `.venv` (3.12.1), then re-tag. Expect the m03/m05/m06 headline figures to move slightly — they will be 3.12 fits — so every table, milestone and quoted number in this file needs re-checking afterwards. Then close the hole: `run_combined` already loads the Phase-5 parquet as `baseline_predictions`, so comparing `mu_log_det` against the `Regime-LSTM-B` column it is holding and failing above a tight tolerance is a three-line guard; and the config dump should carry `sys.version`, `torch.__version__` and `numpy.__version__`.

**Generated-prose defect, fifth of the family.** Both Phase-7 milestones end the tail section with *"Both tails are worst in the same state, so the pooled view captures the structure adequately here"*, emitted by an `else` branch in `run_combined.py` that tests only `argmax(upper_rate) == argmax(lower_rate)`. The premise is true; the conclusion is refuted by the milestone's own table — pooled, calm-vs-rest is Holm p=0.189, while on the upper tail it is 0.089 with the incremental-DQ regime term at 0.035. An argmax comparison is not a test of adequacy. `tests/test_milestone_prose.py` has no coverage of the string.

**Three precision points for Chapter 4.**
- For the **combined** model, calm-vs-rest on the upper tail is Holm p=0.0889 — not significant at 5% — while the transitional row is 0.0494. The Holm-surviving *calm* rows belong to `MC-Dropout-LSTM` (0.019) and `LSTM-Gaussian` (0.013). Calm complacency is a property of the uncertainty-aware **family**; do not attribute it to the Phase-7 model specifically.
- The tested evidence is **one-sided** (upper tail) while the σ-rescale calibrates **two-sided** 90% coverage. In crisis the tails disagree (upper 0.092, lower 0.039), so a symmetric ×1.05 buys upper coverage by over-covering below — an asymmetric collar is the precise remedy (Ch5).
- **RQ3's second clause is now answerable in its own terms.** `m07_*_width_by_regime.csv` gives `mean_sigma_log` 0.5965 / 0.5956 / 0.5998 — flat across states — while MPIW spans 5.1e-5 to 9.6e-4 purely because the band is multiplicative on the point forecast. Width adapts, but only as a fixed collar.

**Smaller items.** All seven tags (`m01-data` … `phases-0-6-reproducible`) point at `9110b18a`, two commits behind HEAD, so no tag marks a distinct milestone and there is no `m07` tag; `2bff94cd` still holds the smoke-run artefacts. Ch2 §2.6 lists non-monotone quantiles as a live caveat while `QuantileLSTMRegressor` removes it by construction (cumulative soft-plus) — true of the literature, no longer true of this model, so Chapter 3 should say so.

**RQ4's asset set decided — `.RUT` and `.FTSE`.** See §1.4 and Phase 8 above. The audit found that RQ4 as written had no target: the Oxford-Man library carries 31 symbols and every one is an index, so AAPL and TSLA could only have been run on a Yang-Zhang proxy that Ch1 §1.8 reserves as a diagnostic. Both replacement series are already in the cached CSV, gap-free, over the identical window.

**2026-08-24 (viii) — Phases 2–7 regenerated in one environment. The fix worked; exactly one number moved, and the interpretation got cleaner.**
`Code\rerun_all.ps1` ran all thirteen steps clean in about fifteen minutes and the suite finished **338 passed**. Every phase's run snapshot now carries the same `_environment` block — python 3.12.1, torch 2.13.0+cpu, numpy 2.4.4, pandas 2.3.3, scipy 1.17.1 — and the artefact timestamps are in strict dependency order, so no downstream file predates its input. The Phase-5 inheritance guard passes at **2.22e-16** on both profiles, against 0.183 before: the combined model is now bit-identical to `Regime-LSTM-B` with dropout switched on, which is what the phase always claimed.

**Almost nothing moved.** Re-derived from the new parquets with from-scratch code: `GARCH` 0.330171, `EGARCH` 0.332751, `HAR-RV` 0.274490, `RW-RV` 0.358432, `LSTM` 0.265019, `LSTM-RVonly` 0.259640, `LSTM-VIX` 0.230147, `Regime-LSTM-A` 0.263209, `-A-RVonly` 0.261748, `-B` 0.262115, `MC-Dropout-LSTM` 0.265233, `Quantile-LSTM` 0.321647, `LSTM-Gaussian` 0.265019 — every one of them the previously published value. So are the Phase-4 posteriors (mean gate entropy agrees to 2.6e-7), the Phase-5 Giacomini–White table (B vs HAR-RV χ²=40.132076, p=9.989868e-09; A vs HAR-RV 22.883780, p=4.269931e-05), the 90% MCS membership, the per-regime QLIKE table, Phase 6's PICP (MC-Dropout 0.864796, quantile 0.821429), its per-regime coverage (0.844 / 0.894 / 0.842) and its σ scales (×1.14 coverage, ×1.24 CRPS). The Phase-3 sweep re-selected hidden 128 / lookback 10 / lr 1e-3, so the hyperparameters hard-coded in `regime_lstm.yaml` and `combined.yaml` are still the validated ones. **RQ1 and RQ2 are untouched.**

**This corrects the (vii) diagnosis.** The two-interpreter split was real, but it was not the cause. Every other model reproduces exactly under 3.12, so CPython 3.10 vs 3.12 does not move a weight in this codebase. What was wrong was confined to the previous Phase-7 run, whose experts differed slightly from Phase 5's — its inherited `√smear_var` was 0.59517 against Phase 5's 0.59604 — while the re-run, executed in one sequential session, produced bit-identical experts. The precise mechanism cannot now be recovered, because the old artefacts are overwritten. What matters is that the guard makes a recurrence impossible to publish: `run_combined` refuses to write a Phase-7 result whose point model is not the Phase-5 point model.

**What moved, all of it downstream of the combined model.**

| quantity | before | after |
|---|---|---|
| `MC-Dropout-Regime-LSTM-B` QLIKE | 0.259854 (rank 2 of 13) | **0.261940 (rank 3 of 13)** — behind `Regime-LSTM-A-RVonly` 0.261748 |
| its PICP @90% | 0.8610 | 0.8661 |
| DM vs `MC-Dropout-LSTM` | −1.734, p=0.083 | **−0.687, p=0.492** |
| DM vs `Regime-LSTM-B` | −1.019, p=0.308 | −1.587, p=0.113 |
| GW vs `HAR-RV` | χ² 39.34, p=1.47e-08 | χ² 39.83, p=1.16e-08 |
| GW vs `MC-Dropout-LSTM` | χ² 5.05, p=0.168 | χ² 4.82, p=0.186 |
| Kupiec LR_uc | 11.98, p=5.4e-04 | 9.16, p=0.0025 |
| Christoffersen LR_ind | 0.436, p=0.509 | 0.426, p=0.514 |
| σ scale, global | ×1.11 coverage / ×1.16 CRPS | ×1.11 coverage / ×1.18 CRPS |
| σ scale, per regime | calm 1.23 / trans 1.02 / crisis 1.05 | calm 1.24 / trans 1.01 / crisis 1.02 |

**Every RQ3 conclusion is unchanged.** The Phase-7 verdict table reads exactly as before: Kupiec rejects for every method, Christoffersen's independence component does not, no t−1 conditioner survives Holm on the pooled hit (0 of 16 at the 90% level), and **5 of 16** upper-tail difference-in-coverage tests do — the same five rows as before (`MC-Dropout-Regime-LSTM-B` transitional; `MC-Dropout-LSTM` calm and transitional; `LSTM-Gaussian` calm and transitional). The comparator profile is still 0 of 16. The width verdict is still **per-regime**, on a spread of 1.01–1.24.

**The interpretation gets cleaner, and one sentence must not survive into Chapter 4.** The old run made the combined model look 0.0054 better than `MC-Dropout-LSTM` at p=0.083 — a near-significant hint that regime conditioning improves the point forecast inside the uncertainty-aware family. On the corrected artefacts that gap is 0.0033 at p=0.492 pooled and p=0.186 conditionally. The combined model is now a null against its regime-agnostic sibling on both tests, while keeping its conditional edge over HAR-RV (χ²=39.8, p=1.2e-08). That is precisely the pre-registered expectation — *combining them should move the interval very little* — and it no longer rests on a fit that could not be reproduced. **Do not carry the old p=0.083 forward.**

**Both tidy-ups closed the same day.**
- **Generated-prose defect #5 is fixed.** The tail section used to conclude *"Both tails are worst in the same state, so the pooled view captures the structure adequately here"* from an `argmax(upper_rate) == argmax(lower_rate)` comparison — a premise that was true and a conclusion its own verdict table refuted (0 of 16 pooled against 5 of 16 upper). The argmax now prints only the descriptive fact, and the claim is decided by a new module-level `pooled_view_verdict(diff_pooled, diff_upper)` from the Holm-adjusted counts on each side, in four cases: *pooled understates* / *pooled is sharper* / *views agree* / **no structure either way**. That fourth branch is the one the old code could not express — when neither side finds anything, no adequacy claim is made in either direction. Every rendered sentence now carries its own counts. **7 unit tests** in `tests/test_milestone_prose.py`, including the regression on the real headline counts and an assertion that the word "adequate" cannot appear on the project's own data. Suite 338 → **345**.
- **The run snapshots are now tracked.** `snapshot_config`'s `_environment` block was being written into `experiments/**/run_*/`, which `.gitignore` excluded — so the record never reached a checkout, which was the whole point of adding it. Git cannot re-include a file whose parent directory is excluded, so the rule now excludes the run directory's *contents* (`experiments/**/run_*/**`) and re-admits the directory and `config.yaml`. 44 files, ~71 KB, and the interpreter behind every committed artefact is auditable from the repo alone.
- Two smaller defects in the same generated block, found while fixing the first: the milestone's *Reproduce* section printed `run_combined --from-predictions` without `--config`, so running it from the comparator's note would have rebuilt the **headline** tables; and it passed a bare filename to `report_coverage --predictions`, which `repo_path` resolves against the repo root and so could never exist. Both now render a command that runs as printed.
- `report_coverage`'s four `m06_uq_*_coverage_{tests,conditional}.csv` were not steps in the re-run script and still carried a 2026-08-22 timestamp; their content was re-verified against the current Phase-6 parquets and is identical to the digit (LSTM-Gaussian Kupiec 10.530278290051228, p=0.0011743452767293489). Regenerated alongside the milestone rebuild so every artefact carries one provenance.

---

*End of roadmap. This file is the canonical source of plan-of-record. Update it (with dated changelog entries above) whenever scope changes.*
