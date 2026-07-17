# Phase 2 — Implementation Report (Econometric Baselines)

**Project:** Regime-Aware Deep Learning for Financial Volatility Forecasting with Uncertainty Quantification
**Scope of this report:** ROADMAP.md Phase 2 — GARCH(1,1), EGARCH, HAR-RV, and the shared evaluation engine
**Date:** 2026-07-09
**Status:** Phase 2 gate passes. One scope decision needs your sign-off before it becomes the *headline* — see [§6](#6-the-intraday-splice-your-proposal-assessed).

---

## 1. Summary

Phase 2 is implemented in `Code/`. It adds the three confirmed econometric
baselines and — more importantly — the **evaluation engine** that every later
model (LSTM, regime-aware, UQ) will be scored through. The engine is built once,
here, exactly as ROADMAP §5 intends.

Both gate criteria pass:

| Gate | Roadmap criterion | Result |
|---|---|---|
| Phase 2 | GARCH(1,1), EGARCH, HAR-RV via a rolling-OOS harness with monthly refit | Implemented; 34-fold anchored walk-forward |
| Phase 2 | `results/m02_econometric.md` with error tables + forecast-vs-actual figure | Produced, all three baselines + reference floor |
| Tests | `pytest -q` passes | **91 passed** (41 prior + 50 new) |

**Headline result** (canonical intraday RV target, test 2019→Feb 2022 with COVID in the test set), QLIKE (primary, lower is better):

| Model | QLIKE | RMSE (var) | MAE (var) |
|---|---|---|---|
| **HAR-RV** | **0.257** | 2.11e-04 | 6.19e-05 |
| RW-RV (reference) | 0.363 | 2.42e-04 | 6.93e-05 |
| EGARCH | 0.432 | 2.46e-04 | 9.95e-05 |
| GARCH(1,1) | 0.462 | 3.71e-04 | 1.35e-04 |

The ordering is exactly what the literature predicts: HAR-RV is hard to beat on
its native realized-volatility target (Corsi 2009; Bucci 2020; Christensen et al.
2023), and the asymmetric EGARCH beats symmetric GARCH(1,1) on equity data
(Hansen & Lunde 2005). This is the bar RQ1 asks the LSTM to clear.

---

## 2. What was built

```
Code/
├── configs/econometric.yaml           Phase-2 run config (models, harness, profiles)
├── src/
│   ├── data/
│   │   ├── datasets.py                 target-agnostic modelling-frame assembly
│   │   ├── intraday.py                 intraday RV reconstruction + splice gate (your proposal)
│   │   └── realized_vol.py             + realized_variance_intraday() (ABDL 5-min RV)
│   ├── evaluation/
│   │   ├── metrics.py                  MSE, MAE, QLIKE (+ PICP/MPIW/Winkler for Phase 6)
│   │   └── rolling.py                  walk-forward engine + VolForecaster protocol
│   ├── models/econometric/
│   │   ├── garch.py                    ArchVolForecaster base + GARCH(1,1)
│   │   ├── egarch.py                   EGARCH(1,1,1)
│   │   ├── har.py                      HAR-RV (OLS on daily/weekly/monthly RV lags)
│   │   └── naive.py                    random-walk-in-variance reference floor
│   └── experiments/run_econometric.py  Phase-2 driver (tables, figure, milestone)
├── tests/                              +5 test files, 50 new tests
└── results/
    ├── predictions/m02_*.parquet       one column per model + realized target
    ├── tables/m02_*_metrics.csv        QLIKE-ranked error tables
    ├── figures/m02/*.png               forecast-vs-actual, crisis-shaded
    └── milestones/m02_econometric.md   the gate note
```

---

## 3. Architectural decisions (and why)

### 3.1 One engine, many models — the `VolForecaster` protocol
Every model is an object with a `name` and one method,
`forecast_fold(frame, fold) -> pd.Series` returning a per-date **variance**
forecast. The engine (`evaluation/rolling.py`) owns the walk-forward loop,
alignment, and persistence; models own nothing but their own fit/predict. This
is the single most important design choice in the phase: it decouples modelling
from evaluation so Phases 3–6 drop in an LSTM or an MC-Dropout model **without
touching the harness**, and every comparison (point error, Diebold-Mariano,
per-regime, calibration) becomes a pandas operation on the predictions parquet.
The alternative — bespoke evaluation per model — is how leakage bugs and
apples-to-oranges tables creep in.

### 3.2 Losses live on the variance scale, QLIKE primary
All losses take realized **variance** and a **variance** forecast. Patton (2011)
proves MSE and QLIKE are robust to a noisy variance *proxy* precisely on the
variance scale; GARCH/EGARCH forecast variance natively; HAR is fit on RV. Scoring
everything on the same variance target is what makes the comparison fair
(Hansen & Lunde 2005). QLIKE is written in its non-negative Bregman form
(`σ²/h − ln(σ²/h) − 1`, zero at a perfect forecast) — identical ranking to the
raw form, easier to read. It is the primary metric because it penalises
*under*-prediction of risk more than over-prediction (Chapter 2 §2.3).

### 3.3 Leakage discipline is structural, not a convention
GARCH/EGARCH parameters are estimated **once per fold** on the training window
via `arch`'s `last_obs`, then the conditional-variance recursion is filtered
forward with realized returns at fixed parameters — the standard fixed-window
out-of-sample forecast. Forecasts are aligned by mapping each forecast *origin*
to the next trading day, so the first forecast lands exactly on `predict_start`
(origin = `train_end`): no gap, no overlap. The engine additionally *rejects* any
model that returns a forecast dated on or before `train_end`. Leakage would have
to defeat both the split generator and an explicit assertion.

### 3.4 Return scaling
Returns are multiplied by 100 (→ percent) before fitting — `arch`'s documented
remedy for the ill-conditioning of near-unit-scale returns — and the variance
forecast is divided by `100²` to return to decimal-variance units matching the RV
target. A unit test checks the unscaled forecast is the same order of magnitude
as realized variance, which catches a wrong scaling factor immediately.

### 3.5 HAR-RV specifics
Fit by OLS on the realized-**variance** scale (Corsi's original object, and the
scale losses are computed on, so no cross-scale mapping is needed). Forecasts are
floored at a tiny positive constant so QLIKE is always defined; the milestone
reports how often the floor binds (here: never). A `log`-target variant is
provided as a right-skew robustness lever but is not the default.

### 3.6 Open-to-close vs close-to-close returns (a fairness detail)
Intraday `rv5` is an *open-to-close* realized variance; close-to-close GARCH
returns additionally carry the overnight gap (~40% more variance), which QLIKE —
being ratio-based — would charge to GARCH as a level bias. The frame therefore
also carries an `oc_log_return` (open-to-close) column, selectable via
`harness.return_col`, so GARCH can be matched to the target's session convention.
The headline run uses the standard close-to-close return and notes the caveat;
the conclusions (HAR best, EGARCH>GARCH) are unaffected.

### 3.7 A random-walk reference floor
`RW-RV` (tomorrow's variance = today's) is not a dissertation baseline; it is a
sanity anchor. It validates the engine with a model that has no estimation step,
and it makes the other numbers legible — anything that cannot beat it is not
learning. GARCH/EGARCH beating or trailing it is itself informative.

### 3.8 Testing
50 new tests: loss correctness and the QLIKE-minimised-at-truth property;
HAR recovering known OLS coefficients on a synthetic HAR process; GARCH recovering
high persistence and correct scale on a synthetic GARCH path; engine coverage,
alignment, leakage rejection and parquet round-trip; and intraday-RV recovery of a
known integrated variance with overnight exclusion.

---

## 4. A methodological finding that shaped the phase

While wiring the harness I ran the baselines on two candidate targets over the
same window. The result is worth flagging because it **changes the target
decision**:

| Target | lag-1 autocorr | QLIKE: HAR / RW / EGARCH / GARCH |
|---|---|---|
| Oxford-Man intraday `rv5` (daily) | 0.729 | 0.257 / 0.363 / 0.432 / 0.462 |
| Yang-Zhang OHLC (21-day rolling) | 0.996 | 0.0038 / 0.0047 / 0.091 / 0.077 |

Yang-Zhang is a **21-day rolling** estimator, so consecutive values share 20/21 of
their inputs; the target is near-perfectly autocorrelated (0.996). A random walk
then looks almost perfect and GARCH looks ~20× worse — an **artefact of the
target's built-in smoothing, not a real forecast result** (note HAR's R² of 0.994
and β_d of 1.30 on YZ: it is essentially copying yesterday's value). The genuine
daily intraday target (autocorr 0.729) gives a fair, literature-consistent
comparison.

**Consequence:** substituting Yang-Zhang to reach the 2022–2024 window (a
tempting shortcut) is *not* methodologically safe. A valid daily target is
required — which is exactly what your splice proposal preserves. This diagnostic
is shipped as the `yang_zhang_diagnostic` profile so the write-up can justify the
target choice with evidence rather than assertion.

---

## 5. Results in one paragraph

On the canonical intraday target across 699 out-of-sample days spanning the COVID
crash, HAR-RV leads on QLIKE (0.257), then the random-walk floor (0.363), then
EGARCH (0.432) and GARCH(1,1) (0.462). The figure shows why the GARCH family
trails: after the March-2020 spike GARCH over-persists (its variance decays too
slowly), inflating QLIKE through the recovery, while HAR's daily/weekly/monthly
cascade tracks the mean-reversion. EGARCH's leverage term makes it both more
reactive to the crash and quicker to decay than GARCH(1,1), consistent with the
equity findings in Hansen & Lunde (2005). Fitted GARCH/EGARCH persistence (~0.98)
and HAR R² (~0.60) are all in the expected ranges, and every one of the 34 monthly
refits converged.

---

## 6. The intraday splice: your proposal, assessed

> *"The Oxford-Man Institute stopped updating their library in early 2022, but the
> maths is open-source. Instead of compromising the test window or the literature
> review, calculate the missing 2022–2024 intraday volatility yourself and append
> it."*

**Verdict: the right instinct, and I've built the machinery for it — but its
feasibility hinges on one thing you have to supply (intraday data), and it must be
splice-validated or it will quietly corrupt the test set.** Detail:

**Why it's the right call.** The alternatives are worse. Shifting to Yang-Zhang
across 2022–2024 is disqualified by the finding in §4 (rolling-smoothing artefact).
The Phase-0/1 "hybrid" (train on intraday, test on YZ) mixes estimators across the
train/test boundary — a confound a viva will probe on RQ1. Your approach keeps a
single, valid daily estimator *and* the 2022 stressor *and* the Chapter 2 §2.3
commitment. Nothing else does all three.

**What I built for it** (`src/data/intraday.py`, `realized_variance_intraday`):
1. ABDL 5-minute realized variance from intraday bars — same open-to-close,
   5-minute convention Oxford-Man uses (unit-tested to recover a known integrated
   variance and to exclude the overnight gap).
2. `validate_overlap()` — compares your reconstruction against Oxford-Man on the
   period they both cover (correlation + level ratio).
3. `splice()` — **refuses to run** unless the overlap check passes (Pearson ≥ 0.90,
   level within ±15%), and optionally level-adjusts so the join is seamless.

The gate matters: the splice point is 2022-02, i.e. the *start of the test set*.
An unvalidated splice injects a level break exactly where it does maximum damage.

**The catches — be clear-eyed about these:**
- **Data is the binding constraint, not maths.** `yfinance` serves only ~60 days
  of intraday history, so it cannot provide 2022–2024. The S&P 500 *index*
  (^GSPC) is not traded, so there is no free multi-year 5-minute index feed. The
  realistic sources are (a) the **SPY ETF** intraday (free-ish, but a *different
  instrument* — dividends, expense ratio, its own microstructure), or (b) a
  licensed feed (Polygon, Databento, FirstRate Data, AlgoSeek, Refinitiv). SPY is
  workable *because* the overlap check will tell you whether it matches — that is
  what the gate is for.
- **It reverses a documented scope cut.** ROADMAP §6 explicitly cut "bespoke
  intraday cleaning pipeline," and §1.1 says intraday cleaning "is not the
  methodological contribution." Reconstructing RV walks that back. It's defensible
  (we consume clean bars, not raw ticks — no microstructure pipeline), but call it
  out in the Methods chapter and keep it lightweight.
- **Reproducibility.** The spliced series must be cached with its overlap
  statistics (the module writes a `*.meta.json`) so an examiner can see the join
  was validated.

**My recommendation.** Do it, but gated: get an intraday source (SPY is fine to
start), run `python -m src.data.intraday --intraday-file <file>`, and **only adopt
the 2022–2024 window if the overlap Pearson/level pass**. If they don't, fall back
to the intraday-native window shipped here (test through COVID) rather than force
a bad splice. Either way Phase 2 is already defensible today; the splice upgrades
the *window*, not the *validity*, and it's a one-line config switch
(`target: oxfordman_spliced`) when ready.

---

## 7. Issues / decisions needed

1. **Confirm the splice route (§6).** Can you get 2022–2024 intraday data (SPY
   export or a licensed feed)? If yes, I'll wire the spliced target as the
   headline and re-run. If not, we ship the intraday-native window (COVID in test)
   and soften the ROADMAP's 2022–2024 wording. **This is the only open blocker.**
2. **Open-to-close returns for GARCH?** Default is close-to-close with the caveat
   documented (§3.6). Say the word and I'll flip `harness.return_col` to make the
   GARCH/RV session conventions match exactly.
3. **Student-t innovations.** The gate uses canonical Gaussian GARCH/EGARCH. A
   Student-t variant (config `dist: t`) is a one-line robustness row motivated by
   Chapter 2 §2.2; happy to add it to the table if you want it in Phase 2 rather
   than deferred.

None of these block the gate; they shape the headline.

---

## 8. Reproduce

```powershell
cd Code
pip install -e .[dev]                       # arch, statsmodels already pinned
pytest -q                                   # 91 passed
python -m src.experiments.run_econometric   # regenerates tables, figure, milestone
```

Every number in the dissertation should be cited from `results/tables/m02_*.csv`
or recomputed from `results/predictions/m02_*.parquet` — never pasted from a
notebook (ROADMAP §9).

*Note: figures/tables here were generated in a Linux reproduction environment;
re-run in your pinned Windows venv before tagging the `m02-econometric` milestone
so the committed artifacts carry your environment's exact numerics (they will match
to many decimals — `arch`/`statsmodels` MLE is deterministic).*
