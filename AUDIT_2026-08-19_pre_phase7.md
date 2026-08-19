# Project audit — alignment, build quality, and Phase-7 readiness

**Reviewed:** 2026-08-19 · `E:\WarWick\MSc Dissertation`
**Scope:** Chapter 1 (Introduction) + Chapter 2 (Literature Review) against `ROADMAP.md`, `Code/src`, `Code/configs`, `Code/results`, `Code/tests`, and the persisted prediction parquets.
**Method:** read the two chapters and the roadmap in full; read the data/model/evaluation modules; independently re-derived the headline QLIKE, per-regime, Diebold–Mariano, Giacomini–White and PICP numbers from `results/predictions/*.parquet` rather than trusting the milestone tables; ran the unit-test suite.

---

## Verdict in one paragraph

The project is aligned with what the chapters promise, and the engineering is genuinely strong — above typical MSc standard and close to what I'd expect from a working paper. Every headline number I recomputed from the persisted parquets matched the milestone tables exactly, the leakage discipline is real and enforced in code (not just asserted in prose), and the significance apparatus (DM with HLN + HAC, Giacomini–White, Model Confidence Set, DM regularity diagnostics) is correctly implemented and unit-tested. **Phases 0–6 are complete and defensible.** However, I found **one methodological defect that affects the headline RQ2 and RQ3 findings** — the per-regime tables bucket days by a regime label inferred *using that day's own return*, which conditions the evaluation on the realized outcome. When I redo the same tables with the correctly-timed (lagged) label, the RQ3 "coverage degrades monotonically with regime severity" claim disappears and two of the RQ2 per-regime rankings flip. This must be resolved **before** Phase 7, because Phase 7's deliverable is the master results table that goes into the dissertation. There is also one **operational** risk (no data or results are under version control) that is more dangerous than the roadmap's "repo hygiene" framing suggests.

**Ready for Phase 7? Not quite — three things first (§4). They are ~1 day of work, not a rebuild.**

---

## 1. Alignment: do the chapters and the code describe the same study?

Yes, and unusually tightly. I checked every substantive commitment in Ch1 §1.5–§1.8 and Ch2 §2.2–§2.8 against the implementation:

| Chapter commitment | Implementation | Status |
|---|---|---|
| Daily RV from 5-min intraday returns, ABDL (2003) sense, Oxford-Man `.SPX` | `datasets.py` → `oxfordman_rv5`, 2000-01-03 → 2022-02-25 | ✅ |
| GARCH(1,1), EGARCH, HAR-RV baselines | `garch.py`, `egarch.py`, `har.py` via `arch`/statsmodels | ✅ |
| LSTM baseline + RV-only control matched to HAR-RV's input set | `LSTM`, `LSTM-RVonly` | ✅ |
| VIX as auxiliary input, **reported separately**, not in the like-for-like benchmark | `LSTM-VIX` + `comparison.exclude_models: [LSTM-VIX]` | ✅ (verified the exclusion leaves the board numerically identical) |
| Yang–Zhang as a **target-validity diagnostic**, not a competing target | `yang_zhang_diagnostic` profile, explicitly labelled "not a result" | ✅ |
| Gaussian-emission HMM, K=3, filtered (not smoothed) posterior, train-only fit | `hmm.py` + `TrainStandardizer` + `filtered_proba` | ✅ |
| Nystrup et al. (2020) jump penalty, λ tuned not fixed a priori | `jump_hmm.py` + `select_jump_penalty_causal` (λ=3 on training rows only) | ⚠️ see §3.4 |
| Two conditioning architectures: regime-as-feature and mixture-of-experts | `Regime-LSTM-A` / `-A-RVonly`, `RegimeExpertForecaster` (gate lagged to t−1) | ✅ |
| MC Dropout (T=100) + pinball quantile regression, monotone head | `MCDropoutLSTMForecaster`, `QuantileLSTMForecaster` (cumulative softplus) | ✅ |
| QLIKE primary (Patton 2011), MSE/MAE robustness | `metrics.qlike` — correct non-negative Bregman form | ✅ |
| Coverage, sharpness, **interval score** for probabilistic variants | PICP / MPIW / Winkler, per regime and pooled | ✅ |
| DM, Giacomini–White, Model Confidence Set | `significance.py`, all three, unit-tested incl. the GW↔DM equivalence identity | ✅ |
| DM regularity conditions "inspected and reported" (Ch2 §2.7) | `report_dm_diagnostics.py` — ACF w/ Bartlett bands, Ljung–Box, HAC-lag sweep | ✅ |
| CRPS (Ch2 §2.6/§2.7 invoke Gneiting–Raftery's "proper scoring rules"; roadmap §6 item 12 lists CRPS) | not implemented; only a 3-point pinball average described as "a discrete approximation to the CRPS" | ⚠️ gap |
| Engle–Manganelli dynamic quantile test (Ch2 §2.6) | not implemented — correctly scheduled for Phase 7 | ⏳ planned |

**Two chapter↔code wording mismatches to fix in the writeup:**

1. **Ch1 §1.7** says all models are "trained under a strict time-ordered split with **anchored walk-forward** out-of-sample evaluation". For the deep models this is true of the *evaluation* but not the *training*: `refit_every_folds: 0` in `lstm_baseline.yaml`, `regime_lstm.yaml` and `uq.yaml` means every LSTM is trained **once** on data ≤ 2018-12-31 and frozen for the whole 2019 → Feb-2022 test window, while GARCH/EGARCH/HAR-RV are refit every 21 trading days on an expanding window. This is leakage-free and *conservative* toward the deep models, but it is an unequal-treatment confound for RQ1/RQ2 that a reader of Ch1 §1.7 would not anticipate. `m03` discloses it in one line; `m05` and `m06` do not mention it at all. **Fix:** state it plainly in Chapter 3, and add one line to the m05/m06 scope sections.
2. **Ch2 §2.4** says the dissertation "adopts the Nystrup et al. (2020) jump-penalised estimator **directly**". The code is a defensible two-step *plug-in*: fit the jump model → read Gaussian emissions and a transition matrix off its hard path (with a Dirichlet pseudo-count) → run the inherited forward filter. `jump_hmm.py`'s docstring is honest about this and the reasoning is sound, but "directly" overstates it. **Fix:** soften to "operationalises" / "estimated via the Nystrup et al. jump-penalised path", and give the three-step construction a short paragraph in Chapter 3.

**Roadmap housekeeping:** the "Still open from the same audit" list at the foot of `ROADMAP.md` is stale. Items 1 (VIX unused), 2 (Yang–Zhang wording), 3 (Ch1 §1.7 omits MCS) and 4 (DM regularity diagnostics) were all closed by the 2026-08-18/19 work — I verified each against the current chapter text and code. Only items **5** (repo hygiene) and **6** (walk-forward off-by-one) remain. Also `ROADMAP.md` §1.1 still defines the target as "the square-root of the sum of squared 5-minute log returns" (volatility) whereas the code and Ch1 correctly use realized *variance*.

---

## 2. What I verified independently (all green)

- **Every headline number reproduces from the persisted parquets.** Recomputed from `m05_regime_dl_intraday_2019_2022.parquet`: LSTM-RVonly 0.2596, Regime-LSTM-A-RVonly 0.2617, Regime-LSTM-B 0.2621, Regime-LSTM-A 0.2632, LSTM 0.2650, HAR-RV 0.2745, GARCH 0.3302, EGARCH 0.3328, RW-RV 0.3584 — all exact to 4 dp.
- **DM statistics reproduce exactly**, including the lag-sensitive pair: LSTM-RVonly vs HAR-RV stat −1.75, p = 0.081, lrv/γ₀ = 1.46. The diagnostics correctly identify this as the pair where the naive iid variance would manufacture significance.
- **GW statistics reproduce exactly**: Regime-LSTM-B vs HAR-RV χ²(3) = 40.13, p = 9.98e-09; LSTM vs HAR-RV χ²(3) = 27.82, p = 3.96e-06; moment signs localise the edge in the transitional state.
- **Unit tests pass.** 155/155 in the torch-free subset (`significance`, `metrics`, `splits`, `har`, `regime`, `jump_hmm`, `calibration`, `realized_vol`, `rolling`, `econometric_models`, `sequence`) in this environment. The remaining ~40 require PyTorch, consistent with the claimed 195.
- **Leakage discipline is real, not decorative.** `make_windows` returns windows ending strictly at t−1; `Standardizer.fit` is only ever called on `index <= fold.train_end`; the MoE gate is explicitly `.shift(1)`; the HMM is fit on `Xz_causal[train_mask]` with a train-only scaler and then filtered forward; `arch`'s `last_obs=predict_start` freezes parameters before the prediction window; HAR's lags are pre-shifted. I could not construct a look-ahead path.
- **Loss functions are correct.** QLIKE Bregman form, Winkler score, PICP, MPIW, pinball — all match their textbook definitions.

---

## 3. Findings, most severe first

### 3.1 🔴 Per-regime tables condition on the outcome — this changes the RQ2 and RQ3 conclusions

`run_regime_lstm.per_regime_qlike` and `calibration.per_regime_interval_metrics` bucket day *t* by `reg_state[t]` — the **contemporaneous** filtered state. But the filtered posterior P(s_t | x₁…x_t) is computed **using the day-t return**. So the bucket label for day *t* is partly determined by the very outcome whose forecast error you are then measuring in that bucket. The forecasts themselves are clean (they only ever see t−1); the *evaluation* is not.

The Giacomini–White path already does this correctly — `regime_test_function(..., shift=1)`, with a good docstring explaining why. The descriptive tables do not. The two therefore disagree, and the disagreement is not cosmetic:

**RQ2 — per-regime QLIKE (recomputed from the parquet):**

| bucket | n | HAR-RV | LSTM | Regime-LSTM-B | winner |
|---|---|---|---|---|---|
| calm — contemporaneous | 379 | 0.2757 | 0.2500 | 0.2468 | deep |
| calm — **lagged** | 379 | **0.3008** | **0.3187** | **0.3172** | **HAR-RV** |
| transitional — contemporaneous | 330 | 0.2435 | 0.2230 | 0.2166 | deep |
| transitional — **lagged** | 329 | 0.2367 | 0.1921 | 0.1851 | deep (stronger) |
| crisis — contemporaneous | 75 | 0.4048 | 0.5257 | 0.5399 | HAR-RV, by a lot |
| crisis — **lagged** | 75 | 0.3107 | 0.3159 | 0.3251 | HAR-RV, essentially tied |

The **calm** conclusion reverses outright, and the **crisis** conclusion goes from "the deep models fail badly in crisis (+33%)" to "everything is within noise". Only the transitional finding survives — and it strengthens, which is exactly why the roadmap's GW-based headline is the trustworthy one. Note that `m05_regime_dl.md`'s Findings section states the *contemporaneous* numbers ("crisis … 5.7% higher") while `ROADMAP.md`'s headline states the *lagged* ones ("QLIKE 0.1851 vs 0.2367 — 21.8% lower on n=329"). An examiner reading both will ask which is right.

**RQ3 — this is the more damaging one. The Phase-6 "distinctive contribution" does not survive:**

| MC-Dropout PICP @ 90% | calm | transitional | crisis |
|---|---|---|---|
| contemporaneous label (as reported in `m06_uq.md`) | 0.871 | 0.870 | **0.813** |
| **lagged label** | **0.844** | **0.894** | **0.840** |

| Quantile PICP @ 90% | calm | transitional | crisis |
|---|---|---|---|
| contemporaneous label (as reported) | 0.834 | 0.818 | **0.773** |
| **lagged label** | **0.807** | **0.842** | **0.800** |

The monotone degradation with regime severity — the headline RQ3 result, and the one the roadmap says "unlike the RQ2 crisis claim, this survives a change of regime estimator" — **is an artefact of the bucketing.** Under the correctly-timed label the worst-covered bucket is *calm*, and crisis is statistically indistinguishable from it.

**The mechanism, quantified.** 51 of 784 test days (6.5%) change bucket when the label is lagged. Those are regime-*transition* days, and their 90% interval miss rate is **0.294** versus **0.124** on non-transition days — 2.4× higher. Which bucket those 51 high-miss days land in is decided purely by the labelling convention. Contemporaneous labelling pushes them toward the state the market just entered, which is disproportionately "crisis".

**The good news:** this points at a *better* finding than the one currently claimed. The real, robust result is not "coverage degrades with regime severity" but **"coverage collapses at regime transitions"** (0.71 vs 0.88) — a sharper claim, more useful to a risk manager, and directly testable with the Christoffersen (1998) conditional-coverage / Engle–Manganelli DQ machinery Phase 7 already plans to add. I'd reframe rather than retract.

**Fix:** add a `shift` argument to both per-regime helpers, default it to 1, regenerate m05/m06, and state in Chapter 3 why the lagged label is the only one that supports a conditional claim. Keep the contemporaneous table if you like, but label it explicitly as an ex-post description that cannot support "the gain is concentrated in state X".

### 3.2 🔴 Nothing empirical is under version control

`git ls-files` returns **zero** `.parquet` or `.csv` files. `Code/.gitignore` excludes `results/{figures,tables,predictions,logs}/**` *and* `data/{raw,interim,processed}/**`. There are **no git tags** (roadmap §9 requires `m01-data` … `m06-uq`), and 9 files are currently uncommitted including all four regenerated milestone notes and both chapters.

Consequences:
- The roadmap's repeated claim that "the committed prediction parquets reproduce the milestone numbers" is not literally true — they are on the E: drive only.
- The Phase-9 gate ("re-run all experiments end-to-end **from a fresh checkout**") would today produce nothing: no results *and no input data*. The 56 MB `data/raw` is untracked and the Oxford-Man library is discontinued — the only ingestion route is a Wayback snapshot URL that may not resolve indefinitely.
- Roadmap Risk #7 is rated Medium/High but is effectively unmitigated.

**Fix (30 minutes, do it today):** force-add `results/predictions/*.parquet`, `results/tables/*.csv` and `data/raw/oxfordman/*` (they are small — the entire predictions set is under 1.5 MB, raw data 56 MB, both well within a normal repo), commit, tag `m01`…`m06`, and push. Keep `figures/` and `logs/` ignored if you prefer. An examiner asking "can I reproduce Table 4.2?" needs to be able to.

### 3.3 🟠 Deep models are trained once; econometric baselines are refit monthly

Covered in §1 as a writeup issue, but it is also a *design* issue worth a deliberate decision before Phase 7. By February 2022, HAR-RV has been refit ~37 times and has seen data through the previous month; the LSTM's weights are three years stale and have never seen a COVID-scale observation in training (the training window ends 2018-12-31). Two consequences:

- The deep models' pooled results are **conservative** — good for the credibility of any win.
- But the crisis-state underperformance, and the near-homoskedastic MC-Dropout band (its aleatoric term is a **single global constant**, 0.600 on the log scale, estimated once pre-2019 and broadcast to all 784 test days — I confirmed `mc_sd_aleatoric` has exactly one unique value), are **both** plausibly artefacts of frozen training rather than architectural properties. Right now you cannot separate the two.

**Fix:** either (a) run one sensitivity with `refit_every_folds: 1` for the best LSTM and report it as a one-figure robustness check in Phase 8, or (b) state the design choice and its direction of bias explicitly in Chapter 3 and in the Limitations section. (a) is stronger and cheap — it's a config change.

### 3.4 🟠 Smaller methodological points, all easily handled in prose

- **λ grid is coarse.** The training-only sweep jumps from λ=1 (mean duration 3.3 days) straight to λ=3 (51.9 days) — nothing in between. The "smallest λ clearing 15 days" rule therefore selects 3.0 by default rather than by discrimination. Add λ ∈ {1.5, 2, 2.5} to the grid so the selection is a genuine choice; the resulting table is much more persuasive.
- **Giacomini–White's regularity conditions.** GW (2006) assumes forecasts come from a *finite/rolling* estimation window; an anchored (expanding) window formally violates it. This actually applies to your GARCH/EGARCH/HAR side, not the frozen LSTMs. Every applied paper using anchored windows has this caveat — just state it in one sentence in Chapter 3 so the viva can't spring it on you.
- **"Duan smearing" naming.** `har.py`, `lstm.py` and `regime_lstm.py` apply `exp(μ̂ + σ̂²/2)`, which is the *parametric log-normal* retransformation correction, not Duan's (1983) non-parametric smearing estimator `mean(exp(ε̂))`. The maths is right under Gaussian log-residuals; the citation is imprecise. One word.
- **CRPS is promised but not implemented.** Ch2 §2.6/§2.7 lean on Gneiting–Raftery's proper scoring rules and roadmap §6 lists CRPS; only a 3-point pinball average exists. Either implement it properly for the quantile and MC-Dropout predictive distributions in Phase 7, or narrow the Ch2 claim to Winkler + coverage.
- **`walk_forward_folds` off-by-one is real.** I reproduced it: when `len(oos_dates) % refit_frequency_days == 1`, the last OOS day is silently never predicted (e.g. n=379 with cadence 21 → last day dropped, no warning). It does not bite at 784/21 = 37 r 7, so no current result is affected — but Phase 8's AAPL/TSLA runs have different calendars and will hit it. Two-line fix plus a regression test.
- **Stale docstring.** `splits.walk_forward_folds` still documents `"test": 2022-2024` and `"val": 2020-2021` from the superseded split plan.

---

## 4. Are you ready for Phase 7?

**Substantively yes — the foundation is sound and Phase 7 is the right next step.** But three things should be done *first*, because Phase 7's deliverable is `m07_combined.md`, the master results table that goes verbatim into Chapter 4. Building it on the current per-regime convention means building the dissertation's centrepiece on a claim that reverses under a one-day relabel.

**Blocking (do before Phase 7, ~1 day total):**

1. **Fix the per-regime bucketing** (§3.1) — add `shift=1` to both per-regime helpers, regenerate m05 and m06, and reconcile the m05 Findings prose with the roadmap headline so one set of numbers is authoritative. Then decide whether RQ3's headline becomes "coverage collapses at regime transitions" (which the data supports) instead of "degrades with regime severity" (which it does not).
2. **Get data + results into git and tag the milestones** (§3.2). Purely mechanical, high consequence.
3. **Decide and document the refit-cadence asymmetry** (§3.3) — one config run or one honest paragraph. Do this before Phase 7 so the combined model inherits a settled convention rather than a retrofitted one.

**Non-blocking (fold into Phase 7/8):** the λ grid, the GW caveat sentence, the Duan naming, CRPS, the off-by-one fix, the stale docstring.

**One more decision Phase 7 needs, which the roadmap already flags:** "best of Phase 5" is ambiguous. Given §3.1, I'd resolve it as the roadmap suggests — pick on the **transitional** result (Regime-LSTM-B), because that is the only per-regime finding that survives both a change of regime estimator *and* the correct label timing. Say so explicitly in the writeup; a stated, defended selection rule is worth more than a marginally better number.

**Per your instruction, I have not started building Phase 7.**

---

## 5. Standing back

The intellectual honesty in this project is its best feature and you should not lose it in the writeup. Retracting the crisis claim when it failed a robustness check, walling `LSTM-VIX` out of the MCS for unequal information, fixing the milestone generator that had hardcoded its own conclusions, keeping the Baum-Welch signal as a named comparator — these are the things that make a viva go well. Chapter 2 §2.8's framing ("a model that adds well-calibrated predictive intervals without necessarily lowering point error remains a useful and defensible result") is exactly the right posture, and the results as they stand support a good dissertation whether or not deep learning wins.

The finding in §3.1 is in the same spirit: it costs you the current RQ3 headline but hands you a better one, and catching it yourself before the examiner does is worth considerably more than the claim you'd be giving up.
