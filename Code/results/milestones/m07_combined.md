# Milestone m07 — Phase 7: combined regime × uncertainty model, and RQ3 made inferential

*Profile:* `intraday_2019_2022` · *test window:* 2019-01-02 → 2022-02-25 (n=784) · *headline level:* 90% · *MC samples:* 100

## Scope and what is being isolated

The combined model is **MC-Dropout-Regime-LSTM-B** — the Phase-5 `Regime-LSTM-B` mixture-of-experts with MC-Dropout inference, training inherited unchanged. "Best of Phase 5" is `Regime-LSTM-B`, selected on the transitional result -- the only regime bucket where the effect is both large and formally tested (per-regime DM p=4.3e-09 vs HAR-RV, and the state carrying the Giacomini-White moment behind chi2(3)=40.1). Calm and crisis are point estimates on 379 and 76 days with no significant test behind them, and crisis reverses sign between the two regime estimators. The rule was fixed before this phase ran.

Because training is inherited and the hyperparameters are still the Phase-3 validation selection, the contrast `MC-Dropout-LSTM` → `MC-Dropout-Regime-LSTM-B` isolates **regime conditioning inside the uncertainty-aware family** and nothing else: same network shape, same seed, same folds, same MC sample count.

## The pre-registered expectation

Recorded in `ROADMAP.md` **before** this phase was run: *Kupiec should reject decisively — the intervals are 3.5–7.9 pp narrow — while Christoffersen's independence component and any t−1 conditioning variable should not, because the failures are unclustered and unpredictable.* If that holds, miscalibration is a **level** problem rather than a **timing** problem, the combined model is a pre-registered null, and the deliverable is a global width correction.

### Verdict, computed from the tables below

| clause | expected | found |
|---|---|---|
| Kupiec rejects (level is wrong) | yes | **yes** — MC-Dropout-Regime-LSTM-B p=0.000537; MC-Dropout-LSTM p=0.00171; LSTM-Gaussian p=0.00117; Quantile-LSTM p=2.31e-11 |
| Christoffersen independence does *not* reject (no clustering) | yes | **yes** — MC-Dropout-Regime-LSTM-B p=0.509; MC-Dropout-LSTM p=0.843; LSTM-Gaussian p=0.909; Quantile-LSTM p=0.455 |
| no t−1 conditioner predicts a miss (pooled hit) | yes | **no conditioner survives Holm** |
| no t−1 subsample differs in *upper-tail* coverage | (not pre-registered) | **5 of 16 survive Holm** |

**The pre-registration holds in its main clauses and is partially falsified in its last one.** The level is wrong and the misses are unclustered, as predicted — but the claim that *nothing* knowable at t−1 predicts a miss does not survive contact with the one-sided tests. That is the more interesting outcome and is reported as such below, not absorbed. On the remedy, the recommendation is **per-regime** — the deliverable section gives the reasoning.


## Master results table (headline level 90%)

| rank_qlike | model | family | qlike | qlike_transitional | mse | mae | picp | mpiw | winkler | crps | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LSTM-RVonly | deep | 0.2596 | 0.2182 | 4.021e-08 | 6.058e-05 |  |  |  |  | 784 |
| 2 | MC-Dropout-Regime-LSTM-B | combined (Phase 7) | 0.2599 | 0.1849 | 4.057e-08 | 5.352e-05 | 0.8610 | 0.0001813 | 0.0004006 | 4.061e-05 | 784 |
| 3 | Regime-LSTM-A-RVonly | regime-aware deep | 0.2617 | 0.2158 | 4.152e-08 | 6.472e-05 |  |  |  |  | 784 |
| 4 | Regime-LSTM-B | regime-aware deep | 0.2621 | 0.1851 | 4.156e-08 | 5.382e-05 |  |  |  |  | 784 |
| 5 | Regime-LSTM-A | regime-aware deep | 0.2632 | 0.1945 | 4.028e-08 | 5.809e-05 |  |  |  |  | 784 |
| 6 | LSTM-Gaussian | uncertainty-aware deep | 0.2650 | 0.1921 | 4.485e-08 | 5.863e-05 | 0.8635 | 0.0001801 | 0.0004198 | 4.306e-05 | 784 |
| 7 | LSTM | deep | 0.2650 | 0.1921 | 4.485e-08 | 5.863e-05 |  |  |  |  | 784 |
| 8 | MC-Dropout-LSTM | uncertainty-aware deep | 0.2652 | 0.1922 | 4.494e-08 | 5.874e-05 | 0.8648 | 0.0001818 | 0.0004199 | 4.306e-05 | 784 |
| 9 | HAR-RV | econometric | 0.2745 | 0.2367 | 4.176e-08 | 6.046e-05 |  |  |  |  | 784 |
| 10 | Quantile-LSTM | uncertainty-aware deep | 0.3216 | 0.2087 | 5.131e-08 | 5.73e-05 | 0.8214 | 0.0001598 | 0.0004317 | 4.306e-05 | 784 |
| 11 | GARCH | econometric | 0.3302 | 0.2863 | 5.625e-08 | 7.172e-05 |  |  |  |  | 784 |
| 12 | EGARCH | econometric | 0.3328 | 0.2797 | 5.666e-08 | 6.902e-05 |  |  |  |  | 784 |
| 13 | RW-RV | econometric | 0.3584 | 0.2664 | 5.284e-08 | 6.579e-05 |  |  |  |  | 784 |

*`qlike_transitional` is the transitional bucket on the **t−1** label (lagged t-1 (conditional / implementable)), n=329. Interval columns are blank for models that produce no predictive distribution — that contrast is the point of the table.*


Pooled, the best model on QLIKE is **LSTM-RVonly** (0.2596); the combined model ranks **2 of 13** at 0.2599.
 Against the regime-agnostic `MC-Dropout-LSTM` (0.2652) that is a change of -0.0054 (-2.0%), and in the transitional bucket 0.1849 vs 0.1922.


## Significance on the Phase-7 pairs

| model_a | model_b | dm_stat | p_value | mean_loss_diff | better | n |
|---|---|---|---|---|---|---|
| MC-Dropout-Regime-LSTM-B | MC-Dropout-LSTM | -1.7339 | 0.0833 | -0.005379 | MC-Dropout-Regime-LSTM-B | 784 |
| MC-Dropout-Regime-LSTM-B | Regime-LSTM-B | -1.0192 | 0.3084 | -0.002261 | MC-Dropout-Regime-LSTM-B | 784 |
| MC-Dropout-Regime-LSTM-B | HAR-RV | -0.9105 | 0.3629 | -0.014635 | MC-Dropout-Regime-LSTM-B | 784 |
| MC-Dropout-Regime-LSTM-B | LSTM | -1.6688 | 0.0956 | -0.005165 | MC-Dropout-Regime-LSTM-B | 784 |
| MC-Dropout-LSTM | LSTM | 1.3159 | 0.1886 | 0.000214018 | LSTM | 784 |

**Giacomini–White, conditional on the lagged regime indicator.** A negative moment means the first model has the lower loss in that state.

| model_a | model_b | n | gw_regime_stat | gw_regime_df | gw_regime_p | moment_calm | moment_transitional | moment_crisis | a_relative_edge_in | a_wins_in |
|---|---|---|---|---|---|---|---|---|---|---|
| MC-Dropout-Regime-LSTM-B | MC-Dropout-LSTM | 784 | 5.0481 | 3 | 0.168312 | -0.000724211 | -0.003054 | -0.001600 | transitional | calm, transitional, crisis |
| MC-Dropout-Regime-LSTM-B | Regime-LSTM-B | 784 | 2.2736 | 3 | 0.517591 | 0.000154932 | -6.28549e-05 | -0.002353 | crisis | transitional, crisis |
| MC-Dropout-Regime-LSTM-B | HAR-RV | 784 | 39.3380 | 3 | 1.47179e-08 | 0.008060 | -0.021718 | -0.000977862 | transitional | transitional, crisis |
| MC-Dropout-Regime-LSTM-B | LSTM | 784 | 4.7583 | 3 | 0.190375 | -0.000592376 | -0.003003 | -0.001569 | transitional | calm, transitional, crisis |
| MC-Dropout-LSTM | LSTM | 784 | 2.0370 | 3 | 0.564772 | 0.000131835 | 5.12326e-05 | 3.09507e-05 | crisis | none |

## RQ3, made inferential — coverage tests on the full 784-day sample

Every regressor is lagged into the t−1 information set, and every test runs on the **full** sample. Two rules make these numbers mean what they appear to mean, both learned from earlier audits of this project:

1. **A subsample selector must be measurable at t−1.** The ex-post transition indicator `1{s_t ≠ s_(t−1)}` is detected by the same one-day surprise that breaks an interval; a test on it has no nominal size. Only `1{s_(t−1) ≠ s_(t−2)}` appears here.

2. **A subsample claim is a *difference*, never a comparison against nominal.** These intervals are all too narrow globally, so a Kupiec test against nominal rejects on *any* subsample with power — that would report a level failure as a timing failure. Subsample rows below are two-sample differences with Holm control; conditional DQ rows are **incremental** over level and hit dynamics, for the same reason one level up.

| method | side | test | stat | df | p_value | violation_rate | expected_violation_rate | n |
|---|---|---|---|---|---|---|---|---|
| MC-Dropout-Regime-LSTM-B | both | Kupiec LR_uc | 11.983 | 1 | 0.00053692 | 0.1390 | 0.1000 | 784 |
| MC-Dropout-Regime-LSTM-B | both | Christoffersen LR_ind | 0.436 | 1 | 0.50929 | 0.1390 | 0.1000 | 784 |
| MC-Dropout-Regime-LSTM-B | both | Christoffersen LR_cc | 12.418 | 2 | 0.00201 | 0.1390 | 0.1000 | 784 |
| MC-Dropout-Regime-LSTM-B | both | Engle-Manganelli DQ | 16.881 | 5 | 0.00473 | 0.1390 | 0.1000 | 784 |
| MC-Dropout-Regime-LSTM-B | upper | Kupiec LR_uc | 16.145 | 1 | 5.8667e-05 | 0.0842 | 0.0500 | 784 |
| MC-Dropout-Regime-LSTM-B | upper | Christoffersen LR_ind | 0.415 | 1 | 0.51959 | 0.0842 | 0.0500 | 784 |
| MC-Dropout-Regime-LSTM-B | upper | Christoffersen LR_cc | 16.560 | 2 | 0.00025355 | 0.0842 | 0.0500 | 784 |
| MC-Dropout-Regime-LSTM-B | upper | Engle-Manganelli DQ | 31.482 | 5 | 7.5244e-06 | 0.0842 | 0.0500 | 784 |
| MC-Dropout-LSTM | both | Kupiec LR_uc | 9.836 | 1 | 0.00171 | 0.1352 | 0.1000 | 784 |
| MC-Dropout-LSTM | both | Christoffersen LR_ind | 0.039 | 1 | 0.84349 | 0.1352 | 0.1000 | 784 |
| MC-Dropout-LSTM | both | Christoffersen LR_cc | 9.875 | 2 | 0.00717 | 0.1352 | 0.1000 | 784 |
| MC-Dropout-LSTM | both | Engle-Manganelli DQ | 14.254 | 5 | 0.01407 | 0.1352 | 0.1000 | 784 |
| MC-Dropout-LSTM | upper | Kupiec LR_uc | 10.992 | 1 | 0.00091484 | 0.0778 | 0.0500 | 784 |
| MC-Dropout-LSTM | upper | Christoffersen LR_ind | 1.116 | 1 | 0.29072 | 0.0778 | 0.0500 | 784 |
| MC-Dropout-LSTM | upper | Christoffersen LR_cc | 12.109 | 2 | 0.00235 | 0.0778 | 0.0500 | 784 |
| MC-Dropout-LSTM | upper | Engle-Manganelli DQ | 30.315 | 5 | 1.2787e-05 | 0.0778 | 0.0500 | 784 |
| LSTM-Gaussian | both | Kupiec LR_uc | 10.530 | 1 | 0.00117 | 0.1365 | 0.1000 | 784 |
| LSTM-Gaussian | both | Christoffersen LR_ind | 0.013 | 1 | 0.90911 | 0.1365 | 0.1000 | 784 |
| LSTM-Gaussian | both | Christoffersen LR_cc | 10.543 | 2 | 0.00514 | 0.1365 | 0.1000 | 784 |
| LSTM-Gaussian | both | Engle-Manganelli DQ | 15.643 | 5 | 0.00794 | 0.1365 | 0.1000 | 784 |
| LSTM-Gaussian | upper | Kupiec LR_uc | 11.954 | 1 | 0.00054532 | 0.0791 | 0.0500 | 784 |
| LSTM-Gaussian | upper | Christoffersen LR_ind | 0.947 | 1 | 0.33046 | 0.0791 | 0.0500 | 784 |
| LSTM-Gaussian | upper | Christoffersen LR_cc | 12.901 | 2 | 0.00158 | 0.0791 | 0.0500 | 784 |
| LSTM-Gaussian | upper | Engle-Manganelli DQ | 31.805 | 5 | 6.4943e-06 | 0.0791 | 0.0500 | 784 |
| Quantile-LSTM | both | Kupiec LR_uc | 44.691 | 1 | 2.3076e-11 | 0.1786 | 0.1000 | 784 |
| Quantile-LSTM | both | Christoffersen LR_ind | 0.559 | 1 | 0.45466 | 0.1786 | 0.1000 | 784 |
| Quantile-LSTM | both | Christoffersen LR_cc | 45.250 | 2 | 1.4933e-10 | 0.1786 | 0.1000 | 784 |
| Quantile-LSTM | both | Engle-Manganelli DQ | 58.161 | 5 | 2.9133e-11 | 0.1786 | 0.1000 | 784 |
| Quantile-LSTM | upper | Kupiec LR_uc | 37.947 | 1 | 7.2693e-10 | 0.1046 | 0.0500 | 784 |
| Quantile-LSTM | upper | Christoffersen LR_ind | 0.024 | 1 | 0.87582 | 0.1046 | 0.0500 | 784 |
| Quantile-LSTM | upper | Christoffersen LR_cc | 37.971 | 2 | 5.6835e-09 | 0.1046 | 0.0500 | 784 |
| Quantile-LSTM | upper | Engle-Manganelli DQ | 79.970 | 5 | 8.5154e-16 | 0.1046 | 0.0500 | 784 |

### Conditional tests (t−1 conditioners only)

| method | side | kind | conditioner | stat | df | p_value | p_holm | n | detail |
|---|---|---|---|---|---|---|---|---|---|
| MC-Dropout-Regime-LSTM-B | both | DQ incremental | regime | 6.294 | 2 | 0.04298 | 0.25177 | 780 | DQ_full=23.175 (df 7, p=0.001589) vs DQ_level+dynamics=16.881 (df 5, p=0.004731) |
| MC-Dropout-Regime-LSTM-B | both | DQ incremental | transition | 0.000515 | 1 | 0.98190 | 1.00000 | 780 | DQ_full=16.881 (df 6, p=0.009729) vs DQ_level+dynamics=16.881 (df 5, p=0.004731) |
| MC-Dropout-Regime-LSTM-B | both | DQ incremental | width | 0.784 | 1 | 0.37594 | 1.00000 | 780 | DQ_full=17.665 (df 6, p=0.007127) vs DQ_level+dynamics=16.881 (df 5, p=0.004731) |
| MC-Dropout-Regime-LSTM-B | both | difference in coverage | regime=calm | 2.212 |  | 0.02695 | 0.18863 | 784 | PICP 0.8338 (n=379) vs 0.8864 (n=405); z_iid=2.129 p_iid=0.03324 |
| MC-Dropout-Regime-LSTM-B | both | difference in coverage | regime=transitional | -2.034 |  | 0.04196 | 0.25177 | 784 | PICP 0.8906 (n=329) vs 0.8396 (n=455); z_iid=-2.038 p_iid=0.04159 |
| MC-Dropout-Regime-LSTM-B | both | difference in coverage | regime=crisis | -0.151 |  | 0.87981 | 1.00000 | 784 | PICP 0.8684 (n=76) vs 0.8602 (n=708); z_iid=-0.198 p_iid=0.8434 |
| MC-Dropout-Regime-LSTM-B | both | difference in coverage | regime transition (ex-ante) | -0.101 |  | 0.91949 | 1.00000 | 784 | PICP 0.8654 (n=52) vs 0.8607 (n=732); z_iid=-0.095 p_iid=0.9241 |
| MC-Dropout-Regime-LSTM-B | upper | DQ incremental | regime | 10.573 | 2 | 0.00506 | 0.03542 | 780 | DQ_full=42.054 (df 7, p=5.076e-07) vs DQ_level+dynamics=31.482 (df 5, p=7.524e-06) |
| MC-Dropout-Regime-LSTM-B | upper | DQ incremental | transition | 0.113 | 1 | 0.73651 | 1.00000 | 780 | DQ_full=31.595 (df 6, p=1.951e-05) vs DQ_level+dynamics=31.482 (df 5, p=7.524e-06) |
| MC-Dropout-Regime-LSTM-B | upper | DQ incremental | width | 1.361 | 1 | 0.24335 | 0.97339 | 780 | DQ_full=32.843 (df 6, p=1.124e-05) vs DQ_level+dynamics=31.482 (df 5, p=7.524e-06) |
| MC-Dropout-Regime-LSTM-B | upper | difference in coverage | regime=calm | 2.371 |  | 0.01776 | 0.08881 | 784 | PICP 0.8918 (n=379) vs 0.9383 (n=405); z_iid=2.341 p_iid=0.01924 |
| MC-Dropout-Regime-LSTM-B | upper | difference in coverage | regime=transitional | -2.642 |  | 0.00823 | 0.04940 | 784 | PICP 0.9453 (n=329) vs 0.8945 (n=455); z_iid=-2.527 p_iid=0.0115 |
| MC-Dropout-Regime-LSTM-B | upper | difference in coverage | regime=crisis | 0.179 |  | 0.85792 | 1.00000 | 784 | PICP 0.9079 (n=76) vs 0.9167 (n=708); z_iid=0.262 p_iid=0.7935 |
| MC-Dropout-Regime-LSTM-B | upper | difference in coverage | regime transition (ex-ante) | 0.326 |  | 0.74479 | 1.00000 | 784 | PICP 0.9038 (n=52) vs 0.9167 (n=732); z_iid=0.322 p_iid=0.7477 |
| MC-Dropout-LSTM | both | DQ incremental | regime | 5.300 | 2 | 0.07067 | 0.42400 | 780 | DQ_full=19.554 (df 7, p=0.006619) vs DQ_level+dynamics=14.254 (df 5, p=0.01407) |
| MC-Dropout-LSTM | both | DQ incremental | transition | 0.247 | 1 | 0.61915 | 1.00000 | 780 | DQ_full=14.501 (df 6, p=0.02451) vs DQ_level+dynamics=14.254 (df 5, p=0.01407) |
| MC-Dropout-LSTM | both | DQ incremental | width | 0.177 | 1 | 0.67426 | 1.00000 | 780 | DQ_full=14.431 (df 6, p=0.02518) vs DQ_level+dynamics=14.254 (df 5, p=0.01407) |
| MC-Dropout-LSTM | both | difference in coverage | regime=calm | 1.608 |  | 0.10775 | 0.53875 | 784 | PICP 0.8443 (n=379) vs 0.8840 (n=405); z_iid=1.621 p_iid=0.1049 |
| MC-Dropout-LSTM | both | difference in coverage | regime=transitional | -1.933 |  | 0.05318 | 0.37226 | 784 | PICP 0.8936 (n=329) vs 0.8440 (n=455); z_iid=-2.007 p_iid=0.04477 |
| MC-Dropout-LSTM | both | difference in coverage | regime=crisis | 0.495 |  | 0.62061 | 1.00000 | 784 | PICP 0.8421 (n=76) vs 0.8672 (n=708); z_iid=0.609 p_iid=0.5427 |
| MC-Dropout-LSTM | both | difference in coverage | regime transition (ex-ante) | -0.482 |  | 0.62950 | 1.00000 | 784 | PICP 0.8846 (n=52) vs 0.8634 (n=732); z_iid=-0.433 p_iid=0.6653 |
| MC-Dropout-LSTM | upper | DQ incremental | regime | 13.726 | 2 | 0.00105 | 0.00627 | 780 | DQ_full=44.041 (df 7, p=2.098e-07) vs DQ_level+dynamics=30.315 (df 5, p=1.279e-05) |
| MC-Dropout-LSTM | upper | DQ incremental | transition | 0.067 | 1 | 0.79539 | 1.00000 | 780 | DQ_full=30.382 (df 6, p=3.325e-05) vs DQ_level+dynamics=30.315 (df 5, p=1.279e-05) |
| MC-Dropout-LSTM | upper | DQ incremental | width | 4.513 | 1 | 0.03364 | 0.13456 | 780 | DQ_full=34.828 (df 6, p=4.654e-06) vs DQ_level+dynamics=30.315 (df 5, p=1.279e-05) |
| MC-Dropout-LSTM | upper | difference in coverage | regime=calm | 2.890 |  | 0.00385 | 0.01924 | 784 | PICP 0.8945 (n=379) vs 0.9481 (n=405); z_iid=2.805 p_iid=0.005039 |
| MC-Dropout-LSTM | upper | difference in coverage | regime=transitional | -3.393 |  | 0.00069137 | 0.00484 | 784 | PICP 0.9574 (n=329) vs 0.8967 (n=455); z_iid=-3.133 p_iid=0.001727 |
| MC-Dropout-LSTM | upper | difference in coverage | regime=crisis | 0.323 |  | 0.74657 | 1.00000 | 784 | PICP 0.9079 (n=76) vs 0.9237 (n=708); z_iid=0.490 p_iid=0.6243 |
| MC-Dropout-LSTM | upper | difference in coverage | regime transition (ex-ante) | -0.026 |  | 0.97893 | 1.00000 | 784 | PICP 0.9231 (n=52) vs 0.9221 (n=732); z_iid=-0.025 p_iid=0.9804 |
| LSTM-Gaussian | both | DQ incremental | regime | 5.808 | 2 | 0.05481 | 0.32887 | 780 | DQ_full=21.451 (df 7, p=0.003157) vs DQ_level+dynamics=15.643 (df 5, p=0.00794) |
| LSTM-Gaussian | both | DQ incremental | transition | 0.286 | 1 | 0.59304 | 1.00000 | 780 | DQ_full=15.929 (df 6, p=0.01414) vs DQ_level+dynamics=15.643 (df 5, p=0.00794) |
| LSTM-Gaussian | both | DQ incremental | width | 0.402 | 1 | 0.52611 | 1.00000 | 780 | DQ_full=16.045 (df 6, p=0.01351) vs DQ_level+dynamics=15.643 (df 5, p=0.00794) |
| LSTM-Gaussian | both | difference in coverage | regime=calm | 1.689 |  | 0.09123 | 0.45614 | 784 | PICP 0.8417 (n=379) vs 0.8840 (n=405); z_iid=1.723 p_iid=0.08497 |
| LSTM-Gaussian | both | difference in coverage | regime=transitional | -1.999 |  | 0.04561 | 0.31929 | 784 | PICP 0.8936 (n=329) vs 0.8418 (n=455); z_iid=-2.087 p_iid=0.03686 |
| LSTM-Gaussian | both | difference in coverage | regime=crisis | 0.467 |  | 0.64082 | 1.00000 | 784 | PICP 0.8421 (n=76) vs 0.8658 (n=708); z_iid=0.572 p_iid=0.5671 |
| LSTM-Gaussian | both | difference in coverage | regime transition (ex-ante) | -0.513 |  | 0.60817 | 1.00000 | 784 | PICP 0.8846 (n=52) vs 0.8620 (n=732); z_iid=-0.459 p_iid=0.6465 |
| LSTM-Gaussian | upper | DQ incremental | regime | 14.841 | 2 | 0.00059897 | 0.00359 | 780 | DQ_full=46.645 (df 7, p=6.545e-08) vs DQ_level+dynamics=31.805 (df 5, p=6.494e-06) |
| LSTM-Gaussian | upper | DQ incremental | transition | 0.072 | 1 | 0.78816 | 1.00000 | 780 | DQ_full=31.877 (df 6, p=1.723e-05) vs DQ_level+dynamics=31.805 (df 5, p=6.494e-06) |
| LSTM-Gaussian | upper | DQ incremental | width | 5.622 | 1 | 0.01773 | 0.07093 | 780 | DQ_full=37.427 (df 6, p=1.453e-06) vs DQ_level+dynamics=31.805 (df 5, p=6.494e-06) |
| LSTM-Gaussian | upper | difference in coverage | regime=calm | 3.011 |  | 0.00260 | 0.01300 | 784 | PICP 0.8918 (n=379) vs 0.9481 (n=405); z_iid=2.921 p_iid=0.003494 |
| LSTM-Gaussian | upper | difference in coverage | regime=transitional | -3.497 |  | 0.00047001 | 0.00329 | 784 | PICP 0.9574 (n=329) vs 0.8945 (n=455); z_iid=-3.223 p_iid=0.001269 |
| LSTM-Gaussian | upper | difference in coverage | regime=crisis | 0.294 |  | 0.76860 | 1.00000 | 784 | PICP 0.9079 (n=76) vs 0.9223 (n=708); z_iid=0.443 p_iid=0.658 |
| LSTM-Gaussian | upper | difference in coverage | regime transition (ex-ante) | -0.064 |  | 0.94858 | 1.00000 | 784 | PICP 0.9231 (n=52) vs 0.9208 (n=732); z_iid=-0.060 p_iid=0.9524 |
| Quantile-LSTM | both | DQ incremental | regime | 3.003 | 2 | 0.22283 | 1.00000 | 780 | DQ_full=61.164 (df 7, p=8.835e-11) vs DQ_level+dynamics=58.161 (df 5, p=2.913e-11) |
| Quantile-LSTM | both | DQ incremental | transition | 0.320 | 1 | 0.57155 | 1.00000 | 780 | DQ_full=58.482 (df 6, p=9.152e-11) vs DQ_level+dynamics=58.161 (df 5, p=2.913e-11) |
| Quantile-LSTM | both | DQ incremental | width | 0.197 | 1 | 0.65708 | 1.00000 | 780 | DQ_full=58.359 (df 6, p=9.693e-11) vs DQ_level+dynamics=58.161 (df 5, p=2.913e-11) |
| Quantile-LSTM | both | difference in coverage | regime=calm | 0.952 |  | 0.34109 | 1.00000 | 784 | PICP 0.8074 (n=379) vs 0.8346 (n=405); z_iid=0.993 p_iid=0.3207 |
| Quantile-LSTM | both | difference in coverage | regime=transitional | -1.204 |  | 0.22863 | 1.00000 | 784 | PICP 0.8419 (n=329) vs 0.8066 (n=455); z_iid=-1.275 p_iid=0.2021 |
| Quantile-LSTM | both | difference in coverage | regime=crisis | 0.381 |  | 0.70300 | 1.00000 | 784 | PICP 0.8026 (n=76) vs 0.8234 (n=708); z_iid=0.450 p_iid=0.6525 |
| Quantile-LSTM | both | difference in coverage | regime transition (ex-ante) | 0.270 |  | 0.78700 | 1.00000 | 784 | PICP 0.8077 (n=52) vs 0.8224 (n=732); z_iid=0.268 p_iid=0.789 |
| Quantile-LSTM | upper | DQ incremental | regime | 14.214 | 2 | 0.00081937 | 0.00574 | 780 | DQ_full=94.184 (df 7, p=1.707e-17) vs DQ_level+dynamics=79.970 (df 5, p=8.515e-16) |
| Quantile-LSTM | upper | DQ incremental | transition | 0.208 | 1 | 0.64798 | 1.00000 | 780 | DQ_full=80.178 (df 6, p=3.283e-15) vs DQ_level+dynamics=79.970 (df 5, p=8.515e-16) |
| Quantile-LSTM | upper | DQ incremental | width | 2.227 | 1 | 0.13564 | 0.54254 | 780 | DQ_full=82.196 (df 6, p=1.256e-15) vs DQ_level+dynamics=79.970 (df 5, p=8.515e-16) |
| Quantile-LSTM | upper | difference in coverage | regime=calm | 2.550 |  | 0.01076 | 0.06456 | 784 | PICP 0.8654 (n=379) vs 0.9235 (n=405); z_iid=2.653 p_iid=0.007981 |
| Quantile-LSTM | upper | difference in coverage | regime=transitional | -2.481 |  | 0.01309 | 0.06545 | 784 | PICP 0.9271 (n=329) vs 0.8725 (n=455); z_iid=-2.462 p_iid=0.01382 |
| Quantile-LSTM | upper | difference in coverage | regime=crisis | -0.280 |  | 0.77956 | 1.00000 | 784 | PICP 0.9079 (n=76) vs 0.8941 (n=708); z_iid=-0.374 p_iid=0.7082 |
| Quantile-LSTM | upper | difference in coverage | regime transition (ex-ante) | 0.275 |  | 0.78361 | 1.00000 | 784 | PICP 0.8846 (n=52) vs 0.8962 (n=732); z_iid=0.263 p_iid=0.7924 |

*Holm adjustment is within each (method, level, side) family; family sizes are in the CSV.*


### Where the band actually breaks

| regime_shift | timing | regime | n | mpiw | mean_log_width | mean_sigma_log | epistemic_share | mean_gate_entropy | upper_violation_rate | lower_violation_rate | expected_one_sided_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lagged t-1 (conditional / implementable) | calm | 379 | 5.13326e-05 | 1.9622 | 0.5965 | 0.00441 | 0.2877 | 0.1082 | 0.0580 | 0.05 |
| 1 | lagged t-1 (conditional / implementable) | transitional | 329 | 0.000151618 | 1.9595 | 0.5956 | 0.00161 | 0.4338 | 0.0547 | 0.0547 | 0.05 |
| 1 | lagged t-1 (conditional / implementable) | crisis | 76 | 0.000958206 | 1.9730 | 0.5998 | 0.01532 | 0.3265 | 0.0921 | 0.0395 | 0.05 |
| 1 | lagged t-1 (conditional / implementable) | all | 784 | 0.000181328 | 1.9621 | 0.5964 | 0.00431 | 0.3528 | 0.0842 | 0.0548 | 0.05 |

The **upper** tail — realized variance above the band, i.e. the day the model understated risk — breaks most often in the **calm** state (0.1082 against a nominal 0.050); the **lower** tail breaks most often in **calm** (0.0580). 
Both tails are worst in the same state, so the pooled view captures the structure adequately here.


Upper-tail coverage differences surviving Holm:

- `MC-Dropout-Regime-LSTM-B` · **regime=transitional** — PICP 0.9453 (n=329) vs 0.8945 (n=455); z_iid=-2.527 p_iid=0.0115; z_HAC=-2.642, p=0.008234, Holm=0.0494
- `MC-Dropout-LSTM` · **regime=calm** — PICP 0.8945 (n=379) vs 0.9481 (n=405); z_iid=2.805 p_iid=0.005039; z_HAC=2.890, p=0.003848, Holm=0.01924
- `MC-Dropout-LSTM` · **regime=transitional** — PICP 0.9574 (n=329) vs 0.8967 (n=455); z_iid=-3.133 p_iid=0.001727; z_HAC=-3.393, p=0.0006914, Holm=0.00484
- `LSTM-Gaussian` · **regime=calm** — PICP 0.8918 (n=379) vs 0.9481 (n=405); z_iid=2.921 p_iid=0.003494; z_HAC=3.011, p=0.0026, Holm=0.013
- `LSTM-Gaussian` · **regime=transitional** — PICP 0.9574 (n=329) vs 0.8945 (n=455); z_iid=-3.223 p_iid=0.001269; z_HAC=-3.497, p=0.00047, Holm=0.00329


**Does the gate widen the band where the regime is ambiguous?** Mean gate entropy ranges 0.288–0.434 nats across states (max possible 1.099), and the epistemic share of predictive variance ranges 0.16%–1.53%. Since the epistemic term is a small fraction of the total everywhere, the gate cannot move the interval width much even where it is maximally uncertain — this is the mechanism behind the combined model being a null on calibration.


## The deliverable: a width correction

| regime_shift | timing | scope | n | nominal | picp_at_fitted_sigma | crps_at_fitted_sigma | coverage_calibrating_scale | picp_at_calibrating_scale | crps_optimal_scale | crps_at_optimal_scale | crps_plateau_lo | crps_plateau_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lagged t-1 (conditional / implementable) | all | 784 | 0.9 | 0.8610 | 4.060831e-05 | 1.11 | 0.8992 | 1.16 | 4.0401372e-05 | 1.10 | 1.23 |
| 1 | lagged t-1 (conditional / implementable) | calm | 379 | 0.9 | 0.8338 | 1.3490187e-05 | 1.23 | 0.8997 | 1.29 | 1.3304322e-05 | 1.22 | 1.37 |
| 1 | lagged t-1 (conditional / implementable) | transitional | 329 | 0.9 | 0.8906 | 2.7111376e-05 | 1.02 | 0.9027 | 1.05 | 2.7093094e-05 | 0.99 | 1.11 |
| 1 | lagged t-1 (conditional / implementable) | crisis | 76 | 0.9 | 0.8684 | 0.00023426962 | 1.05 | 0.8947 | 1.21 | 0.00023248862 | 1.14 | 1.29 |

Globally, σ × **1.11** brings coverage to 0.8992 against a 90% nominal, and σ × **1.16** minimises CRPS (plateau 1.10–1.23, so quote it as approximate).
 Per-regime coverage-calibrating scales span 1.02–1.23 (spread 0.21), and the coverage tests independently find a t−1 conditioner that predicts a miss. Point estimates and formal tests agree, so **the deliverable is the per-regime column**: a single global scale would leave real, testable regime-dependent miscalibration on the table.


## Reproduce

```
python -m src.experiments.run_combined
# tables only, from the persisted forecasts (seconds, no training):
python -m src.experiments.run_combined --from-predictions
# the coverage tests alone, against any UQ predictions file:
python -m src.experiments.report_coverage --predictions m07_combined_intraday_2019_2022.parquet
```
