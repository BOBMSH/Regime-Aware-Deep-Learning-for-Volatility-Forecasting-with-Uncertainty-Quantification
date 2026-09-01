# Milestone m07 — Phase 7: combined regime × uncertainty model, and RQ3 made inferential

*Profile:* `intraday_2019_2022_hmm` · *test window:* 2019-01-02 → 2022-02-25 (n=784) · *headline level:* 90% · *MC samples:* 100

## Scope and what is being isolated

The combined model is **MC-Dropout-Regime-LSTM-B** — the Phase-5 `Regime-LSTM-B` mixture-of-experts with MC-Dropout inference, training inherited unchanged. "Best of Phase 5" is `Regime-LSTM-B`, selected on the transitional result -- the only regime bucket where the effect is both large and formally tested (per-regime DM p=4.3e-09 vs HAR-RV, and the state carrying the Giacomini-White moment behind chi2(3)=40.1). Calm and crisis are point estimates on 379 and 76 days with no significant test behind them, and crisis reverses sign between the two regime estimators. The rule was fixed before this phase ran.

Because training is inherited and the hyperparameters are still the Phase-3 validation selection, the contrast `MC-Dropout-LSTM` → `MC-Dropout-Regime-LSTM-B` isolates **regime conditioning inside the uncertainty-aware family** and nothing else: same network shape, same seed, same folds, same MC sample count.

## The pre-registered expectation

Recorded in `ROADMAP.md` **before** this phase was run: *Kupiec should reject decisively — the intervals are 3.5–7.9 pp narrow — while Christoffersen's independence component and any t−1 conditioning variable should not, because the failures are unclustered and unpredictable.* If that holds, miscalibration is a **level** problem rather than a **timing** problem, the combined model is a pre-registered null, and the deliverable is a global width correction.

### Verdict, computed from the tables below

| clause | expected | found |
|---|---|---|
| Kupiec rejects (level is wrong) | yes | **yes** — MC-Dropout-Regime-LSTM-B p=0.000798; MC-Dropout-LSTM p=0.00171; LSTM-Gaussian p=0.00117; Quantile-LSTM p=2.31e-11 |
| Christoffersen independence does *not* reject (no clustering) | yes | **yes** — MC-Dropout-Regime-LSTM-B p=0.563; MC-Dropout-LSTM p=0.843; LSTM-Gaussian p=0.909; Quantile-LSTM p=0.455 |
| no t−1 conditioner predicts a miss (pooled hit) | yes | **no conditioner survives Holm** |
| no t−1 subsample differs in *upper-tail* coverage | (not pre-registered) | **0 of 16 survive Holm** |

**The pre-registration holds in full.** Miscalibration is a level problem, not a timing problem, and the combined model is a pre-registered null. On the width correction itself the σ-scale table needs reading with care — see the deliverable section, which reconciles the per-regime point estimates against the absence of tested evidence.


## Master results table (headline level 90%)

| rank_qlike | model | family | qlike | qlike_transitional | mse | mae | picp | mpiw | winkler | crps | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MC-Dropout-Regime-LSTM-B | combined (Phase 7) | 0.2582 | 0.1951 | 4.055e-08 | 5.333e-05 | 0.8622 | 0.0001804 | 0.0003993 | 4.055e-05 | 784 |
| 2 | Regime-LSTM-B | regime-aware deep | 0.2583 | 0.1951 | 4.049e-08 | 5.334e-05 |  |  |  |  | 784 |
| 3 | LSTM-RVonly | deep | 0.2596 | 0.2331 | 4.021e-08 | 6.058e-05 |  |  |  |  | 784 |
| 4 | Regime-LSTM-A-RVonly | regime-aware deep | 0.2617 | 0.2271 | 4.2e-08 | 6.507e-05 |  |  |  |  | 784 |
| 5 | Regime-LSTM-A | regime-aware deep | 0.2627 | 0.2049 | 4.035e-08 | 5.813e-05 |  |  |  |  | 784 |
| 6 | LSTM-Gaussian | uncertainty-aware deep | 0.2650 | 0.2043 | 4.485e-08 | 5.863e-05 | 0.8635 | 0.0001801 | 0.0004198 | 4.301e-05 | 784 |
| 7 | LSTM | deep | 0.2650 | 0.2043 | 4.485e-08 | 5.863e-05 |  |  |  |  | 784 |
| 8 | MC-Dropout-LSTM | uncertainty-aware deep | 0.2652 | 0.2044 | 4.494e-08 | 5.874e-05 | 0.8648 | 0.0001818 | 0.0004199 | 4.306e-05 | 784 |
|  | Quantile-LSTM-mean | sensitivity (retransformed point) | 0.2742 | 0.2051 | 4.42e-08 | 5.634e-05 |  |  |  |  | 784 |
| 9 | HAR-RV | econometric | 0.2745 | 0.2409 | 4.176e-08 | 6.046e-05 |  |  |  |  | 784 |
| 10 | Quantile-LSTM | uncertainty-aware deep | 0.3216 | 0.2279 | 5.131e-08 | 5.73e-05 | 0.8214 | 0.0001598 | 0.0004317 |  | 784 |
| 11 | GARCH | econometric | 0.3302 | 0.2907 | 5.625e-08 | 7.172e-05 |  |  |  |  | 784 |
| 12 | EGARCH | econometric | 0.3328 | 0.2778 | 5.666e-08 | 6.902e-05 |  |  |  |  | 784 |
| 13 | RW-RV | econometric | 0.3584 | 0.2810 | 5.284e-08 | 6.579e-05 |  |  |  |  | 784 |

*`qlike_transitional` is the transitional bucket on the **t−1** label (lagged t-1 (conditional / implementable)), n=332. Interval columns are blank for models that produce no predictive distribution — that contrast is the point of the table.*


*Rows with a blank `rank_qlike` are sensitivities, not competitors: **Quantile-LSTM-mean** (0.2742). `Quantile-LSTM-mean` is the quantile model's own forecast retransformed onto the mean scale QLIKE elicits — the correction every other deep model in this table already carries — so it is the row to read when comparing the pinball head's point accuracy against the mean-scale models, and it competes with nothing. Ranks below count the 13 models only.*


Pooled, the best model on QLIKE is **MC-Dropout-Regime-LSTM-B** (0.2582); the combined model ranks **1 of 13** at 0.2582.
 Against the regime-agnostic `MC-Dropout-LSTM` (0.2652) that is a change of -0.0071 (-2.7%), and in the transitional bucket 0.1951 vs 0.2044.


## Significance on the Phase-7 pairs

| model_a | model_b | dm_stat | p_value | mean_loss_diff | better | n |
|---|---|---|---|---|---|---|
| MC-Dropout-Regime-LSTM-B | MC-Dropout-LSTM | -2.2203 | 0.0267 | -0.007050 | MC-Dropout-Regime-LSTM-B | 784 |
| MC-Dropout-Regime-LSTM-B | Regime-LSTM-B | -1.2500 | 0.2117 | -0.000130965 | MC-Dropout-Regime-LSTM-B | 784 |
| MC-Dropout-Regime-LSTM-B | HAR-RV | -1.0497 | 0.2942 | -0.016307 | MC-Dropout-Regime-LSTM-B | 784 |
| MC-Dropout-Regime-LSTM-B | LSTM | -2.1690 | 0.0304 | -0.006836 | MC-Dropout-Regime-LSTM-B | 784 |
| MC-Dropout-LSTM | LSTM | 1.3159 | 0.1886 | 0.000214018 | LSTM | 784 |

**Giacomini–White, conditional on the lagged regime indicator.** A negative moment means the first model has the lower loss in that state.

| model_a | model_b | n | gw_regime_stat | gw_regime_df | gw_regime_p | moment_calm | moment_transitional | moment_crisis | a_relative_edge_in | a_wins_in |
|---|---|---|---|---|---|---|---|---|---|---|
| MC-Dropout-Regime-LSTM-B | MC-Dropout-LSTM | 784 | 9.5807 | 3 | 0.022489 | -0.002080 | -0.003924 | -0.001046 | transitional | calm, transitional, crisis |
| MC-Dropout-Regime-LSTM-B | Regime-LSTM-B | 784 | 3.0358 | 3 | 0.386145 | -0.000148211 | 1.96362e-05 | -2.3896e-06 | calm | calm, crisis |
| MC-Dropout-Regime-LSTM-B | HAR-RV | 784 | 24.1874 | 3 | 2.28275e-05 | 0.003775 | -0.019382 | -0.00069985 | transitional | transitional, crisis |
| MC-Dropout-Regime-LSTM-B | LSTM | 784 | 9.1587 | 3 | 0.027253 | -0.001962 | -0.003892 | -0.000982456 | transitional | calm, transitional, crisis |
| MC-Dropout-LSTM | LSTM | 784 | 1.7100 | 3 | 0.634702 | 0.000118863 | 3.20491e-05 | 6.31057e-05 | transitional | none |

## RQ3, made inferential — coverage tests on the full 784-day sample

Every regressor is lagged into the t−1 information set, and every test runs on the **full** sample. Two rules make these numbers mean what they appear to mean, both learned from earlier audits of this project:

1. **A subsample selector must be measurable at t−1.** The ex-post transition indicator `1{s_t ≠ s_(t−1)}` is detected by the same one-day surprise that breaks an interval; a test on it has no nominal size. Only `1{s_(t−1) ≠ s_(t−2)}` appears here.

2. **A subsample claim is a *difference*, never a comparison against nominal.** These intervals are all too narrow globally, so a Kupiec test against nominal rejects on *any* subsample with power — that would report a level failure as a timing failure. Subsample rows below are two-sample differences with Holm control; conditional DQ rows are **incremental** over level and hit dynamics, for the same reason one level up.

| method | side | test | stat | df | p_value | violation_rate | expected_violation_rate | n |
|---|---|---|---|---|---|---|---|---|
| MC-Dropout-Regime-LSTM-B | both | Kupiec LR_uc | 11.246 | 1 | 0.00079802 | 0.1378 | 0.1000 | 784 |
| MC-Dropout-Regime-LSTM-B | both | Christoffersen LR_ind | 0.335 | 1 | 0.56253 | 0.1378 | 0.1000 | 784 |
| MC-Dropout-Regime-LSTM-B | both | Christoffersen LR_cc | 11.581 | 2 | 0.00306 | 0.1378 | 0.1000 | 784 |
| MC-Dropout-Regime-LSTM-B | both | Engle-Manganelli DQ | 16.182 | 5 | 0.00634 | 0.1378 | 0.1000 | 784 |
| MC-Dropout-Regime-LSTM-B | upper | Kupiec LR_uc | 15.047 | 1 | 0.00010489 | 0.0829 | 0.0500 | 784 |
| MC-Dropout-Regime-LSTM-B | upper | Christoffersen LR_ind | 0.527 | 1 | 0.46803 | 0.0829 | 0.0500 | 784 |
| MC-Dropout-Regime-LSTM-B | upper | Christoffersen LR_cc | 15.573 | 2 | 0.00041526 | 0.0829 | 0.0500 | 784 |
| MC-Dropout-Regime-LSTM-B | upper | Engle-Manganelli DQ | 31.142 | 5 | 8.7812e-06 | 0.0829 | 0.0500 | 784 |
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
| MC-Dropout-Regime-LSTM-B | both | DQ incremental | regime | 5.905 | 2 | 0.05222 | 0.31331 | 780 | DQ_full=22.087 (df 7, p=0.002455) vs DQ_level+dynamics=16.182 (df 5, p=0.006343) |
| MC-Dropout-Regime-LSTM-B | both | DQ incremental | transition | 1.057 | 1 | 0.30398 | 0.98060 | 780 | DQ_full=17.239 (df 6, p=0.008445) vs DQ_level+dynamics=16.182 (df 5, p=0.006343) |
| MC-Dropout-Regime-LSTM-B | both | DQ incremental | width | 0.827 | 1 | 0.36302 | 0.98060 | 780 | DQ_full=17.009 (df 6, p=0.009249) vs DQ_level+dynamics=16.182 (df 5, p=0.006343) |
| MC-Dropout-Regime-LSTM-B | both | difference in coverage | regime=calm | 2.013 |  | 0.04414 | 0.30900 | 784 | PICP 0.8373 (n=381) vs 0.8859 (n=403); z_iid=1.973 p_iid=0.04851 |
| MC-Dropout-Regime-LSTM-B | both | difference in coverage | regime=transitional | -1.215 |  | 0.22456 | 0.98060 | 784 | PICP 0.8795 (n=332) vs 0.8496 (n=452); z_iid=-1.203 p_iid=0.2291 |
| MC-Dropout-Regime-LSTM-B | both | difference in coverage | regime=crisis | -1.293 |  | 0.19612 | 0.98060 | 784 | PICP 0.9155 (n=71) vs 0.8569 (n=713); z_iid=-1.365 p_iid=0.1722 |
| MC-Dropout-Regime-LSTM-B | both | difference in coverage | regime transition (ex-ante) | 0.614 |  | 0.53932 | 0.98060 | 784 | PICP 0.8305 (n=59) vs 0.8648 (n=725); z_iid=0.736 p_iid=0.462 |
| MC-Dropout-Regime-LSTM-B | upper | DQ incremental | regime | 6.332 | 2 | 0.04218 | 0.29525 | 780 | DQ_full=37.474 (df 7, p=3.814e-06) vs DQ_level+dynamics=31.142 (df 5, p=8.781e-06) |
| MC-Dropout-Regime-LSTM-B | upper | DQ incremental | transition | 1.351 | 1 | 0.24514 | 0.90869 | 780 | DQ_full=32.493 (df 6, p=1.312e-05) vs DQ_level+dynamics=31.142 (df 5, p=8.781e-06) |
| MC-Dropout-Regime-LSTM-B | upper | DQ incremental | width | 1.458 | 1 | 0.22717 | 0.90869 | 780 | DQ_full=32.601 (df 6, p=1.251e-05) vs DQ_level+dynamics=31.142 (df 5, p=8.781e-06) |
| MC-Dropout-Regime-LSTM-B | upper | difference in coverage | regime=calm | 1.932 |  | 0.05338 | 0.32026 | 784 | PICP 0.8976 (n=381) vs 0.9355 (n=403); z_iid=1.921 p_iid=0.05476 |
| MC-Dropout-Regime-LSTM-B | upper | difference in coverage | regime=transitional | -1.851 |  | 0.06411 | 0.32057 | 784 | PICP 0.9367 (n=332) vs 0.9027 (n=452); z_iid=-1.711 p_iid=0.08717 |
| MC-Dropout-Regime-LSTM-B | upper | difference in coverage | regime=crisis | -0.305 |  | 0.76062 | 0.90869 | 784 | PICP 0.9296 (n=71) vs 0.9158 (n=713); z_iid=-0.400 p_iid=0.6891 |
| MC-Dropout-Regime-LSTM-B | upper | difference in coverage | regime transition (ex-ante) | 0.786 |  | 0.43191 | 0.90869 | 784 | PICP 0.8814 (n=59) vs 0.9200 (n=725); z_iid=1.035 p_iid=0.3006 |
| MC-Dropout-LSTM | both | DQ incremental | regime | 2.456 | 2 | 0.29292 | 1.00000 | 780 | DQ_full=16.710 (df 7, p=0.01937) vs DQ_level+dynamics=14.254 (df 5, p=0.01407) |
| MC-Dropout-LSTM | both | DQ incremental | transition | 0.237 | 1 | 0.62647 | 1.00000 | 780 | DQ_full=14.491 (df 6, p=0.02461) vs DQ_level+dynamics=14.254 (df 5, p=0.01407) |
| MC-Dropout-LSTM | both | DQ incremental | width | 0.177 | 1 | 0.67426 | 1.00000 | 780 | DQ_full=14.431 (df 6, p=0.02518) vs DQ_level+dynamics=14.254 (df 5, p=0.01407) |
| MC-Dropout-LSTM | both | difference in coverage | regime=calm | 1.331 |  | 0.18310 | 1.00000 | 784 | PICP 0.8478 (n=381) vs 0.8809 (n=403); z_iid=1.356 p_iid=0.1752 |
| MC-Dropout-LSTM | both | difference in coverage | regime=transitional | -1.017 |  | 0.30912 | 1.00000 | 784 | PICP 0.8795 (n=332) vs 0.8540 (n=452); z_iid=-1.033 p_iid=0.3015 |
| MC-Dropout-LSTM | both | difference in coverage | regime=crisis | -0.585 |  | 0.55826 | 1.00000 | 784 | PICP 0.8873 (n=71) vs 0.8626 (n=713); z_iid=-0.582 p_iid=0.5605 |
| MC-Dropout-LSTM | both | difference in coverage | regime transition (ex-ante) | 0.339 |  | 0.73467 | 1.00000 | 784 | PICP 0.8475 (n=59) vs 0.8662 (n=725); z_iid=0.405 p_iid=0.6855 |
| MC-Dropout-LSTM | upper | DQ incremental | regime | 7.776 | 2 | 0.02049 | 0.12292 | 780 | DQ_full=38.091 (df 7, p=2.912e-06) vs DQ_level+dynamics=30.315 (df 5, p=1.279e-05) |
| MC-Dropout-LSTM | upper | DQ incremental | transition | 0.346 | 1 | 0.55635 | 1.00000 | 780 | DQ_full=30.661 (df 6, p=2.942e-05) vs DQ_level+dynamics=30.315 (df 5, p=1.279e-05) |
| MC-Dropout-LSTM | upper | DQ incremental | width | 4.513 | 1 | 0.03364 | 0.13456 | 780 | DQ_full=34.828 (df 6, p=4.654e-06) vs DQ_level+dynamics=30.315 (df 5, p=1.279e-05) |
| MC-Dropout-LSTM | upper | difference in coverage | regime=calm | 2.289 |  | 0.02208 | 0.12292 | 784 | PICP 0.9003 (n=381) vs 0.9429 (n=403); z_iid=2.229 p_iid=0.02581 |
| MC-Dropout-LSTM | upper | difference in coverage | regime=transitional | -2.377 |  | 0.01745 | 0.12212 | 784 | PICP 0.9458 (n=332) vs 0.9049 (n=452); z_iid=-2.113 p_iid=0.03458 |
| MC-Dropout-LSTM | upper | difference in coverage | regime=crisis | -0.180 |  | 0.85701 | 1.00000 | 784 | PICP 0.9296 (n=71) vs 0.9215 (n=713); z_iid=-0.244 p_iid=0.8076 |
| MC-Dropout-LSTM | upper | difference in coverage | regime transition (ex-ante) | 0.537 |  | 0.59129 | 1.00000 | 784 | PICP 0.8983 (n=59) vs 0.9241 (n=725); z_iid=0.712 p_iid=0.4763 |
| LSTM-Gaussian | both | DQ incremental | regime | 2.906 | 2 | 0.23389 | 1.00000 | 780 | DQ_full=18.549 (df 7, p=0.009724) vs DQ_level+dynamics=15.643 (df 5, p=0.00794) |
| LSTM-Gaussian | both | DQ incremental | transition | 0.181 | 1 | 0.67044 | 1.00000 | 780 | DQ_full=15.824 (df 6, p=0.01473) vs DQ_level+dynamics=15.643 (df 5, p=0.00794) |
| LSTM-Gaussian | both | DQ incremental | width | 0.402 | 1 | 0.52611 | 1.00000 | 780 | DQ_full=16.045 (df 6, p=0.01351) vs DQ_level+dynamics=15.643 (df 5, p=0.00794) |
| LSTM-Gaussian | both | difference in coverage | regime=calm | 1.414 |  | 0.15723 | 1.00000 | 784 | PICP 0.8451 (n=381) vs 0.8809 (n=403); z_iid=1.457 p_iid=0.145 |
| LSTM-Gaussian | both | difference in coverage | regime=transitional | -1.092 |  | 0.27463 | 1.00000 | 784 | PICP 0.8795 (n=332) vs 0.8518 (n=452); z_iid=-1.118 p_iid=0.2635 |
| LSTM-Gaussian | both | difference in coverage | regime=crisis | -0.618 |  | 0.53688 | 1.00000 | 784 | PICP 0.8873 (n=71) vs 0.8612 (n=713); z_iid=-0.613 p_iid=0.5401 |
| LSTM-Gaussian | both | difference in coverage | regime transition (ex-ante) | 0.314 |  | 0.75376 | 1.00000 | 784 | PICP 0.8475 (n=59) vs 0.8648 (n=725); z_iid=0.374 p_iid=0.7086 |
| LSTM-Gaussian | upper | DQ incremental | regime | 8.634 | 2 | 0.01334 | 0.09026 | 780 | DQ_full=40.438 (df 7, p=1.038e-06) vs DQ_level+dynamics=31.805 (df 5, p=6.494e-06) |
| LSTM-Gaussian | upper | DQ incremental | transition | 0.305 | 1 | 0.58084 | 1.00000 | 780 | DQ_full=32.110 (df 6, p=1.555e-05) vs DQ_level+dynamics=31.805 (df 5, p=6.494e-06) |
| LSTM-Gaussian | upper | DQ incremental | width | 5.622 | 1 | 0.01773 | 0.09026 | 780 | DQ_full=37.427 (df 6, p=1.453e-06) vs DQ_level+dynamics=31.805 (df 5, p=6.494e-06) |
| LSTM-Gaussian | upper | difference in coverage | regime=calm | 2.411 |  | 0.01592 | 0.09026 | 784 | PICP 0.8976 (n=381) vs 0.9429 (n=403); z_iid=2.349 p_iid=0.01884 |
| LSTM-Gaussian | upper | difference in coverage | regime=transitional | -2.487 |  | 0.01289 | 0.09026 | 784 | PICP 0.9458 (n=332) vs 0.9027 (n=452); z_iid=-2.211 p_iid=0.02703 |
| LSTM-Gaussian | upper | difference in coverage | regime=crisis | -0.211 |  | 0.83272 | 1.00000 | 784 | PICP 0.9296 (n=71) vs 0.9201 (n=713); z_iid=-0.284 p_iid=0.7768 |
| LSTM-Gaussian | upper | difference in coverage | regime transition (ex-ante) | 0.508 |  | 0.61139 | 1.00000 | 784 | PICP 0.8983 (n=59) vs 0.9228 (n=725); z_iid=0.669 p_iid=0.5033 |
| Quantile-LSTM | both | DQ incremental | regime | 0.838 | 2 | 0.65785 | 1.00000 | 780 | DQ_full=58.999 (df 7, p=2.391e-10) vs DQ_level+dynamics=58.161 (df 5, p=2.913e-11) |
| Quantile-LSTM | both | DQ incremental | transition | 1.878 | 1 | 0.17060 | 1.00000 | 780 | DQ_full=60.039 (df 6, p=4.419e-11) vs DQ_level+dynamics=58.161 (df 5, p=2.913e-11) |
| Quantile-LSTM | both | DQ incremental | width | 0.197 | 1 | 0.65708 | 1.00000 | 780 | DQ_full=58.359 (df 6, p=9.693e-11) vs DQ_level+dynamics=58.161 (df 5, p=2.913e-11) |
| Quantile-LSTM | both | difference in coverage | regime=calm | 0.706 |  | 0.48030 | 1.00000 | 784 | PICP 0.8110 (n=381) vs 0.8313 (n=403); z_iid=0.740 p_iid=0.4595 |
| Quantile-LSTM | both | difference in coverage | regime=transitional | -0.419 |  | 0.67532 | 1.00000 | 784 | PICP 0.8283 (n=332) vs 0.8164 (n=452); z_iid=-0.431 p_iid=0.6662 |
| Quantile-LSTM | both | difference in coverage | regime=crisis | -0.536 |  | 0.59225 | 1.00000 | 784 | PICP 0.8451 (n=71) vs 0.8191 (n=713); z_iid=-0.545 p_iid=0.5855 |
| Quantile-LSTM | both | difference in coverage | regime transition (ex-ante) | 0.744 |  | 0.45701 | 1.00000 | 784 | PICP 0.7797 (n=59) vs 0.8248 (n=725); z_iid=0.871 p_iid=0.3837 |
| Quantile-LSTM | upper | DQ incremental | regime | 9.234 | 2 | 0.00988 | 0.06917 | 780 | DQ_full=89.204 (df 7, p=1.803e-16) vs DQ_level+dynamics=79.970 (df 5, p=8.515e-16) |
| Quantile-LSTM | upper | DQ incremental | transition | 1.320 | 1 | 0.25057 | 0.75171 | 780 | DQ_full=81.290 (df 6, p=1.934e-15) vs DQ_level+dynamics=79.970 (df 5, p=8.515e-16) |
| Quantile-LSTM | upper | DQ incremental | width | 2.227 | 1 | 0.13564 | 0.54254 | 780 | DQ_full=82.196 (df 6, p=1.256e-15) vs DQ_level+dynamics=79.970 (df 5, p=8.515e-16) |
| Quantile-LSTM | upper | difference in coverage | regime=calm | 2.043 |  | 0.04102 | 0.24613 | 784 | PICP 0.8714 (n=381) vs 0.9181 (n=403); z_iid=2.137 p_iid=0.03263 |
| Quantile-LSTM | upper | difference in coverage | regime=transitional | -1.635 |  | 0.10203 | 0.51016 | 784 | PICP 0.9157 (n=332) vs 0.8805 (n=452); z_iid=-1.588 p_iid=0.1122 |
| Quantile-LSTM | upper | difference in coverage | regime=crisis | -0.827 |  | 0.40819 | 0.81638 | 784 | PICP 0.9296 (n=71) vs 0.8920 (n=713); z_iid=-0.987 p_iid=0.3239 |
| Quantile-LSTM | upper | difference in coverage | regime transition (ex-ante) | 0.672 |  | 0.50185 | 0.81638 | 784 | PICP 0.8644 (n=59) vs 0.8979 (n=725); z_iid=0.809 p_iid=0.4184 |

*Holm adjustment is within each (method, level, side) family; family sizes are in the CSV.*


### Where the band actually breaks

| regime_shift | timing | regime | n | mpiw | mean_log_width | mean_sigma_log | epistemic_share | mean_gate_entropy | upper_violation_rate | lower_violation_rate | expected_one_sided_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lagged t-1 (conditional / implementable) | calm | 381 | 5.03381e-05 | 1.9637 | 0.5969 | 0.00453 | 0.3140 | 0.1024 | 0.0604 | 0.05 |
| 1 | lagged t-1 (conditional / implementable) | transitional | 332 | 0.000158299 | 1.9607 | 0.5960 | 0.00153 | 0.4433 | 0.0633 | 0.0572 | 0.05 |
| 1 | lagged t-1 (conditional / implementable) | crisis | 71 | 0.000982104 | 1.9741 | 0.6001 | 0.01503 | 0.2925 | 0.0704 | 0.0141 | 0.05 |
| 1 | lagged t-1 (conditional / implementable) | all | 784 | 0.000180438 | 1.9634 | 0.5968 | 0.00422 | 0.3668 | 0.0829 | 0.0548 | 0.05 |

The **upper** tail — realized variance above the band, i.e. the day the model understated risk — breaks most often in the **calm** state (0.1024 against a nominal 0.050); the **lower** tail breaks most often in **calm** (0.0604). 
The two tails peak in the **same** state. 
Neither view finds subsample structure that survives family-wise control (**0 of 16** pooled and **0 of 16** upper-tail differences survive Holm), so the ordering above is descriptive only and no adequacy claim is made in either direction.


**Does the gate widen the band where the regime is ambiguous?** Mean gate entropy ranges 0.292–0.443 nats across states (max possible 1.099), and the epistemic share of predictive variance ranges 0.15%–1.50%. Since the epistemic term is a small fraction of the total everywhere, the gate cannot move the interval width much even where it is maximally uncertain — this is the mechanism behind the combined model being a null on calibration.


## The deliverable: a width correction

| regime_shift | timing | scope | n | nominal | picp_at_fitted_sigma | crps_at_fitted_sigma | coverage_calibrating_scale | picp_at_calibrating_scale | crps_optimal_scale | crps_at_optimal_scale | crps_plateau_lo | crps_plateau_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lagged t-1 (conditional / implementable) | all | 784 | 0.9 | 0.8622 | 4.0547394e-05 | 1.11 | 0.9018 | 1.17 | 4.0334342e-05 | 1.10 | 1.24 |
| 1 | lagged t-1 (conditional / implementable) | calm | 381 | 0.9 | 0.8373 | 1.3080587e-05 | 1.23 | 0.9003 | 1.28 | 1.2919333e-05 | 1.20 | 1.35 |
| 1 | lagged t-1 (conditional / implementable) | transitional | 332 | 0.9 | 0.8795 | 3.0206209e-05 | 1.04 | 0.9006 | 1.11 | 3.0112294e-05 | 1.05 | 1.17 |
| 1 | lagged t-1 (conditional / implementable) | crisis | 71 | 0.9 | 0.9155 | 0.00023629566 | 0.90 | 0.9014 | 1.18 | 0.00023500707 | 1.11 | 1.26 |

Globally, σ × **1.11** brings coverage to 0.9018 against a 90% nominal, and σ × **1.17** minimises CRPS (plateau 1.10–1.24, so quote it as approximate).
 Per-regime coverage-calibrating scales span 0.90–1.23 (spread 0.33), which looks material — but the coverage tests find no t−1 conditioner that predicts a miss once the level is accounted for and the family-wise rate controlled. These scales are point estimates with no standard error attached, on buckets as small as the crisis one, so the spread is **not** evidence of regime-dependent miscalibration. **The defensible deliverable is the global scale**, with the per-regime column reported as indicative and explicitly untested.


## Reproduce

```
python -m src.experiments.run_combined --config combined_hmm
# tables only, from the persisted forecasts (seconds, no training):
python -m src.experiments.run_combined --config combined_hmm --from-predictions
# the coverage tests alone, against any UQ predictions file:
python -m src.experiments.report_coverage --predictions results/predictions/m07_combined_intraday_2019_2022_hmm.parquet
```
