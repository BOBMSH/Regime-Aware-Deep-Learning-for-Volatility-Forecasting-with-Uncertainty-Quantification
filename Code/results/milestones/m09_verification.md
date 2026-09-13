# m09 - Reproducibility gate

_Generated 2026-09-13 03:00 UTC. Relative tolerance 1e-09._

Every number below was recomputed **from the prediction parquets** by code that imports nothing from `src.evaluation`, and compared with what the published table says. The duplication of the metric formulas is deliberate: calling the project's own functions would check that a CSV matches what those functions return today, and agree enthusiastically with a regression in the functions themselves.

## PASS - 943 values, all re-derive

Every published point metric, interval metric, per-regime loss and sample size reproduces from the primary data, and every model carrying a value in more than one table carries the same value in all of them.

## Coverage

| status | values |
|---|---|
| OK | 941 |
| SKIP | 2 |

### By table

| table | checked | mismatched |
|---|---|---|
| m02_intraday_2019_2022_loghar_metrics.csv | 10 | 0 |
| m02_intraday_2019_2022_metrics.csv | 20 | 0 |
| m02_intraday_2019_2022_metrics.csv vs m03_lstm_intraday_2019_2022_metrics.csv | 4 | 0 |
| m02_intraday_2019_2022_metrics.csv vs m05_regime_dl_intraday_2019_2022_metrics.csv | 4 | 0 |
| m02_intraday_2019_2022_metrics.csv vs m07_combined_intraday_2019_2022_master.csv | 4 | 0 |
| m02_intraday_2019_2022_metrics.csv vs m08_rq4_metrics.csv | 4 | 0 |
| m02_rq4_ftse_metrics.csv | 20 | 0 |
| m02_rq4_ftse_metrics.csv vs m05_regime_dl_rq4_ftse_metrics.csv | 4 | 0 |
| m02_rq4_ftse_metrics.csv vs m08_rq4_metrics.csv | 4 | 0 |
| m02_rq4_rut_metrics.csv | 20 | 0 |
| m02_rq4_rut_metrics.csv vs m05_regime_dl_rq4_rut_metrics.csv | 4 | 0 |
| m02_rq4_rut_metrics.csv vs m08_rq4_metrics.csv | 4 | 0 |
| m02_yang_zhang_diagnostic_metrics.csv | 20 | 0 |
| m03_lstm_intraday_2019_2022_metrics.csv | 35 | 0 |
| m03_lstm_intraday_2019_2022_metrics.csv vs m05_regime_dl_intraday_2019_2022_metrics.csv | 2 | 0 |
| m03_lstm_intraday_2019_2022_metrics.csv vs m07_combined_intraday_2019_2022_master.csv | 2 | 0 |
| m03_lstm_intraday_2019_2022_metrics.csv vs m08_rq4_metrics.csv | 2 | 0 |
| m03_lstm_intraday_2019_2022_refit1_metrics.csv | 25 | 0 |
| m05_regime_dl_intraday_2019_2022_gw_per_regime.csv | 45 | 0 |
| m05_regime_dl_intraday_2019_2022_hmm_gw_per_regime.csv | 45 | 0 |
| m05_regime_dl_intraday_2019_2022_hmm_metrics.csv | 45 | 0 |
| m05_regime_dl_intraday_2019_2022_hmm_metrics.csv vs m07_combined_intraday_2019_2022_hmm_master.csv | 9 | 0 |
| m05_regime_dl_intraday_2019_2022_hmm_per_regime.csv | 28 | 0 |
| m05_regime_dl_intraday_2019_2022_metrics.csv | 45 | 0 |
| m05_regime_dl_intraday_2019_2022_metrics.csv vs m07_combined_intraday_2019_2022_master.csv | 3 | 0 |
| m05_regime_dl_intraday_2019_2022_metrics.csv vs m08_rq4_metrics.csv | 3 | 0 |
| m05_regime_dl_intraday_2019_2022_per_regime.csv | 28 | 0 |
| m05_regime_dl_rq4_ftse_metrics.csv | 25 | 0 |
| m05_regime_dl_rq4_ftse_metrics.csv vs m08_rq4_metrics.csv | 1 | 0 |
| m05_regime_dl_rq4_ftse_per_regime.csv | 12 | 0 |
| m05_regime_dl_rq4_rut_metrics.csv | 25 | 0 |
| m05_regime_dl_rq4_rut_metrics.csv vs m08_rq4_metrics.csv | 1 | 0 |
| m05_regime_dl_rq4_rut_per_regime.csv | 12 | 0 |
| m06_uq_intraday_2019_2022_calibration.csv | 25 | 0 |
| m06_uq_intraday_2019_2022_hmm_calibration.csv | 25 | 0 |
| m06_uq_intraday_2019_2022_hmm_per_regime_mc.csv | 20 | 0 |
| m06_uq_intraday_2019_2022_hmm_per_regime_q.csv | 20 | 0 |
| m06_uq_intraday_2019_2022_per_regime_mc.csv | 20 | 0 |
| m06_uq_intraday_2019_2022_per_regime_q.csv | 20 | 0 |
| m07_combined_intraday_2019_2022_calibration.csv | 35 | 0 |
| m07_combined_intraday_2019_2022_hmm_calibration.csv | 35 | 0 |
| m07_combined_intraday_2019_2022_hmm_master.csv | 53 | 0 |
| m07_combined_intraday_2019_2022_master.csv | 53 | 0 |
| m08_rq4_metrics.csv | 95 | 0 |
| m08_rq4_per_regime.csv | 27 | 0 |

## What this gate does not certify

Test statistics (Diebold-Mariano, Giacomini-White, Model Confidence Set, Kupiec, Christoffersen, dynamic quantile) are **not** recomputed here. Their values depend on HAC bandwidth rules, finite-sample corrections and bootstrap draws whose conventions are this project's own choices, so a second implementation would be reimplementing those choices rather than checking them. They are covered by `tests/test_significance.py` against closed-form cases; what this gate certifies for them is that they were computed on the right sample from forecasts that do re-derive.
