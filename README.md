# Regime-Aware Deep Learning for Financial Volatility Forecasting with Uncertainty Quantification

MSc dissertation and accompanying code. **WMG, University of Warwick** — MSc Applied Artificial Intelligence, 2026.

Can deep learning improve on established econometric forecasts of daily realised volatility, and can it say how far its own forecasts deserve to be trusted? This project integrates three literatures that have matured largely apart — regime-switching econometrics, recurrent deep learning, and approximate Bayesian uncertainty estimation — into a single evaluated pipeline.

---

## What the study found

Thirteen specifications, evaluated on five-minute realised variance for the S&P 500 over 5,530 trading days, with the final 784 held out and used once. QLIKE is the ranking criterion.

| Research question | Answer | Evidence |
|---|---|---|
| **RQ1** — Does deep learning beat the econometric benchmarks? | It displaces the GARCH family decisively. It does **not** displace HAR-RV. | Best network 0.2596 vs HAR-RV 0.2745, a 5.4% reduction — but Diebold–Mariano returns *p* = 0.459, and HAR-RV stays inside the Model Confidence Set while GARCH, EGARCH and the random walk are excluded. |
| **RQ2** — Does conditioning on market regime help? | **Not on average. Substantially in the transitional state — and the state is known before the forecast is made.** | Mixture of experts 0.1851 vs HAR-RV 0.2367 in the transitional state, a 21.8% reduction. Giacomini–White conditional test: Wald 40.13 on 3 df, *p* = 9.99e-09. Broad-based: 214 of 329 transitional days favour the mixture. |
| **RQ3** — Are the prediction intervals trustworthy? | No, on both criteria. | A nominal 90% band covers 86.6% of outcomes, and its width does not adapt to the state. The unconditional-coverage test rejects all four specifications; the independence test rejects none — so the failure is one of level, not clustering. |
| **RQ4** — Does it transfer? | To a second index of the same session, yes. To a second national market, no. | Russell 2000 conditional *p* = 0.0087; FTSE 100 *p* = 0.254 — and the FTSE 100 is the cleaner test. |

**The finding that carries** is narrower than the framing anticipated and more usable for being so: on this target, the value of deep learning lies less in lowering average error than in identifying, before a forecast is made, which model to trust that day.

---

## What is in this repository

```
Code/            the full implementation, and the reproduction guide → Code/README.md
  src/           library code: data, models, evaluation, experiment runners
  configs/       one YAML per phase; each carries the reasoning for its settings
  tests/         400+ tests
  results/       prediction parquets and every table quoted in the dissertation
  rerun_all.ps1  regenerates every artefact in dependency order
ROADMAP.md       the plan of record: every methodological decision, its reasoning,
                 and a dated changelog
LICENSE          MIT
```

Start with **[`Code/README.md`](Code/README.md)** — it is the reproduction guide: what the study needs, how to run it, what it writes, and how to verify that what it wrote is what the dissertation reports.

---

## Reproducing the results

```powershell
cd Code
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev,deep]"
python -m src.data.ingest --all          # needs network
python -m src.data.ingest --oxfordman
.\rerun_all.ps1
```

Roughly 45–75 minutes on CPU. **No GPU is required.** Exact versions behind the committed results are pinned in `Code/requirements-lock.txt`.

## The reproducibility gate

The pipeline ends with a check that is unusual enough to be worth naming. `src.experiments.verify_claims` recomputes every published point metric, interval metric, per-regime loss and sample size **directly from the prediction parquets**, using code that imports nothing from `src.evaluation`. The duplication is deliberate: calling the project's own functions would only confirm that a CSV matches what those functions return today, and would agree enthusiastically with a regression inside them.

It re-derives **941 of 943 published values with no disagreement**. The remaining two are documented derivations rather than persisted artefacts. Exit code 0 means every value checks out; the detail lands in `results/tables/m09_verification.csv`.

---

## Data

The method is specified against a **data interface** rather than a source, so any series meeting the requirements in `Code/README.md` §2 drives the pipeline unchanged. This study used the Oxford-Man Institute Realized Library (`rv5`, five-minute realised variance) for `.SPX`, `.RUT` and `.FTSE` from 2000-01-03 to 2022-02-25, with daily OHLC and the CBOE VIX from Yahoo Finance.

Raw data is not redistributed here. `python -m src.data.ingest` fetches it, and every pull writes a sibling `*.meta.json` recording the snapshot date.

## Citation

```bibtex
@mastersthesis{alshamali2026regime,
  author  = {Al Shamali, Ubay},
  title   = {Regime-Aware Deep Learning for Financial Volatility Forecasting
             with Uncertainty Quantification},
  school  = {WMG, University of Warwick},
  year    = {2026},
  type    = {{MSc} dissertation},
  address = {Coventry, United Kingdom}
}
```

## Author and supervision

**Ubay Al Shamali** — MSc Applied Artificial Intelligence, WMG, University of Warwick.
Supervised by Leonardo Alves Dias and Washington Mbonu.

## Licence

MIT — see [LICENSE](LICENSE). The dissertation text is the author's own work; the code is free to reuse under the licence terms.
