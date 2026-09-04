"""Phase 9 -- the compute this study actually used (Appendix A.4, Chapter 4).

What this closes
----------------
Table A.4 gives the parameter count as a formula, ``4H(F + H) + 9H + 1`` per
network times ``K`` for the mixture, and then defers twice to Chapter 4:
"Chapter 4 reports the instantiated counts" and "Chapter 4 reports the measured
wall-clock, hyperparameter search included". Neither had an artefact behind it.
A formula evaluated by hand in a text editor is a derivation, not a measurement,
and it cannot notice when the code stops matching it; a runtime recalled from
memory is worse. This script produces both, plus the machine that produced the
timings, and writes ``results/tables/m09_compute.csv``.

Where each number comes from
----------------------------
* **Parameters** are counted off instantiated ``torch`` modules, built through
  each forecaster's own :meth:`build_network` -- the same expression the
  training path uses -- so a count printed here cannot drift from the network
  actually fitted. The board itself is assembled by calling each phase runner's
  own factory (:func:`run_lstm.make_forecaster`,
  :func:`run_regime_lstm.build_models`, :func:`run_uq.build_models`,
  :func:`run_combined.build_model`) on the committed configs, with the selected
  hyperparameters read from ``m03_*_sweep.csv``, so the board is not restated
  here either. Nothing is trained: instantiating a handful of small LSTMs takes
  well under a second.

  The closed form is then checked against every instantiated count, which turns
  Table A.4's formula from a claim into a test -- the same reasoning that put an
  executable guard on the CRPS table.

* **Wall-clock** is *harvested*, not re-run. ``results/logs/rerun-*/`` holds the
  end-to-end reproduction that produced the committed artefacts, with an ISO
  timestamp on every line. Re-timing today would report a different machine-hour
  on a different thermal state; it would not be a better measurement of the run
  the dissertation reports. The log directory is named in the ``source`` column
  so the provenance travels with the number.

* **Environment** comes from :func:`~src.utils.config.environment_stamp`, the
  same stamp every run snapshot carries, plus physical cores and memory when
  ``psutil`` is importable (it is optional, and its absence costs two rows).

Rows are tidy -- ``section``, ``item``, ``value``, ``unit``, ``detail``,
``source`` -- rather than three wide tables, because Appendix A.4 is a
resource/requirement list and reads straight off this shape.

Usage
-----
    python -m src.experiments.report_compute

Outputs ``results/tables/m09_compute.csv``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.utils.config import environment_stamp, load_config, repo_path
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

log = get_logger("report_compute")

LOG_GLOB = "rerun-*"
# Every project log line opens with "YYYY-MM-DD HH:MM:SS,mmm [LEVEL] logger: ...".
_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
# rolling.py's per-model line: "  HAR-RV     done: 784 forecasts in 0.1s".
_DONE = re.compile(r"\b(\S+)\s+done: (\d+) forecasts in ([\d.]+)s")
# run_lstm brackets its sweep with these two lines.
_SWEEP_OPEN = re.compile(r"sweep: (\d+) combinations")
_SWEEP_BEST = re.compile(r"sweep best:")


# --------------------------------------------------------------------------- #
# Parameters                                                                   #
# --------------------------------------------------------------------------- #
def closed_form_parameters(hidden: int, inputs: int, outputs: int = 1) -> int:
    """Table A.4's formula, generalised to an ``outputs``-wide head.

    A one-layer ``nn.LSTM`` holds ``4H(F + H)`` weights and ``8H`` biases; a
    linear head of width ``Q`` adds ``Q(H + 1)``. At ``Q = 1`` this is exactly
    the appendix's ``4H(F + H) + 9H + 1``. Kept as an independent expression so
    :func:`parameter_rows` can assert the instantiated modules against it.
    """
    h, f, q = int(hidden), int(inputs), int(outputs)
    return 4 * h * (f + h) + 8 * h + q * (h + 1)


def count_parameters(module) -> tuple[int, int]:
    """``(total, trainable)`` parameter counts of a built module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total), int(trainable)


def selected_hyperparameters(profile: str) -> dict:
    """The hyperparameters the Phase-3 sweep actually chose, from its own table.

    Read rather than assumed: the Phase-3 model is defined by the sweep result,
    and a config default that happens to match today is not the same statement.
    """
    path = repo_path("results", "tables", f"m03_lstm_{profile}_sweep.csv")
    if not path.exists():
        raise SystemExit(
            f"sweep table not found: {path}\nRun Phase 3 first, or pass --profile."
        )
    sweep = pd.read_csv(path)
    best = sweep.sort_values("val_qlike").iloc[0]
    return {"hidden": int(best["hidden_size"]), "lookback": int(best["lookback"]),
            "lr": float(best["lr"]), "val_qlike": float(best["val_qlike"])}


def build_board(profile: str) -> list[tuple[str, object, str]]:
    """``(name, forecaster, phase)`` for every trained network in the study.

    Assembled through the phase runners' own factories so this file states no
    architecture of its own.
    """
    from src.experiments import run_combined, run_lstm, run_regime_lstm, run_uq

    board: list[tuple[str, object, str]] = []

    # --- Phase 3: the baseline LSTM and its two feature variants ---
    lcfg = load_config("lstm_baseline")
    best = selected_hyperparameters(profile)
    hp = dict(hidden=best["hidden"], lookback=best["lookback"], lr=best["lr"],
              seed=int(lcfg.seed))
    board.append(("LSTM", run_lstm.make_forecaster(lcfg.model, lcfg.harness,
                                                   name="LSTM", **hp), "3"))
    rob = lcfg.get("robustness", {}) or {}
    if bool(rob.get("rv_only_lstm", False)):
        feats = tuple(OmegaConf.to_container(rob.rv_only_features, resolve=True))
        board.append(("LSTM-RVonly",
                      run_lstm.make_forecaster(lcfg.model, lcfg.harness,
                                               name="LSTM-RVonly", features=feats, **hp), "3"))
    if bool(rob.get("vix_lstm", False)):
        feats = tuple(OmegaConf.to_container(rob.vix_features, resolve=True))
        board.append(("LSTM-VIX",
                      run_lstm.make_forecaster(lcfg.model, lcfg.harness,
                                               name="LSTM-VIX", features=feats, **hp), "3"))

    # --- Phase 5: regime as feature, and the mixture of experts ---
    rcfg = load_config("regime_lstm")
    for m in run_regime_lstm.build_models(rcfg.model, rcfg.harness, rcfg.approaches,
                                          seed=int(rcfg.seed), max_epochs=None):
        board.append((m.name, m, "5"))

    # --- Phase 6: the two uncertainty layers ---
    ucfg = load_config("uq")
    for m in run_uq.build_models(ucfg, max_epochs=None, mc_samples=None):
        board.append((m.name, m, "6"))

    # --- Phase 7: the combined model ---
    ccfg = load_config("combined")
    m = run_combined.build_model(ccfg, max_epochs=None, mc_samples=None)
    board.append((m.name, m, "7"))
    return board


def parameter_rows(profile: str) -> tuple[pd.DataFrame, dict]:
    """One row per trained network, counted from an instantiated module."""
    rows, counts = [], {}
    for name, fc, phase in build_board(profile):
        f_in = fc.network_inputs()
        net = fc.build_network(f_in)
        total, trainable = count_parameters(net)
        k = fc.n_networks()
        outputs = getattr(fc, "n_quantiles", 1)
        expected = closed_form_parameters(fc.hidden_size, f_in, outputs)
        if total != expected:
            raise AssertionError(
                f"{name}: instantiated {total} parameters, Table A.4's closed form "
                f"gives {expected} at H={fc.hidden_size}, F={f_in}, Q={outputs} — "
                "the appendix formula and the code disagree; fix whichever is wrong"
            )
        model_total = total * k
        counts[name] = model_total
        detail = (f"{k} x {total:,} (H={fc.hidden_size}, F={f_in}"
                  + (f", Q={outputs}" if outputs != 1 else "") + ")")
        rows.append({"section": "parameters", "item": name, "value": model_total,
                     "unit": "trainable parameters", "detail": detail,
                     "source": f"instantiated (phase {phase})"})
        if trainable != total:  # pragma: no cover - nothing here freezes weights
            log.warning("%s: %d of %d parameters are frozen", name, total - trainable, total)

    # LSTM-Gaussian is a *reading* of the MC-Dropout network, not a fourth
    # architecture (run_uq's module docstring says so); stating that here stops a
    # reader inferring a model that was never trained.
    if "MC-Dropout-LSTM" in counts:
        rows.append({"section": "parameters", "item": "LSTM-Gaussian",
                     "value": counts["MC-Dropout-LSTM"], "unit": "trainable parameters",
                     "detail": "the MC-Dropout network read deterministically",
                     "source": "same network as MC-Dropout-LSTM"})
    return pd.DataFrame(rows), counts


# --------------------------------------------------------------------------- #
# Wall-clock, harvested from the reproduction run                              #
# --------------------------------------------------------------------------- #
def _read_log(path: Path) -> str:
    """Logs are written by a PowerShell redirect, hence UTF-16 with a BOM."""
    raw = path.read_bytes()
    for enc in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _stamps(text: str) -> list[dt.datetime]:
    out = []
    for line in text.splitlines():
        m = _TS.match(line)
        if m:
            out.append(dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f"))
    return out


def latest_log_dir() -> Path | None:
    dirs = sorted(repo_path("results", "logs").glob(LOG_GLOB))
    return dirs[-1] if dirs else None


def runtime_rows(log_dir: Path) -> pd.DataFrame:
    """Per-phase spans, per-model walk-forward times, and the end-to-end total.

    A phase's span is its first-to-last log timestamp, so it excludes the
    interpreter start-up between phases -- which is why the per-phase spans sum
    to slightly less than the end-to-end figure, and why both are reported.
    """
    rows: list[dict] = []
    src = log_dir.name
    first = last = None
    for path in sorted(log_dir.glob("*.log")):
        text = _read_log(path)
        stamps = _stamps(text)
        if not stamps:
            continue
        a, b = min(stamps), max(stamps)
        first = a if first is None else min(first, a)
        last = b if last is None else max(last, b)
        rows.append({"section": "runtime", "item": path.stem,
                     "value": round((b - a).total_seconds(), 1), "unit": "seconds",
                     "detail": f"{a.time()} -> {b.time()}", "source": src})
        # The Phase-3 sweep is the single largest cost in the study; separating
        # it from the fit it selects is the point of reporting runtime at all.
        opens = [m for m in _SWEEP_OPEN.finditer(text)]
        if opens:
            lines = text.splitlines()
            t_open = t_best = None
            for line in lines:
                ts = _TS.match(line)
                if not ts:
                    continue
                when = dt.datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S,%f")
                if t_open is None and _SWEEP_OPEN.search(line):
                    t_open = when
                if _SWEEP_BEST.search(line):
                    t_best = when
            if t_open and t_best:
                rows.append({"section": "runtime",
                             "item": f"{path.stem} (hyperparameter sweep)",
                             "value": round((t_best - t_open).total_seconds(), 1),
                             "unit": "seconds",
                             "detail": f"{opens[0].group(1)} combinations",
                             "source": src})
        # A model can be walk-forwarded more than once: the econometric board
        # runs two windows inside one log, and Phase 5 runs the same three models
        # again under the Baum-Welch comparator profile. So the label carries the
        # run and the sample size, and every row in the table stays distinct --
        # a duplicated item is a trap for whoever reads this into Chapter 4.
        tag = re.sub(r"^\d+-", "", path.stem)
        for m in _DONE.finditer(text):
            rows.append({"section": "runtime",
                         "item": f"walk-forward: {m.group(1)} [{tag}, n={m.group(2)}]",
                         "value": round(float(m.group(3)), 1), "unit": "seconds",
                         "detail": "", "source": src})
    if first and last:
        total = (last - first).total_seconds()
        rows.insert(0, {"section": "runtime", "item": "end to end",
                        "value": round(total, 1), "unit": "seconds",
                        "detail": f"{round(total / 60, 1)} minutes; "
                                  f"{first.date()} {first.time()} -> {last.time()}",
                        "source": src})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Environment                                                                  #
# --------------------------------------------------------------------------- #
def environment_rows() -> pd.DataFrame:
    """The stamp every run snapshot carries, plus cores and memory if available."""
    rows = [{"section": "environment", "item": k, "value": v, "unit": "",
             "detail": "", "source": "environment_stamp()"}
            for k, v in environment_stamp().items()]
    try:
        import psutil

        rows.append({"section": "environment", "item": "cpu_count_physical",
                     "value": psutil.cpu_count(logical=False), "unit": "cores",
                     "detail": "", "source": "psutil"})
        rows.append({"section": "environment", "item": "memory_total",
                     "value": round(psutil.virtual_memory().total / 1024 ** 3, 1),
                     "unit": "GiB", "detail": "", "source": "psutil"})
    except Exception:  # pragma: no cover - psutil is optional
        log.info("psutil not importable; physical cores and memory omitted")
    rows.append({"section": "environment", "item": "accelerator", "value": "none",
                 "unit": "", "detail": "CPU throughout; no CUDA device is requested "
                                       "anywhere in the pipeline",
                 "source": "configs"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.experiments.report_compute")
    ap.add_argument("--profile", default="intraday_2019_2022",
                    help="profile whose Phase-3 sweep selects the hyperparameters")
    ap.add_argument("--log-dir", default=None,
                    help="reproduction log directory (default: the newest results/logs/rerun-*)")
    args = ap.parse_args(argv)

    params, counts = parameter_rows(args.profile)

    log_dir = Path(args.log_dir) if args.log_dir else latest_log_dir()
    if log_dir is not None and not log_dir.is_absolute():
        log_dir = repo_path(str(log_dir))
    if log_dir is None or not log_dir.exists():
        log.warning("no reproduction log directory under results/logs/%s — the "
                    "runtime section will be empty; run scripts/rerun_all first",
                    LOG_GLOB)
        runtime = pd.DataFrame(columns=params.columns)
    else:
        runtime = runtime_rows(log_dir)
        log.info("wall-clock harvested from %s", log_dir.name)

    out = pd.concat([params, runtime, environment_rows()], ignore_index=True)
    out = out[["section", "item", "value", "unit", "detail", "source"]]

    tables = repo_path("results", "tables")
    ensure_dir(tables)
    path = tables / "m09_compute.csv"
    out.to_csv(path, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20,
                  "display.max_colwidth", 60)
    for section in ("parameters", "runtime", "environment"):
        sub = out[out["section"] == section]
        if len(sub):
            print(f"\n=== {section} ===")
            print(sub.drop(columns=["section"]).to_string(index=False))
    print(f"\nwrote {path}")

    if counts:
        big = max(counts, key=counts.get)
        print(f"\nLargest model: {big} at {counts[big]:,} trainable parameters; "
              f"every count matches Table A.4's closed form.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
