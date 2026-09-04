"""OmegaConf-backed config loading and snapshotting (roadmap §9)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../Code
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name: str, *, overrides: list[str] | None = None) -> DictConfig:
    """Load ``configs/<name>.yaml`` with optional dotlist overrides.

    Example
    -------
    >>> cfg = load_config("data", overrides=["window.start=2010-01-01"])
    """
    path = (CONFIG_DIR / name).with_suffix(".yaml")
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    assert isinstance(cfg, DictConfig)
    return cfg


#: Libraries whose version can move a fitted weight. Recorded in every run
#: snapshot; absent packages are simply omitted rather than raising.
_STAMPED_PACKAGES = (
    "torch", "numpy", "pandas", "scipy", "sklearn", "arch", "hmmlearn",
    "statsmodels",
)


def environment_stamp() -> dict:
    """Interpreter and library versions for the run snapshot (roadmap §9).

    Why this exists -- audit (vii), 2026-08-23
    ------------------------------------------
    §9 asks for config snapshots and pinned dependencies, and until now the
    snapshot recorded the resolved YAML and nothing else. That left one class of
    defect invisible: the Phase-5 predictions were fitted under CPython 3.10 and
    the Phase-7 combined model under 3.12, so the combined model's "training
    inherited unchanged" claim was false on the artefacts -- the dropout-off
    mixture missed the Phase-5 forecast by up to 18% on 2020-03-03, against the
    ~1e-6 a same-environment refit reproduces to. Nothing on disk said which
    interpreter had produced which parquet; the split had to be inferred from
    ``__pycache__`` mtimes.

    A version stamp in every ``config.yaml`` makes that visible in a diff. Note
    that ``requirements-lock.txt`` does not pin ``torch`` at all, which is
    precisely the library whose build most affects a fitted weight -- so the
    per-run stamp is the only record of it.
    """
    import importlib
    import os
    import platform
    import sys

    stamp = {
        "python": sys.version.split()[0],
        "python_build": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        # Timing claims are meaningless without the processor behind them, and
        # every run here is CPU-only, so this is the hardware that matters.
        "processor": platform.processor() or "unknown",
        "cpu_count_logical": os.cpu_count() or 0,
    }
    for name in _STAMPED_PACKAGES:
        try:
            mod = importlib.import_module(name)
        except Exception:  # pragma: no cover - package simply not installed
            continue
        version = getattr(mod, "__version__", None)
        if version is not None:
            stamp[name] = str(version)
    return stamp


def snapshot_config(cfg: DictConfig, run_dir: Path | str) -> Path:
    """Dump the resolved config, a UTC timestamp and the environment stamp."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "config.yaml"
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    resolved["_snapshot_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    resolved["_environment"] = environment_stamp()
    OmegaConf.save(OmegaConf.create(resolved), out)
    return out


def repo_path(*parts: str) -> Path:
    """Resolve a path relative to the Code/ repo root."""
    return REPO_ROOT.joinpath(*parts)


def milestone_path(cfg, default: str, profile_name: str, *, n_profiles: int = 1,
                   key: str = "milestone_file") -> Path:
    """Resolve where one profile's milestone note is written.

    Two things this settles that were previously left to luck.

    **Variant isolation.** ``paths.milestone_file`` overrides the phase default,
    so a robustness variant (the Baum-Welch-conditioned run, a Phase-8 asset)
    cannot overwrite the headline note. Three runners already had this idiom and
    three did not; all six now go through here.

    **Multi-profile safety.** Every runner writes its milestone *inside* the
    profile loop, so a config carrying two profiles silently wrote one note and
    kept only the last. That never bit because every config to date had exactly
    one profile -- and Phase 8 is the first with two, which is exactly when a
    latent trap of this shape fires. A name containing ``{profile}`` is formatted
    per profile; a name without one is **rejected** when there is more than one
    profile, rather than quietly losing a note.

    Parameters
    ----------
    cfg : the config node carrying ``paths`` (the whole config, not ``cfg.paths``).
    default : the phase's default filename, used when ``paths.milestone_file``
        is absent.
    profile_name : substituted into a ``{profile}`` placeholder.
    n_profiles : how many profiles this run will write. Only used to decide
        whether a placeholder is mandatory.
    key : which ``paths`` field names the file. Defaults to ``milestone_file``,
        which belongs to *the runner the config was written for*. A runner that
        reads **another phase's** config must pass its own key, or it inherits a
        name meant for something else: ``run_rq4`` reads
        ``configs/regime_lstm_rq4.yaml`` to learn the robustness profiles, and on
        2026-09-02 it wrote the RQ4 note to ``m08_regime_dl_rq4.md`` -- the
        Phase-5 naming pattern -- because it picked up that file's
        ``milestone_file``. It now passes ``key="rq4_milestone_file"``, which no
        config sets, so it falls back to its own default and a reader can still
        override it deliberately.
    """
    paths = cfg.paths
    name = str(paths.get(key) or default)
    if "{profile}" in name:
        name = name.format(profile=profile_name)
    elif int(n_profiles) > 1:
        raise ValueError(
            f"config has {n_profiles} profiles but paths.milestone_file is "
            f"{name!r}, which has no '{{profile}}' placeholder -- every profile "
            "would write to the same file and only the last would survive. "
            "Use e.g. 'm08_regimes_{profile}.md'."
        )
    return repo_path(paths.milestones, name)
