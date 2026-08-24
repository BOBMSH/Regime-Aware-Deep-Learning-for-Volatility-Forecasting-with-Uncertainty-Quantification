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
    import platform
    import sys

    stamp = {
        "python": sys.version.split()[0],
        "python_build": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
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
