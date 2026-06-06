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


def snapshot_config(cfg: DictConfig, run_dir: Path | str) -> Path:
    """Dump the resolved config plus a UTC timestamp into the run directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "config.yaml"
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    resolved["_snapshot_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OmegaConf.save(OmegaConf.create(resolved), out)
    return out


def repo_path(*parts: str) -> Path:
    """Resolve a path relative to the Code/ repo root."""
    return REPO_ROOT.joinpath(*parts)
