"""Single source of figure styling (roadmap §9: no notebook screenshots in the dissertation)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Crisis markers reused on every time-series figure (see m01_data.md).
CRISIS_PERIODS: dict[str, tuple[str, str]] = {
    "Dot-com bust": ("2000-03-10", "2002-10-09"),
    "2008 GFC": ("2007-10-09", "2009-03-09"),
    "COVID-19": ("2020-02-20", "2020-04-07"),
    "2022 inflation/rates": ("2022-01-03", "2022-10-12"),
}


def set_style() -> None:
    """Apply the dissertation-standard matplotlib style."""
    mpl.rcParams.update(
        {
            "figure.figsize": (9, 4.5),
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )


def shade_crises(ax: plt.Axes, *, alpha: float = 0.12, color: str = "tab:red") -> None:
    """Overlay shaded vertical bands for the canonical crisis windows."""
    for label, (start, end) in CRISIS_PERIODS.items():
        ax.axvspan(start, end, color=color, alpha=alpha, lw=0)
        ax.text(start, ax.get_ylim()[1], label, fontsize=7, va="top", alpha=0.6, rotation=90)


def save_fig(fig: plt.Figure, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


@contextmanager
def styled():
    """Context manager that applies the style temporarily."""
    with mpl.rc_context():
        set_style()
        yield
