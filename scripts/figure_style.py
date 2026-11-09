"""Shared figure style for the agentic repositories.

Copied verbatim into each repository as scripts/figure_style.py so every repo
draws the same arm types in the same colours. Colours come from the Okabe-Ito
colour-blind-safe palette and are hardcoded. Every bar carries a dark edge and
an axis or direct label, so identity never rests on colour alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Arm types. The same type has the same colour in every repository.
BASELINE = "#999999"  # non-LLM baseline: grey
SINGLE = "#0072B2"  # single-call LLM: Okabe-Ito blue
AGENT = "#E69F00"  # iterative agent: Okabe-Ito orange
ABLATION = "#F7D49B"  # agent ablation: lighter orange
EDGE = "#333333"
TEXT = "#222222"
MUTED = "#666666"
GRID = "#E6E6E6"
SECOND_MODEL_HATCH = "///"  # second model: same arm colour, hatched

# Episode outcomes for heatmaps and stacked bars.
OUTCOME = {
    "correct": "#009E73",  # Okabe-Ito bluish green
    "wrong": "#D55E00",  # Okabe-Ito vermillion
    "abstained": "#56B4E9",  # Okabe-Ito sky blue
    "failed": "#555555",  # dark grey
    "not run": "#FFFFFF",
}

WIDTH = 9.0  # inches; every figure is 8 to 10 inches wide
DPI = 200
MIN_FONT = 10


def apply() -> None:
    """Set the shared matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "hatch.color": EDGE,
            "hatch.linewidth": 0.8,
            "svg.hashsalt": "figures",
        }
    )


def arm_colour(arm_type: str) -> str:
    """Colour for an arm type: baseline, single, agent or ablation."""
    return {"baseline": BASELINE, "single": SINGLE, "agent": AGENT, "ablation": ABLATION}[arm_type]


def bar_kwargs(arm_type: str, second_model: bool = False) -> dict[str, Any]:
    """Keyword arguments for ax.bar: arm colour, dark edge, hatch for the second model."""
    return {
        "color": arm_colour(arm_type),
        "edgecolor": EDGE,
        "linewidth": 0.8,
        "hatch": SECOND_MODEL_HATCH if second_model else None,
        "zorder": 2,
    }


def points(ax: Any, x: float, values: list[float], spread: float = 0.12) -> None:
    """Overlay individual runs or tasks as small dark points, evenly offset (no jitter)."""
    n = len(values)
    if n == 0:
        return
    offsets = [0.0] if n == 1 else [-spread + 2 * spread * i / (n - 1) for i in range(n)]
    ax.scatter(
        [x + o for o in offsets],
        values,
        s=22,
        color="white",
        edgecolor=EDGE,
        linewidth=1.0,
        zorder=3,
    )


def save(fig: Any, path: Path) -> Path:
    """Save as an opaque 200 dpi PNG without volatile metadata, then close."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    return path
