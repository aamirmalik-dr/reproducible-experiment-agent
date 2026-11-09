"""Shared publication figure style for the agentic repositories.

Copied verbatim into each repository as scripts/figure_style.py, so every
repository draws the same arm type with the same colour, marker and typeface.

Design rules, at journal print size:
- Typeface Liberation Sans (vendored in scripts/fonts/, SIL Open Font License,
  metric-compatible with Arial), falling back to Arial. Text 7 pt, panel
  letters 9 pt bold. A missing glyph is an error, so no other font can slip in.
- Full width 183 mm, single column 89 mm. Axes are placed in millimetres.
- Left and bottom spines only, 0.6 pt, ticks outward 3 pt, no titles.
- Okabe-Ito colours by arm type, with a marker shape per arm type so nothing
  depends on colour alone. A second model uses the same hue with open markers
  and dashed lines.
- Every figure is saved as SVG (text as paths, white background) and as a
  300 dpi PNG on white. Both files are byte-identical across runs.

Statistics used by every repository:
- wilson(): Wilson score 95% interval for a binomial rate.
- paired_bootstrap(): percentile bootstrap 95% interval of the mean paired
  difference, resampling tasks (not episodes), 10,000 resamples, fixed seed.
- bootstrap_median_band(): percentile bootstrap 95% band of a median curve,
  resampling runs, same resample count and seed.
"""

from __future__ import annotations

import io
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

MM = 1 / 25.4
FULL_WIDTH = 183.0  # mm
SINGLE_WIDTH = 89.0  # mm
DPI = 300
SVG_LIMIT = 600_000  # bytes
PNG_LIMIT = 500_000  # bytes

N_BOOT = 10_000
SEED = 20261109
Z95 = 1.959963984540054

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FILES = ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf")

# Arm types. The same type has the same colour and marker in every repository.
BASELINE = "#7F7F7F"  # non-LLM baseline: grey
SINGLE = "#0072B2"  # single-call LLM: Okabe-Ito blue
AGENT = "#D55E00"  # iterative agent: Okabe-Ito vermillion
ABLATION = "#F4A582"  # agent ablation: light vermillion
ARM_COLOUR = {"baseline": BASELINE, "single": SINGLE, "agent": AGENT, "ablation": ABLATION}
ARM_EDGE = {"baseline": BASELINE, "single": SINGLE, "agent": AGENT, "ablation": AGENT}
ARM_MARKER = {"baseline": "s", "single": "o", "agent": "D", "ablation": "^"}
ARM_MARKER_SIZE = {"baseline": 3.6, "single": 4.0, "agent": 3.6, "ablation": 4.2}

INK = "#1A1A1A"
MUTED = "#5E5E5E"
CONTEXT = "#C8C8C8"  # other arms drawn in the background of small multiples
REFERENCE = "#E3E3E3"  # very light reference lines

# Episode outcomes for tile matrices and stacked bars (Okabe-Ito, no arm hues).
OUTCOME = {
    "correct": "#009E73",  # bluish green
    "partial": "#A3D9C6",  # light bluish green: right answer with a defect
    "abstained": "#F0E442",  # yellow
    "wrong": "#CC79A7",  # reddish purple
    "failed": "#3D3D3D",  # near black: budget or cap stop
    "pending": "#FFFFFF",  # not run
}
OUTCOME_TEXT = {
    "correct": "white",
    "partial": INK,
    "abstained": INK,
    "wrong": INK,
    "failed": "white",
    "pending": MUTED,
}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "okabe_green", ["#F2F2F2", "#A3D9C6", "#009E73"]
)  # completion share: light grey (0) to bluish green (1)

_FAMILY: str | None = None


def _register_fonts() -> str:
    """Register the vendored Liberation Sans files, or fall back to Arial."""
    found = [FONT_DIR / f for f in FONT_FILES if (FONT_DIR / f).exists()]
    for path in found:
        font_manager.fontManager.addfont(str(path))
    for family in ("Liberation Sans", "Arial"):
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=family), fallback_to_default=False
            )
            return family
        except ValueError:
            continue
    raise RuntimeError(
        "Neither the vendored Liberation Sans (scripts/fonts/) nor Arial is available."
    )


def apply() -> str:
    """Set the shared matplotlib defaults and return the typeface in use."""
    global _FAMILY
    _FAMILY = _register_fonts()
    warnings.filterwarnings("error", message=r"Glyph .* missing from")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [_FAMILY],
            "mathtext.fontset": "custom",
            "mathtext.rm": _FAMILY,
            "mathtext.it": f"{_FAMILY}:italic",
            "mathtext.bf": f"{_FAMILY}:bold",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.unicode_minus": True,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.labelpad": 2.5,
            "axes.titlepad": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "patch.linewidth": 0.5,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.4,
            "legend.borderaxespad": 0.2,
            "legend.labelspacing": 0.3,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "savefig.dpi": DPI,
            "svg.fonttype": "path",
            "svg.hashsalt": "agentic-figures",
            "path.simplify": True,
            "image.interpolation": "nearest",
        }
    )
    return _FAMILY


# Layout ---------------------------------------------------------------------


def figure(width_mm: float, height_mm: float) -> plt.Figure:
    """Create a white figure of the given print size in millimetres."""
    return plt.figure(figsize=(width_mm * MM, height_mm * MM))


def axes_mm(fig: plt.Figure, left: float, top: float, width: float, height: float) -> plt.Axes:
    """Add axes placed in millimetres, measured from the figure's top-left corner."""
    fw, fh = fig.get_size_inches() / MM
    return fig.add_axes((left / fw, 1 - (top + height) / fh, width / fw, height / fh))


def panel_label(fig: plt.Figure, letter: str, left: float, top: float) -> None:
    """Write a 9 pt bold lowercase panel letter at a position in millimetres."""
    fw, fh = fig.get_size_inches() / MM
    fig.text(left / fw, 1 - top / fh, letter, fontsize=9, fontweight="bold", va="top", ha="left")


def bare(ax: plt.Axes) -> None:
    """Remove every spine and tick, for tile matrices and keys."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def ref_line(ax: plt.Axes, value: float, axis: str = "x", **kw: Any) -> None:
    """Draw a thin reference line (zero line, cap or y = x) behind the data."""
    style = {"color": MUTED, "linewidth": 0.6, "zorder": 0.5}
    style.update(kw)
    (ax.axvline if axis == "x" else ax.axhline)(value, **style)


def _tick(value: float, _pos: int | None = None) -> str:
    value = 0.0 if abs(value) < 1e-12 else value
    return f"{value:g}".replace("-", chr(0x2212))


def tidy_ticks(ax: plt.Axes, axis: str = "both") -> None:
    """One tick-label convention everywhere: no trailing zeros, true minus sign."""
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(FuncFormatter(_tick))
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(FuncFormatter(_tick))


def light_hlines(ax: plt.Axes, values: Iterable[float]) -> None:
    """Very light horizontal reference lines, used only where values must be read."""
    for v in values:
        ax.axhline(v, color=REFERENCE, linewidth=0.5, zorder=0)


# Markers --------------------------------------------------------------------


def marker_kw(arm_type: str, second: bool = False, size: float | None = None) -> dict[str, Any]:
    """Keyword arguments for a point of one arm type; open markers for a second model."""
    colour = ARM_COLOUR[arm_type]
    edge = ARM_EDGE[arm_type]
    return {
        "marker": ARM_MARKER[arm_type],
        "markersize": size or ARM_MARKER_SIZE[arm_type],
        "markerfacecolor": "white" if second else colour,
        "markeredgecolor": edge,
        "markeredgewidth": 0.9 if second else 0.6,
        "linestyle": "none",
        "zorder": 3,
    }


def line_kw(arm_type: str, second: bool = False, width: float = 1.0) -> dict[str, Any]:
    """Keyword arguments for an interval or curve of one arm type."""
    return {
        "color": ARM_COLOUR[arm_type],
        "linewidth": width,
        "linestyle": (0, (2.2, 1.2)) if second else "-",
        "solid_capstyle": "butt",
        "dash_capstyle": "butt",
        "zorder": 2,
    }


def dot_interval(
    ax: plt.Axes,
    y: float,
    rate: float,
    lo: float,
    hi: float,
    arm_type: str,
    second: bool = False,
    horizontal: bool = True,
) -> None:
    """One Cleveland-style point with its interval line."""
    if horizontal:
        ax.plot([lo, hi], [y, y], **line_kw(arm_type, second))
        ax.plot([rate], [y], **marker_kw(arm_type, second))
    else:
        ax.plot([y, y], [lo, hi], **line_kw(arm_type, second))
        ax.plot([y], [rate], **marker_kw(arm_type, second))


def arm_handles(items: Sequence[tuple[str, str, bool]], line: bool = False) -> list[Line2D]:
    """Legend handles from (label, arm_type, second_model) triples."""
    out = []
    for label, arm_type, second in items:
        kw = marker_kw(arm_type, second)
        if line:
            kw.update({k: v for k, v in line_kw(arm_type, second).items() if k != "zorder"})
        out.append(Line2D([], [], label=label, **kw))
    return out


def outcome_handles(keys: Sequence[tuple[str, str]]) -> list[Patch]:
    """Legend patches from (outcome key, label) pairs."""
    return [
        Patch(
            facecolor=OUTCOME[k],
            edgecolor=MUTED if k == "pending" else OUTCOME[k],
            linewidth=0.5,
            label=label,
        )
        for k, label in keys
    ]


# Statistics -----------------------------------------------------------------


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """Rate and Wilson score 95% interval for k successes out of n."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    z2 = Z95**2
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z95 * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_boot: int = N_BOOT, seed: int = SEED
) -> tuple[float, float, float]:
    """Mean of a - b over paired tasks with a percentile bootstrap 95% interval.

    a and b hold one value per task (already averaged over that task's runs),
    in the same task order. Tasks are resampled with replacement.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def bootstrap_median_band(
    curves: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise median of curves (runs x points) with a bootstrap 95% band over runs."""
    curves = np.asarray(curves, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, curves.shape[0], size=(n_boot, curves.shape[0]))
    meds = np.median(curves[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5], axis=0)
    return np.median(curves, axis=0), lo, hi


def ecdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Step coordinates of an empirical CDF, starting at zero height."""
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, x.size + 1) / x.size
    return np.concatenate([[x[0]], x]), np.concatenate([[0.0], y])


def kde(values: Sequence[float], grid: np.ndarray) -> np.ndarray:
    """Gaussian kernel density with Scott's bandwidth, evaluated on grid."""
    v = np.asarray(values, dtype=float)
    sd = v.std(ddof=1) if v.size > 1 else 0.0
    bw = max(sd * v.size ** (-1 / 5), 1e-9) if sd > 0 else 1e-3
    z = (grid[:, None] - v[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(axis=1) / (v.size * bw * np.sqrt(2 * np.pi))


def raincloud(
    ax: plt.Axes,
    y: float,
    values: Sequence[float],
    arm_type: str,
    second: bool = False,
    height: float = 0.28,
    jitter: float = 0.32,
    seed: int = SEED,
    marker: str | None = None,
    shade: str | None = None,
) -> None:
    """Horizontal raincloud: half violin above, box at y, jittered points below.

    Designed for rows spaced 1.3 data units apart. The half violin is a Gaussian
    kernel density (Scott's bandwidth) drawn only between the observed minimum
    and maximum, with a clear gap above the box and no outline on its base. The
    box spans the quartiles with the median as a black tick; whiskers run
    through its centre to the minimum and maximum. A second model is drawn with
    dashed lines throughout. "Above" is on screen, so set the y limits
    (inverted or not) before calling. marker and shade override the arm type's
    marker and colour, for two baselines of the same type in one panel.
    """
    up = -1.0 if ax.yaxis_inverted() else 1.0
    v = np.asarray(values, dtype=float)
    colour = shade or ARM_COLOUR[arm_type]
    edge = shade or ARM_EDGE[arm_type]
    dash = (0, (2.2, 1.2)) if second else "-"
    half, base = 0.15, 0.35
    if v.size > 1 and np.ptp(v) > 0:
        grid = np.linspace(v.min(), v.max(), 200)
        dens = kde(v, grid)
        dens = dens / dens.max() * height
        ax.fill_between(
            grid,
            y + up * base,
            y + up * (base + dens),
            facecolor=colour,
            linewidth=0,
            alpha=0.12 if second else 0.35,
            zorder=1,
        )
        ax.plot(grid, y + up * (base + dens), color=edge, linewidth=0.6, linestyle=dash, zorder=1)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([v.min(), q1], [y, y], color=edge, linewidth=0.6, linestyle=dash, zorder=2)
    ax.plot([q3, v.max()], [y, y], color=edge, linewidth=0.6, linestyle=dash, zorder=2)
    ax.add_patch(
        plt.Rectangle(
            (q1, y - half),
            q3 - q1,
            2 * half,
            facecolor="white",
            edgecolor=edge,
            linewidth=0.7,
            linestyle=dash,
            zorder=2,
        )
    )
    ax.plot([med, med], [y - half, y + half], color=INK, linewidth=1.0, zorder=3)
    rng = np.random.default_rng(seed)
    jy = y - up * (half + 0.06 + rng.uniform(0, jitter, size=v.size))
    scale = {"baseline": 0.62, "single": 0.6, "agent": 0.52, "ablation": 0.74}[arm_type]
    kw = marker_kw(arm_type, second, size=ARM_MARKER_SIZE[arm_type] * scale)
    kw["markeredgewidth"] = 0.5
    if marker:
        kw["marker"] = marker
    if shade:
        kw.update(markeredgecolor=shade, markerfacecolor="white" if second else shade)
    ax.plot(v, jy, **kw)


# Export ---------------------------------------------------------------------


def save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    """Save stem.svg (text as paths) and stem.png (300 dpi), both on white."""
    out_dir.mkdir(parents=True, exist_ok=True)
    svg, png = out_dir / f"{stem}.svg", out_dir / f"{stem}.png"
    buf = io.StringIO()
    fig.savefig(buf, format="svg", facecolor="white", metadata={"Date": None})
    svg.write_bytes(buf.getvalue().replace("\r\n", "\n").encode("utf-8"))
    fig.savefig(png, format="png", dpi=DPI, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    for path, limit in ((svg, SVG_LIMIT), (png, PNG_LIMIT)):
        size = path.stat().st_size
        if size > limit:
            raise RuntimeError(f"{path.name} is {size} bytes, over the {limit} byte limit")
    return [svg, png]
