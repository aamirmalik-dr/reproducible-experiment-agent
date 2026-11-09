"""Draw the README figures from saved run outputs only.

Reads reports/live (v1, gpt-4.1-mini), reports/live_v2_gpt-4.1-mini (v2 run A)
and reports/live_v2_claude-sonnet-4.6 (v2 run B): task_results.jsonl,
trial_results.jsonl and the agent traces. Writes each figure to docs/figures/
as SVG and as a 300 dpi PNG, plus figure_stats.json with every plotted value
and interval. No model is called, nothing is refitted and nothing is
estimated. Cap-stopped episodes are pending and are never drawn as zero.

Each PNG is re-encoded after saving with a 256-colour palette built from its
own most frequent colours, to keep the repository small: hardcoded colours are
kept exactly and only rare anti-aliasing shades move to the nearest kept
colour. Running the script twice gives byte-identical files.

Usage: python scripts/make_figures.py [--out docs/figures]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style as fs  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, MultipleLocator  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# (output dir, task family, label, second model)
SETS = [
    ("live", "heldout", "v1", False),
    ("live_v2_gpt-4.1-mini", "heldout_v2", "v2", False),
    ("live_v2_claude-sonnet-4.6", "heldout_v2", "v2, Sonnet", True),
]
ARMS = ["random_search", "single_call", "agent"]
ARM_TYPE = {"fixed": "baseline", "random_search": "baseline", "single_call": "single",
            "agent": "agent"}  # fmt: skip
ARM_NAME = {
    "fixed": "Fixed pipeline",
    "random_search": "Random search",
    "single_call": "Single call",
    "agent": "Agent",
}
STEMS = ["fig1_main", "figS1_paired_differences"]
STATS: dict[str, dict] = {}
FIG_H = 150.0


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def task_rows(name: str) -> list[dict]:
    return jsonl(REPORTS / name / "task_results.jsonl")


def trial_rows(name: str) -> list[dict]:
    return jsonl(REPORTS / name / "trial_results.jsonl")


def measured(rows: list[dict], arm: str) -> list[dict]:
    """Episodes of an arm that ran; cap-stopped episodes are pending, not zero."""
    return [r for r in rows if r["arm"] == arm and r["stop_reason"] != "run_budget_exhausted"]


def arms_of(second: bool) -> list[str]:
    """The second-model run repeats no baseline (they equal v2 run A, run 0)."""
    return ["single_call", "agent"] if second else ARMS


def per_task_gain(rows: list[dict], arm: str) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in measured(rows, arm):
        acc[r["task_id"]].append(r["gain_vs_fixed"])
    return {t: float(np.mean(v)) for t, v in acc.items()}


def paired(rows_a: list[dict], arm_a: str, rows_b: list[dict], arm_b: str, tag: str) -> tuple:
    if arm_b == "fixed":
        a = per_task_gain(rows_a, arm_a)
        b = {t: 0.0 for t in a}
    else:
        a, b = per_task_gain(rows_a, arm_a), per_task_gain(rows_b, arm_b)
    tasks = sorted(set(a) & set(b))
    m, lo, hi = fs.paired_bootstrap([a[t] for t in tasks], [b[t] for t in tasks])
    STATS[tag] = {"mean_difference": m, "bootstrap95": [lo, hi], "n_tasks": len(tasks)}
    return m, lo, hi, len(tasks)


# Figure 1 -------------------------------------------------------------------


SONNET = "claude-sonnet-4.6"


def minus(text: str) -> str:
    """True minus sign in annotation text."""
    return text.replace("-", "−")


def bounded_minor(ax, lo: float, hi: float, step: float, axis: str = "x") -> None:
    ticks = np.arange(lo, hi + 1e-9, step)
    (ax.xaxis if axis == "x" else ax.yaxis).set_minor_locator(FixedLocator(ticks))


def open_kw(arm_type: str, size: float) -> dict:
    """Unfilled open marker drawn on top, so points underneath stay visible."""
    kw = fs.marker_kw(arm_type, True, size=size)
    kw.update(markerfacecolor="none", zorder=5)
    return kw


def panel_tasks(fig, top: float) -> None:
    width, height, gap, left0 = 30.0, 66.0, 16.0, 20.0
    offsets = {
        "random_search": -0.3,
        "single_call": -0.15,
        "agent": 0.0,
        "sonnet_single": 0.15,
        "sonnet_agent": 0.3,
    }
    for j, (name, _, label, _) in enumerate(SETS[:2]):
        rows = task_rows(name)
        sonnet = task_rows(SETS[2][0]) if j == 1 else []
        tasks = sorted({r["task_id"] for r in rows})
        mean = {
            t: np.mean([r["gain_vs_fixed"] for r in rows if r["task_id"] == t and r["arm"] in ARMS])
            for t in tasks
        }
        tasks.sort(key=lambda t: (-mean[t], t))
        ax = fs.axes_mm(fig, left0 + j * (width + gap), top, width, height)
        for i, t in enumerate(tasks):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#F5F5F5", linewidth=0, zorder=0)
            for arm in ARMS:
                vals = [r["gain_vs_fixed"] for r in measured(rows, arm) if r["task_id"] == t]
                y = i + offsets[arm]
                ax.plot(vals, [y] * len(vals), **fs.marker_kw(ARM_TYPE[arm], size=3.0))
                if len(vals) > 1:
                    ax.plot([min(vals), max(vals)], [y, y], **fs.line_kw(ARM_TYPE[arm], width=0.6))
            if sonnet:
                for arm in ("single_call", "agent"):
                    vals = [r["gain_vs_fixed"] for r in measured(sonnet, arm) if r["task_id"] == t]
                    y = i + offsets[f"sonnet_{arm.split('_')[0] if arm == 'agent' else 'single'}"]
                    if vals:
                        ax.plot(vals, [y] * len(vals), **open_kw(ARM_TYPE[arm], 3.0))
                if not any(r["task_id"] == t for r in measured(sonnet, "agent")):
                    ax.text(-0.57, i + 0.22, "pending", ha="left", va="center", color=fs.MUTED)
        fs.ref_line(ax, 0.0, "x")
        ax.set_ylim(len(tasks) - 0.5, -0.5)
        ax.set_yticks(range(len(tasks)), tasks)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(-0.6, 0.55)
        ax.spines["bottom"].set_bounds(-0.5, 0.5)
        ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
        bounded_minor(ax, -0.5, 0.5, 0.05)
        fs.tidy_ticks(ax, "x")
        ax.set_xlabel("Holdout gain over the\nfixed pipeline (fraction)")
        ax.set_title(f"{label}, gpt-4.1-mini" + (f"\nand {SONNET}" if j else ""))
    handles = fs.arm_handles([(ARM_NAME[a], ARM_TYPE[a], False) for a in ARMS])
    for arm in ("single_call", "agent"):
        h = fs.arm_handles([(f"{ARM_NAME[arm]}, {SONNET}", ARM_TYPE[arm], True)])[0]
        h.set_markerfacecolor("none")
        handles.append(h)
    key = fs.axes_mm(fig, 6.0, top + height + 13.0, 2 * width + gap + 20, 8.0)
    key.set_axis_off()
    key.legend(handles=handles, loc="center", ncol=2, columnspacing=1.5)
    fs.panel_label(fig, "a", 1.5, top - 6.0)


def best_so_far(name: str) -> tuple[dict, dict, dict]:
    """Per agent episode: dev gain of each completed trial, the frozen trial, the last trial."""
    rows = task_rows(name)
    metric = {r["task_id"]: r["metric"] for r in rows}
    trials = trial_rows(name)
    fixed = {
        (t["task_id"], t["run"]): t["dev_metrics"][metric[t["task_id"]]]
        for t in trials
        if t["arm"] == "fixed"
    }
    episodes: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    last: dict[tuple[str, int], int] = {}
    for t in trials:
        if t["arm"] != "agent":
            continue
        key = (t["task_id"], t["run"])
        idx = int(t["run_id"][1:])
        last[key] = max(last.get(key, 0), idx)
        if t["status"] != "ok":
            continue
        f = fixed[key]
        episodes[key][idx] = (f - t["dev_metrics"][metric[t["task_id"]]]) / f
    frozen = {}
    for r in measured(rows, "agent"):
        rid = r.get("frozen_run_id")
        key = (r["task_id"], r["run"])
        if rid and key in episodes:
            frozen[key] = (int(rid[1:]), episodes[key].get(int(rid[1:])))
    return episodes, frozen, last


TRIALS = np.arange(1, 7)


def curve(vals: dict[int, float], last: int) -> np.ndarray:
    """Best dev gain after each trial index; NaN after the episode's last trial."""
    out, best = [], np.nan
    for k in TRIALS:
        if k in vals:
            best = vals[k] if np.isnan(best) else max(best, vals[k])
        out.append(best if k <= last else np.nan)
    return np.array(out)


def panel_trajectories(fig, top: float) -> None:
    left, width, height, gap = 122.0, 36.0, 19.0, 6.0
    for i, (name, _, label, second) in enumerate(SETS):
        episodes, frozen, last = best_so_far(name)
        ax = fs.axes_mm(fig, left, top + i * (height + gap), width, height)
        curves, tasks = [], []
        for key in sorted(episodes):
            c = curve(episodes[key], last[key])
            curves.append(c)
            tasks.append(key[0])
            ok = ~np.isnan(c)
            ax.step(TRIALS[ok], c[ok], where="post", color=fs.AGENT, linewidth=0.4,
                    alpha=0.45, zorder=2)  # fmt: skip
        arr = np.array(curves)
        running = np.array([sum(last[k] >= t for k in sorted(episodes)) for t in TRIALS])
        ids = sorted(set(tasks))
        groups = [np.where(np.asarray(tasks) == t)[0] for t in ids]
        rng = np.random.default_rng(fs.SEED)
        meds = np.empty((fs.N_BOOT, len(TRIALS)))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for b in range(fs.N_BOOT):
                pick = np.concatenate([groups[k] for k in rng.integers(0, len(ids), len(ids))])
                meds[b] = np.nanmedian(arr[pick], axis=0)
            lo, hi = np.nanpercentile(meds, [2.5, 97.5], axis=0)
            med = np.nanmedian(arr, axis=0)
        ax.fill_between(TRIALS, lo, hi, step="post", color=fs.AGENT, alpha=0.18, linewidth=0,
                        zorder=1)  # fmt: skip
        okm = ~np.isnan(med)
        ax.step(TRIALS[okm], med[okm], where="post", **fs.line_kw("agent", second, width=1.6))
        fx = [v[0] for v in frozen.values() if v[1] is not None]
        fy = [v[1] for v in frozen.values() if v[1] is not None]
        ax.plot(fx, fy, linestyle="none", marker="o", markersize=2.6, markerfacecolor="white",
                markeredgecolor=fs.INK, markeredgewidth=0.5, zorder=4)  # fmt: skip
        # the frozen trial counts as the best only if it is the first trial reaching the max
        best_frozen = 0
        for key, (idx, _) in frozen.items():
            vals = episodes[key]
            top_val = max(vals.values())
            best_frozen += idx == min(k for k, v in vals.items() if v == top_val)
        STATS[f"best_so_far/{name}"] = {
            "median": med.tolist(),
            "bootstrap95": [lo.tolist(), hi.tolist()],
            "episodes_still_running": running.tolist(),
            "episodes": len(curves),
            "froze_best_dev_trial": best_frozen,
        }
        for k, nrun in zip(TRIALS, running, strict=True):
            ax.text(k, 1.03, str(nrun), transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", color=fs.MUTED, fontsize=6)  # fmt: skip
        model = SONNET if second else "gpt-4.1-mini"
        ax.text(1.03, 0.5, f"{label.split(',')[0]},\n{model}\nfroze best\ndev trial "
                f"{best_frozen}/{len(curves)}", transform=ax.transAxes, ha="left",
                va="center", color=fs.MUTED)  # fmt: skip
        ax.set_xlim(0.8, 6.2)
        ax.set_xticks(TRIALS)
        ax.set_ylim(-1.2, 0.6)
        ax.spines["left"].set_bounds(-1.2, 0.5)
        ax.set_yticks([-1, -0.5, 0, 0.5])
        bounded_minor(ax, -1.0, 0.5, 0.25, "y")
        fs.tidy_ticks(ax, "y")
        fs.ref_line(ax, 0.0, "y", color=fs.REFERENCE)
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Agent trial index")
        if i == 1:
            ax.set_ylabel("Best dev gain so far (fraction)")
    fs.panel_label(fig, "b", 104.0, top - 6.0)


def panel_dev_holdout(fig, top: float) -> None:
    size, gap, left0 = 30.0, 9.0, 20.0
    for j, (name, _, label, _) in enumerate(SETS[:2]):
        ax = fs.axes_mm(fig, left0 + j * (size + gap), top, size, size)
        groups = [(name, ARMS, False)] + ([(SETS[2][0], arms_of(True), True)] if j else [])
        pts: list[tuple[str, float]] = []
        below = 0
        for gname, arms, second in groups:
            rows = task_rows(gname)
            trials = trial_rows(gname)
            metric = {r["task_id"]: r["metric"] for r in rows}
            fixed_dev = {
                (t["task_id"], t["run"]): t["dev_metrics"][metric[t["task_id"]]]
                for t in trials
                if t["arm"] == "fixed"
            }
            for arm in arms:
                xs, ys = [], []
                for r in measured(rows, arm):
                    if not r["frozen"]:
                        continue
                    f = fixed_dev[(r["task_id"], r["run"])]
                    xs.append((f - r["dev_primary"]) / f)
                    ys.append(r["gain_vs_fixed"])
                    if not second:
                        pts.append((r["task_id"], ys[-1] - xs[-1]))
                if second:
                    ax.plot(xs, ys, **open_kw(ARM_TYPE[arm], 3.4))
                else:
                    ax.plot(
                        xs,
                        ys,
                        **{
                            **fs.marker_kw(ARM_TYPE[arm], size=3.0),
                            "markeredgecolor": "white",
                            "markeredgewidth": 0.4,
                        },
                    )
                    below += sum(y < x - 0.01 for x, y in zip(xs, ys, strict=True))
        n = len(pts)
        g = np.array([v for _, v in pts])
        ids = sorted({t for t, _ in pts})
        groups_idx = [np.array([k for k, (t, _) in enumerate(pts) if t == i]) for i in ids]
        rng = np.random.default_rng(fs.SEED)
        meds = np.empty(fs.N_BOOT)
        for b in range(fs.N_BOOT):
            pick = np.concatenate([groups_idx[k] for k in rng.integers(0, len(ids), len(ids))])
            meds[b] = np.median(g[pick])
        mlo, mhi = np.percentile(meds, [2.5, 97.5])
        p, wlo, whi = fs.wilson(below, n)
        STATS[f"dev_vs_holdout/{name}"] = {
            "median_gap_holdout_minus_dev": float(np.median(g)),
            "median_gap_bootstrap95": [float(mlo), float(mhi)],
            "below_y_eq_x_by_0.01": below,
            "n": n,
            "below_fraction": p,
            "below_wilson95": [wlo, whi],
        }
        lim = (-0.6, 0.6)
        ax.plot(lim, lim, color=fs.MUTED, linewidth=0.6, zorder=0)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MultipleLocator(0.5))
            axis.set_minor_locator(MultipleLocator(0.1))
        fs.tidy_ticks(ax)
        ax.set_xlabel("Dev gain (fraction)")
        if j == 0:
            ax.set_ylabel("Holdout gain (fraction)")
        ax.set_title(label + (f", gpt-4.1-mini\nand {SONNET}" if j else ", gpt-4.1-mini"))
    fs.panel_label(fig, "c", 1.5, top - 8.0)


def invalid_agent_specs(name: str, family: str) -> int:
    """run_experiment calls the policy rejected as invalid arguments, from agent traces."""
    count = 0
    for path in sorted((REPORTS / name / "traces" / family).glob("*.jsonl")):
        for event in jsonl(path):
            if event.get("event") != "step" or event["action"]["tool"] != "run_experiment":
                continue
            err = (event.get("result") or {}).get("error") or {}
            count += err.get("code") == "invalid_arguments"
    return count


OUTCOMES = [
    ("correct", "Completed"),
    ("failed", "Fit failed or timed out"),
    ("wrong", "Refused by the column policy"),
    ("abstained", "Invalid spec, not run"),
]


def panel_outcomes(fig, top: float) -> None:
    rows_y, y = [], 0.0
    for name, family, label, second in SETS:
        for arm in arms_of(second):
            rows_y.append((y, name, family, label, second, arm))
            y += 1
        y += 0.5
    height = (rows_y[-1][0] + 1) * 3.0
    ax = fs.axes_mm(fig, 134.0, top, 34.0, height)
    for yy, name, family, _label, second, arm in rows_y:
        rows = task_rows(name)
        ts = [t for t in trial_rows(name) if t["arm"] == arm]
        rs = measured(rows, arm)
        invalid = (
            invalid_agent_specs(name, family)
            if arm == "agent"
            else sum(r.get("invalid_trial_requests") or 0 for r in rs)
        )
        counts = [
            sum(t["status"] == "ok" for t in ts),
            sum(t["status"] in ("failed", "timeout") for t in ts),
            sum(r["forbidden_attempts"] for r in rs),
            invalid,
        ]
        total = sum(counts)
        STATS[f"trial_requests/{name}/{arm}"] = dict(
            zip(["completed", "failed", "refused", "invalid"], counts, strict=True)
        ) | {"total": total}
        start = 0.0
        for (key, _), c in zip(OUTCOMES, counts, strict=True):
            if c:
                ax.barh(yy, c / total, left=start, height=0.64, color=fs.OUTCOME[key],
                        edgecolor="white", linewidth=0.3)  # fmt: skip
            start += c / total
        if second:
            ax.add_patch(fs.plt.Rectangle((0, yy - 0.32), 1, 0.64, fill=False,
                                          edgecolor=fs.MUTED, linewidth=0.5,
                                          linestyle=(0, (2.2, 1.2)), zorder=3))  # fmt: skip
        ax.text(1.03, yy, f"{sum(counts[1:])} of {total}", va="center", ha="left",
                transform=ax.get_yaxis_transform(), color=fs.MUTED)  # fmt: skip
    ax.set_ylim(rows_y[-1][0] + 0.6, -2.0)
    labels = []
    for r in rows_y:
        cond = f"v2, {SONNET}" if r[4] else r[3]
        labels.append(f"{ARM_NAME[r[5]]}, {cond}")
    ax.set_yticks([r[0] for r in rows_y], labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    fs.tidy_ticks(ax, "x")
    ax.set_xlabel("Trial requests (fraction)")
    ax.text(1.03, -1.35, "Not\ncompleted", va="center", ha="left",
            transform=ax.get_yaxis_transform(), color=fs.MUTED)  # fmt: skip
    key = fs.axes_mm(fig, 92.0, top + height + 9.5, 90.0, 7.0)
    key.set_axis_off()
    key.legend(handles=fs.outcome_handles(OUTCOMES), loc="center", ncol=2, handlelength=1.0,
               columnspacing=1.0)  # fmt: skip
    fs.panel_label(fig, "d", 92.0, top - 8.0)


def figure1(out: Path) -> list[Path]:
    fig = fs.figure(fs.FULL_WIDTH, FIG_H)
    panel_tasks(fig, top=9.0)
    panel_trajectories(fig, top=9.0)
    panel_dev_holdout(fig, top=106.0)
    panel_outcomes(fig, top=106.0)
    return save(fig, out, "fig1_main")


# Supplementary --------------------------------------------------------------


def figure_s1(out: Path) -> list[Path]:
    rows = []
    for name, _, label, second in SETS:
        tr = task_rows(name)
        specs = [("agent", "single_call"), ("agent", "fixed"), ("single_call", "fixed")]
        if not second:
            specs += [("agent", "random_search"), ("single_call", "random_search")]
        cond = f"v2, {SONNET}" if second else f"{label}, gpt-4.1-mini"
        for a, b in specs:
            m, lo, hi, n = paired(tr, a, tr, b, f"paired/{name}/{a}_minus_{b}")
            rows.append((f"{ARM_NAME[a]} minus {ARM_NAME[b].lower()}", cond, a, second, m, lo,
                         hi, n))  # fmt: skip
    fig = fs.figure(fs.FULL_WIDTH * 0.66, 86)
    ys, heads, y = [], [], 0.0
    for i, r in enumerate(rows):
        if i == 0 or rows[i - 1][1] != r[1]:
            heads.append((y, r[1]))
            y += 0.9
        ys.append(y)
        y += 1
    ax = fs.axes_mm(fig, 48, 6, 56, 64)
    for yy, (_, _, arm, second, m, lo, hi, n) in zip(ys, rows, strict=True):
        fs.dot_interval(ax, yy, m, lo, hi, ARM_TYPE[arm], second)
        for edge in (lo, hi):
            ax.plot([edge, edge], [yy - 0.15, yy + 0.15], color=fs.ARM_COLOUR[ARM_TYPE[arm]],
                    linewidth=0.6, zorder=2)  # fmt: skip
        ax.text(1.02, yy, f"n = {n}", transform=ax.get_yaxis_transform(), va="center",
                ha="left", color=fs.MUTED)  # fmt: skip
    for yh, text in heads:
        ax.text(-0.82, yh, text, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontweight="bold")  # fmt: skip
    fs.ref_line(ax, 0.0, "x")
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(ys[-1] + 0.6, -1.4)
    ax.set_xlim(-0.12, 0.42)
    ax.spines["bottom"].set_bounds(-0.1, 0.4)
    ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])
    bounded_minor(ax, -0.1, 0.4, 0.05)
    fs.tidy_ticks(ax, "x")
    ax.set_xlabel("Paired difference in holdout gain\n(fraction of the fixed pipeline's loss)")
    ax.text(-0.012, -1.1, "second arm better", ha="right", va="center", color=fs.MUTED)
    ax.text(0.012, -1.1, "first arm better", ha="left", va="center", color=fs.MUTED)
    return save(fig, out, "figS1_paired_differences")


def compact(path: Path) -> None:
    """Re-encode a saved PNG with a palette of its 256 most frequent colours."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)
    flat = rgb.reshape(-1, 3)
    colours, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    order = np.lexsort((colours[:, 2], colours[:, 1], colours[:, 0], -counts))
    keep = colours[order[:256]]
    dist = ((colours[:, None, :] - keep[None, :, :]) ** 2).sum(-1)
    index = dist.argmin(1).astype(np.uint8)[inverse.ravel()].reshape(rgb.shape[:2])
    img = Image.fromarray(index, mode="P")
    pal = keep.astype(np.uint8).ravel().tolist()
    img.putpalette(pal + [0] * (768 - len(pal)))
    img.save(path, optimize=True, dpi=(fs.DPI, fs.DPI))


def save(fig, out: Path, stem: str) -> list[Path]:
    paths = fs.save(fig, out, stem)
    compact(out / f"{stem}.png")
    return paths


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args(argv)
    fs.apply()
    paths = figure1(args.out)
    paths += figure_s1(args.out)

    def rnd(v):
        if isinstance(v, float):
            return None if np.isnan(v) else round(v, 4)
        if isinstance(v, list):
            return [rnd(x) for x in v]
        if isinstance(v, dict):
            return {k: rnd(x) for k, x in v.items()}
        return v

    meta = {
        "interval": "percentile bootstrap 95% over tasks for medians and paired differences",
        "bootstrap_resamples": fs.N_BOOT,
        "seed": fs.SEED,
        "resampling_unit": "task",
    }
    stats = args.out / "figure_stats.json"
    text = json.dumps({"method": meta, "values": rnd(dict(sorted(STATS.items())))}, indent=1)
    stats.write_text(text + "\n", encoding="utf-8", newline="\n")
    return paths + [stats]


if __name__ == "__main__":
    for p in main():
        print(f"wrote {p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p}")
