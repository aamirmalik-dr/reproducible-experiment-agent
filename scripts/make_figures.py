"""Draw the README figures from saved run outputs only.

Reads, for each live result set (reports/live for v1, reports/live_v2_gpt-4.1-mini
and reports/live_v2_claude-sonnet-4.6 for v2): task_results.jsonl,
trial_results.jsonl, the frozen candidate manifests under frozen/ and the agent
traces, plus reports/per_task_breakdown.csv. No model is called, no trial is
refit and nothing is estimated. Running this script twice gives identical files.

Usage: python scripts/make_figures.py [--out docs/figures]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# (output dir, task family, group label, model, second model?)
SETS = [
    ("live", "heldout", "v1", "gpt-4.1-mini", False),
    ("live_v2_gpt-4.1-mini", "heldout_v2", "v2", "gpt-4.1-mini", False),
    ("live_v2_claude-sonnet-4.6", "heldout_v2", "v2, second model", "claude-sonnet-4.6", True),
]
ARMS = ["random_search", "single_call", "agent"]
ARM_TYPE = {
    "fixed": "baseline",
    "random_search": "baseline",
    "single_call": "single",
    "agent": "agent",
}
ARM_LABEL = {"fixed": "fixed", "random_search": "random", "single_call": "single", "agent": "agent"}
MARKER = {"random_search": "D", "single_call": "o", "agent": "s"}
SAMPLE = (
    "Live, held-out families. v1 and v2: gpt-4.1-mini, 10 tasks x 2 runs.\n"
    "Second model: claude-sonnet-4.6, 10 tasks x 1 run, 4 tasks pending at the run cap."
)
KEY = (
    "fixed: gradient-boosting defaults, no model (the zero line).  "
    "random: 6 random specs, no model.\n"
    "single: one model reply with 6 specs.  agent: iterative agent, up to 6 trials."
)
FIGURES = [
    "gain_by_arm.png",
    "per_task_gain.png",
    "trial_trajectories.png",
    "dev_vs_holdout.png",
    "trial_outcomes.png",
]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def task_rows(name: str) -> list[dict]:
    return jsonl(REPORTS / name / "task_results.jsonl")


def trial_rows(name: str) -> list[dict]:
    return jsonl(REPORTS / name / "trial_results.jsonl")


def measured(rows: list[dict], arm: str) -> list[dict]:
    """Episodes of an arm that ran; cap-stopped episodes are pending, not zero."""
    return [r for r in rows if r["arm"] == arm and r["stop_reason"] != "run_budget_exhausted"]


def pending(rows: list[dict], arm: str) -> int:
    return sum(r["arm"] == arm and r["stop_reason"] == "run_budget_exhausted" for r in rows)


def marker_kwargs(arm: str, second: bool, size: float = 30) -> dict:
    colour = fs.arm_colour(ARM_TYPE[arm])
    return {
        "marker": MARKER[arm],
        "s": size,
        "facecolor": "white" if second else colour,
        "edgecolor": colour if second else fs.EDGE,
        "linewidth": 1.3 if second else 0.6,
        "zorder": 3,
    }


# ---------------------------------------------------------------- figure 1


def gain_by_arm(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 5.0))
    x, ticks, labels, group_mid = 0.0, [], [], []
    for name, _, group, _, second in SETS:
        rows = task_rows(name)
        start = x
        for arm in ARMS:
            if second and arm == "random_search":
                continue
            rs = measured(rows, arm)
            gains = [r["gain_vs_fixed"] for r in rs if r["gain_vs_fixed"] is not None]
            mean = float(np.mean(gains)) if gains else 0.0
            ax.bar(x, mean, width=0.7, **fs.bar_kwargs(ARM_TYPE[arm], second))
            fs.points(ax, x, gains, spread=0.25)
            wins = sum(r["win_vs_fixed"] for r in rs)
            losses = sum(r["loss_vs_fixed"] for r in rs)
            note = f"W/L {wins}/{losses}\nn={len(rs)}"
            if pending(rows, arm):
                note += f"\n{pending(rows, arm)} pending"
            ax.text(x, 0.53, note, ha="center", va="bottom", fontsize=10, color=fs.TEXT)
            ticks.append(x)
            labels.append(ARM_LABEL[arm])
            x += 1.25
        group_mid.append(((start + x - 1.25) / 2, group))
        x += 0.7
    ax.axhline(0, color=fs.EDGE, linewidth=1.0, zorder=1)
    ax.set_xticks(ticks, labels)
    ax.set_xlim(-0.75, x - 0.6)
    ax.set_ylim(-0.25, 0.72)
    ax.set_ylabel(
        "relative gain over the fixed pipeline\n(holdout log loss or MAE, higher is better)"
    )
    for mid, group in group_mid:
        ax.text(
            mid,
            -0.36,
            group,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            transform=ax.transData,
        )
    ax.set_title(
        "Mean gain over the fixed pipeline by arm and round\n"
        + "Bars: mean of measured episodes; points: episodes; W / L: wins and losses beyond 1%",
        fontsize=11,
    )
    fig.text(0.01, 0.01, SAMPLE + "\n" + KEY, fontsize=10, color=fs.MUTED, va="bottom")
    fig.subplots_adjust(left=0.11, right=0.99, top=0.88, bottom=0.32)
    fs.save(fig, out / "gain_by_arm.png")


# ---------------------------------------------------------------- figure 2


def breakdown() -> list[dict]:
    with (REPORTS / "per_task_breakdown.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def per_task_gain(out: Path) -> None:
    rows = breakdown()
    fig, axes = plt.subplots(1, 2, figsize=(fs.WIDTH + 1.0, 6.8), sharex=True)
    panels = [
        ("heldout", ["live"], "v1, held-out v1"),
        ("heldout_v2", ["live_v2_gpt-4.1-mini", "live_v2_claude-sonnet-4.6"], "v2, held-out v2"),
    ]
    offsets = {"random_search": 0.27, "single_call": 0.0, "agent": -0.27}
    for ax, (family, sets, title) in zip(axes, panels, strict=True):
        tasks = sorted({r["task_id"] for r in rows if r["family"] == family})
        for i, task in enumerate(tasks):
            y = len(tasks) - 1 - i
            ax.axhspan(y - 0.45, y + 0.45, color="#F5F5F5" if i % 2 else "white", zorder=0)
            for name in sets:
                second = name.endswith("sonnet-4.6")
                for arm in ARMS:
                    if second and arm == "random_search":
                        continue
                    sel = [
                        r
                        for r in rows
                        if r["result_set"] == name and r["task_id"] == task and r["arm"] == arm
                    ]
                    vals = [float(r["gain_vs_fixed"]) for r in sel if r["status"] == "complete"]
                    shift = -0.09 if second else 0.0
                    ax.scatter(
                        vals,
                        [y + offsets[arm] + shift] * len(vals),
                        **marker_kwargs(arm, second, size=34),
                    )
                    if (
                        second
                        and arm == "agent"
                        and any(r["stop_reason"] == "run_budget_exhausted" for r in sel)
                    ):
                        ax.text(
                            -0.58,
                            y - 0.27,
                            "2nd model: pending",
                            fontsize=10,
                            color=fs.MUTED,
                            ha="left",
                            va="center",
                        )
        ax.axvline(0, color=fs.EDGE, linewidth=1.0, zorder=1)
        ax.set_yticks(range(len(tasks)), list(reversed(tasks)))
        ax.set_ylim(-0.6, len(tasks) - 0.4)
        ax.set_xlim(-0.6, 0.65)
        ax.grid(axis="x", color=fs.GRID)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("relative gain over the fixed pipeline (holdout)")
    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            markersize=7,
            marker=MARKER[a],
            markerfacecolor=fs.arm_colour(ARM_TYPE[a]),
            markeredgecolor=fs.EDGE,
            label=f"{ARM_LABEL[a]}, gpt-4.1-mini" if a != "random_search" else "random, no model",
        )
        for a in ARMS
    ] + [
        Line2D(
            [],
            [],
            linestyle="",
            markersize=7,
            marker=MARKER[a],
            markerfacecolor="white",
            markeredgecolor=fs.arm_colour(ARM_TYPE[a]),
            markeredgewidth=1.3,
            label=f"{ARM_LABEL[a]}, claude-sonnet-4.6",
        )
        for a in ("single_call", "agent")
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.075)
    )
    fig.suptitle(
        "Holdout gain per task; each marker is one episode (two runs per task, one for the "
        "second model)",
        fontsize=11,
    )
    fig.text(0.01, 0.005, SAMPLE, fontsize=10, color=fs.MUTED, va="bottom")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.25, wspace=0.18)
    fs.save(fig, out / "per_task_gain.png")


# ---------------------------------------------------------------- figure 3


def dev_gain_by_trial(name: str) -> tuple[dict, dict]:
    """Per agent episode: dev gains in trial order and the frozen run id."""
    rows = task_rows(name)
    metric = {r["task_id"]: r["metric"] for r in rows}
    trials = trial_rows(name)
    fixed = {
        (t["task_id"], t["run"]): t["dev_metrics"][metric[t["task_id"]]]
        for t in trials
        if t["arm"] == "fixed"
    }
    episodes: dict[tuple[str, int], list[tuple[int, float | None]]] = {}
    for t in trials:
        if t["arm"] != "agent":
            continue
        idx = int(t["run_id"][1:])
        f = fixed[(t["task_id"], t["run"])]
        val = (f - t["dev_metrics"][metric[t["task_id"]]]) / f if t["status"] == "ok" else None
        episodes.setdefault((t["task_id"], t["run"]), []).append((idx, val))
    frozen = {(r["task_id"], r["run"]): r.get("frozen_run_id") for r in rows if r["arm"] == "agent"}
    return episodes, frozen


def trial_trajectories(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(fs.WIDTH + 1.0, 5.2), sharey=True)
    for ax, (name, _, group, model, second) in zip(axes, SETS, strict=True):
        episodes, frozen = dev_gain_by_trial(name)
        n_fail = 0
        n_best = 0
        for key in sorted(episodes):
            pts = sorted(episodes[key])
            ok = [(i, v) for i, v in pts if v is not None]
            n_fail += len(pts) - len(ok)
            ax.plot(
                [i for i, _ in ok],
                [v for _, v in ok],
                color=fs.AGENT,
                linewidth=0.9,
                alpha=0.55,
                zorder=2,
            )
            ax.scatter(
                [i for i, _ in ok],
                [v for _, v in ok],
                s=12,
                facecolor="white" if second else fs.AGENT,
                edgecolor=fs.AGENT if second else fs.EDGE,
                linewidth=0.6,
                zorder=3,
            )
            fid = frozen.get(key)
            hit = [(i, v) for i, v in ok if fid and i == int(fid[1:])]
            if hit:
                ax.scatter([hit[0][0]], [hit[0][1]], marker="*", s=110, color=fs.EDGE, zorder=4)
                n_best += hit[0][0] == max(ok, key=lambda t: t[1])[0]
        ax.axhline(0, color=fs.EDGE, linewidth=1.0, zorder=1)
        ax.set_xticks(range(1, 7))
        ax.set_xlim(0.6, 6.4)
        ax.set_ylim(-1.5, 0.6)
        ax.set_xlabel("trial index")
        extra = f"\n{n_fail} failed trial, no metric" if n_fail else "\n "
        ax.set_title(
            f"{group}, {model}\nfroze its best dev trial in {n_best}/{len(episodes)}{extra}",
            fontsize=10,
        )
    axes[0].set_ylabel("dev gain over the fixed pipeline\n(dev log loss or MAE)")
    handles = [
        Line2D(
            [],
            [],
            color=fs.AGENT,
            marker="o",
            markersize=4,
            markeredgecolor=fs.EDGE,
            label="one agent episode, trials in order",
        ),
        Line2D(
            [],
            [],
            color=fs.EDGE,
            marker="*",
            markersize=11,
            linestyle="",
            label="trial the agent froze",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.1)
    )
    fig.suptitle(
        "Agent trials: dev-set gain at each trial index, frozen trial starred", fontsize=11
    )
    fig.text(0.01, 0.005, SAMPLE, fontsize=10, color=fs.MUTED, va="bottom")
    fig.subplots_adjust(left=0.1, right=0.99, top=0.8, bottom=0.3, wspace=0.08)
    fs.save(fig, out / "trial_trajectories.png")


# ---------------------------------------------------------------- figure 4


def dev_vs_holdout(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(fs.WIDTH + 1.0, 5.3), sharex=True, sharey=True)
    for ax, (name, _family, group, model, second) in zip(axes, SETS, strict=True):
        rows = task_rows(name)
        trials = trial_rows(name)
        metric = {r["task_id"]: r["metric"] for r in rows}
        fixed_dev = {
            (t["task_id"], t["run"]): t["dev_metrics"][metric[t["task_id"]]]
            for t in trials
            if t["arm"] == "fixed"
        }
        n = 0
        below = 0
        for arm in ARMS:
            if second and arm == "random_search":
                continue
            xs, ys = [], []
            for r in measured(rows, arm):
                if not r["frozen"]:
                    continue
                f = fixed_dev[(r["task_id"], r["run"])]
                xs.append((f - r["dev_primary"]) / f)
                ys.append(r["gain_vs_fixed"])
            ax.scatter(xs, ys, **marker_kwargs(arm, second, size=30))
            n += len(xs)
            below += sum(y < x - 0.01 for x, y in zip(xs, ys, strict=True))
        lim = (-0.6, 0.6)
        ax.plot(lim, lim, color=fs.MUTED, linestyle="--", linewidth=1.0, zorder=1)
        ax.axhline(0, color=fs.GRID, linewidth=0.8, zorder=0)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.grid(axis="x", color=fs.GRID)
        ax.set_xlabel("dev gain over fixed")
        ax.set_title(
            f"{group}, {model}\n{n} frozen candidates\n{below}/{n} more than 0.01 below y = x",
            fontsize=10,
        )
    axes[0].set_ylabel("holdout gain over fixed")
    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker=MARKER[a],
            markersize=7,
            markerfacecolor=fs.arm_colour(ARM_TYPE[a]),
            markeredgecolor=fs.EDGE,
            label=ARM_LABEL[a] + (", no model" if a == "random_search" else ""),
        )
        for a in ARMS
    ] + [
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=fs.MUTED,
            markeredgewidth=1.3,
            label="white fill: second model",
        ),
        Line2D([], [], color=fs.MUTED, linestyle="--", label="y = x (no generalisation gap)"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.075)
    )
    fig.suptitle(
        "Frozen candidates: dev gain against holdout gain, both relative to the fixed pipeline",
        fontsize=11,
    )
    fig.text(0.01, 0.005, SAMPLE, fontsize=10, color=fs.MUTED, va="bottom")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.33, wspace=0.1)
    fs.save(fig, out / "dev_vs_holdout.png")


# ---------------------------------------------------------------- figure 5


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
    ("completed", fs.OUTCOME["correct"]),
    ("fit failed or timed out", fs.OUTCOME["failed"]),
    ("refused by the column policy", fs.OUTCOME["wrong"]),
    ("invalid spec, not run", fs.OUTCOME["abstained"]),
]


def trial_outcomes(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 5.6))
    y, yt, yl, groups = 0.0, [], [], []
    for name, family, group, _, second in SETS:
        rows = task_rows(name)
        trials = trial_rows(name)
        start = y
        for arm in ARMS:
            if second and arm == "random_search":
                continue
            ts = [t for t in trials if t["arm"] == arm]
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
            left = 0
            for (label, colour), c in zip(OUTCOMES, counts, strict=True):
                ax.barh(
                    y,
                    c,
                    left=left,
                    height=0.7,
                    color=colour,
                    edgecolor=fs.EDGE,
                    linewidth=0.6,
                    hatch=fs.SECOND_MODEL_HATCH if second and label == "completed" else None,
                    zorder=2,
                )
                left += c
            short = ("failed", "refused", "invalid")
            other = [f"{c} {s}" for s, c in zip(short, counts[1:], strict=True) if c]
            text = f"{counts[0]}/{total} completed" + (", " + ", ".join(other) if other else "")
            ax.text(left + 2, y, text, va="center", fontsize=10, color=fs.TEXT)
            yt.append(y)
            yl.append(ARM_LABEL[arm])
            y += 1.0
        groups.append(((start + y - 1.0) / 2, group))
        y += 0.7
    ax.set_yticks(yt, yl)
    ax.invert_yaxis()
    ax.set_xlim(0, 215)
    ax.grid(axis="x", color=fs.GRID)
    ax.grid(axis="y", visible=False)
    for mid, group in groups:
        ax.text(
            -0.13,
            mid,
            group,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_xlabel("trial requests, summed over episodes")
    handles = [Patch(facecolor=c, edgecolor=fs.EDGE, label=lab) for lab, c in OUTCOMES]
    ax.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.45, -0.42)
    )
    ax.set_title(
        "What happened to every trial request, by arm and round\n"
        "(hatched: second model; refused and invalid requests use no trial)",
        fontsize=11,
    )
    fig.text(0.01, 0.005, SAMPLE, fontsize=10, color=fs.MUTED, va="bottom")
    fig.subplots_adjust(left=0.27, right=0.99, top=0.88, bottom=0.3)
    fs.save(fig, out / "trial_outcomes.png")


def compact(path: Path) -> None:
    """Re-encode a saved PNG with a palette of its 256 most frequent colours.

    Kept colours are stored exactly; every other colour (rare anti-aliasing
    shades) is mapped to the nearest kept colour in RGB space.
    """
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
    img.save(path, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    fs.apply()
    gain_by_arm(args.out)
    per_task_gain(args.out)
    trial_trajectories(args.out)
    dev_vs_holdout(args.out)
    trial_outcomes(args.out)
    for name in FIGURES:
        p = args.out / name
        compact(p)
        print(f"{p.name}: {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
