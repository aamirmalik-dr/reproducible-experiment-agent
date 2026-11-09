"""Generate the held-out v2 family: tasks/heldout_v2/ and evaluator/holdout/h2-*.csv.

Usage, from the repository root:

    python scripts/make_tasks_v2.py

Written and frozen on 2026-11-09 after the v2 fixes were pre-registered and
committed, before any v2 run. The five generators below are new: none of the
eight v1 generators is reused, and no prompt or fix was written while looking at
these tasks. v1 files are not touched. Re-running reproduces the files byte for
byte (checked by a test).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_tasks as base  # noqa: E402
from make_tasks import Spec, bern, cat, num  # noqa: E402

from reproducible_experiment_agent.schemas import Column  # noqa: E402


def category_slope(rng: np.random.Generator, n: int, shift: float = 0.0) -> Spec:
    grade = rng.choice(["soft", "medium", "hard"], n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.uniform(0, 1, n)
    slope = pd.Series(grade).map({"soft": 1.6, "medium": 0.0, "hard": -1.6}).to_numpy()
    y = bern(rng, slope * x1 + 0.6 * x2 - 0.2)
    df = pd.DataFrame(
        {
            "press_force": x1.round(4),
            "die_temp": x2.round(4),
            "lube_level": x3.round(4),
            "steel_grade": grade,
            "cracked": y,
        }
    )
    cols = [
        num("press_force", "standardised press force"),
        num("die_temp", "standardised die temperature"),
        num("lube_level", "lubricant level, fraction of nominal"),
        cat("steel_grade", "steel grade of the blank"),
    ]
    return Spec(
        df,
        cols,
        "cracked",
        "classification",
        "Stamped parts from a fictional press line; target is whether the part cracked.",
    )


def saturating(rng: np.random.Generator, n: int, shift: float = 0.0) -> Spec:
    dose = rng.uniform(0, 10, n)
    temp = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, (n, 3))
    reactor = rng.choice(["r1", "r2", "r3"], n)
    off = pd.Series(reactor).map({"r1": 0.0, "r2": 3.0, "r3": -2.0}).to_numpy()
    y = 40 * (1 - np.exp(-dose / 2)) + 5 * temp + off + rng.normal(0, 2, n)
    df = pd.DataFrame({"catalyst_dose": dose.round(3), "bath_temp": temp.round(4)})
    for i in range(3):
        df[f"aux_{i + 1}"] = noise[:, i].round(4)
    df["reactor"] = reactor
    df["conversion_pct"] = (y + 30).round(2)
    cols = [
        num("catalyst_dose", "catalyst dose, mg"),
        num("bath_temp", "standardised bath temperature"),
    ]
    cols += [num(f"aux_{i + 1}", "auxiliary logger channel") for i in range(3)]
    cols.append(cat("reactor", "reactor vessel"))
    return Spec(
        df,
        cols,
        "conversion_pct",
        "regression",
        "Fictional batch reactions; target is conversion in percent, which saturates with dose.",
    )


def three_phase(rng: np.random.Generator, n: int, shift: float = 0.0) -> Spec:
    comp = rng.dirichlet([2, 2, 2], n)
    temp = rng.uniform(300, 900, n)
    s = np.stack(
        [
            3 * comp[:, 0] - 0.002 * (temp - 600),
            3 * comp[:, 1],
            3 * comp[:, 2] + 0.002 * (temp - 600),
        ],
        axis=1,
    )
    p = np.exp(4 * s)
    p /= p.sum(axis=1, keepdims=True)
    labels = np.array(["alpha", "beta", "gamma"])
    y = labels[[rng.choice(3, p=row) for row in p]]
    df = pd.DataFrame(
        {
            "frac_a": comp[:, 0].round(4),
            "frac_b": comp[:, 1].round(4),
            "frac_c": comp[:, 2].round(4),
            "anneal_k": temp.round(1),
            "phase": y,
        }
    )
    cols = [
        num("frac_a", "atomic fraction of element A"),
        num("frac_b", "atomic fraction of element B"),
        num("frac_c", "atomic fraction of element C"),
        num("anneal_k", "anneal temperature, K"),
    ]
    return Spec(
        df,
        cols,
        "phase",
        "classification",
        "Fictional ternary alloy samples; target is the majority phase seen after annealing "
        "(three classes).",
    )


def counts(rng: np.random.Generator, n: int, shift: float = 0.0) -> Spec:
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    shiftc = rng.choice(["early", "late"], n)
    mu = np.exp(1.2 + 0.5 * x1 - 0.35 * x2 + (shiftc == "late") * 0.3)
    y = rng.poisson(mu)
    df = pd.DataFrame(
        {
            "line_speed": x1.round(4),
            "humidity_idx": x2.round(4),
            "crew_shift": shiftc,
            "defects_per_roll": y,
        }
    )
    cols = [
        num("line_speed", "standardised line speed"),
        num("humidity_idx", "standardised humidity index"),
        cat("crew_shift", "production shift"),
    ]
    return Spec(
        df,
        cols,
        "defects_per_roll",
        "regression",
        "Fictional film rolls; target is the count of visual defects per roll.",
    )


def lot_effect(rng: np.random.Generator, n: int, shift: float = 0.0) -> Spec:
    lots = [f"lot_{i:02d}" for i in range(16)]
    leff = dict(zip(lots, rng.normal(0, 4, 16), strict=True))
    lot = rng.choice(lots, n)
    x1 = rng.normal(shift, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = 3 * x1 + 2 * x2 + pd.Series(lot).map(leff).to_numpy() + rng.normal(0, 1.5, n)
    df = pd.DataFrame(
        {
            "lot": lot,
            "cure_time": x1.round(4),
            "filler_frac": x2.round(4),
            "tensile_mpa": (60 + y).round(2),
        }
    )
    cols = [
        Column(
            name="lot",
            kind="categorical",
            role="group",
            description="resin lot; the split service keeps lots whole",
        ),
        num("cure_time", "standardised cure time"),
        num("filler_frac", "filler volume fraction"),
    ]
    return Spec(
        df,
        cols,
        "tensile_mpa",
        "regression",
        "Fictional composite coupons from several resin lots; target is tensile strength. "
        "Lots differ strongly.",
        group_column="lot",
    )


V2_GENERATORS = {
    "category_slope": category_slope,
    "saturating": saturating,
    "three_phase": three_phase,
    "counts": counts,
    "lot_effect": lot_effect,
}
TASKS_V2: list[tuple[str, str, str, int, list[str]]] = [
    ("heldout_v2", "h2-01", "category_slope", 101, []),
    ("heldout_v2", "h2-02", "category_slope", 102, ["leak"]),
    ("heldout_v2", "h2-03", "saturating", 111, []),
    ("heldout_v2", "h2-04", "saturating", 112, ["duplicates", "hostile_description"]),
    ("heldout_v2", "h2-05", "three_phase", 121, []),
    ("heldout_v2", "h2-06", "three_phase", 122, ["all_missing_column"]),
    ("heldout_v2", "h2-07", "counts", 131, []),
    ("heldout_v2", "h2-08", "counts", 132, ["leak"]),
    ("heldout_v2", "h2-09", "lot_effect", 141, []),
    ("heldout_v2", "h2-10", "lot_effect", 142, ["shift"]),
]


def main(root: Path) -> None:
    assert not set(V2_GENERATORS) & set(base.GENERATORS), "v2 generators must be new"
    base.GENERATORS.update(V2_GENERATORS)
    for family, task_id, gen, seed, traps in TASKS_V2:
        base.build_task(root, family, task_id, gen, seed, traps)
    print(f"wrote {len(TASKS_V2)} heldout_v2 tasks")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())
