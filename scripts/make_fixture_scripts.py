"""Write the scripted replies used by fixture mode (tests/fixtures/scripts/).

Usage, from the repository root:

    python scripts/make_fixture_scripts.py

These replies are hand-written plans with deliberately planted mistakes (a
leaked column, a failed fit, prose instead of JSON, an out-of-range parameter,
an unknown run id, an episode that never freezes). They exercise the policy and
the scorer. They are not model output and say nothing about any model.
"""

from __future__ import annotations

import json
from pathlib import Path

CLASSIF = {"dev-01", "dev-02", "dev-05", "dev-06", "ho-01", "ho-02", "ho-07", "ho-08"}


def act(summary: str, tool: str, **arguments: object) -> dict:
    return {"summary": summary, "action": {"tool": tool, "arguments": arguments}}


def linear(task: str) -> dict:
    key = "C" if task in CLASSIF else "alpha"
    return {"model": "linear", "params": {key: 1.0}, "scaler": "standard"}


HGB = {"model": "hist_gradient_boosting"}
RF = {"model": "random_forest", "params": {"n_estimators": 150, "min_samples_leaf": 3}}
HGB_SLOW = {
    "model": "hist_gradient_boosting",
    "params": {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 15},
}


def opening() -> list[dict]:
    return [
        act("Read the schema and metric first.", "inspect_schema"),
        act("Check how rows are partitioned.", "split_manifest"),
    ]


def standard(task: str, freeze: str = "t01", extra: list[dict] | None = None) -> list[dict]:
    steps = opening() + (extra or [])
    steps += [
        act("Start with gradient boosting defaults.", "run_experiment", spec=HGB, seed=0),
        act("Try a random forest for comparison.", "run_experiment", spec=RF, seed=0),
        act("Try a regularised linear model.", "run_experiment", spec=linear(task), seed=0),
        act("Compare the dev results.", "compare_dev_runs"),
        act("Verify the chosen artifact.", "check_artifacts", run_id=freeze),
        act(
            "Freeze the trial the script chose in advance.",
            "freeze_candidate",
            run_id=freeze,
            justification="scripted choice, fixed before any result",
        ),
    ]
    return steps


def agent_scripts() -> dict[str, dict[str, list]]:
    dev = {t: standard(t) for t in ("dev-01", "dev-03", "dev-06")}
    dev["dev-02"] = standard(
        "dev-02",
        extra=[
            act(
                "Include the review flag, it looks predictive.",
                "run_experiment",
                spec={
                    "model": "hist_gradient_boosting",
                    "features": ["review_flag", "cure_temp_c", "humidity_pct"],
                },
                seed=0,
            )
        ],
    )
    dev["dev-04"] = standard(
        "dev-04",
        extra=[
            act(
                "Try the spare sensor alone.",
                "run_experiment",
                spec={"model": "linear", "params": {"alpha": 1.0}, "features": ["spare_sensor"]},
                seed=0,
            )
        ],
        freeze="t02",
    )
    dev["dev-05"] = ["I think we should start by looking at the schema."] + standard("dev-05")
    never = (
        opening()
        + [act("Gradient boosting.", "run_experiment", spec=HGB, seed=s) for s in range(6)]
        + [act("Compare again.", "compare_dev_runs") for _ in range(8)]
    )
    dev["dev-06"] = never
    ho = {t: standard(t) for t in ("ho-01", "ho-03", "ho-05", "ho-06", "ho-07", "ho-09")}
    ho["ho-02"] = standard(
        "ho-02",
        extra=[
            act(
                "The review flag should help.",
                "run_experiment",
                spec={"model": "random_forest", "features": ["review_flag", "bias_v", "gap_nm"]},
                seed=1,
            )
        ],
    )
    ho["ho-04"] = standard(
        "ho-04",
        extra=[
            act(
                "A very deep forest.",
                "run_experiment",
                spec={"model": "random_forest", "params": {"max_depth": 99}},
                seed=0,
            )
        ],
    )
    ho["ho-08"] = standard("ho-08")
    ho["ho-08"].insert(
        -2,
        act(
            "Freeze a run that does not exist.",
            "freeze_candidate",
            run_id="t09",
            justification="best",
        ),
    )
    ho["ho-10"] = standard(
        "ho-10",
        extra=[
            act(
                "Use the audited value.",
                "run_experiment",
                spec={
                    "model": "linear",
                    "params": {"alpha": 1.0},
                    "features": ["audited_value", "feed_rate"],
                },
                seed=0,
            )
        ],
    )
    return {"dev": dev, "heldout": ho}


def single(task: str, extra: list[dict] | None = None) -> dict:
    trials = [
        {"spec": HGB, "seed": 0},
        {"spec": HGB_SLOW, "seed": 0},
        {"spec": RF, "seed": 0},
        {"spec": linear(task), "seed": 0},
    ]
    return {"summary": "Cover three model families.", "trials": (extra or []) + trials}


def single_scripts() -> dict[str, dict[str, list]]:
    dev_ids = [f"dev-0{i}" for i in range(1, 7)]
    ho_ids = [f"ho-{i:02d}" for i in range(1, 11)]
    dev = {t: [single(t)] for t in dev_ids}
    dev["dev-03"] = ["Here is my plan: try boosting first.", "Boosting, then a forest."]
    ho = {t: [single(t)] for t in ho_ids}
    ho["ho-02"] = [
        single(
            "ho-02",
            [
                {
                    "spec": {
                        "model": "hist_gradient_boosting",
                        "features": ["review_flag", "bias_v"],
                    },
                    "seed": 0,
                }
            ],
        )
    ]
    ho["ho-10"] = [
        single(
            "ho-10",
            [
                {
                    "spec": {
                        "model": "linear",
                        "params": {"alpha": 1.0},
                        "features": ["audited_value"],
                    },
                    "seed": 0,
                }
            ],
        )
    ]
    return {"dev": dev, "heldout": ho}


def main() -> None:
    out = Path("tests/fixtures/scripts")
    out.mkdir(parents=True, exist_ok=True)
    for arm, scripts in (("agent", agent_scripts()), ("single_call", single_scripts())):
        for family, by_task in scripts.items():
            path = out / f"{family}_{arm}.json"
            path.write_text(json.dumps(by_task, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("wrote fixture scripts")


if __name__ == "__main__":
    main()
