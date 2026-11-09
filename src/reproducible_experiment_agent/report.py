"""Evidence files and README tables, built only from saved run outputs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

MODES = ("fixture", "live")
NO_MODEL_ARMS = ("fixed", "random_search")
MARKERS = {
    "results": ("<!-- results-table:start -->", "<!-- results-table:end -->"),
    "scorer": ("<!-- scorer-table:start -->", "<!-- scorer-table:end -->"),
}
LIMITATIONS = [
    "Tasks are small synthetic tables (about 360 rows, 72 holdout rows) from eight original "
    "generators. They test discipline, not state-of-the-art modelling.",
    "The fixed pipeline is gradient boosting with library defaults, a strong default; beating "
    "it on small tables is hard and often within noise.",
    "Wins and losses use a 1 percent relative threshold on one holdout of about 72 rows; small "
    "differences are not meaningful.",
    "Windows enforces only the trial time limit; the memory limit applies on POSIX only.",
    "Fixture results for the LLM arms are scripted and say nothing about any model.",
]


def _num(v: str) -> Any:
    if v == "":
        return None
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() and "." not in v else f


def load_mode(reports: Path, mode: str) -> dict[str, Any] | None:
    """Saved outputs for one mode, or None."""
    d = reports / mode
    if not (d / "run_meta.json").exists():
        return None
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    with (d / "comparison.csv").open(encoding="utf-8") as fh:
        comparison = [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    rows = [
        json.loads(x)
        for x in (d / "task_results.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    notable = [
        {
            k: r.get(k)
            for k in (
                "family",
                "arm",
                "task_id",
                "run",
                "status",
                "stop_reason",
                "traps",
                "trials_failed",
                "forbidden_attempts",
                "leak_attempts",
                "malformed_actions",
                "gain_vs_fixed",
                "reproduced",
            )
        }
        for r in rows
        if r["status"] != "complete"
        or r["forbidden_attempts"]
        or r["trials_failed"]
        or r.get("malformed_actions")
        or (r["frozen"] and not r["reproduced"])
    ]
    return {"meta": meta, "metrics": comparison, "notable": notable, "episodes": len(rows)}


def build_evidence(root: Path) -> dict[str, Any]:
    """Assemble evidence.json from saved outputs; missing modes are 'pending'."""
    ev: dict[str, Any] = {"schema": "evidence-v1", "modes": {}}
    for mode in MODES:
        data = load_mode(root / "reports", mode)
        if data is None:
            ev["modes"][mode] = "pending"
            continue
        m = data["meta"]
        ev["modes"][mode] = {
            "mode": mode,
            "interpretation": (
                "fixed and random_search use no model and are real measurements; the "
                "single_call and agent arms replay scripted replies and only check the scorer"
                if mode == "fixture"
                else "live model results"
            ),
            "git_commit": m["git_commit"],
            "git_dirty": m["git_dirty"],
            "model": m["model"],
            "agentcore_version": m["agentcore_version"],
            "prompt_version": m["prompt_version"],
            "prompt_sha256": m["prompt_sha256"],
            "config_sha256": m["config_sha256"],
            "input_hashes": m["input_hashes"],
            "hardware": m["hardware"],
            "started_at": m["started_at"],
            "finished_at": m["finished_at"],
            "caps": {"episode": m["config"]["episode"], "run": m["config"]["run_caps"]},
            "cost": {
                "tokens": m["tokens"],
                "usd_at_listed_prices": m["usd"],
                "usd_reported_by_provider": m["usd_reported_by_provider"],
                "adapter_retries": m["adapter_retries"],
            },
            "metrics": data["metrics"],
            "notable_episodes": data["notable"],
            "episodes": data["episodes"],
        }
    live_done = ev["modes"]["live"] != "pending"
    ev["limitations"] = LIMITATIONS + [
        (
            "Live pilot: one model, 10 held-out tasks, two runs at temperature 0; exploratory."
            if live_done
            else "No live model has been evaluated yet; live results are pending."
        )
    ]
    ev["reproduce"] = {
        "fixture": "python -m reproducible_experiment_agent evaluate --config "
        "configs/experiment_fixture.yaml && python -m reproducible_experiment_agent report",
        "live": "python -m reproducible_experiment_agent evaluate --config "
        "configs/experiment_live.yaml --require-live && python -m reproducible_experiment_agent "
        "report",
        "frozen artifact": "python -m reproducible_experiment_agent reproduce --manifest "
        "reports/frozen_candidate.json",
    }
    return ev


def _frac(k: Any, n: Any) -> str:
    return f"{k}/{n}" if n else "n/a"


def _gain(r: dict[str, Any]) -> str:
    if r["mean_gain_vs_fixed"] is None:
        return "n/a"
    return (
        f"{r['mean_gain_vs_fixed']:+.3f} ({r['min_gain_vs_fixed']:+.3f} to "
        f"{r['max_gain_vs_fixed']:+.3f})"
    )


def _label(mode: str, arm: str) -> str:
    if arm in NO_MODEL_ARMS:
        return f"no model, {mode} run"
    return "live" if mode == "live" else "fixture, scripted"


HEAD = (
    "| Mode | Family | Arm | Frozen | Mean gain vs fixed (min to max) | Wins / losses vs fixed "
    "| Wins / losses vs random search | Reproduced | Leak-clean episodes | Refused columns "
    "(executed) | Failed trials | Tokens | USD (provider) |\n" + "|---" * 13 + "|\n"
)


def _line(mode: str, r: dict[str, Any]) -> str:
    usd = r.get("reported_cost_usd_total")
    return (
        f"| {_label(mode, r['arm'])} | {r['family']} | {r['arm']} "
        f"| {_frac(r['frozen_k'], r['frozen_n'])} | {_gain(r)} "
        f"| {r['wins_vs_fixed_k']} / {r['losses_vs_fixed_k']} "
        f"| {r['wins_vs_random_k']} / {r['losses_vs_random_k']} "
        f"| {_frac(r['reproduced_k'], r['frozen_k'])} "
        f"| {_frac(r['leak_clean_k'], r['leak_clean_n'])} "
        f"| {r['forbidden_attempts_total']} ({r['executed_forbidden_total']}) "
        f"| {r['trials_failed_total']} | {r['tokens_total'] or 0} "
        f"| {'0' if usd is None else f'{usd:.4f}'} |"
    )


def results_table(ev: dict[str, Any]) -> str:
    """Headline: no-model baselines and live rows only."""
    lines = []
    for mode in MODES:
        data = ev["modes"][mode]
        if data == "pending":
            lines.append(f"| {mode}: pending |" + " |" * 12)
            continue
        for r in data["metrics"]:
            if mode == "live" or r["arm"] in NO_MODEL_ARMS:
                lines.append(_line(mode, r))
    return HEAD + "\n".join(lines) + "\n"


def scorer_table(ev: dict[str, Any]) -> str:
    """Scripted fixture rows for the LLM arms."""
    data = ev["modes"]["fixture"]
    if data == "pending":
        return HEAD + "| fixture: pending |" + " |" * 12 + "\n"
    lines = [_line("fixture", r) for r in data["metrics"] if r["arm"] not in NO_MODEL_ARMS]
    return HEAD + "\n".join(lines) + "\n"


def evidence_markdown(ev: dict[str, Any]) -> str:
    """Human-readable evidence report."""
    out = [
        "# Evidence",
        "",
        "Generated from saved outputs by "
        "`python -m reproducible_experiment_agent report`. Do not edit by hand.",
        "",
    ]
    for mode in MODES:
        data = ev["modes"][mode]
        out += [f"## {mode.capitalize()} mode", ""]
        if data == "pending":
            out += ["Pending. No saved outputs exist for this mode.", ""]
            continue
        c = data["cost"]
        out += [
            data["interpretation"],
            "",
            f"- Commit: `{data['git_commit']}` (dirty: {data['git_dirty']})",
            f"- Model: {data['model']}; agentcore {data['agentcore_version']}",
            f"- Prompt: {data['prompt_version']}, sha256 `{(data['prompt_sha256'] or '')[:16]}`",
            f"- Config sha256: `{data['config_sha256'][:16]}`",
            f"- Run: {data['started_at']} to {data['finished_at']}",
            f"- Hardware: {data['hardware']['platform']}, Python {data['hardware']['python']}",
            f"- Episodes: {data['episodes']}",
            f"- Tokens: {c['tokens']}; USD at listed prices: {c['usd_at_listed_prices']}; "
            f"USD reported by provider: {c['usd_reported_by_provider']}; "
            f"adapter retries: {c['adapter_retries']}",
            "",
            "### Episodes that failed, refused a column, lost a trial or did not reproduce",
            "",
            "| Family | Arm | Task | Run | Status | Stop | Traps | Failed trials | Refused | "
            "Leak attempts | Malformed | Gain vs fixed | Reproduced |",
            "|---" * 13 + "|",
        ]
        out += [
            "| "
            + " | ".join(
                str(n.get(k))
                for k in (
                    "family",
                    "arm",
                    "task_id",
                    "run",
                    "status",
                    "stop_reason",
                    "traps",
                    "trials_failed",
                    "forbidden_attempts",
                    "leak_attempts",
                    "malformed_actions",
                    "gain_vs_fixed",
                    "reproduced",
                )
            )
            + " |"
            for n in data["notable_episodes"]
        ]
        out.append("")
    out += ["## Results (no-model baselines and live rows)", "", results_table(ev)]
    out += ["## Scorer checks (scripted replies, not model results)", "", scorer_table(ev)]
    out += ["## Limitations", ""] + [f"- {x}" for x in ev["limitations"]]
    out += ["", "## Reproduce", ""] + [f"- {k}: `{v}`" for k, v in ev["reproduce"].items()]
    return "\n".join(out) + "\n"


def write_reports(root: Path, update_readme: bool = False) -> dict[str, Any]:
    """Write evidence.json and evidence.md; optionally refresh the README tables."""
    ev = build_evidence(root)
    (root / "reports" / "evidence.json").write_text(
        json.dumps(ev, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (root / "reports" / "evidence.md").write_text(
        evidence_markdown(ev), encoding="utf-8", newline="\n"
    )
    if update_readme:
        readme = root / "README.md"
        text = readme.read_text(encoding="utf-8")
        for key, table in (("results", results_table(ev)), ("scorer", scorer_table(ev))):
            start, end = MARKERS[key]
            pattern = re.compile(re.escape(start) + ".*?" + re.escape(end), re.S)
            if not pattern.search(text):
                raise ValueError(f"README lacks {start}")
            text = pattern.sub(lambda _, t=table, a=start, b=end: f"{a}\n{t}{b}", text)
        readme.write_text(text, encoding="utf-8", newline="\n")
    return ev
