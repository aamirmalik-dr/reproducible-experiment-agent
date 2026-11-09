"""Evidence files, the per-task breakdown and README tables, built only from saved outputs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

# (output folder under reports/, label used in tables, show no-model rows in the headline)
RESULT_SETS: tuple[tuple[str, str, bool], ...] = (
    ("fixture", "fixture", True),
    ("live", "live v1", True),
    ("live_v2_gpt-4.1-mini", "live v2", True),
    ("live_v2_claude-sonnet-4.6", "live v2", False),
)
LIVE_SETS = tuple(k for k, _, _ in RESULT_SETS if k != "fixture")
NO_MODEL_ARMS = ("fixed", "random_search")
MARKERS = {
    "results": ("<!-- results-table:start -->", "<!-- results-table:end -->"),
    "scorer": ("<!-- scorer-table:start -->", "<!-- scorer-table:end -->"),
}
LIMITATIONS = [
    "Tasks are small synthetic tables (about 360 rows, 46 to 78 holdout rows) from thirteen "
    "original generators. They test discipline, not state-of-the-art modelling.",
    "The fixed pipeline is gradient boosting with library defaults, a strong default; beating "
    "it on small tables is hard and often within noise.",
    "Wins and losses use a 1 percent relative threshold on one holdout of about 72 rows; small "
    "differences are not meaningful.",
    "Windows enforces only the trial time limit; the memory limit applies on POSIX only.",
    "Fixture results for the LLM arms are scripted and say nothing about any model.",
    "Each live run is one model, ten held-out tasks and one or two runs at temperature 0; "
    "exploratory samples, not rankings.",
]
BREAKDOWN_FIELDS = [
    "result_set",
    "model",
    "family",
    "task_id",
    "run",
    "arm",
    "task_type",
    "traps",
    "status",
    "stop_reason",
    "frozen_model",
    "trials_run",
    "model_families_tried",
    "holdout_primary",
    "gain_vs_fixed",
    "leak_attempts",
]


def _num(v: str) -> Any:
    if v == "":
        return None
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() and "." not in v else f


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_set(reports: Path, key: str) -> dict[str, Any] | None:
    """Saved outputs for one result set, or None."""
    d = reports / key
    if not (d / "run_meta.json").exists():
        return None
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    with (d / "comparison.csv").open(encoding="utf-8") as fh:
        comparison = [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    rows = _jsonl(d / "task_results.jsonl")
    keys = (
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
    notable = [
        {k: r.get(k) for k in keys}
        for r in rows
        if r["status"] != "complete"
        or r["forbidden_attempts"]
        or r["trials_failed"]
        or r.get("malformed_actions")
        or (r["frozen"] and not r["reproduced"])
    ]
    return {"meta": meta, "metrics": comparison, "notable": notable, "episodes": len(rows)}


def short_model(name: str) -> str:
    """Model id without the provider prefix, for table labels."""
    return name.split("/", 1)[-1]


def build_evidence(root: Path) -> dict[str, Any]:
    """Assemble evidence.json from saved outputs; missing result sets are 'pending'."""
    ev: dict[str, Any] = {"schema": "evidence-v2", "modes": {}}
    for key, tag, _ in RESULT_SETS:
        data = load_set(root / "reports", key)
        if data is None:
            ev["modes"][key] = "pending"
            continue
        m = data["meta"]
        ev["modes"][key] = {
            "result_set": key,
            "label": tag,
            "interpretation": (
                "fixed and random_search use no model and are real measurements; the "
                "single_call and agent arms replay scripted replies and only check the scorer"
                if key == "fixture"
                else f"live model results ({tag})"
            ),
            "git_commit": m["git_commit"],
            "git_dirty": m["git_dirty"],
            "model": m["model"],
            "agentcore_version": m["agentcore_version"],
            "prompt_version": m["prompt_version"],
            "prompt_sha256": m["prompt_sha256"],
            "config_sha256": m["config_sha256"],
            "families": m["config"]["families"],
            "runs": m["config"]["runs"],
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
    aborted = sorted(p.name for p in (root / "reports" / "aborted").glob("*") if p.is_dir())
    ev["aborted_runs"] = [
        {"folder": f"reports/aborted/{name}", "reason": "see reports/aborted/README.md"}
        for name in aborted
    ]
    ev["limitations"] = LIMITATIONS
    ev["reproduce"] = {
        "fixture": "python -m reproducible_experiment_agent evaluate --config "
        "configs/experiment_fixture.yaml && python -m reproducible_experiment_agent report",
        "live v1": "python -m reproducible_experiment_agent evaluate --config "
        "configs/experiment_live.yaml --require-live",
        "live v2, first model": "python -m reproducible_experiment_agent evaluate --config "
        "configs/experiment_live_v2_gpt41mini.yaml --require-live",
        "live v2, second model": "python -m reproducible_experiment_agent evaluate --config "
        "configs/experiment_live_v2_sonnet46.yaml --require-live",
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


def _label(tag: str, model: str, arm: str) -> str:
    if arm in NO_MODEL_ARMS:
        return f"no model, {tag} run"
    if tag == "fixture":
        return "fixture, scripted"
    return f"{tag}, {short_model(model)}"


HEAD = (
    "| Mode | Family | Arm | Frozen | Mean gain vs fixed (min to max) | Wins / losses vs fixed "
    "| Wins / losses vs random search | Reproduced | Leak-clean episodes | Refused columns "
    "(executed) | Failed trials | Tokens | USD (provider) |\n" + "|---" * 13 + "|\n"
)


def _line(tag: str, model: str, r: dict[str, Any]) -> str:
    usd = r.get("reported_cost_usd_total")
    frozen = _frac(r["frozen_k"], r["frozen_n"])
    if r.get("pending_run_cap"):
        frozen += f" ({r['pending_run_cap']} pending, run cap)"
    return (
        f"| {_label(tag, model, r['arm'])} | {r['family']} | {r['arm']} "
        f"| {frozen} | {_gain(r)} "
        f"| {r['wins_vs_fixed_k']} / {r['losses_vs_fixed_k']} "
        f"| {r['wins_vs_random_k']} / {r['losses_vs_random_k']} "
        f"| {_frac(r['reproduced_k'], r['frozen_k'])} "
        f"| {_frac(r['leak_clean_k'], r['leak_clean_n'])} "
        f"| {r['forbidden_attempts_total']} ({r['executed_forbidden_total']}) "
        f"| {r['trials_failed_total']} | {r['tokens_total'] or 0} "
        f"| {'0' if usd is None else f'{usd:.4f}'} |"
    )


def results_table(ev: dict[str, Any]) -> str:
    """Headline: no-model baselines and live rows only, one block per result set."""
    lines = []
    for key, tag, show_baselines in RESULT_SETS:
        data = ev["modes"][key]
        if data == "pending":
            if key != "fixture":
                lines.append(f"| {tag} ({key}): pending |" + " |" * 12)
            continue
        for r in data["metrics"]:
            if r["arm"] in NO_MODEL_ARMS and not show_baselines:
                continue
            if key == "fixture" and r["arm"] not in NO_MODEL_ARMS:
                continue
            lines.append(_line(tag, data["model"], r))
    return HEAD + "\n".join(lines) + "\n"


def scorer_table(ev: dict[str, Any]) -> str:
    """Scripted fixture rows for the LLM arms."""
    data = ev["modes"]["fixture"]
    if data == "pending":
        return HEAD + "| fixture: pending |" + " |" * 12 + "\n"
    lines = [
        _line("fixture", data["model"], r) for r in data["metrics"] if r["arm"] not in NO_MODEL_ARMS
    ]
    return HEAD + "\n".join(lines) + "\n"


def per_task_breakdown(root: Path) -> list[dict[str, Any]]:
    """One row per live episode, with the model families each episode tried."""
    out: list[dict[str, Any]] = []
    for key in LIVE_SETS:
        d = root / "reports" / key
        if not (d / "run_meta.json").exists():
            continue
        model = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))["model"]
        tried: dict[tuple[str, str, int], set[str]] = {}
        for t in _jsonl(d / "trial_results.jsonl"):
            tried.setdefault((t["arm"], t["task_id"], t["run"]), set()).add(t["spec"]["model"])
        for r in _jsonl(d / "task_results.jsonl"):
            families = sorted(tried.get((r["arm"], r["task_id"], r["run"]), set()))
            out.append(
                {
                    "result_set": key,
                    "model": "none" if r["arm"] in NO_MODEL_ARMS else model,
                    **{k: r.get(k) for k in BREAKDOWN_FIELDS[2:12] if k != "model_families_tried"},
                    "model_families_tried": ";".join(families),
                    "holdout_primary": r.get("holdout_primary"),
                    "gain_vs_fixed": r.get("gain_vs_fixed"),
                    "leak_attempts": r.get("leak_attempts"),
                }
            )
    return out


def evidence_markdown(ev: dict[str, Any]) -> str:
    """Human-readable evidence report."""
    out = [
        "# Evidence",
        "",
        "Generated from saved outputs by "
        "`python -m reproducible_experiment_agent report`. Do not edit by hand.",
        "",
    ]
    for key, tag, _ in RESULT_SETS:
        data = ev["modes"][key]
        out += [f"## {tag} ({key})", ""]
        if data == "pending":
            out += ["Pending. No saved outputs exist for this result set.", ""]
            continue
        c = data["cost"]
        out += [
            data["interpretation"],
            "",
            f"- Commit: `{data['git_commit']}` (dirty: {data['git_dirty']})",
            f"- Model: {data['model']}; agentcore {data['agentcore_version']}",
            f"- Prompt: {data['prompt_version']}, sha256 `{(data['prompt_sha256'] or '')[:16]}`",
            f"- Config sha256: `{data['config_sha256'][:16]}`; families {data['families']}; "
            f"runs {data['runs']}",
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
        cols = (
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
        out += [
            "| " + " | ".join(str(n.get(k)) for k in cols) + " |" for n in data["notable_episodes"]
        ]
        out.append("")
    if ev["aborted_runs"]:
        out += ["## Aborted runs (excluded from every table)", ""]
        out += [f"- `{a['folder']}`: {a['reason']}" for a in ev["aborted_runs"]]
        out.append("")
    out += ["## Results (no-model baselines and live rows)", "", results_table(ev)]
    out += ["## Scorer checks (scripted replies, not model results)", "", scorer_table(ev)]
    out += ["## Limitations", ""] + [f"- {x}" for x in ev["limitations"]]
    out += ["", "## Reproduce", ""] + [f"- {k}: `{v}`" for k, v in ev["reproduce"].items()]
    return "\n".join(out) + "\n"


def reaggregate(root: Path) -> None:
    """Rebuild the v2 comparison.csv files from saved task_results.jsonl (pure, no re-running).

    v1 outputs are left exactly as they were written.
    """
    from .evaluation import aggregate

    for key in (k for k in LIVE_SETS if k.startswith("live_v2")):
        d = root / "reports" / key
        rows = _jsonl(d / "task_results.jsonl")
        if not rows:
            continue
        comparison = aggregate(rows)
        fields = list(dict.fromkeys(k for row in comparison for k in row))
        with (d / "comparison.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(comparison)


def write_reports(root: Path, update_readme: bool = False) -> dict[str, Any]:
    """Write evidence.json, evidence.md and per_task_breakdown.csv; optionally the README tables."""
    reaggregate(root)
    ev = build_evidence(root)
    reports = root / "reports"
    (reports / "evidence.json").write_text(
        json.dumps(ev, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (reports / "evidence.md").write_text(evidence_markdown(ev), encoding="utf-8", newline="\n")
    rows = per_task_breakdown(root)
    if rows:
        with (reports / "per_task_breakdown.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=BREAKDOWN_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
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
