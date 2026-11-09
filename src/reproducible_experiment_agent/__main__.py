"""Command line: demo, evaluate, reproduce and report."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .agent import EpisodeSettings, make_context, run_agent
from .agentcore.adapters import FixtureAdapter, LiveModelUnavailable, OpenAICompatibleAdapter
from .evaluation import load_config, load_scorer, reproduce, run_evaluation
from .report import write_reports
from .schemas import ExperimentState, FrozenCandidate
from .task_registry import visible_schema

ROOT = Path.cwd()
DEMO_TASK = ("dev", "dev-02")


def _short(obj: object, n: int = 70) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def _steps(state: ExperimentState) -> Table:
    t = Table(title=f"{state.task_id}: agent steps")
    for col in ("#", "tool", "arguments", "result", "model summary"):
        t.add_column(col, overflow="fold")
    for h in state.history:
        if h.result.ok:
            out = h.result.output or {}
            res = _short(out.get("dev_metrics") or out.get("frozen") or list(out)[:4], 50)
        else:
            res = f"REFUSED {h.result.error.code}: {_short(h.result.error.message, 60)}"
        t.add_row(
            str(h.step), h.action.tool, _short(h.action.arguments, 60), res, _short(h.summary, 50)
        )
    return t


def run_demo(mode: str, root: Path, console: Console, model: str | None = None) -> dict:
    """Run the agent on the leak-trap task, score the holdout once, then reproduce."""
    family, task_id = DEMO_TASK
    work = root / "runs" / "demo"
    shutil.rmtree(work, ignore_errors=True)
    ctx = make_context(root, family, task_id, work / "artifacts", "subprocess", 60.0)
    schema = visible_schema(ctx.parts)
    console.print(f"[bold]Task {task_id}[/bold] ({schema['task_type']}, metric {schema['metric']})")
    cols = Table(title="Visible schema (train rows only)")
    for c in ("column", "kind", "role", "missing", "description"):
        cols.add_column(c, overflow="fold")
    for c in schema["columns"]:
        cols.add_row(
            c["name"],
            c["kind"],
            c["role"],
            str(c["missing_fraction_train"]),
            _short(c["description"], 60),
        )
    console.print(cols)
    console.print("Dataset description (untrusted): " + schema["dataset_description"])
    console.print("Split summary: " + json.dumps(ctx.parts.summary()))
    adapter = (
        OpenAICompatibleAdapter.from_env(model=model)
        if mode == "live"
        else FixtureAdapter.from_file(root / "tests/fixtures/scripts/dev_agent.json", task_id)
    )
    settings = EpisodeSettings(record_timing=False)
    res = run_agent(ctx, adapter, 0, settings, checkpoint_path=work / "demo.ckpt.json")
    state = ExperimentState.model_validate_json(
        json.loads((work / "demo.ckpt.json").read_text(encoding="utf-8"))["state"]
    )
    console.print(_steps(state))
    out: dict = {"task_id": task_id, "status": res.status, "refused": res.refused_columns}
    if res.frozen is None:
        console.print("[red]no candidate was frozen[/red]")
        return out
    path = root / "reports" / "frozen_candidate.json"
    path.write_text(res.frozen.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n")
    scorer = load_scorer(root).HoldoutScorer()
    blob = (ctx.runner.artifact_dir / f"{res.frozen.run_id}.pkl").read_bytes()
    receipt = scorer.score(res.frozen, ctx.parts.manifest, blob)
    console.print(
        f"Frozen {res.frozen.run_id} ({res.frozen.spec.model}); manifest written to "
        f"reports/frozen_candidate.json"
    )
    console.print("Holdout, scored once outside the loop: " + json.dumps(receipt["metrics"]))
    rep = reproduce(root, res.frozen, work / "repro")
    console.print(
        f"Reproduction in a fresh process: artifact identical {rep['artifact_identical']}, "
        f"predictions identical {rep['predictions_identical']}"
    )
    label = "scripted fixture replies, no model" if mode == "fixture" else adapter.name
    console.print(f"Mode: {mode} ({label}).")
    return out | {"frozen": res.frozen.run_id, "holdout": receipt["metrics"], "reproduce": rep}


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m reproducible_experiment_agent``."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="reproducible_experiment_agent", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("demo", help="agent on the leak-trap task, holdout once, reproduce")
    p.add_argument("--mode", choices=("fixture", "live"), default="fixture")
    p.add_argument("--model", help="model id for live mode; overrides AGENT_MODEL_NAME")
    p = sub.add_parser("evaluate", help="run every arm on every task in a config")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--require-live", action="store_true")
    p = sub.add_parser("reproduce", help="refit a frozen candidate and compare hashes")
    p.add_argument("--manifest", type=Path, default=Path("reports/frozen_candidate.json"))
    p = sub.add_parser("report", help="regenerate evidence files from saved outputs")
    p.add_argument("--update-readme", action="store_true")
    args = parser.parse_args(argv)
    console = Console(width=150)
    try:
        if args.cmd == "demo":
            run_demo(args.mode, ROOT, console, model=args.model)
        elif args.cmd == "evaluate":
            cfg, sha = load_config(args.config)
            if args.require_live and cfg.mode != "live":
                raise LiveModelUnavailable("--require-live needs a config with mode: live")
            out = run_evaluation(cfg, sha, ROOT, require_live=args.require_live)
            console.print(f"wrote {out.relative_to(ROOT).as_posix()}/")
        elif args.cmd == "reproduce":
            frozen = FrozenCandidate.model_validate_json(args.manifest.read_text(encoding="utf-8"))
            if frozen.manifest_sha256 != frozen.body_sha256():
                console.print("[red]manifest hash does not match its contents[/red]")
                return 1
            rep = reproduce(ROOT, frozen, ROOT / "runs" / "reproduce")
            console.print(json.dumps(rep, indent=1))
            ok = rep["artifact_identical"] and rep["predictions_identical"]
            console.print("reproduced: " + ("yes, byte-identical" if ok else "NO"))
            return 0 if ok else 1
        elif args.cmd == "report":
            ev = write_reports(ROOT, update_readme=args.update_readme)
            console.print(
                "wrote reports/evidence.json and reports/evidence.md; "
                + ", ".join(
                    f"{k}={'pending' if v == 'pending' else 'saved'}"
                    for k, v in ev["modes"].items()
                )
            )
    except LiveModelUnavailable as exc:
        console.print(f"[red]live model unavailable:[/red] {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
