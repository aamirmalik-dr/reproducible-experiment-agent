"""Evaluation: every arm on every task, one-shot holdout scoring, reproduction, aggregation.

Metrics and the normalisation were declared in docs/evaluation.md before any
run: the primary holdout metric is log loss (classification) or MAE
(regression), lower is better, and the relative gain of an arm over the fixed
pipeline on the same task and run is ``(fixed - arm) / fixed``. A win is a gain
above 0.01, a loss a gain below -0.01. Episodes without a frozen candidate
count in every denominator and never as a win.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from . import __version__
from .agent import (
    EpisodeResult,
    EpisodeSettings,
    make_context,
    run_agent,
    run_fixed,
    run_random_search,
    run_single_call,
)
from .agentcore import AGENTCORE_VERSION
from .agentcore.adapters import (
    FixtureAdapter,
    LiveModelUnavailable,
    ModelAdapter,
    OpenAICompatibleAdapter,
)
from .agentcore.policy import ToolRegistry
from .prompts import PROMPT_VERSION, prompt_hash
from .runner import TrialRunner
from .task_registry import list_tasks, load_task
from .tools import build_tools

ARMS = ("fixed", "random_search", "single_call", "agent")
WIN = 0.01


class RunCaps(BaseModel):
    max_total_tokens: int | None = None
    max_usd: float | None = None
    usd_per_mtok_input: float | None = None
    usd_per_mtok_output: float | None = None


class EvalConfig(BaseModel):
    """Frozen evaluation settings from configs/*.yaml."""

    mode: Literal["fixture", "live"]
    model: str | None = None
    families: list[str]
    arms: list[str] = list(ARMS)
    runs: int = 1
    task_limit: int | None = None
    isolation: Literal["subprocess", "inprocess"] = "subprocess"
    fixture_scripts: str = "tests/fixtures/scripts"
    episode: EpisodeSettings = Field(default_factory=EpisodeSettings)
    run_caps: RunCaps = Field(default_factory=RunCaps)
    output_dir: str = "reports/fixture"
    work_dir: str = "runs"
    adapter_max_retries: int = 2
    adapter_backoff_s: float = 2.0


def load_config(path: Path) -> tuple[EvalConfig, str]:
    """Load a config and return it with the file's SHA-256."""
    raw = Path(path).read_bytes()
    cfg = EvalConfig.model_validate(yaml.safe_load(raw))
    if set(cfg.arms) - set(ARMS):
        raise ValueError(f"unknown arms {sorted(set(cfg.arms) - set(ARMS))}")
    return cfg, hashlib.sha256(raw).hexdigest()


def load_scorer(root: Path) -> Any:
    """Import evaluator/holdout_scorer.py by path; it is not part of the package."""
    path = root / "evaluator" / "holdout_scorer.py"
    spec = importlib.util.spec_from_file_location("holdout_scorer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cost_usd(tokens_in: int, tokens_out: int, caps: RunCaps) -> float | None:
    """Dollar cost at the configured listed prices, or None if unset."""
    if caps.usd_per_mtok_input is None or caps.usd_per_mtok_output is None:
        return None
    return (tokens_in * caps.usd_per_mtok_input + tokens_out * caps.usd_per_mtok_output) / 1e6


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def reproduce(root: Path, frozen: Any, work: Path, isolation: str = "subprocess") -> dict[str, Any]:
    """Refit a frozen candidate from its spec and seed in a fresh process and compare hashes."""
    parts = load_task(root, frozen.family, frozen.task_id)
    runner = TrialRunner(root, work, isolation=isolation)  # type: ignore[arg-type]
    redo = runner.run(parts, frozen.spec, frozen.seed, frozen.features, "repro")
    return {
        "status": redo.status,
        "artifact_identical": redo.artifact_sha256 == frozen.artifact_sha256,
        "predictions_identical": redo.prediction_sha256 == frozen.prediction_sha256,
        "partition_identical": redo.partition_sha256 == frozen.partition_sha256,
        "artifact_sha256": redo.artifact_sha256,
        "prediction_sha256": redo.prediction_sha256,
    }


def run_evaluation(cfg: EvalConfig, cfg_sha: str, root: Path, require_live: bool = False) -> Path:
    """Run every arm on every task and write all outputs.

    Raises:
        LiveModelUnavailable: Live mode or ``require_live`` without an endpoint.
    """
    live: ModelAdapter | None = None
    if cfg.mode == "live":
        live = OpenAICompatibleAdapter.from_env(
            model=cfg.model, max_retries=cfg.adapter_max_retries, backoff_s=cfg.adapter_backoff_s
        )
    elif require_live:
        raise LiveModelUnavailable("--require-live was given but the config is in fixture mode")
    out = root / cfg.output_dir
    for sub in ("traces", "frozen"):
        shutil.rmtree(out / sub, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    work = root / cfg.work_dir / cfg.mode
    shutil.rmtree(work, ignore_errors=True)
    settings = cfg.episode
    if cfg.mode == "fixture":
        settings = settings.model_copy(update={"record_timing": False})
    scorer = load_scorer(root).HoldoutScorer()
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    trials_out: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    tok_in = tok_out = 0
    reported = 0.0
    reserve: dict[str, float] = {}  # largest per-episode cost seen so far, per arm
    p_hash = None

    def adapter(family: str, arm: str, task_id: str) -> ModelAdapter:
        if live is not None:
            return live
        path = root / cfg.fixture_scripts / f"{family}_{arm}.json"
        return FixtureAdapter.from_file(path, task_id)

    for family in cfg.families:
        task_ids = list_tasks(root, family)[: cfg.task_limit]
        for tid in task_ids:
            raw = (root / "tasks" / family / tid / "manifest.json").read_bytes()
            hashes[f"{family}/{tid}/manifest"] = hashlib.sha256(raw).hexdigest()
            hold = (root / "evaluator" / "holdout" / f"{tid}.csv").read_bytes()
            hashes[f"{family}/{tid}/holdout"] = hashlib.sha256(hold).hexdigest()
        if cfg.mode == "fixture":
            for arm in ("agent", "single_call"):
                if arm in cfg.arms:
                    path = root / cfg.fixture_scripts / f"{family}_{arm}.json"
                    hashes[f"{family}/script/{arm}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        for run in range(cfg.runs):
            for tid in task_ids:
                for arm in cfg.arms:
                    ep = f"{arm}_{tid}_run{run}"
                    ctx = make_context(
                        root,
                        family,
                        tid,
                        work / family / ep,
                        cfg.isolation,
                        settings.trial_time_limit_s,
                    )
                    if p_hash is None:
                        p_hash = prompt_hash(ToolRegistry(build_tools(ctx)), settings.max_trials)
                    spent = cost_usd(tok_in, tok_out, cfg.run_caps)
                    caps = cfg.run_caps
                    capped = (
                        caps.max_total_tokens is not None
                        and tok_in + tok_out >= caps.max_total_tokens
                    ) or (
                        caps.max_usd is not None
                        and spent is not None
                        and spent + reserve.get(arm, 0.0) >= caps.max_usd
                    )
                    if capped and arm in ("agent", "single_call"):
                        res = EpisodeResult(
                            tid, arm, run, "failed", "run_budget_exhausted", [], None, [], {}
                        )
                    elif arm == "fixed":
                        res = run_fixed(ctx, run, settings)
                    elif arm == "random_search":
                        res = run_random_search(ctx, run, settings)
                    elif arm == "single_call":
                        res = run_single_call(ctx, adapter(family, arm, tid), run, settings)
                    else:
                        ckpt = work / family / f"{ep}.ckpt.json"
                        res = run_agent(
                            ctx,
                            adapter(family, arm, tid),
                            run,
                            settings,
                            trace_path=out / "traces" / family / f"{tid}_run{run}.jsonl",
                            checkpoint_path=ckpt,
                        )
                    ep_cost = cost_usd(
                        res.info.get("prompt_tokens") or 0,
                        res.info.get("completion_tokens") or 0,
                        cfg.run_caps,
                    )
                    reserve[arm] = max(reserve.get(arm, 0.0), ep_cost or 0.0)
                    tok_in += res.info.get("prompt_tokens") or 0
                    tok_out += res.info.get("completion_tokens") or 0
                    reported += res.info.get("reported_cost_usd") or 0.0
                    rows.append(
                        score_episode(root, family, ctx, res, scorer, receipts, work, cfg, settings)
                    )
                    trials_out += [
                        {"family": family, "arm": arm, "run": run, **t.model_dump()}
                        | ({} if settings.record_timing else {"elapsed_s": None})
                        for t in res.trials
                    ]
    add_gains(rows)
    meta = {
        "mode": cfg.mode,
        "started_at": started,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(
            _git(
                root, "status", "--porcelain", "--", "src", "configs", "tasks", "evaluator", "tests"
            )
        ),
        "package_version": __version__,
        "agentcore_version": AGENTCORE_VERSION,
        "model": live.name if live else "none (scripted fixture adapter)",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": p_hash,
        "config_sha256": cfg_sha,
        "config": cfg.model_dump(),
        "input_hashes": hashes,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "tokens": {"prompt": tok_in, "completion": tok_out} if live else None,
        "usd": cost_usd(tok_in, tok_out, cfg.run_caps) if live else None,
        "usd_reported_by_provider": round(reported, 6) if live else None,
        "adapter_retries": getattr(live, "retries", None),
    }
    write_outputs(out, rows, trials_out, receipts, meta)
    return out


def score_episode(
    root: Path,
    family: str,
    ctx: Any,
    res: EpisodeResult,
    scorer: Any,
    receipts: list[dict[str, Any]],
    work: Path,
    cfg: EvalConfig,
    s: EpisodeSettings,
) -> dict[str, Any]:
    """Score one episode: holdout once, reproduction, discipline counters."""
    m = ctx.parts.manifest
    leak_cols = set(m.forbidden_columns) - {m.target, "__row_id"} - {m.group_column}
    used_forbidden = sum(bool(set(t.features_used) & set(m.forbidden_columns)) for t in res.trials)
    row: dict[str, Any] = {
        "family": family,
        "arm": res.arm,
        "task_id": res.task_id,
        "run": res.run,
        "mode": cfg.mode,
        "task_type": m.task_type,
        "metric": m.metric,
        "traps": ",".join(m.traps),
        "status": res.status,
        "stop_reason": res.stop_reason,
        "frozen": int(res.frozen is not None),
        "trials_run": len(res.trials),
        "trials_failed": sum(t.status == "failed" for t in res.trials),
        "trials_timeout": sum(t.status == "timeout" for t in res.trials),
        "forbidden_attempts": len(res.refused_columns),
        "leak_task": int(bool(leak_cols)),
        "leak_attempts": sum(c in leak_cols for c in res.refused_columns),
        "executed_forbidden": used_forbidden,
        "compute_s": round(sum(t.elapsed_s for t in res.trials), 3) if s.record_timing else None,
        **res.info,
    }
    if res.frozen is None:
        return row | {"holdout_primary": None, "dev_primary": None, "reproduced": 0}
    frozen = res.frozen
    artifact = ctx.runner.artifact_dir / f"{frozen.run_id}.pkl"
    receipt = scorer.score(frozen, m, artifact.read_bytes())
    receipts.append({"family": family, "arm": res.arm, "run": res.run, **receipt})
    rep = reproduce(root, frozen, work / "repro" / family / f"{res.arm}_{res.task_id}_{res.run}")
    path = root / cfg.output_dir / "frozen" / family / res.arm / f"{res.task_id}_run{res.run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frozen.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n")
    hm = receipt["metrics"]
    second = "macro_f1" if m.task_type == "classification" else "rmse"
    return row | {
        "frozen_run_id": frozen.run_id,
        "frozen_model": frozen.spec.model,
        "dev_primary": frozen.dev_metrics[m.metric],
        "holdout_primary": hm[m.metric],
        "holdout_secondary": hm[second],
        "dev_holdout_gap": round(hm[m.metric] - frozen.dev_metrics[m.metric], 6),
        "reproduced": int(rep["artifact_identical"] and rep["predictions_identical"]),
        "reproduced_predictions": int(rep["predictions_identical"]),
    }


def add_gains(rows: list[dict[str, Any]]) -> None:
    """Relative gain over the fixed pipeline and over random search, per task and run."""
    ref: dict[tuple[str, str, str, int], float | None] = {}
    for r in rows:
        if r["arm"] in ("fixed", "random_search"):
            ref[(r["arm"], r["family"], r["task_id"], r["run"])] = r.get("holdout_primary")
    for r in rows:
        for base in ("fixed", "random_search"):
            b = ref.get((base, r["family"], r["task_id"], r["run"]))
            a = r.get("holdout_primary")
            gain = None if (a is None or not b) else round((b - a) / b, 6)
            r[f"gain_vs_{base}"] = gain
            r[f"win_vs_{base}"] = int(gain is not None and gain > WIN)
            r[f"loss_vs_{base}"] = int(gain is None or gain < -WIN)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (family, arm) with numerators and denominators."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["family"], r["arm"]), []).append(r)
    out = []
    for (family, arm), all_rs in groups.items():
        # Episodes the run-level cap never started are pending, not failures of the arm.
        rs = [r for r in all_rs if r["stop_reason"] != "run_budget_exhausted"]
        if not rs:
            continue
        n = len(rs)
        frozen = [r for r in rs if r["frozen"]]
        gains = [r["gain_vs_fixed"] for r in frozen if r["gain_vs_fixed"] is not None]
        gaps = [r["dev_holdout_gap"] for r in frozen]
        leak = [r for r in rs if r["leak_task"]]
        row: dict[str, Any] = {
            "family": family,
            "arm": arm,
            "mode": rs[0]["mode"],
            "episodes": n,
            "runs": len({r["run"] for r in rs}),
            "pending_run_cap": len(all_rs) - len(rs),
            "frozen_k": len(frozen),
            "frozen_n": n,
            "wins_vs_fixed_k": sum(r["win_vs_fixed"] for r in rs),
            "losses_vs_fixed_k": sum(r["loss_vs_fixed"] for r in rs),
            "wins_vs_random_k": sum(r["win_vs_random_search"] for r in rs),
            "losses_vs_random_k": sum(r["loss_vs_random_search"] for r in rs),
            "mean_gain_vs_fixed": round(sum(gains) / len(gains), 4) if gains else None,
            "min_gain_vs_fixed": round(min(gains), 4) if gains else None,
            "max_gain_vs_fixed": round(max(gains), 4) if gains else None,
            "mean_dev_holdout_gap": round(sum(gaps) / len(gaps), 4) if gaps else None,
            "reproduced_k": sum(r["reproduced"] for r in frozen),
            "trials_total": sum(r["trials_run"] for r in rs),
            "trials_failed_total": sum(r["trials_failed"] + r["trials_timeout"] for r in rs),
            "leak_clean_k": sum(r["leak_attempts"] == 0 for r in leak),
            "leak_clean_n": len(leak),
            "forbidden_attempts_total": sum(r["forbidden_attempts"] for r in rs),
            "executed_forbidden_total": sum(r["executed_forbidden"] for r in rs),
            "failures": sum(r["status"] != "complete" for r in rs),
            "malformed_actions": sum(r.get("malformed_actions", 0) or 0 for r in rs),
            "model_calls_total": sum(r.get("model_calls", 0) or 0 for r in rs),
            "tool_calls_total": sum(r.get("tool_calls", 0) or 0 for r in rs),
        }
        toks = [r.get("prompt_tokens") for r in rs]
        row["tokens_total"] = (
            None
            if all(t is None for t in toks)
            else sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in rs)
        )
        costs = [r.get("reported_cost_usd") for r in rs]
        row["reported_cost_usd_total"] = (
            None if all(c is None for c in costs) else round(sum(c or 0 for c in costs), 6)
        )
        out.append(row)
    return out


def write_outputs(
    out: Path, rows: list[dict], trials: list[dict], receipts: list[dict], meta: dict
) -> None:
    """Write task_results, trial_results, scoring receipts, comparison and run_meta."""

    def jsonl(name: str, items: list[dict]) -> None:
        with (out / name).open("w", encoding="utf-8", newline="\n") as fh:
            for item in items:
                fh.write(json.dumps(item, sort_keys=True) + "\n")

    jsonl("task_results.jsonl", rows)
    jsonl("trial_results.jsonl", trials)
    stable = [{k: v for k, v in r.items() if k != "scored_at"} for r in receipts]
    if meta["mode"] == "live":
        stable = receipts
    jsonl("scoring_receipts.jsonl", stable)
    comparison = aggregate(rows)
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(comparison[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison)
    (out / "run_meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
