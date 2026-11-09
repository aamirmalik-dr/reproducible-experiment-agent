"""Episodes for every arm: fixed pipeline, bounded random search, single-call
planner and the iterative agent. All arms act through the same tools, so trial
accounting, refusals and freezing are identical across arms.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from .agentcore.adapters import AdapterError, ModelAdapter
from .agentcore.loop import AgentLoop, CheckpointStore
from .agentcore.policy import Budget, Policy, ToolRegistry
from .agentcore.schemas import Action, PendingAction, ToolResult
from .agentcore.trace import TraceWriter
from .prompts import (
    agent_system_prompt,
    render_observation,
    single_call_system_prompt,
    single_call_user,
)
from .runner import TrialRunner
from .schemas import PARAM_RANGES, ExperimentState, FrozenCandidate, TrialResult
from .task_registry import load_task, manifest_bytes_sha256, visible_schema
from .tools import ToolContext, build_tools, primary


class EpisodeSettings(BaseModel):
    """Per-episode caps shared by every arm."""

    max_trials: int = 6
    compute_seconds: float = 180.0
    trial_time_limit_s: float = 60.0
    max_steps: int = 18
    max_tool_calls: int = 14
    max_tokens: int | None = 60000
    max_wall_seconds: float | None = 900
    recent_k: int = 4
    max_completion_tokens: int = 900
    record_timing: bool = True


@dataclass
class EpisodeResult:
    """What evaluation needs from one episode."""

    task_id: str
    arm: str
    run: int
    status: str
    stop_reason: str | None
    trials: list[TrialResult]
    frozen: FrozenCandidate | None
    refused_columns: list[str]
    info: dict[str, Any] = field(default_factory=dict)


def make_context(
    root: Path, family: str, task_id: str, artifact_dir: Path, isolation: str, limit: float
) -> ToolContext:
    """Bind tools to one task, with a fresh artifact store."""
    parts = load_task(root, family, task_id)
    runner = TrialRunner(root, artifact_dir, isolation=isolation, time_limit_s=limit)  # type: ignore[arg-type]
    return ToolContext(parts, runner, manifest_bytes_sha256(root, family, task_id), limit)


def new_state(task_id: str, arm: str, run: int, s: EpisodeSettings) -> ExperimentState:
    """Fresh agent-visible state."""
    return ExperimentState(
        episode_id=f"{arm}:{task_id}:run{run}",
        task_id=task_id,
        remaining_tool_calls=s.max_tool_calls,
        remaining_trials=s.max_trials,
        remaining_compute_seconds=s.compute_seconds,
    )


def _info(state: ExperimentState, live: bool, s: EpisodeSettings, **extra: Any) -> dict[str, Any]:
    return {
        "model_calls": state.model_calls,
        "steps": state.steps,
        "tool_calls": s.max_tool_calls - state.remaining_tool_calls,
        "malformed_actions": state.malformed_actions,
        "reprompts": state.reprompts,
        "prompt_tokens": state.token_usage.prompt if live else None,
        "completion_tokens": state.token_usage.completion if live else None,
        "reported_cost_usd": round(state.reported_cost_usd, 6) if live else None,
        "latency_s": round(state.elapsed_seconds, 3) if s.record_timing else None,
        **extra,
    }


def _result(arm: str, run: int, state: ExperimentState, info: dict[str, Any]) -> EpisodeResult:
    return EpisodeResult(
        state.task_id,
        arm,
        run,
        state.status,
        state.stop_reason,
        list(state.trials),
        state.frozen,
        list(state.refused_columns),
        info,
    )


def run_agent(
    ctx: ToolContext,
    adapter: ModelAdapter,
    run: int,
    s: EpisodeSettings,
    trace_path: Path | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> EpisodeResult:
    """The iterative agent: the model picks every tool call from earlier results."""
    registry = ToolRegistry(build_tools(ctx))
    store = CheckpointStore(checkpoint_path) if checkpoint_path else None
    state = store.load(ExperimentState) if (resume and store) else None
    if state is None:
        state = new_state(ctx.parts.manifest.task_id, "agent", run, s)
    loop = AgentLoop(
        adapter,
        Policy(
            registry,
            Budget(
                max_steps=s.max_steps,
                max_tool_calls=s.max_tool_calls,
                max_tokens=s.max_tokens,
                max_wall_seconds=s.max_wall_seconds,
            ),
        ),
        agent_system_prompt(registry),
        render_observation,
        checkpoint=store,
        trace=TraceWriter(trace_path, s.record_timing) if trace_path else None,
        recent_k=s.recent_k,
        max_completion_tokens=s.max_completion_tokens,
    )
    state = loop.run(state)
    return _result("agent", run, state, _info(state, adapter.is_live, s))


def _call(policy: Policy, state: ExperimentState, tool: str, args: dict[str, Any]) -> ToolResult:
    state.steps += 1
    pending = PendingAction(
        step=state.steps,
        summary="planned",
        action=Action(tool=tool, arguments=args),
        idempotency_key=f"{state.episode_id}:{state.steps}",
    )
    result, consumed = policy.execute(state, pending)
    state.remaining_tool_calls -= int(consumed)
    return result


def run_planned(
    ctx: ToolContext,
    arm: str,
    run: int,
    s: EpisodeSettings,
    plan: list[dict[str, Any]],
    state: ExperimentState | None = None,
    live: bool = False,
) -> EpisodeResult:
    """Run a fixed list of trials, then freeze the best dev trial by rule."""
    registry = ToolRegistry(build_tools(ctx))
    policy = Policy(registry, Budget(max_steps=99, max_tool_calls=99))
    state = state or new_state(ctx.parts.manifest.task_id, arm, run, s)
    invalid = 0
    for item in plan[: s.max_trials + 4]:
        if state.remaining_trials <= 0:
            break
        res = _call(policy, state, "run_experiment", item)
        invalid += int(
            not res.ok and res.error is not None and res.error.code == "invalid_arguments"
        )
    metric = ctx.parts.manifest.metric
    ok = [t for t in state.trials if t.status == "ok"]
    if ok:
        best = min(ok, key=lambda t: (primary(t, metric), t.run_id))
        res = _call(
            policy,
            state,
            "freeze_candidate",
            {"run_id": best.run_id, "justification": "lowest dev metric (fixed selection rule)"},
        )
        if res.ok:
            state.status, state.stop_reason = "complete", "finished"
    if state.status == "running":
        state.status, state.stop_reason = "failed", "no_successful_trial"
    return _result(arm, run, state, _info(state, live, s, invalid_trial_requests=invalid))


def fixed_plan() -> list[dict[str, Any]]:
    """Task-appropriate fixed pipeline: gradient boosting defaults, all permitted features."""
    return [{"spec": {"model": "hist_gradient_boosting"}, "seed": 0}]


def random_plan(task_id: str, run: int, n: int) -> list[dict[str, Any]]:
    """Bounded random search over the allowlist, seeded by task and run."""
    seed = int(hashlib.sha256(f"{task_id}:{run}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        model = str(rng.choice(list(PARAM_RANGES)))
        params: dict[str, Any] = {}
        for name, (lo, hi) in PARAM_RANGES[model].items():
            if (model == "linear") and name != ("C" if i % 2 == 0 else "alpha"):
                continue
            if name in ("C", "alpha", "learning_rate"):
                params[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            elif name in ("l2_regularization",):
                params[name] = float(rng.uniform(lo, hi))
            else:
                params[name] = int(rng.integers(int(lo), int(hi) + 1))
        if model == "linear":
            params = {k: round(v, 6) for k, v in params.items()}
        out.append(
            {
                "spec": {
                    "model": model,
                    "params": params,
                    "numeric_imputer": str(rng.choice(["median", "mean", "zero"])),
                    "categorical_encoder": str(rng.choice(["onehot", "ordinal"])),
                    "scaler": str(rng.choice(["standard", "robust", "none"])),
                },
                "seed": int(rng.integers(0, 1000)),
            }
        )
    return out


def _linear_param_fix(plan: list[dict[str, Any]], task_type: str) -> list[dict[str, Any]]:
    """Keep only the linear parameter that applies to the task type."""
    keep = "C" if task_type == "classification" else "alpha"
    for item in plan:
        spec = item["spec"]
        if spec["model"] == "linear":
            value = next(iter(spec["params"].values()), 1.0)
            spec["params"] = {keep: value}
    return plan


def run_random_search(ctx: ToolContext, run: int, s: EpisodeSettings) -> EpisodeResult:
    """Bounded random search with the same trial budget as the agent."""
    m = ctx.parts.manifest
    plan = _linear_param_fix(random_plan(m.task_id, run, s.max_trials), m.task_type)
    return run_planned(ctx, "random_search", run, s, plan)


def run_fixed(ctx: ToolContext, run: int, s: EpisodeSettings) -> EpisodeResult:
    """The fixed pipeline baseline (one trial)."""
    return run_planned(ctx, "fixed", run, s, fixed_plan())


def parse_trials(
    text: str, max_chars: int = 8000
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Parse the single-call reply into a list of {spec, seed} items."""
    if len(text) > max_chars:
        return None, f"reply was {len(text)} characters, the limit is {max_chars}"
    start = text.find("{")
    if start < 0:
        return None, "no JSON object found"
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg} at position {exc.pos}"
    trials = obj.get("trials") if isinstance(obj, dict) else None
    if not isinstance(trials, list) or not trials:
        return None, 'expected a non-empty "trials" list'
    if not all(isinstance(t, dict) and "spec" in t for t in trials):
        return None, 'every trial needs a "spec" object'
    return trials, None


def run_single_call(
    ctx: ToolContext, adapter: ModelAdapter, run: int, s: EpisodeSettings
) -> EpisodeResult:
    """One model reply proposing up to max_trials specs; best dev trial frozen by rule."""
    state = new_state(ctx.parts.manifest.task_id, "single_call", run, s)
    messages = [
        {"role": "system", "content": single_call_system_prompt(s.max_trials)},
        {
            "role": "user",
            "content": single_call_user(visible_schema(ctx.parts), ctx.parts.summary()),
        },
    ]
    trials = None
    for attempt in range(2):
        try:
            resp = adapter.complete(messages, s.max_completion_tokens * 2)
        except AdapterError:
            state.status, state.stop_reason = "failed", "adapter_error"
            return _result("single_call", run, state, _info(state, adapter.is_live, s))
        state.model_calls += 1
        state.token_usage.prompt += resp.prompt_tokens or 0
        state.token_usage.completion += resp.completion_tokens or 0
        state.reported_cost_usd += resp.cost_usd or 0.0
        trials, error = parse_trials(resp.text)
        if trials is not None:
            break
        state.reprompts += int(attempt == 0)
        messages += [
            {"role": "assistant", "content": resp.text[:8000]},
            {"role": "user", "content": f"Your reply could not be used: {error}. Try again."},
        ]
    if trials is None:
        state.malformed_actions += 1
        state.status, state.stop_reason = "failed", "malformed_output"
        return _result("single_call", run, state, _info(state, adapter.is_live, s))
    plan = [{"spec": t.get("spec"), "seed": t.get("seed", 0)} for t in trials]
    return run_planned(ctx, "single_call", run, s, plan, state=state, live=adapter.is_live)
