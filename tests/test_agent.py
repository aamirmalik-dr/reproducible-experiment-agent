from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from reproducible_experiment_agent.agent import (
    random_plan,
    run_agent,
    run_fixed,
    run_random_search,
    run_single_call,
)
from reproducible_experiment_agent.agentcore.adapters import FixtureAdapter
from reproducible_experiment_agent.agentcore.policy import ToolRegistry
from reproducible_experiment_agent.evaluation import load_scorer
from reproducible_experiment_agent.schemas import ExperimentState
from reproducible_experiment_agent.tools import ALL_TOOLS, build_tools

from .conftest import ROOT, SCRIPTS

PKG = ROOT / "src" / "reproducible_experiment_agent"


def script(family: str, arm: str, task_id: str) -> FixtureAdapter:
    return FixtureAdapter.from_file(SCRIPTS / f"{family}_{arm}.json", task_id)


def test_agent_refuses_leak_ignores_hostile_text_and_freezes(ctx_factory, settings) -> None:
    ctx = ctx_factory("dev", "dev-02")
    res = run_agent(ctx, script("dev", "agent", "dev-02"), 0, settings)
    assert res.status == "complete" and res.frozen is not None
    assert res.refused_columns == ["review_flag"]
    assert all("review_flag" not in t.features_used for t in res.trials)
    assert "id_rsa" in ctx.parts.manifest.description
    assert {s.name for s in build_tools(ctx)} == set(ALL_TOOLS)


def test_failed_trial_counts_and_freeze_needs_a_successful_trial(ctx_factory, settings) -> None:
    res = run_agent(ctx_factory("dev", "dev-04"), script("dev", "agent", "dev-04"), 0, settings)
    assert [t.status for t in res.trials][0] == "failed"
    assert res.frozen is not None and res.frozen.run_id == "t02"


def test_prose_reply_gets_one_reprompt(ctx_factory, settings, tmp_path: Path) -> None:
    res = run_agent(ctx_factory("dev", "dev-05"), script("dev", "agent", "dev-05"), 0, settings)
    assert res.info["reprompts"] == 1 and res.info["malformed_actions"] == 0
    assert res.status == "complete"


def test_episode_that_never_freezes_is_a_failure(ctx_factory, settings) -> None:
    res = run_agent(ctx_factory("dev", "dev-06"), script("dev", "agent", "dev-06"), 0, settings)
    assert res.frozen is None and res.status == "stopped"
    assert len(res.trials) == settings.max_trials


def test_no_trial_after_freeze(ctx_factory, settings) -> None:
    ctx = ctx_factory("dev", "dev-01")
    res = run_agent(ctx, script("dev", "agent", "dev-01"), 0, settings)
    state = ExperimentState(
        episode_id="x",
        task_id="dev-01",
        remaining_tool_calls=5,
        remaining_trials=5,
        remaining_compute_seconds=100,
        frozen=res.frozen,
    )
    from reproducible_experiment_agent.agentcore.policy import Budget, Policy
    from reproducible_experiment_agent.agentcore.schemas import Action, PendingAction

    policy = Policy(ToolRegistry(build_tools(ctx)), Budget())
    pending = PendingAction(
        step=1,
        summary="s",
        idempotency_key="k",
        action=Action(tool="run_experiment", arguments={"spec": {}}),
    )
    result, _ = policy.execute(state, pending)
    assert not result.ok and "frozen" in result.error.message


class Crash(Exception):
    pass


class CrashingAdapter(FixtureAdapter):
    """Raises a non-adapter exception after n replies, like a killed process."""

    def __init__(self, responses: list, n: int) -> None:
        super().__init__(responses)
        self.n = n

    def complete(self, messages, max_tokens):
        if self.cursor >= self.n:
            raise Crash()
        return super().complete(messages, max_tokens)


def test_resume_does_not_refit_completed_trials(ctx_factory, settings, tmp_path: Path) -> None:
    replies = json.loads((SCRIPTS / "dev_agent.json").read_text())["dev-01"]
    ctx = ctx_factory("dev", "dev-01")
    fits = []
    original = ctx.runner.run

    def counting(*a, **k):
        fits.append(a[4] if len(a) > 4 else k.get("run_id"))
        return original(*a, **k)

    ctx.runner.run = counting  # type: ignore[method-assign]
    ckpt = tmp_path / "ep.ckpt.json"
    with pytest.raises(Crash):
        run_agent(ctx, CrashingAdapter(replies, 4), 0, settings, checkpoint_path=ckpt)
    assert fits == ["t01", "t02"]
    res = run_agent(ctx, FixtureAdapter(replies), 0, settings, checkpoint_path=ckpt, resume=True)
    assert fits == ["t01", "t02", "t03"] and res.frozen is not None
    assert [t.run_id for t in res.trials] == ["t01", "t02", "t03"]


def test_holdout_scorer_is_one_shot_and_unreachable_from_tools(ctx_factory, settings) -> None:
    ctx = ctx_factory("dev", "dev-01")
    res = run_agent(ctx, script("dev", "agent", "dev-01"), 0, settings)
    scorer = load_scorer(ROOT).HoldoutScorer()
    blob = (ctx.runner.artifact_dir / f"{res.frozen.run_id}.pkl").read_bytes()
    receipt = scorer.score(res.frozen, ctx.parts.manifest, blob)
    assert receipt["holdout_rows"] == ctx.parts.manifest.rows["holdout"]
    with pytest.raises(Exception, match=res.frozen.manifest_sha256[:16]):
        scorer.score(res.frozen, ctx.parts.manifest, blob)
    tampered = res.frozen.model_copy(update={"seed": 99})
    with pytest.raises(ValueError):
        scorer.score(tampered, ctx.parts.manifest, blob)
    for path in PKG.rglob("*.py"):
        if path.name == "evaluation.py":
            continue
        for literal in code_strings(path):
            assert not re.search(r"holdout_scorer|evaluator|holdout/", literal), (
                path.name,
                literal,
            )


def code_strings(path: Path) -> list[str]:
    """Imported module names and string literals, excluding docstrings."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        first = body[0] if isinstance(body, list) and body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            docs.add(id(first.value))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            out.append(node.module or "")
        elif (
            isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs
        ):
            out.append(node.value)
    return out


def test_tool_outputs_never_contain_holdout_row_ids(ctx_factory, settings) -> None:
    import pandas as pd

    ctx = ctx_factory("heldout", "ho-07")
    res = run_agent(ctx, script("heldout", "agent", "ho-07"), 0, settings)
    assert res.frozen is not None
    hold_ids = set(pd.read_csv(ROOT / "evaluator" / "holdout" / "ho-07.csv")["__row_id"])
    assert hold_ids.isdisjoint(set(ctx.parts.train["__row_id"]) | set(ctx.parts.dev["__row_id"]))


def test_baselines_and_single_call(ctx_factory, settings) -> None:
    fixed = run_fixed(ctx_factory("dev", "dev-03"), 0, settings)
    assert fixed.status == "complete" and len(fixed.trials) == 1
    rs = run_random_search(ctx_factory("dev", "dev-03"), 0, settings)
    assert len(rs.trials) == settings.max_trials and rs.frozen is not None
    assert random_plan("dev-03", 0, 6) == random_plan("dev-03", 0, 6)
    assert random_plan("dev-03", 0, 6) != random_plan("dev-03", 1, 6)
    bad = run_single_call(
        ctx_factory("dev", "dev-03"), script("dev", "single_call", "dev-03"), 0, settings
    )
    assert bad.status == "failed" and bad.stop_reason == "malformed_output"
    leak = run_single_call(
        ctx_factory("heldout", "ho-02"), script("heldout", "single_call", "ho-02"), 0, settings
    )
    assert leak.refused_columns == ["review_flag"] and leak.frozen is not None
