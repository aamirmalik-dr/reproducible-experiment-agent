"""The six experiment tools. All are finite, typed operations.

No tool reads holdout rows, imports the evaluator, writes outside the
artifact store, runs model-written code or touches the network. Selection ends
with ``freeze_candidate``; after it, the episode is over and no trial can run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .agentcore.policy import ToolFailure, ToolSpec
from .runner import TrialRunner
from .schemas import ExperimentSpec, ExperimentState, FrozenCandidate, TrialResult
from .split_service import Partitions
from .task_registry import resolve_features, visible_schema

ALL_TOOLS = (
    "inspect_schema",
    "split_manifest",
    "run_experiment",
    "compare_dev_runs",
    "check_artifacts",
    "freeze_candidate",
)
REFUSAL = (
    "not permitted as features: {cols}. Identifiers, the target, group identifiers and "
    "columns recorded after the outcome are refused by policy."
)


@dataclass
class ToolContext:
    """What the tools of one episode are bound to."""

    parts: Partitions
    runner: TrialRunner
    manifest_sha256: str
    time_limit_s: float = 60.0


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(_Args):
    pass


class RunArgs(_Args):
    spec: ExperimentSpec
    seed: int = Field(default=0, ge=0, le=1_000_000)


class RunIdArgs(_Args):
    run_id: str = Field(min_length=1, max_length=20)


class FreezeArgs(_Args):
    run_id: str = Field(min_length=1, max_length=20)
    justification: str = Field(min_length=1, max_length=400)


def primary(trial: TrialResult, metric: str) -> float:
    """Primary dev metric of a trial (lower is better); inf for failures."""
    return trial.dev_metrics.get(metric, float("inf")) if trial.status == "ok" else float("inf")


def trial_view(t: TrialResult) -> dict[str, Any]:
    """Compact trial record shown to the model."""
    out: dict[str, Any] = {
        "run_id": t.run_id,
        "status": t.status,
        "spec": t.spec.model_dump(),
        "seed": t.seed,
        "n_features_used": len(t.features_used),
        "dev_metrics": t.dev_metrics,
        "elapsed_s": t.elapsed_s,
    }
    if t.error:
        out["error"] = t.error
    return out


def build_tools(ctx: ToolContext) -> list[ToolSpec]:
    """Tool specs bound to one task episode."""
    m = ctx.parts.manifest

    def inspect_schema(state: ExperimentState, a: NoArgs) -> dict[str, Any]:
        return visible_schema(ctx.parts) | {"trials_left": state.remaining_trials}

    def split_manifest(state: ExperimentState, a: NoArgs) -> dict[str, Any]:
        return ctx.parts.summary()

    def run_experiment(state: ExperimentState, a: RunArgs) -> dict[str, Any]:
        if state.frozen is not None:
            raise ToolFailure("precondition_failed", "selection is frozen; no further trials")
        if state.remaining_trials <= 0 or state.remaining_compute_seconds <= 0:
            raise ToolFailure("budget_exhausted", "no trials or compute seconds left; freeze now")
        try:
            features, refused = resolve_features(m, a.spec)
        except KeyError as exc:
            raise ToolFailure("not_found", str(exc.args[0])) from None
        if refused:
            state.refused_columns.extend(refused)
            raise ToolFailure("precondition_failed", REFUSAL.format(cols=", ".join(refused)))
        run_id = f"t{len(state.trials) + 1:02d}"
        result = ctx.runner.run(
            ctx.parts,
            a.spec,
            a.seed,
            features,
            run_id,
            time_limit_s=min(ctx.time_limit_s, state.remaining_compute_seconds),
        )
        state.trials.append(result)
        state.trial_ids.append(run_id)
        state.remaining_trials -= 1
        state.remaining_compute_seconds -= result.elapsed_s
        return trial_view(result) | {"trials_left": state.remaining_trials}

    def compare_dev_runs(state: ExperimentState, a: NoArgs) -> dict[str, Any]:
        ranked = sorted(state.trials, key=lambda t: (primary(t, m.metric), t.run_id))
        return {
            "metric": m.metric,
            "direction": "lower is better",
            "trials": [trial_view(t) for t in ranked],
        }

    def _trial(state: ExperimentState, run_id: str) -> TrialResult:
        found = next((t for t in state.trials if t.run_id == run_id), None)
        if found is None:
            raise ToolFailure("not_found", f"no trial {run_id!r} in this episode")
        return found

    def check_artifacts(state: ExperimentState, a: RunIdArgs) -> dict[str, Any]:
        t = _trial(state, a.run_id)
        stored = ctx.runner.artifact_sha256(t.run_id)
        return {
            "run_id": t.run_id,
            "status": t.status,
            "artifact_present": stored is not None,
            "artifact_hash_matches": stored is not None and stored == t.artifact_sha256,
            "partition_hash_matches": t.partition_sha256 == ctx.parts.sha256,
            "artifact_sha256": (t.artifact_sha256 or "")[:16],
            "prediction_sha256": (t.prediction_sha256 or "")[:16],
        }

    def check_freeze(state: ExperimentState, a: FreezeArgs) -> None:
        t = _trial(state, a.run_id)
        if t.status != "ok":
            raise ToolFailure("precondition_failed", f"trial {t.run_id} did not succeed")
        if ctx.runner.artifact_sha256(t.run_id) != t.artifact_sha256:
            raise ToolFailure("precondition_failed", "stored artifact does not match its hash")

    def freeze_candidate(state: ExperimentState, a: FreezeArgs) -> dict[str, Any]:
        t = _trial(state, a.run_id)
        frozen = FrozenCandidate(
            task_id=m.task_id,
            family=m.family,
            episode_id=state.episode_id,
            run_id=t.run_id,
            features=t.features_used,
            spec=t.spec,
            seed=t.seed,
            dev_metrics=t.dev_metrics,
            artifact_sha256=t.artifact_sha256 or "",
            prediction_sha256=t.prediction_sha256 or "",
            partition_sha256=t.partition_sha256,
            train_dev_sha256=m.train_dev_sha256,
            package_version=__version__,
            justification=a.justification,
        )
        frozen.manifest_sha256 = frozen.body_sha256()
        state.frozen = frozen
        state.frozen_candidate_id = t.run_id
        return {"frozen": t.run_id, "manifest_sha256": frozen.manifest_sha256[:16]}

    return [
        ToolSpec(
            "inspect_schema",
            "Columns, types, roles, missingness, metric and the " "dataset description.",
            NoArgs,
            inspect_schema,
        ),
        ToolSpec(
            "split_manifest",
            "Row-partition strategy, counts and hashes (no holdout rows).",
            NoArgs,
            split_manifest,
        ),
        ToolSpec(
            "run_experiment",
            "Fit one allowlisted pipeline on train rows and score it on "
            "dev rows. Uses one trial.",
            RunArgs,
            run_experiment,
            side_effect=True,
        ),
        ToolSpec(
            "compare_dev_runs",
            "All trials so far, ranked by the dev metric.",
            NoArgs,
            compare_dev_runs,
        ),
        ToolSpec(
            "check_artifacts",
            "Verify a trial's stored artifact and partition hashes.",
            RunIdArgs,
            check_artifacts,
        ),
        ToolSpec(
            "freeze_candidate",
            "End selection with one successful trial. The holdout is "
            "scored once, later, outside your control.",
            FreezeArgs,
            freeze_candidate,
            precheck=check_freeze,
            terminal="complete",
        ),
    ]
