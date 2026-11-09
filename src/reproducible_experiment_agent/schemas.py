"""Typed records: task manifests, experiment specs, trial results, frozen candidates.

Two meanings of "test" never meet in this code base. Task families (``dev`` and
``heldout``) partition the benchmark tasks used to evaluate the agent. Row
partitions (``train``, ``dev`` and ``holdout``) split the rows inside one ML
task. Row partitions are always called partitions, never splits of tasks.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agentcore.schemas import BaseAgentState

TaskType = Literal["classification", "regression"]
ModelName = Literal["linear", "random_forest", "hist_gradient_boosting"]

PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "linear": {"C": (1e-4, 1e4), "alpha": (1e-4, 1e4)},
    "random_forest": {
        "n_estimators": (10, 200),
        "max_depth": (1, 20),
        "min_samples_leaf": (1, 50),
    },
    "hist_gradient_boosting": {
        "learning_rate": (0.01, 1.0),
        "max_iter": (20, 300),
        "max_leaf_nodes": (4, 64),
        "l2_regularization": (0.0, 10.0),
        "min_samples_leaf": (1, 50),
    },
}
INT_PARAMS = {"n_estimators", "max_depth", "min_samples_leaf", "max_iter", "max_leaf_nodes"}


class Column(BaseModel):
    """One column of a task table as described in its manifest."""

    name: str
    kind: Literal["numeric", "categorical"]
    role: Literal["feature", "target", "id", "group"]
    description: str


class TaskManifest(BaseModel):
    """Everything about one ML task that the trusted services know.

    ``forbidden_columns`` is policy: the run_experiment tool refuses specs that
    select them. It is never shown to the model as a list; the schema tool shows
    only column descriptions.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    family: Literal["dev", "heldout"]
    generator: str
    seed: int
    task_type: TaskType
    metric: Literal["log_loss", "mae"]
    target: str
    columns: list[Column]
    forbidden_columns: list[str]
    traps: list[str] = Field(default_factory=list)
    description: str
    partition_strategy: Literal["random", "group"]
    group_column: str | None = None
    rows: dict[str, int]
    duplicate_rows_merged: int = 0
    train_dev_sha256: str


class ExperimentSpec(BaseModel):
    """A typed trial request from the finite allowlist. No code, no paths."""

    model_config = ConfigDict(extra="forbid")

    features: list[str] | Literal["all_permitted"] = "all_permitted"
    numeric_imputer: Literal["median", "mean", "zero"] = "median"
    categorical_encoder: Literal["onehot", "ordinal"] = "onehot"
    scaler: Literal["standard", "robust", "none"] = "standard"
    model: ModelName = "hist_gradient_boosting"
    params: dict[str, float | int | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> ExperimentSpec:
        allowed = PARAM_RANGES[self.model]
        for name, value in self.params.items():
            if name not in allowed:
                raise ValueError(f"parameter {name!r} is not allowed for {self.model}")
            if value is None:
                if name != "max_depth":
                    raise ValueError(f"parameter {name!r} cannot be null")
                continue
            lo, hi = allowed[name]
            if not lo <= float(value) <= hi:
                raise ValueError(f"{name}={value} outside [{lo}, {hi}]")
            if name in INT_PARAMS and float(value) != int(value):
                raise ValueError(f"{name} must be an integer")
        if isinstance(self.features, list) and not self.features:
            raise ValueError("features must be a non-empty list or 'all_permitted'")
        return self


class TrialResult(BaseModel):
    """One trial, successful or not. Failures are trials too."""

    run_id: str
    task_id: str
    spec: ExperimentSpec
    seed: int
    status: Literal["ok", "failed", "timeout"]
    error: str | None = None
    dev_metrics: dict[str, float] = Field(default_factory=dict)
    train_rows: int = 0
    dev_rows: int = 0
    features_used: list[str] = Field(default_factory=list)
    artifact_sha256: str | None = None
    prediction_sha256: str | None = None
    elapsed_s: float = 0.0
    partition_sha256: str


class FrozenCandidate(BaseModel):
    """The immutable selection record. Once written, no further trials run."""

    task_id: str
    family: str
    episode_id: str
    run_id: str
    features: list[str]
    spec: ExperimentSpec
    seed: int
    dev_metrics: dict[str, float]
    artifact_sha256: str
    prediction_sha256: str
    partition_sha256: str
    train_dev_sha256: str
    package_version: str
    justification: str
    manifest_sha256: str = ""

    def body_sha256(self) -> str:
        """Hash of every field except the hash itself."""
        import hashlib

        payload = self.model_dump_json(exclude={"manifest_sha256"})
        return hashlib.sha256(payload.encode()).hexdigest()


class ExperimentState(BaseAgentState):
    """Agent-visible state. Holdout data and forbidden-column policy are never here."""

    task_id: str
    trial_ids: list[str] = Field(default_factory=list)
    remaining_trials: int
    remaining_compute_seconds: float
    frozen_candidate_id: str | None = None
    refused_columns: list[str] = Field(default_factory=list)
    trials: list[TrialResult] = Field(default_factory=list)
    frozen: FrozenCandidate | None = None

    def observable(self) -> dict[str, Any]:
        """Compact summary shown to the model each step."""
        return {
            "task_id": self.task_id,
            "steps_used": self.steps,
            "tool_calls_left": self.remaining_tool_calls,
            "trials_left": self.remaining_trials,
            "compute_seconds_left": round(self.remaining_compute_seconds, 1),
            "trials_so_far": self.trial_ids,
        }
