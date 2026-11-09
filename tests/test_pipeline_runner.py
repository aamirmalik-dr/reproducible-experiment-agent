from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from reproducible_experiment_agent.pipeline_factory import build_pipeline
from reproducible_experiment_agent.runner import TrialRunner
from reproducible_experiment_agent.schemas import ExperimentSpec
from reproducible_experiment_agent.task_registry import (
    load_task,
    permitted_features,
    resolve_features,
)

from .conftest import ROOT


def test_imputer_and_scaler_are_fit_on_train_rows_only() -> None:
    parts = load_task(ROOT, "dev", "dev-01")
    feats = ["cure_temp_c", "humidity_pct"]
    pipe = build_pipeline(ExperimentSpec(model="linear"), "classification", feats, [], 0)
    pipe.fit(parts.train[feats], parts.train["defect"])
    num = pipe.named_steps["pre"].named_transformers_["num"]
    assert np.allclose(num.named_steps["impute"].statistics_, parts.train[feats].median())
    assert np.allclose(
        num.named_steps["scale"].mean_,
        parts.train[feats].fillna(parts.train[feats].median()).mean(),
    )
    both = parts.train[feats].median() != parts.dev[feats].median()
    assert both.any()


def test_unseen_categories_and_missing_values_are_handled() -> None:
    parts = load_task(ROOT, "heldout", "ho-03")
    dev = parts.dev.copy()
    dev.loc[:4, "furnace"] = "f99"
    dev.loc[5:9, "furnace"] = np.nan
    dev.loc[10:14, "quench_temp"] = np.nan
    assert "f99" not in set(parts.train["furnace"])
    for enc in ("onehot", "ordinal"):
        spec = ExperimentSpec(model="linear", categorical_encoder=enc)
        pipe = build_pipeline(spec, "regression", ["quench_temp", "cool_rate"], ["furnace"], 0)
        pipe.fit(parts.train, parts.train["hardness_hv"])
        assert np.isfinite(pipe.predict(dev)).all()


@pytest.mark.parametrize(
    "spec",
    [
        {"model": "random_forest", "params": {"max_depth": 99}},
        {"model": "linear", "params": {"n_estimators": 10}},
        {"model": "hist_gradient_boosting", "params": {"max_iter": 12.5}},
        {"model": "svm"},
        {"model": "linear", "features": []},
        {"model": "linear", "code": "import os"},
    ],
)
def test_illegal_specs_are_rejected(spec: dict) -> None:
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(spec)


def test_policy_refuses_target_id_group_and_leak_columns() -> None:
    m = load_task(ROOT, "dev", "dev-02").manifest
    assert "review_flag" not in permitted_features(m) and "defect" not in permitted_features(m)
    for col in ("review_flag", "defect", "__row_id"):
        used, refused = resolve_features(m, ExperimentSpec(features=[col, "dwell_min"]))
        assert used == ["dwell_min"] and refused == [col]
    with pytest.raises(KeyError):
        resolve_features(m, ExperimentSpec(features=["no_such_column"]))
    g = load_task(ROOT, "dev", "dev-05").manifest
    assert resolve_features(g, ExperimentSpec(features=["site"]))[1] == ["site"]


def test_failed_fit_is_a_recorded_trial(tmp_path: Path) -> None:
    parts = load_task(ROOT, "dev", "dev-04")
    runner = TrialRunner(ROOT, tmp_path, isolation="inprocess")
    res = runner.run(parts, ExperimentSpec(model="linear"), 0, ["spare_sensor"], "t01")
    assert res.status == "failed" and res.error and res.artifact_sha256 is None
    assert not (tmp_path / "t01.pkl").exists()


def test_subprocess_trial_matches_in_process_byte_for_byte(tmp_path: Path) -> None:
    parts = load_task(ROOT, "heldout", "ho-01")
    spec = ExperimentSpec(model="random_forest", params={"n_estimators": 30})
    feats = permitted_features(parts.manifest)
    a = TrialRunner(ROOT, tmp_path / "a", isolation="inprocess").run(parts, spec, 3, feats, "t1")
    b = TrialRunner(ROOT, tmp_path / "b", isolation="subprocess").run(parts, spec, 3, feats, "t1")
    assert a.status == b.status == "ok"
    assert a.artifact_sha256 == b.artifact_sha256 and a.prediction_sha256 == b.prediction_sha256
    assert a.dev_metrics == b.dev_metrics


def test_runner_kills_a_trial_over_its_time_limit(tmp_path: Path) -> None:
    parts = load_task(ROOT, "dev", "dev-03")
    runner = TrialRunner(ROOT, tmp_path, isolation="subprocess", time_limit_s=0.05)
    res = runner.run(parts, ExperimentSpec(), 0, permitted_features(parts.manifest), "t01")
    assert res.status == "timeout" and "exceeded" in (res.error or "")
