from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproducible_experiment_agent.__main__ import main
from reproducible_experiment_agent.evaluation import add_gains, load_config, run_evaluation
from reproducible_experiment_agent.report import build_evidence, results_table, scorer_table

from .conftest import ROOT


def test_small_fixture_evaluation_runs_and_is_deterministic(tmp_path: Path) -> None:
    cfg, sha = load_config(ROOT / "configs" / "experiment_fixture.yaml")
    outs = []
    for i in range(2):
        c = cfg.model_copy(
            update={
                "families": ["dev"],
                "task_limit": 2,
                "output_dir": str(tmp_path / f"out{i}"),
                "work_dir": str(tmp_path / f"work{i}"),
            }
        )
        out = run_evaluation(c, sha, ROOT)
        outs.append((out / "task_results.jsonl").read_text(encoding="utf-8"))
    assert outs[0] == outs[1]
    rows = [json.loads(x) for x in outs[0].splitlines()]
    assert {r["arm"] for r in rows} == {"fixed", "random_search", "single_call", "agent"}
    assert all(r["executed_forbidden"] == 0 for r in rows)
    assert all(r["reproduced"] == 1 for r in rows if r["frozen"])
    fixed = [r for r in rows if r["arm"] == "fixed"]
    assert all(r["gain_vs_fixed"] == 0.0 for r in fixed)


def test_gain_and_failure_accounting() -> None:
    rows = [
        {"arm": "fixed", "family": "d", "task_id": "t", "run": 0, "holdout_primary": 1.0},
        {"arm": "random_search", "family": "d", "task_id": "t", "run": 0, "holdout_primary": 0.9},
        {"arm": "agent", "family": "d", "task_id": "t", "run": 0, "holdout_primary": 0.95},
        {"arm": "single_call", "family": "d", "task_id": "t", "run": 0, "holdout_primary": None},
    ]
    add_gains(rows)
    agent, single = rows[2], rows[3]
    assert agent["gain_vs_fixed"] == pytest.approx(0.05) and agent["win_vs_fixed"] == 1
    assert agent["loss_vs_random_search"] == 1
    assert single["win_vs_fixed"] == 0 and single["loss_vs_fixed"] == 1


def test_require_live_fails_clearly(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    for var in ("AGENT_MODEL_BASE_URL", "AGENT_MODEL_NAME", "AGENT_MODEL_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(ROOT)
    meta = ROOT / "reports" / "live" / "run_meta.json"
    before = meta.read_bytes() if meta.exists() else None
    assert main(["evaluate", "--config", "configs/experiment_fixture.yaml", "--require-live"]) == 2
    assert main(["evaluate", "--config", "configs/experiment_live.yaml", "--require-live"]) == 2
    assert "fixture fallback is disabled" in " ".join(capsys.readouterr().out.split())
    assert (meta.read_bytes() if meta.exists() else None) == before


def test_headline_table_holds_only_real_measurements() -> None:
    ev = build_evidence(ROOT)
    head, scorer = results_table(ev), scorer_table(ev)
    assert "fixture, scripted" not in head
    if ev["modes"]["fixture"] != "pending":
        assert "| no model, fixture run |" in head and "fixture, scripted" in scorer


def test_committed_frozen_candidate_reproduces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    assert main(["reproduce", "--manifest", "reports/frozen_candidate.json"]) == 0


def test_live_configs_carry_adapter_retry_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from reproducible_experiment_agent.agentcore.adapters import OpenAICompatibleAdapter

    cfg, _ = load_config(ROOT / "configs" / "experiment_live_v2_sonnet46.yaml")
    assert (cfg.adapter_max_retries, cfg.adapter_backoff_s) == (6, 5.0)
    assert load_config(ROOT / "configs" / "experiment_live.yaml")[0].adapter_max_retries == 2
    monkeypatch.setenv("AGENT_MODEL_BASE_URL", "http://x/v1")
    ad = OpenAICompatibleAdapter.from_env(
        model=cfg.model, max_retries=cfg.adapter_max_retries, backoff_s=cfg.adapter_backoff_s
    )
    assert (ad.max_retries, ad.backoff_s, ad.name) == (6, 5.0, "anthropic/claude-sonnet-4.6")
