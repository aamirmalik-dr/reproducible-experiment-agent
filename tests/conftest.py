from __future__ import annotations

from pathlib import Path

import pytest

from reproducible_experiment_agent.agent import EpisodeSettings, make_context
from reproducible_experiment_agent.tools import ToolContext

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "tests" / "fixtures" / "scripts"


@pytest.fixture
def ctx_factory(tmp_path: Path):
    def make(family: str, task_id: str, isolation: str = "inprocess") -> ToolContext:
        return make_context(ROOT, family, task_id, tmp_path / task_id, isolation, 60.0)

    return make


@pytest.fixture
def settings() -> EpisodeSettings:
    return EpisodeSettings(record_timing=False)
