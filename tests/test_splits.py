from __future__ import annotations

import hashlib
import importlib.util
import io
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from reproducible_experiment_agent.split_service import (
    PARTITION,
    ROW_ID,
    content_key,
    load_partitions,
)
from reproducible_experiment_agent.task_registry import list_tasks, load_manifest, load_task

from .conftest import ROOT

FAMILIES = ("dev", "heldout", "heldout_v2")
ALL = [(f, t) for f in FAMILIES for t in list_tasks(ROOT, f)]


def _holdout(task_id: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "evaluator" / "holdout" / f"{task_id}.csv")


def test_families_use_disjoint_generators() -> None:
    gens = {f: {load_manifest(ROOT, f, t).generator for t in list_tasks(ROOT, f)} for f in FAMILIES}
    assert [len(list_tasks(ROOT, f)) for f in FAMILIES] == [6, 10, 10]
    assert not gens["dev"] & gens["heldout"]
    assert not (gens["dev"] | gens["heldout"]) & gens["heldout_v2"]


@pytest.mark.parametrize("family,task_id", ALL)
def test_partitions_are_disjoint_and_holdout_is_absent(family: str, task_id: str) -> None:
    parts = load_task(ROOT, family, task_id)
    hold = _holdout(task_id)
    ids = [set(parts.train[ROW_ID]), set(parts.dev[ROW_ID]), set(hold[ROW_ID])]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    raw = (ROOT / "tasks" / family / task_id / "train_dev.csv").read_text(encoding="utf-8")
    assert "holdout" not in set(pd.read_csv(io.StringIO(raw))[PARTITION])
    m = parts.manifest
    assert len(hold) == m.rows["holdout"] and len(parts.dev) == m.rows["dev"]


@pytest.mark.parametrize("family,task_id", [a for a in ALL if load_manifest(ROOT, *a).group_column])
def test_groups_never_straddle_partitions(family: str, task_id: str) -> None:
    parts = load_task(ROOT, family, task_id)
    g = parts.manifest.group_column
    sets = [set(parts.train[g]), set(parts.dev[g]), set(_holdout(task_id)[g])]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert parts.summary()["groups_shared_between_train_and_dev"] == 0


@pytest.mark.parametrize(
    "family,task_id", [a for a in ALL if "duplicates" in load_manifest(ROOT, *a).traps]
)
def test_duplicate_rows_stay_on_one_side(family: str, task_id: str) -> None:
    parts = load_task(ROOT, family, task_id)
    cols = [c for c in parts.train.columns if c != ROW_ID]
    keys = [set(content_key(df, cols)) for df in (parts.train, parts.dev, _holdout(task_id))]
    assert parts.manifest.duplicate_rows_merged > 0
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2])


def test_tampered_train_dev_is_detected(tmp_path: Path) -> None:
    src = ROOT / "tasks" / "dev" / "dev-01"
    dst = tmp_path / "dev-01"
    shutil.copytree(src, dst)
    text = (dst / "train_dev.csv").read_text(encoding="utf-8").replace(",train\n", ",dev\n", 1)
    (dst / "train_dev.csv").write_text(text, encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="does not match"):
        load_partitions(dst, load_manifest(ROOT, "dev", "dev-01"))


def test_split_summary_has_counts_not_rows() -> None:
    summary = load_task(ROOT, "heldout", "ho-07").summary()
    assert summary["holdout_rows"] > 0 and "none" in summary["holdout_access"]
    assert not any(isinstance(v, (list, dict)) for v in summary.values())


def test_task_generation_is_byte_reproducible(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("make_tasks", ROOT / "scripts" / "make_tasks.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main(tmp_path)
    v2 = importlib.util.spec_from_file_location(
        "make_tasks_v2", ROOT / "scripts" / "make_tasks_v2.py"
    )
    assert v2 and v2.loader
    module2 = importlib.util.module_from_spec(v2)
    sys.modules[v2.name] = module2
    v2.loader.exec_module(module2)
    module2.main(tmp_path)
    for family, task_id in ALL:
        for rel in (
            f"tasks/{family}/{task_id}/train_dev.csv",
            f"tasks/{family}/{task_id}/manifest.json",
            f"evaluator/holdout/{task_id}.csv",
        ):
            a = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            b = hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest()
            assert a == b, rel
