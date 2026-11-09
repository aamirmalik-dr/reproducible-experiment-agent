"""Task manifests, permitted features and the visible schema.

Each task lives in ``tasks/<family>/<task_id>/`` with a manifest and the train
and dev rows. Families never share a generator: different seeds of one
generator stay inside one family.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import ExperimentSpec, TaskManifest
from .split_service import Partitions, load_partitions


def task_dir(root: Path, family: str, task_id: str) -> Path:
    """Folder holding one task's manifest and train/dev rows."""
    return root / "tasks" / family / task_id


def load_manifest(root: Path, family: str, task_id: str) -> TaskManifest:
    """Read and validate one manifest."""
    path = task_dir(root, family, task_id) / "manifest.json"
    return TaskManifest.model_validate_json(path.read_text(encoding="utf-8"))


def list_tasks(root: Path, family: str) -> list[str]:
    """Task ids of a family, sorted."""
    base = root / "tasks" / family
    return sorted(p.name for p in base.iterdir() if (p / "manifest.json").exists())


def load_task(root: Path, family: str, task_id: str) -> Partitions:
    """Manifest plus verified train and dev rows."""
    manifest = load_manifest(root, family, task_id)
    return load_partitions(task_dir(root, family, task_id), manifest)


def permitted_features(manifest: TaskManifest) -> list[str]:
    """Feature columns the policy allows: role feature and not forbidden."""
    forbidden = set(manifest.forbidden_columns)
    return [c.name for c in manifest.columns if c.role == "feature" and c.name not in forbidden]


def resolve_features(manifest: TaskManifest, spec: ExperimentSpec) -> tuple[list[str], list[str]]:
    """Return (features to use, refused column names) for a spec."""
    if spec.features == "all_permitted":
        return permitted_features(manifest), []
    allowed = set(permitted_features(manifest))
    known = {c.name for c in manifest.columns}
    refused = [f for f in spec.features if f not in allowed]
    unknown = [f for f in refused if f not in known]
    if unknown:
        raise KeyError(f"unknown columns {unknown}")
    return [f for f in spec.features if f in allowed], refused


def visible_schema(parts: Partitions) -> dict[str, Any]:
    """What inspect_schema returns: types, roles, missingness from train rows only."""
    m = parts.manifest
    cols = []
    for c in m.columns:
        series = parts.train[c.name]
        item: dict[str, Any] = {
            "name": c.name,
            "kind": c.kind,
            "role": c.role,
            "description": c.description,
            "missing_fraction_train": round(float(series.isna().mean()), 3),
        }
        if c.kind == "categorical":
            item["levels_in_train"] = int(series.nunique(dropna=True))
        cols.append(item)
    return {
        "task_id": m.task_id,
        "task_type": m.task_type,
        "metric": m.metric,
        "metric_direction": "lower is better",
        "secondary_metric": "macro_f1" if m.task_type == "classification" else "rmse",
        "target": m.target,
        "columns": cols,
        "dataset_description": m.description,
    }


def manifest_bytes_sha256(root: Path, family: str, task_id: str) -> str:
    """Hash of the manifest file, recorded in every frozen candidate."""
    import hashlib

    raw = (task_dir(root, family, task_id) / "manifest.json").read_bytes()
    return hashlib.sha256(raw).hexdigest()


def dump_manifest(manifest: TaskManifest) -> str:
    """Canonical JSON for a manifest file."""
    return json.dumps(manifest.model_dump(), indent=1, ensure_ascii=False) + "\n"
