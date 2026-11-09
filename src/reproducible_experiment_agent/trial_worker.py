"""Fit one trial on training rows and score it on dev rows.

Runs either in-process (tests) or as ``python -m
reproducible_experiment_agent.trial_worker <job.json>`` inside an isolated
temporary directory that holds a copy of the task files. It never sees
holdout rows: they are not in the task folder.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, log_loss, mean_absolute_error, mean_squared_error

from .pipeline_factory import build_pipeline
from .schemas import ExperimentSpec, TaskManifest
from .split_service import ROW_ID, Partitions, load_partitions


def prediction_hash(values: np.ndarray) -> str:
    """Hash of predictions rounded to 12 decimals (stable across processes)."""
    return hashlib.sha256(np.round(np.asarray(values, dtype=float), 12).tobytes()).hexdigest()


def predict_for_metric(pipe: Any, task_type: str, X: Any) -> np.ndarray:
    """Positive-class probability for classification, values for regression."""
    if task_type == "classification":
        return pipe.predict_proba(X)
    return pipe.predict(X)


def score(task_type: str, y_true: Any, pred: np.ndarray, classes: Any = None) -> dict[str, float]:
    """Primary and secondary metrics. Lower is better for log loss, MAE and RMSE."""
    if task_type == "classification":
        labels = list(classes)
        hard = np.asarray(labels)[np.argmax(pred, axis=1)]
        return {
            "log_loss": float(log_loss(y_true, pred, labels=labels)),
            "macro_f1": float(f1_score(y_true, hard, average="macro")),
            "accuracy": float(np.mean(hard == np.asarray(y_true))),
        }
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
    }


def fit_trial(
    parts: Partitions, spec: ExperimentSpec, seed: int, features: list[str]
) -> tuple[dict[str, Any], bytes | None]:
    """Fit on train rows, score on dev rows.

    Returns:
        A result dict (status, metrics, hashes) and the pickled pipeline, or
        None for a failed fit. Failures are reported, never raised.
    """
    m = parts.manifest
    kinds = {c.name: c.kind for c in m.columns}
    numeric = [f for f in features if kinds[f] == "numeric"]
    categorical = [f for f in features if kinds[f] == "categorical"]
    start = time.perf_counter()
    try:
        pipe = build_pipeline(spec, m.task_type, numeric, categorical, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(parts.train[features], parts.train[m.target])
            pred = predict_for_metric(pipe, m.task_type, parts.dev[features])
        classes = getattr(pipe, "classes_", None)
        metrics = score(m.task_type, parts.dev[m.target], pred, classes)
        if not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError("non-finite dev metric")
    except Exception as exc:  # noqa: BLE001, a failed fit is a recorded trial
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            "elapsed_s": round(time.perf_counter() - start, 3),
        }, None
    blob = pickle.dumps(pipe, protocol=4)
    return {
        "status": "ok",
        "dev_metrics": {k: round(v, 6) for k, v in metrics.items()},
        "train_rows": len(parts.train),
        "dev_rows": len(parts.dev),
        "artifact_sha256": hashlib.sha256(blob).hexdigest(),
        "prediction_sha256": prediction_hash(pred),
        "elapsed_s": round(time.perf_counter() - start, 3),
    }, blob


def main(job_path: str) -> int:
    """Subprocess entry point: read a job file, write result.json and artifact.pkl."""
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    work = Path(job_path).parent
    manifest = TaskManifest.model_validate(job["manifest"])
    parts = load_partitions(work, manifest)
    if ROW_ID in job["features"]:
        raise SystemExit("row id is never a feature")
    result, blob = fit_trial(
        parts, ExperimentSpec.model_validate(job["spec"]), job["seed"], job["features"]
    )
    if blob is not None:
        (work / "artifact.pkl").write_bytes(blob)
    (work / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
