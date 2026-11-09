"""One-shot holdout scorer. Evaluator-only: not part of the installed package.

The evaluation runner loads this file by path after an episode has frozen its
candidate. No tool imports it, and the package never reads evaluator/. Each
frozen candidate can be scored once; a second request for the same manifest
hash is refused, so a holdout score can never feed back into selection.
"""

from __future__ import annotations

import hashlib
import io
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from reproducible_experiment_agent.schemas import FrozenCandidate, TaskManifest
from reproducible_experiment_agent.trial_worker import predict_for_metric, score

HOLDOUT_DIR = Path(__file__).resolve().parent / "holdout"


class AlreadyScored(RuntimeError):
    """The frozen candidate already has a scoring receipt."""


class HoldoutScorer:
    """Scores frozen candidates on holdout rows and keeps receipts."""

    def __init__(self) -> None:
        self.receipts: dict[str, dict[str, Any]] = {}

    def score(
        self, frozen: FrozenCandidate, manifest: TaskManifest, artifact: bytes
    ) -> dict[str, Any]:
        """Verify the artifact, score it once on the holdout rows and return a receipt."""
        if frozen.manifest_sha256 != frozen.body_sha256():
            raise ValueError("frozen manifest hash does not match its contents")
        if frozen.manifest_sha256 in self.receipts:
            raise AlreadyScored(frozen.manifest_sha256[:16])
        if hashlib.sha256(artifact).hexdigest() != frozen.artifact_sha256:
            raise ValueError("artifact bytes do not match the frozen hash")
        pipe = pickle.loads(artifact)  # noqa: S301, our own artifact, hash-verified above
        raw = (HOLDOUT_DIR / f"{frozen.task_id}.csv").read_bytes()
        holdout = pd.read_csv(io.BytesIO(raw))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = predict_for_metric(pipe, manifest.task_type, holdout[frozen.features])
        metrics = score(
            manifest.task_type, holdout[manifest.target], pred, getattr(pipe, "classes_", None)
        )
        receipt = {
            "task_id": frozen.task_id,
            "episode_id": frozen.episode_id,
            "run_id": frozen.run_id,
            "manifest_sha256": frozen.manifest_sha256,
            "holdout_sha256": hashlib.sha256(raw).hexdigest(),
            "holdout_rows": len(holdout),
            "metrics": {k: round(v, 6) for k, v in metrics.items()},
            "scored_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        self.receipts[frozen.manifest_sha256] = receipt
        return receipt
