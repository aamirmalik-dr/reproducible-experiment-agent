"""Trial execution: subprocess isolation on a copy of the task, artifact store.

Each trial runs in a fresh temporary directory that holds a copy of the task
manifest and train/dev rows, under a wall-time limit. On POSIX an address-space
limit is also set; on Windows only the time limit is enforced. This protects
against runaway computation in this project's own code. It is not a sandbox
for untrusted code, and none is needed: no model-written code is ever run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from .schemas import ExperimentSpec, TrialResult
from .split_service import Partitions
from .task_registry import task_dir
from .trial_worker import fit_trial


def _limit_memory(mb: int) -> Any:
    def apply() -> None:  # pragma: no cover, POSIX only
        import resource

        limit = mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return apply


class TrialRunner:
    """Runs trials and stores their artifacts.

    Args:
        root: Repository root (task folders live under ``root/tasks``).
        artifact_dir: Where fitted pipelines are stored, one file per run id.
        isolation: ``subprocess`` (default) or ``inprocess`` (fast, for tests).
        time_limit_s: Wall-time limit per trial.
        memory_mb: Address-space limit per trial (POSIX only).
    """

    def __init__(
        self,
        root: Path,
        artifact_dir: Path,
        isolation: Literal["subprocess", "inprocess"] = "subprocess",
        time_limit_s: float = 60.0,
        memory_mb: int = 2048,
    ) -> None:
        self.root, self.artifact_dir = Path(root), Path(artifact_dir)
        self.isolation, self.time_limit_s, self.memory_mb = isolation, time_limit_s, memory_mb
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        parts: Partitions,
        spec: ExperimentSpec,
        seed: int,
        features: list[str],
        run_id: str,
        time_limit_s: float | None = None,
    ) -> TrialResult:
        """Fit and score one trial. Failures and timeouts come back as results."""
        limit = min(self.time_limit_s, time_limit_s or self.time_limit_s)
        if self.isolation == "inprocess":
            raw, blob = fit_trial(parts, spec, seed, features)
        else:
            raw, blob = self._subprocess(parts, spec, seed, features, limit)
        if blob is not None:
            (self.artifact_dir / f"{run_id}.pkl").write_bytes(blob)
        return TrialResult(
            run_id=run_id,
            task_id=parts.manifest.task_id,
            spec=spec,
            seed=seed,
            features_used=features,
            partition_sha256=parts.sha256,
            **raw,
        )

    def _subprocess(
        self, parts: Partitions, spec: ExperimentSpec, seed: int, features: list[str], limit: float
    ) -> tuple[dict[str, Any], bytes | None]:
        m = parts.manifest
        with tempfile.TemporaryDirectory(prefix="trial-") as tmp:
            work = Path(tmp)
            shutil.copy2(task_dir(self.root, m.family, m.task_id) / "train_dev.csv", work)
            job = {
                "manifest": m.model_dump(),
                "spec": spec.model_dump(),
                "seed": seed,
                "features": features,
            }
            (work / "job.json").write_text(json.dumps(job), encoding="utf-8")
            kwargs: dict[str, Any] = {}
            if sys.platform != "win32":
                kwargs["preexec_fn"] = _limit_memory(self.memory_mb)
            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "reproducible_experiment_agent.trial_worker",
                        "job.json",
                    ],
                    cwd=work,
                    capture_output=True,
                    text=True,
                    timeout=limit,
                    check=False,
                    **kwargs,
                )
            except subprocess.TimeoutExpired:
                elapsed = round(time.perf_counter() - start, 3)
                return {
                    "status": "timeout",
                    "error": f"exceeded {limit} s",
                    "elapsed_s": elapsed,
                }, None
            result_file = work / "result.json"
            if proc.returncode != 0 or not result_file.exists():
                tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no output"]
                return {
                    "status": "failed",
                    "error": f"worker exited {proc.returncode}: {tail[0][:200]}",
                    "elapsed_s": round(time.perf_counter() - start, 3),
                }, None
            raw = json.loads(result_file.read_text(encoding="utf-8"))
            art = work / "artifact.pkl"
            blob = art.read_bytes() if art.exists() else None
            return raw, blob

    def artifact_sha256(self, run_id: str) -> str | None:
        """Hash of the stored artifact file, or None if it is missing."""
        path = self.artifact_dir / f"{run_id}.pkl"
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
