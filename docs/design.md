# Design

## Problem

An LLM can propose ML experiments, but the interesting failures of an experimenter are not about model choice. They are about discipline: fitting a scaler on all the rows, picking a column that was recorded after the outcome, letting duplicated rows sit on both sides of a split, peeking at the test set, or reporting a number that cannot be reproduced. This project keeps all of those decisions out of the model's reach and measures what the model does with what is left: which allowlisted pipelines to try, in what order, and when to stop.

## Loop

```
model ── JSON decision ──> agentcore policy (allowlist, schema, budgets)
                                  │
                ┌─────────────────┼──────────────────────────┐
         inspect_schema    run_experiment(spec, seed)   freeze_candidate(run_id)
         split_manifest           │                          │
         compare_dev_runs   resolve_features (policy)   FrozenCandidate (hashed)
         check_artifacts          │                          │
                           TrialRunner: subprocess,     episode ends
                           copy of train/dev rows,           │
                           time limit                        ▼
                                  │                 evaluation runner:
                           TrialResult (ok, failed,   holdout_scorer (once)
                           timeout), artifact.pkl     reproduce (fresh process)
```

The agent sees its observable state (trials left, compute seconds left, trial ids) and the last four tool results. It must answer with one JSON object, `{"summary": ..., "action": {...}}`, parsed by agentcore 0.1.2.

## Trusted services

- `split_service`: partitions assigned once at generation, by whole units (groups or duplicate rows); integrity hash checked on every load; a visible summary with counts and hashes only.
- `task_registry`: manifests, permitted features (role feature and not forbidden), the visible schema computed from train rows only.
- `pipeline_factory`: one scikit-learn Pipeline per trial, so every transform is fitted on training rows and only applied to dev rows.
- `runner`: subprocess isolation, artifact store, artifact hashes.
- `evaluator/holdout_scorer.py`: outside the package, one receipt per frozen candidate.

## Reproducibility

A trial is identified by its spec, its seed, the partition hash (train and dev row ids) and the task's `train_dev.csv` hash. The artifact hash is the SHA-256 of the pickled fitted pipeline (protocol 4); the prediction hash is the SHA-256 of dev predictions rounded to 12 decimals. `reproduce` refits from the frozen manifest in a fresh process and compares both. The committed `reports/frozen_candidate.json` is reproduced by a test, and pickles are byte-identical across processes on the same platform and library versions. Across scikit-learn or numpy versions, byte identity is not promised; prediction identity within 1e-12 usually holds but is not promised either.

## Why the arms look the way they do

The fixed pipeline is a strong, honest default (gradient boosting with library defaults). Random search with the same budget is the classic non-LLM alternative. The single-call planner sees exactly the schema and split summary the agent could read, and proposes the same number of trials without seeing any result. The agent's only advantage is iteration. If iteration does not beat random search on these tasks, the README says so.
