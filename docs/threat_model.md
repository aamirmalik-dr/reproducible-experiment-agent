# Threat model

## What is protected

1. The holdout. No tool, prompt or trial can read holdout rows or labels, and a holdout score can never feed back into selection.
2. Leakage discipline. No trial can use the target, identifiers, group identifiers or a column recorded after the outcome.
3. The record. Every trial, including failures, is kept; a frozen candidate is immutable and can be re-derived byte for byte.

## Capabilities the model has

Six typed tools: `inspect_schema`, `split_manifest`, `run_experiment`, `compare_dev_runs`, `check_artifacts`, `freeze_candidate`. A trial spec is a closed schema: three imputers, two encoders, three scalers, three model families, numeric parameters inside fixed ranges, and a feature list checked against the manifest. There is no string that is executed, no path, no import, no shell, no network and no file access. The model never writes code.

## Threats and controls

| Threat | Control | Evidence |
|---|---|---|
| Model selects a leaked or forbidden column | `resolve_features` refuses any column that is not a permitted feature; the attempt is counted, no trial runs | `test_policy_refuses_target_id_group_and_leak_columns`, `test_agent_refuses_leak_ignores_hostile_text_and_freezes` |
| Instructions inside a dataset description (read key files, export the model) | descriptions reach the model only inside `<tool_result trust="untrusted-data">`; no tool can read files or export; the tool set is fixed | same test, plus the fixed tool list |
| Holdout access through a tool | holdout rows live only in `evaluator/holdout/`; the package never reads `evaluator/` (checked by a source scan in the tests); the scorer is loaded by path only in the evaluation runner, after freeze | `test_holdout_scorer_is_one_shot_and_unreachable_from_tools`, `test_partitions_are_disjoint_and_holdout_is_absent` |
| Holdout scored repeatedly to steer selection | the scorer keeps receipts and refuses a second request for the same frozen manifest; there is no tool to request a score | `test_holdout_scorer_is_one_shot_and_unreachable_from_tools` |
| Tampered data or frozen manifest | `train_dev.csv` is checked against its manifest hash on every load; a frozen manifest carries a hash of its own body and the artifact hash | `test_tampered_train_dev_is_detected`, scorer tamper check |
| Trial after freeze | `run_experiment` refuses once a candidate is frozen; `freeze_candidate` also ends the episode | `test_no_trial_after_freeze` |
| Runaway computation | each trial runs in a subprocess on a copy of the task files in a temporary directory, with a wall-time limit; on POSIX an address-space limit is also set | `test_runner_kills_a_trial_over_its_time_limit` |
| Duplicate fits after a crash | the agent loop writes each action ahead to an atomic checkpoint and skips completed actions on resume | `test_resume_does_not_refit_completed_trials` |
| Secrets in traces | agentcore sanitizes keys, bearer tokens and absolute paths before writing traces | vendored agentcore tests in evidence-ledger-agent |

## What the subprocess runner does not do

It protects against runaway computation in this project's own fitting code: an oversized forest, a model that will not converge. It is not a sandbox. It does not defend against malicious code, and it does not need to, because no model-written or untrusted code is executed; the worker only builds scikit-learn objects from a validated spec. On Windows only the wall-time limit is enforced; the memory limit uses `resource.setrlimit` and applies on POSIX systems only.

## Residual risks

- A crash after a trial finishes but before its result is checkpointed re-runs that trial on resume. Trials only write to a per-episode artifact store, so the cost is time, not correctness.
- Pickled artifacts are loaded by the scorer and the reproduction check. They are produced by this code and hash-verified before loading; never load an artifact from an untrusted source.
