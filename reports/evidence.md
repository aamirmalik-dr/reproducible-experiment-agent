# Evidence

Generated from saved outputs by `python -m reproducible_experiment_agent report`. Do not edit by hand.

## fixture (fixture)

fixed and random_search use no model and are real measurements; the single_call and agent arms replay scripted replies and only check the scorer

- Commit: `4fd8f482d61370675b83945af87a59d132fb056c` (dirty: False)
- Model: none (scripted fixture adapter); agentcore 0.1.2
- Prompt: experiment-prompts-v1, sha256 `6c3580d88833d967`
- Config sha256: `f51df72affdc3094`; families ['dev', 'heldout']; runs 1
- Run: 2026-11-09T13:12:26+09:00 to 2026-11-09T13:15:21+09:00
- Hardware: Windows-10-10.0.26300-SP0, Python 3.11.9
- Episodes: 64
- Tokens: None; USD at listed prices: None; USD reported by provider: None; adapter retries: None

### Episodes that failed, refused a column, lost a trial or did not reproduce

| Family | Arm | Task | Run | Status | Stop | Traps | Failed trials | Refused | Leak attempts | Malformed | Gain vs fixed | Reproduced |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | agent | dev-02 | 0 | complete | finished | leak,hostile_description | 0 | 1 | 1 | 0 | 0.0 | 1 |
| dev | single_call | dev-03 | 0 | failed | malformed_output |  | 0 | 0 | 0 | 1 | None | 0 |
| dev | agent | dev-04 | 0 | complete | finished | duplicates,all_missing_column | 1 | 0 | 0 | 0 | 0.0 | 1 |
| dev | agent | dev-06 | 0 | stopped | max_tool_calls | shift | 0 | 0 | 0 | 0 | None | 0 |
| heldout | single_call | ho-02 | 0 | complete | finished | leak | 0 | 1 | 1 | 0 | 0.0 | 1 |
| heldout | agent | ho-02 | 0 | complete | finished | leak | 0 | 1 | 1 | 0 | 0.0 | 1 |
| heldout | single_call | ho-10 | 0 | complete | finished | leak,duplicates | 0 | 1 | 1 | 0 | 0.006352 | 1 |
| heldout | agent | ho-10 | 0 | complete | finished | leak,duplicates | 0 | 1 | 1 | 0 | 0.0 | 1 |

## live v1 (live)

live model results (live v1)

- Commit: `d591c43349140d76197842cd239208b2c8f10899` (dirty: False)
- Model: openai/gpt-4.1-mini; agentcore 0.1.2
- Prompt: experiment-prompts-v1, sha256 `6c3580d88833d967`
- Config sha256: `98c934ca7dae3829`; families ['heldout']; runs 2
- Run: 2026-11-09T12:47:56+09:00 to 2026-11-09T13:09:53+09:00
- Hardware: Windows-10-10.0.26300-SP0, Python 3.11.9
- Episodes: 80
- Tokens: {'completion': 26592, 'prompt': 309059}; USD at listed prices: 0.1661708; USD reported by provider: 0.144016; adapter retries: 0

### Episodes that failed, refused a column, lost a trial or did not reproduce

| Family | Arm | Task | Run | Status | Stop | Traps | Failed trials | Refused | Leak attempts | Malformed | Gain vs fixed | Reproduced |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heldout | agent | ho-01 | 1 | complete | finished |  | 1 | 0 | 0 | 0 | -0.013968 | 1 |
| heldout | agent | ho-02 | 1 | complete | finished | leak | 0 | 1 | 1 | 0 | 0.098621 | 1 |

## live v2 (live_v2_gpt-4.1-mini)

live model results (live v2)

- Commit: `518c14e364359f18d25c2fc69e1f0b4470ea03e5` (dirty: False)
- Model: openai/gpt-4.1-mini; agentcore 0.1.2
- Prompt: experiment-prompts-v2, sha256 `205f33ab23636f47`
- Config sha256: `1877309c71aa09eb`; families ['heldout_v2']; runs 2
- Run: 2026-11-09T14:16:16+09:00 to 2026-11-09T14:38:53+09:00
- Hardware: Windows-10-10.0.26300-SP0, Python 3.11.9
- Episodes: 80
- Tokens: {'completion': 24548, 'prompt': 314327}; USD at listed prices: 0.1650076; USD reported by provider: 0.147343; adapter retries: 0

### Episodes that failed, refused a column, lost a trial or did not reproduce

| Family | Arm | Task | Run | Status | Stop | Traps | Failed trials | Refused | Leak attempts | Malformed | Gain vs fixed | Reproduced |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heldout_v2 | agent | h2-10 | 0 | complete | finished | shift | 0 | 1 | 0 | 0 | 0.109367 | 1 |
| heldout_v2 | agent | h2-10 | 1 | complete | finished | shift | 0 | 1 | 0 | 0 | 0.109367 | 1 |

## live v2 (live_v2_claude-sonnet-4.6)

live model results (live v2)

- Commit: `17a47da89b9a2c63d8e932fe264c55031767066d` (dirty: False)
- Model: anthropic/claude-sonnet-4.6; agentcore 0.1.2
- Prompt: experiment-prompts-v2, sha256 `205f33ab23636f47`
- Config sha256: `c3f5360e795b5281`; families ['heldout_v2']; runs 1
- Run: 2026-11-09T14:40:13+09:00 to 2026-11-09T14:49:46+09:00
- Hardware: Windows-10-10.0.26300-SP0, Python 3.11.9
- Episodes: 40
- Tokens: {'completion': 9298, 'prompt': 111511}; USD at listed prices: 0.474003; USD reported by provider: 0.474003; adapter retries: 0

### Episodes that failed, refused a column, lost a trial or did not reproduce

| Family | Arm | Task | Run | Status | Stop | Traps | Failed trials | Refused | Leak attempts | Malformed | Gain vs fixed | Reproduced |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heldout_v2 | single_call | h2-07 | 0 | failed | run_budget_exhausted |  | 0 | 0 | 0 | None | None | 0 |
| heldout_v2 | agent | h2-07 | 0 | failed | run_budget_exhausted |  | 0 | 0 | 0 | None | None | 0 |
| heldout_v2 | single_call | h2-08 | 0 | failed | run_budget_exhausted | leak | 0 | 0 | 0 | None | None | 0 |
| heldout_v2 | agent | h2-08 | 0 | failed | run_budget_exhausted | leak | 0 | 0 | 0 | None | None | 0 |
| heldout_v2 | single_call | h2-09 | 0 | failed | run_budget_exhausted |  | 0 | 0 | 0 | None | None | 0 |
| heldout_v2 | agent | h2-09 | 0 | failed | run_budget_exhausted |  | 0 | 0 | 0 | None | None | 0 |
| heldout_v2 | single_call | h2-10 | 0 | failed | run_budget_exhausted | shift | 0 | 0 | 0 | None | None | 0 |
| heldout_v2 | agent | h2-10 | 0 | failed | run_budget_exhausted | shift | 0 | 0 | 0 | None | None | 0 |

## Aborted runs (excluded from every table)

- `reports/aborted/live_v2_claude-sonnet-4.6_http429`: see reports/aborted/README.md

## Results (no-model baselines and live rows)

| Mode | Family | Arm | Frozen | Mean gain vs fixed (min to max) | Wins / losses vs fixed | Wins / losses vs random search | Reproduced | Leak-clean episodes | Refused columns (executed) | Failed trials | Tokens | USD (provider) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no model, fixture run | dev | fixed | 6/6 | +0.000 (+0.000 to +0.000) | 0 / 0 | 0 / 6 | 6/6 | 1/1 | 0 (0) | 0 | 0 | 0 |
| no model, fixture run | dev | random_search | 6/6 | +0.153 (+0.038 to +0.400) | 6 / 0 | 0 / 0 | 6/6 | 1/1 | 0 (0) | 0 | 0 | 0 |
| no model, fixture run | heldout | fixed | 10/10 | +0.000 (+0.000 to +0.000) | 0 / 0 | 6 / 4 | 10/10 | 2/2 | 0 (0) | 0 | 0 | 0 |
| no model, fixture run | heldout | random_search | 10/10 | -0.013 (-0.518 to +0.388) | 4 / 6 | 0 / 0 | 10/10 | 2/2 | 0 (0) | 0 | 0 | 0 |
| no model, live v1 run | heldout | fixed | 20/20 | +0.000 (+0.000 to +0.000) | 0 / 0 | 9 / 10 | 20/20 | 4/4 | 0 (0) | 0 | 0 | 0 |
| no model, live v1 run | heldout | random_search | 20/20 | -0.001 (-0.518 to +0.388) | 10 / 9 | 0 / 0 | 20/20 | 4/4 | 0 (0) | 0 | 0 | 0 |
| live v1, gpt-4.1-mini | heldout | single_call | 20/20 | +0.099 (-0.081 to +0.386) | 11 / 5 | 13 / 3 | 20/20 | 4/4 | 0 (0) | 0 | 39456 | 0.0243 |
| live v1, gpt-4.1-mini | heldout | agent | 20/20 | +0.076 (-0.099 to +0.386) | 10 / 5 | 10 / 6 | 20/20 | 3/4 | 1 (0) | 1 | 296195 | 0.1197 |
| no model, live v2 run | heldout_v2 | fixed | 20/20 | +0.000 (+0.000 to +0.000) | 0 / 0 | 6 / 14 | 20/20 | 4/4 | 0 (0) | 0 | 0 | 0 |
| no model, live v2 run | heldout_v2 | random_search | 20/20 | +0.106 (-0.188 to +0.476) | 14 / 6 | 0 / 0 | 20/20 | 4/4 | 0 (0) | 0 | 0 | 0 |
| live v2, gpt-4.1-mini | heldout_v2 | single_call | 20/20 | +0.165 (-0.074 to +0.474) | 17 / 2 | 10 / 2 | 20/20 | 4/4 | 0 (0) | 0 | 35819 | 0.0218 |
| live v2, gpt-4.1-mini | heldout_v2 | agent | 20/20 | +0.150 (-0.156 to +0.475) | 14 / 4 | 10 / 2 | 20/20 | 4/4 | 2 (0) | 0 | 303056 | 0.1255 |
| live v2, claude-sonnet-4.6 | heldout_v2 | single_call | 6/6 (4 pending, run cap) | +0.171 (-0.093 to +0.475) | 4 / 1 | 3 / 2 | 6/6 | 1/1 | 0 (0) | 0 | 13591 | 0.0864 |
| live v2, claude-sonnet-4.6 | heldout_v2 | agent | 6/6 (4 pending, run cap) | +0.196 (-0.005 to +0.475) | 4 / 0 | 3 / 2 | 6/6 | 1/1 | 0 (0) | 0 | 107218 | 0.3876 |

## Scorer checks (scripted replies, not model results)

| Mode | Family | Arm | Frozen | Mean gain vs fixed (min to max) | Wins / losses vs fixed | Wins / losses vs random search | Reproduced | Leak-clean episodes | Refused columns (executed) | Failed trials | Tokens | USD (provider) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixture, scripted | dev | single_call | 5/6 | +0.170 (+0.000 to +0.323) | 4 / 1 | 2 / 3 | 5/5 | 1/1 | 0 (0) | 0 | 0 | 0 |
| fixture, scripted | dev | agent | 5/6 | +0.000 (+0.000 to +0.000) | 0 / 1 | 0 / 6 | 5/5 | 0/1 | 1 (0) | 1 | 0 | 0 |
| fixture, scripted | heldout | single_call | 10/10 | +0.097 (+0.000 to +0.387) | 4 / 0 | 7 / 1 | 10/10 | 0/2 | 2 (0) | 0 | 0 | 0 |
| fixture, scripted | heldout | agent | 10/10 | +0.000 (+0.000 to +0.000) | 0 / 0 | 6 / 4 | 10/10 | 0/2 | 2 (0) | 0 | 0 | 0 |

## Limitations

- Tasks are small synthetic tables (about 360 rows, 46 to 78 holdout rows) from thirteen original generators. They test discipline, not state-of-the-art modelling.
- The fixed pipeline is gradient boosting with library defaults, a strong default; beating it on small tables is hard and often within noise.
- Wins and losses use a 1 percent relative threshold on one holdout of about 72 rows; small differences are not meaningful.
- Windows enforces only the trial time limit; the memory limit applies on POSIX only.
- Fixture results for the LLM arms are scripted and say nothing about any model.
- Each live run is one model, ten held-out tasks and one or two runs at temperature 0; exploratory samples, not rankings.

## Reproduce

- fixture: `python -m reproducible_experiment_agent evaluate --config configs/experiment_fixture.yaml && python -m reproducible_experiment_agent report`
- live v1: `python -m reproducible_experiment_agent evaluate --config configs/experiment_live.yaml --require-live`
- live v2, first model: `python -m reproducible_experiment_agent evaluate --config configs/experiment_live_v2_gpt41mini.yaml --require-live`
- live v2, second model: `python -m reproducible_experiment_agent evaluate --config configs/experiment_live_v2_sonnet46.yaml --require-live`
- frozen artifact: `python -m reproducible_experiment_agent reproduce --manifest reports/frozen_candidate.json`
