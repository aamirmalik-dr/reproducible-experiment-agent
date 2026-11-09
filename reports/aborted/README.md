# Aborted runs (kept, not reported as model results)

## live_v2_claude-sonnet-4.6_http429

First v2 run with `anthropic/claude-sonnet-4.6` (configs/experiment_live_v2_sonnet46.yaml, commit 518c14e,
2026-11-09). OpenRouter answered HTTP 429 (rate limited) while several sessions on the same key were calling the
same model. With the default two short retries, 6 of 20 LLM episodes stopped with `adapter_error` (5 agent, 1
single_call; 12 retries in total). This is an infrastructure failure, not model behaviour, so these outputs are
kept here for the record and excluded from every results table.

Provider-reported cost of this aborted run: 0.466 USD. The run was repeated once on the same frozen tasks after
adding config-level retry settings (6 retries, 5 s exponential backoff); nothing else changed.
