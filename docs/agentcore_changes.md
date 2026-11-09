# agentcore changes

`src/reproducible_experiment_agent/agentcore/` is vendored from evidence-ledger-agent, agentcore version 0.1.2 (commit 45f6ff8 of that repository). The six files are byte-identical to the source.

Local changes: none.

The project-specific behaviour (trial budget, compute seconds, freeze) lives in this package's `tools.py` and `schemas.py`, on top of agentcore's generic step, tool-call, token and wall-time caps.
