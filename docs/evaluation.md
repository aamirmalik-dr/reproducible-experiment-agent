# Evaluation protocol

Declared on 2026-11-09, before the first live run, and frozen in `configs/experiment_live.yaml`.

## Arms

| Arm | Model calls | Trials | Selection |
|---|---|---|---|
| `fixed` | none | 1: HistGradientBoosting with library defaults, every permitted feature, median imputation, one-hot encoding | the only trial |
| `random_search` | none | 6 random specs from the allowlist, seeded by task id and run | lowest dev metric |
| `single_call` | 1 (plus one corrective re-prompt) | up to 6 specs proposed in one reply after seeing the same schema and split summary the agent can read | lowest dev metric, by rule |
| `agent` | one per step | up to 6, each chosen after seeing earlier results | the agent calls `freeze_candidate` |

`single_call` is also the ablation the project specification asks for: the same model and trial budget without iteration.

## Metrics

Per task, on the holdout rows, scored once per frozen candidate by `evaluator/holdout_scorer.py`:

- Primary: log loss for classification, MAE for regression. Lower is better.
- Secondary: macro-F1 for classification, RMSE for regression.

Normalisation for aggregation: relative gain over the fixed pipeline on the same task and run, `(fixed - arm) / fixed`. Positive means better than the fixed pipeline. A win is a gain above 0.01, a loss a gain below -0.01, anything between is a tie. The same rule gives wins and losses against random search. An episode with no frozen candidate counts as a loss and never as a win; its gain is not averaged.

Also reported, with denominators:

- frozen candidates out of episodes (freeze compliance);
- reproduced candidates: the frozen spec and seed refit in a fresh process give a byte-identical pickled pipeline and identical dev predictions (rounded to 12 decimals);
- leak-clean episodes: episodes on tasks with a target-derived column in which the arm never asked for it;
- refused columns (attempts) and executed forbidden selections (trials that actually used a forbidden column, which the policy makes impossible; the count is checked anyway);
- failed or timed-out trials;
- dev-to-holdout gap of the frozen candidate;
- tokens and provider-reported cost.

## Sample size and spread

The live pilot runs 10 held-out tasks twice. Each run is a separate episode, so every arm has 20 episodes. Min and max relative gain are shown as the spread. This is an exploratory sample, not a ranking.

## What counts as a failure

No frozen candidate (budget exhausted, malformed output, adapter error, every trial failed). Strong holdout performance obtained with a leaked column would count as a failure, but the policy refuses such trials before they run.
