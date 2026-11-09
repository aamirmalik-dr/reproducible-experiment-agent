# evaluator/

Evaluator-only material. Nothing in `src/` reads this folder, and no tool can reach it.

- `holdout/<task_id>.csv`: holdout rows with labels, written by `scripts/make_tasks.py`.
- `holdout_scorer.py`: scores a frozen candidate once and returns a receipt. The evaluation runner loads it by file path after an episode has frozen its candidate.
