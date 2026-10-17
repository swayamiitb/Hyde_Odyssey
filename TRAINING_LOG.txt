# Odyssey training log

Team Odyssey. Supplement for the frozen learned release.

**Status:** This log reconstructs the original training process from preserved source, data and metrics. It also includes a new, timestamped refit of the saved dataset. Reconstructed counts and new refit output are labeled separately. No historical timestamps or per-iteration measurements have been invented.

## Release identity

- Runtime source: [submission/agent.py](submission/agent.py).
- Agent SHA-256: a43d291a1cd984f67e59b7e24a4105c79c3110fa8ad164e45b3619e98927ba06
- Model export: [models/value_model.json](models/value_model.json).
- Fitted weights are embedded in the runtime as ML_BIAS and ML_TREES.
- The original standalone training console transcript is unavailable in this snapshot. The original summary metrics, dataset and model are preserved.

## Original data generation

The heuristic teacher played F in 1,000 offline games, alternating seats. The trainer collected end-of-turn states from both player perspectives and labeled each state with the eventual winner. Each player's random state was isolated from the simulated dice stream. Collection is an offline learning process with a constant 0.8-second clock observation, not a competition benchmark.

| Item | Preserved value |
|---|---:|
| Games represented | 1000 |
| Labeled positions | 124,344 |
| Features | 91 |
| Fit games / held-out games | 800 / 200 |
| Fit positions / held-out positions | 98,598 / 25,746 |
| Positive / negative labels | 62,172 / 62,172 |
| Game overlap across the split | 0 |

Board seed = 1500000 + game ID. Dice seed = 6000001 + game ID. Agent random-state seed = 90001 + 2 × game ID + player index. Whole games with game ID divisible by five form the external validation split. The [collection-count log](training_logs/reconstructed_collection_counts.jsonl) recovers exact 20-game batch counts from the saved dataset. It does not reconstruct historical elapsed times.

Features describe information available to the player: permitted points, production, resource hands, ports, robber effects, buildings, roads, played knights and build affordability. Opponent unplayed development-card types are excluded.

## Original fitting configuration

| Parameter | Value |
|---|---|
| Estimator | HistGradientBoostingClassifier |
| Maximum boosting iterations | 64 |
| Maximum depth / leaves per tree | 3 / 7 |
| Learning rate | 0.1 |
| L2 regularization | 15 |
| Minimum samples per leaf | 80 |
| Random state | 483 |
| Internal validation fraction | 0.1, scikit-learn default |
| Internal early stopping | auto, scikit-learn default |
| Numeric input | float64 |

The internal validation split is drawn from the fit subset. The separate held-out game subset never enters model fitting. Full resolved estimator settings appear in [training_logs/manifest.json](training_logs/manifest.json).

## Preserved original metrics

- Held-out-game AUC: **0.8847908708**.
- Held-out log loss: **0.4285112022**.
- Recorded end-to-end collection and fitting wall time: **85.662986 seconds**.
- Runtime model: **64 trees**. Exported pure-Python inference was checked against the fitted model.

The original timing includes collection and fitting. It is not a per-epoch timing or a gameplay CPU measurement. AUC describes outcome prediction, not win rate.

## Newly executed reproduction

- Started: 2026-10-17T11:57:13.227615+00:00.
- Finished: 2026-10-17T11:57:14.379205+00:00.
- Refit wall time: 0.953522 seconds.
- Trees exactly match the preserved release model: **True**.
- Refit AUC: 0.8847908708.
- Refit log loss: 0.4285112022.
- Maximum pure-Python logit error on 100 held-out vectors: 0.

The [actual refit console log](training_logs/refit_stdout.log), [iteration scores](training_logs/refit_iterations.jsonl) and [machine-readable result](training_logs/reproduction_result.json) belong to this new verification run. Iteration 0 is the pre-boosting score. The iteration log reports negative log loss on the fit and internal validation subsets. It must not be presented as a recovered historical training transcript.

## Policy selection and gameplay validation

| Run | Wins / games | Vs F | Mean local CPU ms | Errors | Over-clock |
|---|---:|---:|---:|---:|---:|
| [learned_control](evidence/learned_control/summary.json) | 316/400 | 79.00% | 347.8 | 0 | 0 |
| [learned_04](evidence/learned_04/summary.json) | 318/400 | 79.50% | 443.5 | 0 | 0 |
| [learned_09](evidence/learned_09/summary.json) | 317/400 | 79.25% | 441.1 | 0 | 0 |
| [learned_final_1000](evidence/learned_final_1000/summary.json) | 797/1000 | 79.70% | 393.4 | 0 | 0 |

The first three runs share the same 400-game screen. The highest-win clean candidate, the 0.4 model blend, was frozen before the fresh 1,000-game test. Its two-win screening lead is weak evidence of improvement. The final 79.7% is a local official-harness result, not an organizer leaderboard score. O and group robustness are unmeasured.

## Reproduce and inspect

After setting CATAN_KIT_ROOT and using the pinned Python 3.11 environment:

~~~bash
python tools/verify_evidence.py
python tools/build_training_log.py
~~~

The log builder refits the saved dataset and writes only the log supplement. It does not replace the packaged agent, the model export or the saved dataset. [manifest.json](training_logs/manifest.json) records input SHA-256 hashes, software versions and all settings. The report, model, runtime ZIP and benchmark evidence remain tied to the same source hash.
