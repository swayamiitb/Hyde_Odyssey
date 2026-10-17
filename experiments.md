# Experiment ledger

This table comes from completed run records included in evidence/. Local F benchmarks are distinct from organizer scoring. Each run includes its manifest, exact source and per-game records.

| Run | Games | Wins | Win rate | Mean CPU ms | Errors | Over-clock |
|---|---:|---:|---:|---:|---:|---:|
| [army_control](evidence/army_control/summary.json) | 400 | 334 | 83.50% | 345.0 | 0 | 0 |
| [army_knight_005](evidence/army_knight_005/summary.json) | 400 | 326 | 81.50% | 330.5 | 0 | 0 |
| [army_tactical](evidence/army_tactical/summary.json) | 400 | 323 | 80.75% | 468.6 | 0 | 0 |
| [baseline_v4](evidence/baseline_v4/summary.json) | 1000 | 537 | 53.70% | 275.0 | 0 | 0 |
| [final_guarded_wins_2000](evidence/final_guarded_wins_2000/summary.json) | 2000 | 1592 | 79.60% | 316.3 | 0 | 0 |
| [learned_04](evidence/learned_04/summary.json) | 400 | 318 | 79.50% | 443.5 | 0 | 0 |
| [learned_09](evidence/learned_09/summary.json) | 400 | 317 | 79.25% | 441.1 | 0 | 0 |
| [learned_control](evidence/learned_control/summary.json) | 400 | 316 | 79.00% | 347.8 | 0 | 0 |
| [learned_final_1000](evidence/learned_final_1000/summary.json) | 1000 | 797 | 79.70% | 393.4 | 0 | 0 |
| [screen_beam_broad](evidence/screen_beam_broad/summary.json) | 400 | 329 | 82.25% | 528.6 | 0 | 0 |
| [screen_beam_compact](evidence/screen_beam_compact/summary.json) | 400 | 331 | 82.75% | 336.8 | 0 | 0 |
| [screen_beam_wide](evidence/screen_beam_wide/summary.json) | 400 | 327 | 81.75% | 443.9 | 0 | 0 |

## Interpretation

The learned_control, learned_04 and learned_09 runs share the same 400-game screen. The selected 0.4 blend gains only two wins. It is not a statistically established improvement.

The three screen_beam runs share a common screen. Their small win differences are uncertain, while the observed broad and wide beam CPU costs are higher than the compact beam.

The army variants share their own control and screen. Reweighting spare knights or broader card values did not beat that control.

final_guarded_wins_2000 and learned_final_1000 are separate frozen holdouts on different boards. Do not treat their 79.6% and 79.7% point estimates as a paired ablation.

baseline_v4 is historical context measured on Python 3.14.5. Current frozen holdouts use Python 3.11.17. It is not a clean one-variable comparison with the release.

Only the completed runs explicitly listed here support this snapshot. O and group robustness are unmeasured. Later active experiments are excluded.

## Earlier research

The original ledger remains at [docs/history/experiments.md](docs/history/experiments.md). It records earlier measured failures and legacy claims, including notes that later entries corrected. It is separate from the current evidence table.
