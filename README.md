# Hyde_Odyssey — Hyde × WnCC Catan AI Hackathon (17 Oct 2026, IIT Bombay)

Our build for the 2-player Catan agent challenge on the pinned catanatron
engine (`lyric-hyde/catan-hackathon-agent` kit): one CPU core, a chess-style
CPU clock (300 ms + 12.5 ms/decision, ≤1.5 s/game), Python-only, scored vs
the anchor bots F and O.

## Final result

Three validated submissions (statistical twins of one measured distribution,
submitted as independent draws under the best-of-submissions rule):

| Zip | Agent | Win rate vs F (1,000 games, fresh seeds) | Mean CPU/game |
|---|---|---|---|
| `sub1.zip` | v4 — corrected eval + joint placement + gated depth-2 | 0.5230 ±0.0310 | 338 ms |
| `sub2.zip` | v17 — v4 + hidden-VP denial gate + build-tempo bonus | 0.5260 ±0.0309 | 313 ms |
| `sub3.zip` | v8 — Texel(logistic)-fitted linear eval (ships encoder.py + texel_eval.npz) | 0.5250 ±0.0310 | 379 ms |

All: 0 faults, 0 games over the clock, ~⅓ of the CPU allowance used.
Robustness portfolio: 200/200 vs road-rusher / city-builder / dev-spammer /
VP-chaser; 0.52 vs tuned contender-F.

## Contents

- `my_team/` — every agent version (v1…v17), shared encoder, fitted weights,
  sparring/zoo bots, STRATEGY_TERMS.md (book distillation).
- `experiments.md` — **the ledger**: 27 measured variants, every failure with
  its mechanism, every seed range logged.
- `REPORT_FINAL.md` — the 4-page competition report. `REPORT.md` — long-form.
- `PRESENTATION.md` — 6-minute script + measured-answer question bank.
- `*_gen*.py`, `train_*.py`, `texel_fit.py`, `tuner.py`, `diag_score.py`,
  `wf_eval.py` — the day's tooling (self-play generators, trainers, tuner,
  diagnostics, cwd-safe eval driver).
- `*_CONTEXT*.md` — the full engineering record of the day.

## Reproduce any number

From a clone of the official kit (Python 3.11, `pip install -r requirements.txt`):

```bash
python -m catan_kit.selfcheck
python -m catan_kit.evaluate --a my_team/v4.py --b examples/f_agent.py --games 1000 --seed0 2000000
```

Seed ranges per experiment are recorded in `experiments.md`.

## Highlights (see report)

- F's evaluation reconstructed from the pinned engine source; five surviving
  corrections, led by the opponent-VP term that prices LR/LA denial.
- Same-turn search proven exhausted: a 10×-depth teacher disagrees with the
  shipped agent on **0 of 391** decisions.
- Outcome-label distillation proven gradient-free at this scale (deployed
  scorer matches its model at ρ=1.000 and still ranks badly); preference
  rankers fail to transfer at all eight measured scales.
- A cross-turn (dice-sampled opponent-reply) teacher disagrees with greedy on
  30% of decisions — the identified but clock-gated frontier.

Public material used (per fair-play rule 2): catanatron (GPL-3.0, pinned
commit), its author's post-mortem writings, Texel tuning (chessprogramming.org),
Buro 2002, Tesauro TD-Gammon, Anthony et al. 2017, Guhe & Lascarides 2014,
EfficientZero/ReZero, Oh et al. — ideas only; all code, weights, and
experiments are our own work built during the event.
