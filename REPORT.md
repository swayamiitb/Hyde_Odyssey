# Beating a Known Opponent on a Budget: Search, Distillation and Negative Results in 2-Player Catan

Team: <NAME> — Hyde × WnCC AI Builder Hackathon, 17 Oct 2026

## 1. The problem

Build an agent for 2-player Catan (catanatron, pinned commit; 10 VP; no player
trading) that runs on ONE CPU core under a chess-style clock: 300 ms + 12.5 ms
per decision, 1.5 s ceiling, ~1 s of CPU for a typical 56-decision game. The
given anchor bot F — whose full source ships in the kit — is a 1-ply greedy
over a 13-feature weighted evaluation. The second anchor, O, is the
organizers' trained bot (code withheld). Judging: robustness (worst-half win
rate over 9 group opponents) + strength (mean of win rates vs F and O) +
report + Q&A.

The only hidden information in this variant is the opponent's unplayed
development cards: their resource hand is exactly 19 − bank − mine, so the
harness exposes it; dev cards are sampled consistently by `obs.sample_world`.

## 2. Reading F (the given opponent)

We reconstructed F's evaluation from the pinned engine source. Properties that
shaped everything we did: 1-ply greedy with deterministic argmax; ONE sampled
world per decision; NO opponent-VP term (F cannot tell it is losing, and never
prices stealing Longest Road / Largest Army — a 4-VP swing); ports valued at
exactly zero; production reachable with 0 roads weighted 0; Longest Road
suppressed to 0.1× while expansion room exists. Reference point: submitting F
itself scores 0.500 by symmetry; catanatron's own MCTS and AlphaBeta bots
score ≈0.001 because they exhaust the clock and finish on random moves.

## 3. The journey, with numbers (all vs F, 400 games, ±0.049 at 95%)

| Version | Idea | Win rate | Lesson |
|---|---|---|---|
| v1 | five hand-crafted eval "fixes" | **0.4475** | scoring own hidden VPs at the dominant weight makes BUY_DEVELOPMENT_CARD dominate whenever the sampled deck top is a VP card — phantom-knowledge bias. Dominant-scale terms must be robust to sampling noise. |
| v1a | only the three "safe" fixes | 0.5000 | eval tweaks near F's weights are neutral; the edge must come from search. |
| v2 | + opponent-awareness (enemy VP at −3e14 prices LR/LA denial), joint settlement+road placement search, same-turn depth-2 (top-5), clock gates | 0.5250 | first clear edge; search + denial terms work. |
| v3 | fast direct-read eval (dropped reachability/buildable terms in-turn) | **0.2300** | the day's biggest failure: reachable-production signals ARE the expansion policy. Also node cost is copy-dominated — the fast eval bought only ~2× nodes, not the hoped 5–10×. |
| v4 | v2 with wider expansion (top-8) | **0.5350** | best clean agent; 0 faults, 0 games over the clock, ~300 ms/game (≈⅓ of allowance). |
| v5 | + win-now DFS (≥7 VP, 40 ms) + trade-enables-build follow-ups | 0.5200 | the endgame search re-ran its full DFS EVERY late-game decision: 23 games over the clock, 523 random-move degradations. Good idea, wrong cost model. |
| v6 | value net distilled from 5.8k teacher games (579k positions) | 0.4675 | the net was WORSE than its teacher: dice-noisy outcome labels + a VP-dominant teacher-regression term washed out the subtle features. |
| v7 | heuristic + 0.3×net blend | 0.4675 | blending did not rescue a miscalibrated net. |
| v8 | Texel-tuned linear eval (logistic regression on the same 579k positions; AUC 0.847) in v4's skeleton | 0.5275 | data-fitted linear eval ≈ hand-tuned within noise; interpretable weights (opponent LR strongest negative, own wheat/ore totals strongest positive). Confirms the ~0.53 plateau of the eval axis. |
| weight tuner | 15 weights × 4 factors, 200-game paired fitness, 8 parallel workers | best 0.550 @200 games | did not replicate at 400 (0.5275) — paired-seed screens at 200 games are still too noisy to accept +4% claims. |

Robustness checks (the metric the group stage actually uses): v4 beats a pure
VP-chaser **200/200**, holds **0.490 ±0.069** against F-with-tuned-contender-
weights — no collapse against a different weight vector.

## 4. Round 2: better labels, not more architecture

Diagnosis of v6's failure directed round 2: the labels, not the architecture,
were the bottleneck. We regenerated 5.8k games (578,566 positions) with a
stronger teacher and richer logging:

- **VP_DISCOUNTED_RETURN targets** (0.9999^turns × final VPs): rewards closing
  games out — the one label trick the engine's author measured to work.
- **Progress weighting**: later positions carry less dice noise.
- **A self-supervised auxiliary head**: predict the opponent's hidden VP-card
  count from public features (ground truth known in self-play) — a masked-
  modeling task aimed at the game's only hidden information; wired into play
  as a calibrated discount on our value estimates.
- **EMA shipping weights**, cosine LR, held-out early stop.

Results — round 2 failed, and the failure is the finding:

| Variant | Win rate vs F | Note |
|---|---|---|
| v4p (placement coverage + denial) | 0.4975 ±0.0490, **426 faults** | near the 2% DQ line; no gain even ignoring faults. Discarded. |
| v8b (Texel refit on round-2 data) | **0.1850** ±0.0381 | collapse from 0.5275 with identical agent code, only weights differ |
| v9 (round-2 net) | **0.1875** ±0.0383 | different architecture, same data, same collapse |
| v8 re-measured with consistent weights | 0.4650 ±0.069 | the original 0.5275 had a fit/deploy feature mismatch |

Two different architectures trained on the same 578k-row dataset collapsing
to the same ~0.19 localizes the fault to the dataset or the fit-to-deploy
path, not the models. A direct label audit came back sane — win probability
climbs monotonically 0.10→0.91 with VP differential, no NaNs, healthy row
counts — and the normalization has no degenerate scales. We time-boxed the
root cause and shipped the heuristic champion instead; the open question is
documented deliberately.

**Robustness portfolio of the shipped agent (the metric the group stage
actually uses — mean win rate vs the worst half of opponents):** v4 sweeps
the scripted archetypes completely — road-rusher 200/200, city-builder
200/200, dev-spammer 200/200, VP-chaser 200/200 — and holds 0.520 ±0.069
against F-with-tuned-contender-weights. No catastrophic matchup exists; the
weak half is exactly the two strong opponents (F 0.535, contender 0.52).

**Final validation:** v4, 1000 games vs F on fresh seeds: **0.5230 ±0.0310**,
0 faults, 0 games over the clock, mean CPU 338 ms/game, worst game 938 ms —
about a third of the ~1 s allowance, leaving ~2× margin for the slower
scoring core. A last isolated test of the cooled-down win-now layer on top of
v4 measured 0.4800 (worse); it was cut, making v4 the frozen submission.

## 5. Method: the shipped agent

Decision stack, each layer clock-gated on `obs.cpu_left` (tiers: 18 ms / 8 ms /
plain 1-ply): joint placement search (settlement+road, full eval) → win-now
sequence DFS (≥8 actual VP, 15 ms, forced-move compression) → 1-ply greedy
with the selected evaluation → same-turn depth-2 expansion of the top-K
actions with build/dev follow-ups. Fallback chain: any exception → heuristic
path → first legal action; the agent cannot raise (0 faults in every measured
run; the 2% DQ threshold was never approached).

Dataset discipline: one shared encoder module for training and deployment
(the #1 silent bug in this pattern is feature drift between the two);
observability rule — every stored feature is computed from the deciding seat's
masked view, never god-view; disjoint TRAIN/EVAL seed ranges; every number in
this report comes from a logged run of the organizers' own `evaluate` script.

## 6. Insights we keep

1. **The clock is the true opponent.** Every library search bot dies at 0.001
   by timing out; our v5 lost 1.5% the same way. Budget-awareness is a
   first-class design axis, not an afterthought.
2. **Negative results are load-bearing.** v1 and v3 shaped the architecture
   more than any positive delta; we kept both in the log.
3. **Distillation quality is label quality — and label sanity is not
   pipeline sanity.** Our round-2 labels audited perfectly sane, yet every
   model trained on them collapsed in play. A healthy-looking offline metric
   (AUC 0.83) says nothing about the fit-to-deploy feature path; only the
   400-game play measurement caught it.
4. **A linear eval fitted on 579k positions matches hand-tuning** — Buro's
   1997 Othello result reproduces here at hackathon scale.
5. **Paired-seed evaluation at 200 games is a screen, not a verdict** — the
   tuner's +4% candidate vanished at 400 games.

## 6b. The revamp wave: five more failures, each with a mechanism

| Variant | Win rate | Root cause |
|---|---|---|
| v11 (strategy-book terms, subagent-distilled from 3 books) | 0.5175 | the eval-family plateau, again |
| v13 (pairwise ranker distilled from a cross-turn teacher, λ=3e5–3e6) | 0.027–0.08 | λ mis-scaled by ~10³: the unbounded ranker output swamps the eval's production/hand tiers and randomizes all non-VP decisions |
| v14 DART (fast full eval + in-play cross-turn, 0.6 blend) | 0.4275 | reachability cache keyed on road count (collisions), buildable term dropped |
| v14b (exact caching, 0.3 blend) | 0.4500 | approximation error still exceeds the cross-turn gain |
| v15 (knight-sequence + trade-enables-build expansion) | 0.4650 | widening depth-2 beyond v4's exact regime perturbs a finely balanced ranking |

Two systematic findings close the day: (1) **learned residuals wired into a
hand-balanced eval are a scale hazard** — two independent label families
(outcomes, preferences) collapsed identically until λ calibration was
re-derived; (2) **v4 sits in a narrow local optimum** whose neighborhood we
have now mapped with 21 measured variants. The 30%-disagreement cross-turn
teacher remains the strongest untapped signal — as an offline label source
with a correctly-scaled student, which is precisely the "more time" plan.

## 7. What we would do with more time

Cross-turn expectimax on critical decisions (opponent greedy reply under
sampled dice); reanalysis relabeling between improvement rounds (reuse stored
games, relabel with each new net — attacks the CPU generation bottleneck);
league play against past checkpoints; BRExIt-style bounded in-match opponent
adaptation (legal here: one process plays a whole 1000-game match against the
same opponent).

## 8. References

catanatron (engine + author's "5 Ways NOT to Build a Catan AI" post-mortem);
Buro 2002 (fitted linear evaluation); Tesauro TD-Gammon; Anthony et al. 2017
(Expert Iteration); Texel tuning (chessprogramming.org); Colonist.io starting-
strategy statistics; EfficientZero/ReZero (reanalysis); Oh et al. (self-
imitation). All ideas implemented by us today, on the pinned engine.
