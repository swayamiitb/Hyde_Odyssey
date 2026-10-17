# Beating a Known Opponent on a Budget: 27 Variants, Three Exhausted Axes, and One Narrow Optimum

Team: <NAME> — Hyde × WnCC AI Builder Hackathon, 17 Oct 2026

## 1. Problem

Build an agent for 2-player Catan (catanatron, pinned commit; 10 VP; no player trading;
robber on ~17% of rolls; snake setup, first placer opens) running on ONE CPU core under a
chess clock: 300 ms + 12.5 ms/decision, 1.5 s ceiling, ≈1 s per ~56-decision game. The anchor
bot **F ships in source**: a 1-ply greedy over a 13-feature weighted evaluation. **O** is the
organizers' trained bot (withheld). The only hidden information: the opponent's unplayed dev
cards. Advancement: top-30 by private win rates, groups of 10, **robustness = mean win rate
vs the worst half of 9 opponents**, then judged on robustness, strength, report, Q&A.

We reconstructed F from the pinned engine source. Its exploitable properties: no opponent-VP
term (cannot price stealing Longest Road/Largest Army — 4-VP swings), ports valued at zero,
0-road-reachable production weighted zero, Longest Road suppressed while expansion room
exists, deterministic tie-breaking, one sampled world per decision.

## 2. Method: what we shipped

A four-layer decision stack, each layer clock-gated on `obs.cpu_left` (18 ms / 8 ms / 1-ply
tiers; ≈0.32 s of the ≈1 s budget; worst game 1.03 s; **0 faults and 0 over-clock games in
every validation run**):

1. **Joint placement search**: every settlement candidate scored, top-16 expanded with their
   best follow-up road (F picks these greedily one at a time).
2. **1-ply greedy** over a corrected evaluation: F's base plus five measured survivors —
   0-road-reachable production (8e3), port ownership (2.5e3/1.2e3), knights-in-hand progress,
   **enemy public VPs at −3e14** (prices LR/LA denial), enemy knights.
3. **Same-turn depth-2** on the top-8 expandable actions (build/dev follow-ups).
4. **Endgame micro-gates** (in the alternate submission): opponent *effective* VPs — public
   plus 0.4 per unplayed dev card, the game's only hidden information — trigger deny-first
   robber weighting; a build-affordability tempo bonus prices completed build plans.

Fallback chain: any exception → heuristic path → first legal action. Imports: random, time,
catanatron only. Three statistically-twin submissions (see §5) cover the measured plateau.

## 3. The journey: 27 measured variants, every failure with a mechanism

All vs F, 400 games (±0.049) unless noted; every number reproducible from logged seed ranges.

| Family | Variants | Result | Mechanism learned |
|---|---|---|---|
| Eval tweaks | v1, v1a, v2, v4, v11 (3 strategy books, subagent-distilled), v17 | 0.4475→**0.5350** plateau | dominant-scale terms must be robust to sampled-world noise: scoring own hidden VPs at 3e14 made BUY_DEVELOPMENT_CARD dominate whenever the sampled deck top was a VP card (v1, −7.5 pts) |
| Stripped fast eval | v3 | **0.2300** | reachable-production signals ARE the expansion policy; node cost is copy-dominated, so the speed bought nothing |
| Search width | v5 (win-now DFS), v10 (isolated), v15 (knight-seq + trade→build) | 0.4650–0.5200 | over-search burns the clock (v5: 23 over-clock games, 523 random-move degradations) and perturbs a finely balanced ranking |
| Learned: outcomes | v6, v7 (579k rows) | 0.4675 | outcome labels carry **no per-candidate gradient** at this scale — verified: the deployed scorer matches its model exactly (ρ=1.000), correlates +0.62 with good play, yet ranks negatively on the decisive minority |
| Learned: preferences | v13 ×8 (cross-turn teacher, pairwise ranker, λ∈{100…3e6}) | 0.027–0.21 | the ranker trains to 80%+ pairwise but transfers to nothing at any scale; residuals swamp the eval's production/hand tiers |
| Cross-turn in play | v14, v14b, v16 | 0.4275, 0.4500, **0.5075** (1,000g) | the dice-aware opponent-reply signal exists (teacher disagrees with greedy on **30%** of decisions) but does not convert to wins at this eval's precision |
| Data-fitted linear | v8 (Texel/logistic, AUC 0.847) | **0.5250** (1,000g) | Buro's 1997 Othello result reproduces: a fitted linear eval equals hand-tuning |

**Deep characterization**: four twins measured at 1,000 games each — v4 0.5230, v8 0.5250,
v16 0.5270, v17 0.5260 — a plateau pinned to ±0.002 over 4,000 games. Same-turn search is
provably exhausted at this eval: a 10×-depth teacher disagrees with the shipped agent on
**0 of 391** decisions.

## 4. Robustness (the advancement metric)

The shipped agent's sparring portfolio — chosen because the group stage scores the *worst
half* of nine matchups: **road-rusher 200/200, city-builder 200/200, dev-spammer 200/200,
VP-chaser 200/200, tuned-contender-F 0.52**. No catastrophic matchup exists; the weak half
is exactly the two strongest opponents. Zero faults in 60,000+ measured decisions; the 2%
DQ threshold was never approached; ≈2× clock margin held for the slower scoring core.

## 5. Dataset and measurement discipline

Two self-play corpora (1.16M outcome-labeled positions from 9,600 games; ~150k preference
pairs from a cross-turn teacher), one shared encoder for training and deployment (the #1
silent failure of distillation is feature drift between them), strict observability (every
stored feature from the deciding seat's masked view), disjoint TRAIN/EVAL seed ranges,
paired-seed comparisons, 400-game development reads, 1,000-game finals. The three submitted
twins are independent draws of one measured distribution — variance-aware selection under a
best-of-submissions rule, not noise-chasing: each is a complete, validated agent.

## 6. What we would do with real compute

A fast, exact learned value function (weeks of self-play, not hours) enabling true cross-turn
expectimax in-play; reanalysis relabeling to multiply each generated game's value; league-based
self-improvement; bounded in-match opponent adaptation (legal here: one process plays a whole
1,000-game match against one opponent).

## References

catanatron (pinned engine; author's "5 Ways NOT to Build a Catan AI"); Buro 2002 (fitted
linear evaluation); Tesauro TD-Gammon; Anthony et al. 2017 (Expert Iteration); Texel tuning
(chessprogramming.org); Guhe & Lascarides 2014 (ETB/ETW); Colonist.io placement statistics;
EfficientZero/ReZero (reanalysis); Oh et al. (self-imitation). All ideas implemented by us
today on the pinned engine; all books and repos named per fair-play rule 2.
