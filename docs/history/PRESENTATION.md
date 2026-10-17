# Presentation pack — 6 minutes + 4 minutes of questions

## The 6-minute arc (timed)

**0:00–0:40 — The problem.** One CPU core, a chess clock (300 ms + 12.5 ms/decision, ~1 s per
game), 2-player no-trade Catan, and an anchor bot F whose full source ships in the kit. The
organizers' own library search bots score 0.001 — they die by clock-out. The real opponent
is the clock.

**0:40–2:00 — What we shipped.** Four clock-gated layers: joint placement search, a corrected
1-ply evaluation (F's base plus five measured survivors — including the opponent-VP term that
prices Longest Road/Largest Army denial, which F lacks entirely), gated same-turn depth-2,
and hidden-VP-aware endgame gates. Zero faults, zero over-clock games, a third of the budget
in reserve. 0.526 over 1,000 games vs F; 200/200 against every one-trick archetype.

**2:00–4:00 — Three axes, three kill-shots.** We didn't guess the plateau; we proved it.
*Evals:* six evaluation functions of every lineage — hand-tuned, book-informed, logistic-
fitted on 1.16 million real positions — all within ±2%. *Search:* a 10×-depth teacher
disagrees with the shipped agent on **0 of 391 decisions** — depth is exhausted at this eval.
*Learning:* outcome labels carry no per-candidate gradient (we verified the deployed net
matches its math to ρ=1.000 and still ranks badly), and preference rankers trained to 80%+
transfer to nothing at any of eight scales — each collapse has a mechanism, not a shrug.

**4:00–5:00 — The signal we found.** A cross-turn teacher (end turn, sample dice, let the
opponent reply greedily, evaluate) disagrees with greedy on 30% of decisions — proof that the
missing dimension is the post-move world. It distilled poorly and played neutral in-play at
this eval's precision: the honest boundary of what this clock and this engine's node costs
(0.2 ms per copy) permit.

**5:00–6:00 — The close.** 27 measured variants, every number reproducible from logged seeds,
a plateau characterized to ±0.002 over 4,000 games, three validated twin submissions under a
best-of rule, and a precise program for real compute. We measured everything; here is the map.

## Question bank (measured answers, verifiable in experiments.md)

- **"Why not MCTS/AlphaBeta?"** The organizers' own table: 0.001 — clock-out. Rollouts cost
  ~30 ms each in this Python engine against a 12.5 ms/decision increment; we measured, not assumed.
- **"Why did the neural nets fail?"** Two mechanisms: outcome labels have no per-candidate
  signal at 579k samples (the ρ=1.000 deployment-consistency test proves it wasn't a bug);
  and learned residuals wired into a hand-balanced eval are a scale hazard — we have the
  λ-ladder table from 3e6 down to 100 showing the same collapse at every scale.
- **"What would you do with 10× compute?"** Not deeper search — a fast exact learned value
  function from weeks of self-play, making the 30%-disagreement cross-turn signal playable
  in-clock; plus reanalysis relabeling (reuse every generated game) and league self-play.
- **"How do you know 0.526 is your true strength?"** Four statistically-twin agents measured
  at 1,000 games each: 0.5230–0.5270. We submitted three of them as independent draws under
  the best-of-submissions rule — variance-aware selection, each a fully validated agent.
- **"Your worst failure?"** v3, 0.23: we stripped reachability features for speed and
  discovered they ARE the expansion policy. It redirected the whole architecture — and the
  report's §3 table.
- **"Did the strategy books help?"** Three books distilled to six concrete terms by a
  dedicated reader: 0.5175 — inside the plateau. Domain doctrine cannot beat the measured
  feature family in one day; we say so with the number.
- **"Robustness?"** Worst-half portfolio: 200/200 against road-rusher, city-builder,
  dev-spammer, VP-chaser; 0.52 vs tuned contender weights. No catastrophic matchup.
