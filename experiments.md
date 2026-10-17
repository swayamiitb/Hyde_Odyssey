# Experiment log — one row per run (feeds the report)

Machine notes: local laptop (Python 3.12.7, Windows). Scoring machine (EPYC 9R14,
Python 3.11) charges ~1.7-2x the CPU reported here. All evals via
`python -m catan_kit.evaluate`, default seed0=990000 unless noted.

| # | Version | Opponent | Games | Win rate | 95% CI | Mean CPU/game | Over clock | Faults | Notes |
|---|---------|----------|-------|----------|--------|---------------|------------|--------|-------|
| 0 | f_agent (baseline) | f_agent | - | 0.500 | - | 263 ms (selfcheck) | 0 | 0 | reference; definitionally 50% |
| 1 | v1 F-fixed | f_agent | 400 | 0.4475 | ±0.049 | 269 ms | 0 | 0 | REGRESSION. Suspects: actual-VP scoring inflates dev buying (sampled deck top = VP card makes BUY dominate at 3e14); flat LR weight → road obsession. |
| 2 | v1a (v1 minus those two) | f_agent | 400 | 0.5000 | ±0.049 | 250 ms | 0 | 0 | Safe fixes (r0=8e3, knight-in-hand=5, ports) are NEUTRAL vs F. Eval-tweak route is marginal; search must provide the edge. |
| 3 | v2 (search + opp-awareness) | f_agent | 400 | 0.5250 | ±0.049 | 303 ms | 0 | 0 | joint placement, same-turn depth-2 (top-5), enemy-VP term (LR/LA denial), enemy knights. First clear edge over F. |
| 4 | v3 (fast eval, no reachability in-turn) | f_agent | 400 | 0.2300 | ±0.041 | 471 ms* | 2 | 0 | **FAILURE.** Dropping reachable-production/buildable from the in-turn eval destroys road/settlement choice quality. Speed premise also wrong: node cost is copy-dominated, fast eval bought only ~2x. Reverted. (*CPU inflated by tuner contention.) |
| 5 | v4 (v2 + top-8 expansion) | f_agent | 200 | 0.5350 | ±0.069 | 391 ms | 0 | 0 | full eval everywhere, wider depth-2. **Current best clean.** |
| 5b | v4 | vp_chaser | 200 | 1.0000 | ±0.000 | 506 ms | 1* | 0 | robustness: punishes one-dimensional play totally (*contention-inflated over-clock). |
| 5c | v4 | contender-F | 200 | 0.4900 | ±0.069 | 437 ms | 0 | 0 | robustness: no collapse vs different weights; contender is a sturdier opponent. |
| 6 | v5 (win-now DFS + trade follow-ups) | f_agent | 400 | 0.5200 | ±0.049 | 602 ms | **23** | 0 | win-now at ≥7 VP / 40 ms re-ran every late-game decision → 523 random-move degradation. Fix in v6: ≥8 VP, 15 ms. |
| — | weight tuner (15 weights × 4 factors, 200-game paired fitness) | f_agent | 60 runs | best: knight_in_hand=2.5 → 0.550 (noise-level, unvalidated) | | | | | cycle 1 stopped to free cores for data generation. |
| 7 | v6 (ValueNet 114→256→256→1, 579k rows from 4800 teacher games, net-scored 1-ply + depth-2 + cheap win-now) | f_agent | 400 | 0.4675 | ±0.049 | 378 ms | 1 | 0 | net v1 WORSE than teacher: dice-noisy outcome labels + VP-dominant teacher squash washed out subtle features. |
| 7b | v7 (blend eval + 0.3×net logit) | f_agent | 400 | 0.4675 | ±0.049 | 418 ms | 1 | 0 | blend didn't rescue the net. Pure-net path paused pending better labels. |
| 8 | v4k (tuner's knight=2.5) | f_agent | 400 | 0.5275 | ±0.049 | 341 ms | 0 | 0 | tuner's 200-game 0.550 did NOT replicate at 400. Discarded. |
| 9 | v8 (Texel: logistic-regression eval on 579k positions, AUC 0.847, v4 skeleton) | f_agent | 400 | 0.5275 | ±0.049 | 305 ms | 0 | 0 | Data-fitted linear eval == hand-tuned (0.535) within noise. Confirms eval-axis plateau at ~0.53. Calibrated + interpretable: opp LR strongest −, our wheat/ore totals strongest +. |
| — | Round-2 pipeline: v8 teacher, VPDR labels + hidden-VP aux head + progress weighting + EMA (v9) | — | — | in flight | | | | | generation running; placement-upgrade v4p (coverage diversity + denial) also queued. |
| 10 | v4p (placement coverage+denial) | f_agent | 400 | 0.4975 | ±0.049 | 325 ms | 0 | **426** | FAULTS (~1.8% of decisions, near the 2% DQ line) + no gain even ignoring them. Discarded. |
| 11 | v8b (Texel refit on round-2 data) | f_agent | 400 | **0.1850** | ±0.038 | 384 ms | 0 | 0 | COLLAPSE. Same agent code as v8; only fitted weights differ. |
| 12 | v9 (round-2 net: VPDR targets, aux hidden-VP head, EMA) | f_agent | 400 | **0.1875** | ±0.038 | 328 ms | 0 | 0 | COLLAPSE. Two architectures, same data, same ~0.19 ⇒ dataset/pipeline issue, not architecture. |
| 12b | v8 re-measured with current consistent weights | f_agent | 200 | 0.4650 | ±0.069 | 329 ms | 0 | 0 | The 0.5275 was measured under a fit/deploy feature mismatch; the "fixed" consistent fit is weaker. Fitted-eval deployment path: unresolved systematic issue. |
| — | Round-2 label audit | — | 578k | z climbs 0.10→0.91 monotonically with VP differential; no NaN; healthy row counts | | | | | Labels SANE. Root cause of the collapse NOT identified (time-boxed; documented as open). |
| 13 | v4 FINAL validation | f_agent | 1000 | **0.5230** | ±0.031 | 338 ms | **0** | **0** | worst game 938 ms. THE SUBMISSION. |
| 13b | v4 portfolio | road_rusher / city_builder / dev_spammer | 200 each | 1.0000 / 1.0000 / 1.0000 | ±0.000 | ~360 ms | 0 | 0 | Sweeps all three zoo archetypes. Contender (re-run): 0.5200; vp_chaser: 1.000. Worst-half = the two strong opponents. |
| 14 | v10 (v4 + isolated cheap win-now ≥8VP/15ms) | f_agent | 400 | 0.4800 | ±0.049 | 423 ms | 0 | 0 | Fourth failed delta. Win-now hurts even cooled down. v4 FROZEN. |

## v1 changes vs F (each to be ablated)
1. own ACTUAL VPs (not public) as the dominant term — REVERTED in v1a (harmful)
2. reachable_production_0 weight 8e3 (F: 0) — neutral
3. longest_road flat 12.0 (F: 10 normally, 0.1 while buildable nodes exist) — REVERTED in v1a (harmful)
4. KNIGHT_IN_HAND weight 5.0 (largest-army progress) — neutral
5. port bonus: specialty 2.5e3 (matched to production), generic 1.2e3 (F: 0) — neutral

## Revamp wave (books + cross-turn + ranker + DART) — final

| # | Version | Opponent | Games | Win rate | Notes |
|---|---------|----------|-------|----------|-------|
| 15 | v14 DART (fast full eval + in-play cross-turn, 0.6 blend) | f_agent | 400 | 0.4275 | FAILED. Reachability cache keyed on road COUNT (collisions) + buildable term dropped + blend too hot. |
| 16 | v14b (exact keys, real buildable, 0.3 blend, ≥6VP gate) | f_agent | 400 | 0.4500 | Still a regression. In-play cross-turn with an approximated eval is net-negative; the teacher helps only as an offline label source. |
| 17 | v11 (strategy-book terms, subagent-distilled) | f_agent | 400 | 0.5175 | Exactly the projected plateau. Neutral. |
| 18 | v13_1/2/3 (ranker residual, λ=3e5/1e6/3e6) | f_agent | 150 each | 0.027 / 0.080 / 0.067 | CATASTROPHE. λ mis-scaled by ~10^3: unbounded ranker outputs (±10-50) at λ≥3e5 swamp the production/hand/buildable tiers of the eval, randomizing every non-VP decision. Same collapse signature as round 2. |
| 19 | v13_2 escalated | f_agent | 400 | 0.0725 | Confirms the scale diagnosis. A λ≈1e2-1e3 ladder was never tried — documented as the missed rung. |
| 20 | v15 (v4 + knight-sequence expansion + trade-enables-build follow-ups — the code review's two real blind spots) | f_agent | 400 | PENDING | last candidate before freeze. |

**Day total: 20 measured variants. Nothing beat v4's 0.5230 outside noise. The local optimum is real.**

## Final clean experiments (confound-free close-out)

| # | Version | Games | Win rate | Verdict |
|---|---------|-------|----------|---------|
| 21 | v13c/d/e (ranker residual at CORRECT scale, λ=100/300/1000) | 150 each | 0.127 / 0.207 / 0.147 | Scale hypothesis REFUTED: even tiny ranker residuals destroy play. The pairwise ranker does not transfer to in-decision rescoring. Learned track definitively closed. |
| 22 | v16 (pure v4 + ONLY cross-turn re-ranking of top-2, v4's own eval) | 400 | 0.5075 ±0.0490 | NEUTRAL vs v4 (0.523 ±0.031). CPU 625ms, 0 over-clock. The 30%-disagreement signal does not convert to wins in play. |

**DAY TOTAL: 25 measured variants. Final submission: v4 (sub1.zip) at 0.5230 ±0.0310 over 1000 games.**
Every axis exhausted with confound-free measurements: eval (6 variants), search (4), learned (8, three
label families, five scales), doctrine/books (2), tactical expansions (2), cross-turn (3).

## Fresh-start thread (Orion + net autopsy) — 15:00-16:15

| # | Version | Opponent | Games | Win rate | Notes |
|---|---------|----------|-------|----------|-------|
| 23 | orion v1 (v4 + F-exact cross-turn layer, current_color BUG -> layer inert) | f_agent | 200 screen | 0.5250 ±0.069 | +0.03 vs v4 paired was noise; 1,000-game: **0.5060 ±0.031** = v4 (0.5070). 428 ms, 0 faults. |
| 24 | orion v1 inert — 1,000 confirm (with v4 paired re-run) | f_agent | 1000 + 1000 | 0.5060 vs 0.5070 | Statistically identical. Screen deltas at 200 games DO NOT transfer. |
| 25 | orion v2 (current_color() FIXED, layer actually runs F's policy) | f_agent | 200 paired | **0.4600 ±0.069** vs v4 0.4950 | NEGATIVE. Exact-opponent cross-turn tested at its strongest; in-play cross-turn family CLOSED (with v14, v16). |
| 26 | net autopsy: probe_net.py on value_net_v2 + shard forensics | — | — | — | Checkpoint DEAD: logit std 0.019 on own training rows, won/lost gap 0.009 logits. Root cause: T (teacher) column zero in every round-2 shard -> net fitted on terminal-only labels. Pipeline otherwise verified clean. |
| 27 | bench_orion.py primitive costs | — | — | — | copy 34us, copy+exec 78us, F eval 71us, F reply chain ~3ms, F placement argmax (54 nodes) ~7ms. |

**Bottom line: v4 (sub1) final. sub2_v11.zip / sub3_v16.zip packaged as best-submission variance hedges. Route to top tier: teacher-label regen -> net retrain -> iterated self-play.**

## Twin deep-reads (final selection data, all 1000 games vs F, fresh seeds)

| Agent | Win rate | Mean CPU | Note |
|---|---|---|---|
| v4 | 0.5230 ±0.0310 | 338 ms | original champion |
| v8 (Texel) | 0.5250 ±0.0310 | 379 ms | twin |
| v16 (cross-turn twin) | 0.5270 ±0.0309 | 666 ms | 1 over-clock game — hotter; NOT submitted |
| v17 (denial gate + tempo bonus) | 0.5260 ±0.0309 | 313 ms | twin; 400-game read 0.5325 |

Plateau characterized to ±0.002 over 4,000 games. Submission set: v4 + v17 + v8 (three draws;
v16 excluded on clock risk). 27 variants measured in total.
