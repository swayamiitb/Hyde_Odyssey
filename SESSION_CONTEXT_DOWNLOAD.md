# Session Context Download — The Build Thread (Hyde × WnCC Catan Hackathon)

> THE complete record of this assistant session, 17 Oct 2026, ~10:30–14:45 IST. Companion to
> `FULL_CONTEXT_DOWNLOAD.md` (parallel team thread + master facts), `GAME_DAY_CONTEXT.md`
> (repo/engine deep reference), `DECISION_MODEL_BLUEPRINT.md` (build spec), `RESEARCH_SOTA_DOSSIER.md`
> (techniques), `DATASET_CONTEXT.md` (data plan). Self-contained: a fresh session or teammate
> can read only this to know everything this thread did, found, and decided.

---

## PART A — What this session did, in order

### A1. Context gathering (~10:30–11:00)
Cloned `lyric-hyde/catan-hackathon-agent` to `D:\Odyssey\catan-hackathon-agent`; read all ~2,150
lines (PROBLEM_STATEMENT, README, catan_kit/{obs,match,host,evaluate,sandbox,selfcheck,_patch},
examples). Fetched and verified pinned catanatron engine sources (commit d3f4ad0) into
`D:\Odyssey\.engine_ref` — F's exact brain (value.py), the 614-vector (features.py), setup order
and LA/LR semantics (apply_action/state_functions). Deliverable: `GAME_DAY_CONTEXT.md`.
**Corrections to prep brief:** Monopoly IS in (25-card deck); no player trading at all; learned
component NOT mandatory; funnel/judging precise.

### A2. Q&A settled (~11:00–11:20)
- **First steps / compute picture:** three machines — their AWS scoring box (constraint only:
  1 core/2GB/1.5s), team machine via SSH (primary dev + only upload path), our laptops
  (unlimited practice). First submission = early pipeline proof.
- **Neural nets:** explicitly allowed (torch/sklearn preinstalled on the ring); LLMs at play
  banned; any model must load with numpy/scipy/sklearn/torch only, ship in the ≤100MB zip.
- **"Sure win" honesty clause:** kept all day — max-probability build on the actual formula,
  no guaranteed-win claims.

### A3. Build wave 1 (v1→v5) + tuner (~11:00–12:50)
Local env: Python 3.12.7 venv (kit passes selfcheck on 3.12; scoring is 3.11 — final validation
on team machine), pinned catanatron 3.3.0 + networkx 3.6.1. 32 logical cores.
- v1 (5 eval fixes) → **0.4475** (dev-buy phantom bias: scoring own actual VPs at 3e14 makes
  BUY_DEVELOPMENT_CARD dominate when the sampled deck top is VP)
- v1a (safe fixes only) → 0.5000 (eval tweaks near F's weights are neutral)
- v2 (+enemy-VP term pricing LR/LA denial, joint placement, gated depth-2) → 0.5250
- v3 (fast eval without reachability) → **0.2300** (reachability IS the expansion policy;
  node cost is copy-dominated anyway)
- v4 (v2 + top-8 expansion) → **0.5350** — the champion
- v5 (win-now DFS 40ms/≥7VP) → 0.5200 with 23 over-clock games (self-destructed)
- Parallel coordinate-descent tuner (15 weights × 4 factors, 8 workers): best 200-game candidate
  0.550 did NOT replicate at 400 (0.5275) — 200-game fitness is a screen, not a verdict.
- Robustness: v4 200/200 vs vp_chaser; 0.490 vs contender-weighted F.

### A4. Neural round 1 (v6–v8) (~12:00–13:00)
- Self-play generator (`selfplay_gen.py`): 4,800 teacher games → **579,090 rows** in ~7 min
  (≈20 games/s aggregate; observability rule satisfied — features from the masked seat's view,
  one shared encoder).
- v6 (ValueNet 114→256→256→1, outcome labels) → **0.4675** — worse than teacher.
- v7 (eval + 0.3×net blend) → 0.4675.
- Texel/Logistello fit (logistic regression on same data, AUC 0.847) → v8 → 0.5275.
  **Eval-axis plateau confirmed.**
- GPU: RTX 4070 present but PyPI serves CPU wheel; per team's own B10 rule, CPU training
  (10 s/epoch for the tiny net) — GPU not the bottleneck; self-play generation is.

### A5. Round 2 + first workflow (~13:00–13:45)
- Round-2 data (578,566 rows) with VPDR labels, hidden-VP ground truth (H), turn (TN); teacher
  = v8. Trainer `train_value_net2.py` (VPDR blend + aux hidden-VP head + progress weighting + EMA).
- **/workflow #1** ("Round-2 bot build and measurement"): pure world.run orchestration; three
  submissions to get it running (Windows spawn ENOENT on relative exe path → absolute path;
  then doubled path after discovering cwd = repo root). Results: v4p (placement doctrine)
  **0.4975 + 426 faults**; v8b (Texel r2) **0.1850**; v9 (round-2 net) **0.1875**.
- **Diagnostics:** label audit SANE (z climbs 0.10→0.91 with VP differential); deployed scorer
  matches model math exactly (ρ=1.000); correlation with good heuristic +0.62 average, negative
  on a minority. Verdict: outcome labels carry no per-candidate gradient — the models rank on noise.

### A6. Dataset plan review + revamp workflow (~13:45–14:45)
- Deep review of team's DATASET_CONTEXT.md: adopted D2/D3 (done), D10 zoo (road_rusher,
  city_builder, dev_spammer written — dual use: portfolio + diversity), D8 enabler (candidate
  scores), D4 league; rejected D11/12/13, D5, memmap (reasoned in-thread).
- **Deep teacher discovery:** v4deep (10× same-eval search) disagrees with v4 on **0/391
  decisions** — same-turn search is EXHAUSTED at this eval. Built **v4xct** (cross-turn
  teacher: END_TURN → 3 sampled rolls → opponent greedy per F's eval → our eval): **30%
  disagreement**, ~2 s/game offline. Preference-pair generator `selfplay_gen3x.py` (~150k pairs)
  + pairwise margin-loss `train_ranker.py`.
- Extracted three strategy books to text (books/book1..3.txt).
- **/workflow #2** ("Bot revamp: books + cross-turn self-improvement"): strategist subagent
  (6 terms from books), integrator subagent (v11; v13 λ-variants), world.run gates. Results:
  v11 (book terms) **0.5175**; v13 λ∈{3e5,1e6,3e6} **0.027–0.080** (ranker residual swamps
  the eval's production/hand tiers — λ mis-scaled ~10³).
- **v14 DART** (fast full eval + in-play cross-turn): **0.4275** (reachability cache keyed on
  road COUNT = collisions; buildable term dropped) → **v14b fixed: 0.4500** — still regression;
  approximation error exceeds cross-turn gain.
- **v15** (team code review's two real blind spots: PLAY_KNIGHT_CARD expansion + MOVE_ROBBER
  follow-ups + MARITIME_TRADE→build): **0.4650** — correct fixes, still worse; v4's regime is
  a narrow local optimum.

### A7. Final clean experiments (~14:15–14:45)
- λ ladder at correct scale {100,300,1000}: **0.127 / 0.207 / 0.147** — scale hypothesis
  REFUTED; the ranker does not transfer to in-decision rescoring at any scale.
- **v16** (pure v4 + ONLY cross-turn re-ranking of top-2, v4's own eval): **0.5075 ±0.0490** —
  statistical twin of v4; the cross-turn signal does not convert to wins in play.

**DAY TOTAL: 25 measured variants. FINAL SUBMISSION: v4 (`sub1.zip`), 0.5230 ±0.0310 over
1,000 games, 0 faults, 0 over-clock, portfolio-swept.**

---

## PART B — Every question in this thread, and the settled answer

| # | Question | Settled answer |
|---|---|---|
| 1 | Gain full context; deep-dive the repo | Done; corrections: Monopoly in, no trading, learning optional; engine verified at source level |
| 2 | Detail it in-chat, not just files | Full briefing delivered; condensed into GAME_DAY_CONTEXT.md |
| 3 | What first? Laptop vs their machines? | 3-machine picture; first submission = pipeline proof; build order settled |
| 4 | Must win at all costs; compute? nets allowed? | Team machine ≠ laptop; nets explicitly allowed (torch on ring); honesty clause on "sure win" |
| 5 | Check FULL_CONTEXT_DOWNLOAD/blueprint | Adopted P0–P5 pipeline, portfolio selection, zero-fault chain; GPU = CPU-torch decision |
| 6 | /workflow for parallel tasks | Workflow #1 (3 compile/spawn fixes) → round-2 triple failure, honestly labeled |
| 7 | Deep go-thru of dataset plan, put into action | D-catalog adoption/rejection (A6); zoo built; observability invariant confirmed |
| 8 | What was picked from the research list | Item-by-item verdict table (ExIt core; Gumbel/ISMCTS/GNN rejected with reasons) |
| 9 | "It's 12:08 not 14:05" | Clock corrected; schedule re-derived with MORE time |
| 10 | Mediocre scores; revamp with books + max compute + self-learning | Workflow #2: books + cross-turn teacher + ranker; v14 DART built; all measured |
| 11 | Projected sub2? DART doc? conclusive why-not-improving? | Projections (v11 ~0.52, v13 0.53–0.56 conditional); DART mapped adopt/have/reject; three-mechanism plateau answer: eval family exhausted, search exhausted, learning label-starved |
| 12 | Workflow outputs + R&D + giant leap | Revamp results reported; v14 built/measured (0.4275) |
| 13 | Code-review doc + wire to redesign; need guaranteed 0.75+ | v15 fixes implemented exactly and measured (0.4650); straight talk: no 80-minute path to 0.75; explained what 0.8-claims likely are; redirected to formula-optimal play (report/presentation = half the score) |
| 14 | Fresh start, ultimate bot, 2h left | v16 + corrected λ ladder = the two remaining confound-free tests; both neutral/catastrophic; day closed with complete evidence map |

---

## PART C — The complete measurement ledger (all vs F unless noted)

| Version | Idea | Games | Win rate |
|---|---|---|---|
| v1 | 5 eval fixes | 400 | 0.4475 |
| v1a | safe fixes only | 400 | 0.5000 |
| v2 | +opp-awareness, joint placement, depth-2 | 400 | 0.5250 |
| v3 | fast eval, no reachability | 400 | 0.2300 |
| **v4** | **v2 + top-8 expansion** | **200→1000** | **0.5350 → 0.5230 ±0.031 (final)** |
| v4 | vs vp_chaser / contender / road_rusher / city_builder / dev_spammer | 200 ea | 1.000 / 0.490+0.520 / 1.000 / 1.000 / 1.000 |
| v5 | win-now DFS (hot) | 400 | 0.5200 (23 over-clock) |
| v4k | tuner knight=2.5 | 400 | 0.5275 |
| v6 | ValueNet v1, outcome labels | 400 | 0.4675 |
| v7 | eval+net blend | 400 | 0.4675 |
| v8 | Texel linear eval (AUC 0.847) | 400 | 0.5275 |
| v4p | placement doctrine | 400 | 0.4975 + 426 faults |
| v8b | Texel on round-2 data | 400 | 0.1850 |
| v9 | round-2 net (VPDR+aux+EMA) | 400 | 0.1875 |
| v8 re-measured | consistent r1 weights | 200 | 0.4650 |
| v10 | v4 + isolated cheap win-now | 400 | 0.4800 |
| v11 | strategy-book terms | 400 | 0.5175 |
| v13 λ=3e5/1e6/3e6 | ranker residual | 150 ea | 0.027 / 0.080 / 0.067 |
| v13 best λ escalated | | 400 | 0.0725 |
| v14 / v14b | DART fast-eval + cross-turn | 400 ea | 0.4275 / 0.4500 |
| v15 | knight-seq + trade→build expansion | 400 | 0.4650 |
| v13c/d/e | ranker λ=100/300/1000 | 150 ea | 0.127 / 0.207 / 0.147 |
| v16 | v4 + only cross-turn top-2 | 400 | 0.5075 |

---

## PART D — Mechanistic know-how (what survives the day)

1. **The clock is the true opponent.** Library search bots die at 0.001 by timing out; our own
   v5 lost ranking the same way. Budget tiers on `obs.cpu_left` are non-negotiable.
2. **The eval family is exhausted at ~0.52.** Six evals of every lineage within ±2%; a logistic
   fit on 1.16M real outcome-labeled positions matches hand-tuning (Buro reproduces in miniature).
3. **Same-turn search is exhausted at that eval**: 0/391 disagreement at 10× depth. Deeper = same
   moves (and the engine author measured depth-3 playing WORSE).
4. **Learned components are a transfer hazard**: outcome labels have no per-candidate gradient;
   preference rankers train fine (80%+ pairwise) but do not transfer to in-decision rescoring at
   ANY λ; dominant-scale eval terms must be robust to sampled-world noise (the v1 phantom-buy).
5. **The cross-turn signal exists (30% teacher disagreement) but converts to zero in-play** at
   this eval's precision and this engine's node costs (~0.2ms copy, ~0.3ms full eval).
6. **Negative results compound**: the v1/v3/v5/v13/v14 post-mortems each eliminated a whole
   design family and are the report's strongest material.
7. **Measurement religion**: same-seed comparisons, disjoint TRAIN(1k/2k/3k)/EVAL(990k+) ranges,
   400-game development reads, 1000-game finals; 200-game fitness is a screen only.

## PART E — Final state and outstanding actions

- **Submission: `sub1.zip` = `my_team/agent.py` = v4** (self-contained; imports random/time/
  catanatron only). 0.5230 ±0.0310 vs F over 1,000 games; 0 faults; 0 over-clock; ~⅓ CPU budget
  used with ~2× margin for the slower scoring core.
- **Report: `REPORT.md`** — complete narrative incl. all failures with mechanisms; convert to PDF,
  upload by 17:30 (`~/upload.sh report.pdf`).
- **Presentation**: 6+4 min; lead with the map (three exhausted axes), the cross-turn teacher,
  the λ post-mortem; "10× compute" answer = cross-turn expectimax at compiled speed + reanalysis
   + weeks of self-play.
- **Portal**: https://main.d2sgimrfzis7m5.amplifyapp.com/ · SSH `team<name>@32.190.233.160`.
- Deadlines: uploads 17:00, report 17:30, announcements 17:50, presentations 18:00, end 19:30.

## PART F — Workspace index & reproducibility

| Path | Contents |
|---|---|
| `D:\Odyssey\catan-hackathon-agent\my_team\` | agent.py (=v4, THE submission), v1…v16 variants, encoder.py, texel_eval*.npz, value_net*.pt/json, ranker_v1.pt, STRATEGY_TERMS.md, sparring/ (zoo + contender) |
| `…\experiments.md` | the full 25-variant ledger (the report's data spine) |
| `…\REPORT.md` | final report draft |
| `…\selfplay_gen*.py, train_*.py, texel_fit.py, tuner.py, diag_score.py, wf_eval.py` | the day's tooling |
| `…\data\ (r2), data_r1\, data3\ (pairs)` | 1.16M rows + ~150k preference pairs |
| `…\books\book1-3.txt` | extracted strategy books |
| `…\.zcode\workflow-drafts\` | the two workflow scripts |
| `D:\Odyssey\*.md` | the five context documents incl. THIS file |

Reproduce any number: `python -m catan_kit.evaluate --a <variant> --b examples/f_agent.py
--games N --seed0 <range>` with `PYTHONHASHSEED=0` (wf_eval.py handles it). Every seed range
used is logged in the ledger rows above.
