# SOTA Research Dossier — Catan Bot Build

> Compiled 17 Oct 2026 (~12:00 IST) from a web sweep of papers, theses, blogs, wikis and repos,
> filtered for ONE criterion: **usable today inside our constraints** (Python-only, numpy/scipy/
> sklearn/torch, 1 CPU core, ~12.5 ms/decision, ~5 h remaining). Companion to
> `DECISION_MODEL_BLUEPRINT.md` (the build spec) and `FULL_CONTEXT_DOWNLOAD.md` (master context).

---

## 0. The five findings that CHANGE our plan (read first)

These came out of the sweep and materially alter decisions in the blueprint:

1. **In-play rollouts are mathematically off the table — distillation is now mandatory, not optional.**
   The catanatron author measured **~3 s per 100 MCTS playouts** in this exact Python engine
   (≈30 ms per playout). Our whole decision budget is ~12.5 ms. Even ONE full-game rollout
   overdraws a decision. Consequence: value labels from rollouts are generated OFFLINE
   (laptop, unclocked) and compressed into the net; the net IS our rollout replacement. This is
   precisely the ExIt/"Thinking Fast and Slow" architecture — and it validates the blueprint.

2. **Depth-3 pathology warning.** In his experiments, expectimax-style Alpha-Beta **depth 2
   improved** the ValueFunctionPlayer only slightly, and **depth 3 played WORSE** (suspected
   high-risk/high-reward overoptimism or bug). Our plan's "selective 2-ply" is the right call;
   if we try 3-ply anywhere, it must pass a 1,000-game ablation before shipping. Never assume
   deeper = better in this engine.

3. **A proven label-shaping trick: VP_DISCOUNTED_RETURN.** `label = 0.9999^num_turns × final_VPs`
   — faster wins score higher; this label showed real improvement when plain outcomes didn't.
   We adopt it as a value-net target variant (alongside win/loss and search-score targets): it
   teaches the net to prefer closing games out, which matters under the 1000-turn double-loss rule.

4. **Texel tuning replaces CMA-ES as the primary weight-fitting method.** The chess world's
   standard: collect (position, outcome) pairs from self-play, fit eval weights by logistic
   regression — sigmoid(eval) ≈ win probability. It's convex-ish, gradient-based, runs in
   sklearn/numpy in minutes, and directly gives a *calibrated* eval (its output IS a win
   probability). CMA-ES/SPSA remain the fallback for knobs that only affect win rate noisily.

5. **Value-net-over-policy-net is validated by his failure.** His RL/policy attempts collapsed
   on class imbalance (≈30% of decisions are END_TURN → nets learned to predict END_TURN) and a
   5,000-action flattened action space. Our architecture (score successors with a value net,
   argmax over the real legal list) is immune to both failure modes by construction.

## 1. Catan-specific intelligence

### 1.1 "5 Ways NOT to Build a Catan AI" — Bryan Collazo (catanatron's author) ★★★ READ
<https://medium.com/@bcollazo2010/5-ways-not-to-build-a-catan-ai-e01bc491af17>
Post-mortem of every failed approach ON OUR EXACT ENGINE: CEM/online-DQN (failed: class
imbalance, garbage random-play data, 5k-dim action space), supervised value from random games
(weak until VP_DISCOUNTED_RETURN shaping), MCTS (not superhuman even at 10k sims; ~30 ms/playout
in Python), TensorForce DQN (8 h training → still below ValueFunctionPlayer), expectimax
alpha-beta (depth 2 slight win, depth 3 worse). His conclusion — the Stockfish path
(eval + search + incremental community improvement) is the ripe one; he explicitly wanted
alpha-zero-general / muzero-general integrated (§2.1 below) and flags supervised learning on
stronger-than-random games as the promising direction. **That is literally our pipeline.**

### 1.2 Szita, Chaslot & Spronck, "Monte-Carlo Tree Search in Settlers of Catan" (ACG 2009) ★★
<https://spronck.net/pubs/ACG12Szita.pdf> · <https://link.springer.com/chapter/10.1007/978-3-642-12993-3_3>
The foundational Catan-MCTS paper (~170 citations). Use: the academic framing of Catan's
challenges (imperfect info, long horizon, trading) and confirmation that flat MCTS plateaus.

### 1.3 Dobre & Lascarides, "POMCP with Human Preferences in Settlers of Catan" (AAAI 2020) ★
<https://cdn.aaai.org/ojs/13014/13014-52-16531-1-2-20201228.pdf>
POMCP = partially-observable MCTS on sampled belief states — the same move as our
`obs.sample_world` averaging. Too slow for our clock, but its belief-state handling justifies
our PIMC design (hidden info here is only the dev deck — small; PIMC's known failure modes
[strategy fusion, non-locality] are mild at this information size).

### 1.4 RL-Catan attempts (for the report's related-work section)
- settlers-rl (deep RL + GAE value function): <https://settlers-rl.github.io>
- van der Kooij, actor-critic Catan thesis (Utrecht 2020):
  <https://studenttheses.uu.nl/bitstreams/9ced96a3-9a86-4d73-abdb-4b62f5698c2a/download>
- Stanford AA228 RL-Catan project: <https://web.stanford.edu/class/aa228/reports/2020/final128.pdf>
- Whelan, RL in Catan: <https://sidhardhburre.com/final>
All confirm: end-to-end RL underperforms tuned eval+search at Catan-scale compute — cite as
evidence for our distillation choice.

### 1.5 Placement doctrine — Colonist.io data-backed starting guide ★★★ IMPLEMENT
<https://blog.colonist.io/guide-to-catan-starting-strategies>
Converts directly into placement-eval features for our teacher:
- **Four archetypes**: wood/brick+Longest-Road; balanced hybrid expansion; sheep/wheat/ore
  cities+dev; 2:1-port mono-resource. Placement scoring should reward the BEST executable
  archetype given the board, not a single fixed weight vector.
- **Complementary placements**: settlement 2 covers the resources settlement 1 misses (target:
  all 5 across the pair); number diversity across BOTH placements (8-5-10 + 6-9-4 ideal);
  count dots, not just numbers.
- **Scarcity/monopoly**: grab rare resources early — in 2-player no-trade this becomes
  **denial**: if we hold most of a resource, the opponent must pay 4:1 bank rates for it.
- **Deny-by-placement**: take the spot the opponent needs (we see their first settlement when
  placing our second — snake order gives P1 back-to-back picks, P0 the last+first-turn combo).
- **Roads point outward** at expandable (not stealable-goldmine) spots; late placers should
  avoid road-pointing at contested goldmines.
- **Ports**: 2:1 next to matching high production = jackpot; port-vs-production tradeoff is
  board-dependent → a learned/tuned term, not a constant.
Also: pip-count math at <https://kingofcatan.net/initial-settlement-placements-production>.

## 2. Search algorithms (the clock-aware toolbox)

### 2.1 AlphaZero-General / MuZero-General (the author's own wishlist) ★★
<https://github.com/suragnair/alpha-zero-general> · <https://github.com/werner-duvaud/muzero-general>
Minimal PyTorch self-play+MCTS codebases. NOT to run as-is (too slow at our clock) but to port:
their training loop structure, PUCT constants, and replay-buffer hygiene are battle-tested.
Also: AI Singapore's from-scratch Connect4 AlphaZero
(<https://aisingapore.org/from-scratch-implementation-of-alphazero-for-connect4>) and the
applied-data-science Keras AlphaZero tutorial
(<https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188>)
— fastest reading path for the loop mechanics.

### 2.2 Chance-node pruning: Star1/Star2 + Monte Carlo *-Minimax ★★
Ballard 1983 (in organizers' list); Lanctot et al., "Monte Carlo *-Minimax Search" (IJCAI 2013):
<https://www.ijcai.org/Proceedings/13/Papers/093.pdf> — probing phase at chance nodes to bound
and prune. Background: <https://en.wikipedia.org/wiki/Expectiminimax> ·
<https://inst.eecs.berkeley.edu/~cs188/textbook/games/expectimax.html>.
Use ONLY if we ever expand the dice-roll chance layer explicitly; our default is to fold dice
expectation into the value net (learned expectation ≈ free at inference).

### 2.3 Small-game self-play hyperparameters ★
Kemeny et al.-style analysis: "Analysis of Hyper-Parameters for Small Games"
<https://arxiv.org/abs/2003.05988> — 12 AlphaZero hyperparameters ablated on small games.
Consult when picking simulations-per-move, c_puct, replay ratios for OFFLINE label generation.

## 3. Learning architecture & pipelines

### 3.1 Expert Iteration (the loop we run) ★★★
Anthony, Tian & Barber, "Thinking Fast and Slow with Deep Learning and Tree Search" (NeurIPS
2017): slow search expert → distill into fast apprentice → expert uses apprentice → repeat.
<https://www.researchgate.net/publication/317088029_Thinking_Fast_and_Slow_with_Deep_Learning_and_Tree_Search>
**BRExIt — Opponent Modelling in Expert Iteration (IJCAI 2023)**
<https://www.ijcai.org/proceedings/2023/0422.pdf> — ExIt extended with opponent models: cite and
mimic in miniature for our legal in-match adaptation (track opponent archetype from played_devs/
robber pattern → switch eval knobs).

### 3.2 Student of Games + 2026 successor ★ (report ammo; conceptual)
Schmid et al., Science Advances 2023: <https://www.science.org/doi/10.1126/sciadv.abg3256> —
unifies perfect+imperfect info (grows from DeepStack's continual re-solving). 2026 follow-up on
self-play under imperfect information: <https://arxiv.org/html/2605.28863v2>. Our
sample_world-averaging is the budget version of the same principle.

### 3.3 Robustness via populations ★★ (feeds §5.1 of the blueprint)
- "Reducing Exploitability with Population Based Training": <https://arxiv.org/html/2208.05083v3>
- PBT survey (MDPI 2023): <https://www.mdpi.com/2227-739/11/10/2234>
- Mixture policies for robustness (PMLR 2022): <https://proceedings.mlr.press/v164/seyde22a/seyde22a.pdf>
Takeaway for us: the sparring PORTFOLIO (F, F-family perturbations, our old checkpoints,
archetype bots from §1.5) is a miniature population-based defense — evaluate/select on the
worst half, exactly the metric the group stage uses.

## 4. Weight fitting & auto-tuning

### 4.1 Texel tuning ★★★ — our primary method
Chess Programming Wiki, "Automated Tuning": <https://chessprogramming.org/Automated_Tuning> ·
walkthrough: <https://jamesswaffield.dev/automated-parameter-tuning-in-chess4j> ·
reference impl: <https://github.com/GediminasMasaitis/texel-tuner>
Method: sigmoid(eval) fit to game outcomes by logistic regression. For us: run self-play with
the heuristic teacher → (features, outcome) dataset → fit the eval's linear weights with
sklearn `LogisticRegression` (minutes) → teacher instantly stronger and CALIBRATED (eval value
≈ win probability → sane thresholds for endgame/denial overrides).

### 4.2 SPSA ★ — fallback for noisy/no-gradient knobs
<https://chessprogramming.org/SPSA> · Kocsis & Szepesvári, "Universal Parameter Optimisation in
Games Based on SPSA" (ACG 2005): <https://sites.ualberta.ca/~szepesva/papers/kocsis_acg05.pdf>
Two noisy objective evaluations per iteration (the Stockfish fishtest method). Use for discrete
behavioral knobs (robber aggression, dev-buy thresholds) where the objective is directly win
rate over ~400-game batches. CMA-ES (organizers' list) stays the heavier alternative.

## 5. CPU inference engineering

- **Dynamic int8 quantization** (pure torch — legal): 1-liner
  `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`; known to
  speed Linear-heavy MLPs; per-op overhead can limit gains on tiny nets → benchmark, ship only
  if measurably faster AND win-neutral. <https://leimao.github.io/blog/PyTorch-Dynamic-Quantization> ·
  <https://docs.pytorch.org/docs/2.1.0/serve/performance_checklist.html>
- **Batch all candidates into one forward pass** (biggest lever; ~1 ms for ~100 successors).
- `torch.inference_mode()`, warmup in `__init__`, `set_num_threads(1)`.
- `torch.compile` is possible inside the 60 s init but adds fragility — only if profiling
  demands it. (ONNX Runtime/OpenVINO are BANNED — not in the package whitelist.)

## 6. Prioritized integration list (remaining ~5 h)

| # | Action | Source | Cost | Expected gain |
|---|---|---|---|---|
| 1 | Placement eval with archetype/complementarity/denial/dots features | §1.5 | 45 min | High — placement is the highest-leverage decision; F ignores most of this |
| 2 | VP_DISCOUNTED_RETURN value target (+win/loss, +teacher-score) | §0.3 | 5 min | Medium-High — speed-to-win shaping |
| 3 | Texel-tune the teacher's linear weights on self-play outcomes | §4.1 | 60 min | High — instant, calibrated eval upgrade |
| 4 | ExIt rounds: teacher+search → labels → v-net → redeploy | §3.1 | 90 min | High — the core edge |
| 5 | Worst-half portfolio selection + old checkpoints as opponents | §3.3 | 20 min | High vs the actual metric |
| 6 | Batched inference + (benchmarked) dynamic quantization | §5 | 30 min | Medium — clock headroom |
| 7 | Depth-3 ablation ONLY as gated experiment (expect worse) | §0.2 | 30 min | Risk control |
| 8 | BRExIt-style bounded opponent knobs for in-match adaptation | §3.1 | 40 min | Medium — group-stage compounding |
| 9 | Reanalysis relabeling between ExIt rounds (reuse games, not just collect more) | §8.2 | 20 min | High — attacks our #1 bottleneck (CPU data generation) |
| 10 | Masked opponent-hand auxiliary head (self-supervised, from ground-truth self-play) | §8.3 | 30 min | Medium-High — attacks the game's only hidden info |
| 11 | EMA weights + AdamW/OneCycle + phase-conditioned FiLM gating | §10.1–10.3 | 30 min | Medium — free quality, no runtime cost |

## 8. Self-improving & self-learning techniques (deep cut)

The 2024–2026 thread (EfficientZero → ReZero → 2026 self-play replay work) converged on one
insight we should steal wholesale: **in compute-limited regimes, reusing data beats collecting
data.** Our bottleneck is CPU self-play (§0.1), so every technique below maximizes learning per
generated game.

### 8.1 Expert Iteration with fresh-data discipline (baseline loop — already planned)
Rounds of: deploy policy π_k → generate games → distill π_{k+1}. Two disciplines from the
literature: (a) retrain from scratch each round or use data mixing (≤30% old data) to avoid
staleness collapse — Fedus et al., "Revisiting Fundamentals of Experience Replay" (ICML 2020),
<https://proceedings.mlr.press/v119/fedus20a/fedus20a.pdf>, shows big buffers help rare outcomes
but stale data hurts off-policy; (b) keep past checkpoints as OPPONENTS (league play) —
"Self-play and Autocurricula in the Age of Agents" (2025),
<https://www.amplifypartners.com/blog-posts/self-play-and-autocurricula-in-the-age-of-agents>,
and AlphaStar's league are the reference; our worst-half portfolio IS a miniature league.

### 8.2 Reanalysis / just-in-time relabeling ★★★ — the highest-ROI steal
EfficientZero (self-imitation auxiliary + reanalyze):
<https://www.alignmentforum.org/posts/jYNT3Qihn2aAYaaPb/efficientzero-human-ale-sample-efficiency-w-muzero-self> ·
ReZero (just-in-time reanalysis): <https://arxiv.org/html/2404.16364v1> ·
M0RV (MuZero data reuse, 2024): <https://ieeexplore.ieee.org/iel8/6287639/10380310/10648685.pdf> ·
"Mastering Go with Self-play Experience Replay" (Jan 2026):
<https://www.alphaxiv.org/abs/2601.03306>.
**Adapted to us:** store every position + cheap features during self-play (not just final
labels). After training value net v_k, sweep the STORED dataset and relabel every position with
v_k's 1-ply search verdict — no new games needed — then train v_{k+1} on the relabeled set.
Each ExIt round thus costs minutes of tensor math instead of an hour of engine self-play.
This directly attacks §0.1's bottleneck and is our "self-improvement engine."

### 8.3 Self-imitation channel (SIL) ★★
Oh et al., "Self-Imitation Learning" (486+ cites):
<https://www.cs.ru.nl/~arwen/RSDS2019/RU_RSDS2019_paper_11.pdf> · "Match or Replay: Self
Imitating Proximal Policy" (2025): <https://openreview.net/forum?id=54TO3KUoYf> ·
Episodic SIL with hindsight: <https://arxiv.org/html/2011.13467v1>.
**Adapted:** weight training samples by advantage = (game outcome − v(s)): positions from won
games where the net was pessimistic get upweighted. Implement as a weighted sampler over the
memmap shards — zero new machinery, better credit assignment under dice noise (a won game with
bad dice = genuinely good decisions; SIL mines exactly those).

### 8.4 Masked opponent-hand auxiliary head ★★ (our novel self-supervised task)
In OUR generated games we know the ground truth of the opponent's hidden dev cards. Add a small
auxiliary output to the value net: predict P(opponent holds ≥1 VP card) and expected hidden-VP
count from PUBLIC-only features. This is masked-modeling self-supervision (the MLM idea,
ported to game state) and it attacks the game's ONLY hidden information. At play time the
auxiliary output sharpens endgame/denial thresholds (§5.3 of the blueprint) with a calibrated
probability instead of a guess. Cost: one extra head + one extra loss term.

## 9. Memory-efficient management (training side + scoring side)

### 9.1 Training-side dataset pipeline (laptop / 8-core team machine)
- **numpy `memmap` shards**: store positions as float32 `.npy` memmaps (1M × ~200 feats × 4 B
  ≈ 800 MB — fits RAM but memmap keeps workers independent and the team machine's 16 GB free
  for parallel self-play). One shard per worker per round; concatenate by index arithmetic.
- **Ring/streaming discipline** for continuous generation: fixed shard count, overwrite oldest
  (ExStream-style rehearsal logic — Hayes & Kanan, "Memory Efficient Experience Replay for
  Streaming Learning": <https://arxiv.org/abs/1809.05922>).
- **Prioritized sampling without a resident buffer**: SIL advantages (§8.3) live in a small
  float32 side-array; sample minibatches by np.random.choice with p=advantage (sum-tree PER is
  overkill at our scale — labml reference if wanted:
  <https://nn.labml.ai/rl/dqn/replay_buffer.html>).
- **Checkpoints = state_dict only** (no optimizer state) ≈ 0.5 MB each — save every round.

### 9.2 Scoring-side memory plan (the ring's hard 2 GB / one core)
- Model ≈ 0.5 MB fp32; weights loaded once in `__init__`; no growth.
- **Bounded per-process state**: opponent-model accumulators capped at a few hundred floats;
  position→decision memo as a bounded LRU (≤1k entries) — memoizing repeated placements is
  nearly free win-rate on a budget.
- **No object hoarding**: sampled worlds and successor `Game` copies are the memory hogs —
  `del` them per candidate loop; encode to features immediately, never keep lists of Games.
- `torch.inference_mode()` everywhere; preallocated output buffers; int8 dynamic quantization
  (§5) also shrinks the resident model 4×.
- Verify with psutil during local `evaluate` runs on the team machine — RSS must stay flat
  across a 1000-game chunk.

## 10. Modern training mechanics — the 2026 crop, triaged honestly

**Triage verdicts** (the constraint filter is the point — a 110k-param CPU MLP is not an LLM):

| Technique | Verdict | Why |
|---|---|---|
| FlashAttention / fused SDPA kernels | ✗ | GPU kernels; no GPU on the ring; sequences ≤ ~100 tokens anyway |
| Gated DeltaNet / DeltaNet-2 / Kimi-Linear hybrids (2026 SOTA linear attention) | ✗ runtime, ✓ report | Squeezes unbounded KV cache into fixed recurrent state — solves a problem we don't have; our net is one MLP pass. Cite as 2026 context: <https://openreview.net/forum?id=r8H7xhYPwz>, <https://arxiv.org/html/2605.22791v1> |
| Set-transformer attention over variable candidate sets | ⚠ gated stretch | Elegant for variable action lists, but our batched-MLP argmax already handles that in ~1 ms; only if v2 encodes raw sets |
| **Muon optimizer** (2025's headline result: ~48% FLOP savings vs AdamW, Moonlight 16B; Keller Jordan's speedruns) | ✓ optional | ~60 lines of pure torch (Newton–Schulz on 2D weight matrices) — legal, CPU-fine. Adopt ONLY after AdamW baseline works: <https://kellerjordan.github.io/posts/muon>, <https://arxiv.org/html/2509.02046v2> |
| **EMA of weights ("mean teacher")** | ✓ always | Copy of weights updated as `ema = β·ema + (1−β)·w`; ship EMA weights. Free stability/quality at zero runtime cost (Tarvainen & Valpola lineage) |
| **FiLM-style phase conditioning** — the "dynamic mechanism" that survives our constraints | ✓ cheap win | Instead of dynamic attention, dynamically GATE feature groups by game phase: `γ,β = g(phase_features)` modulating the MLP's first hidden layer. Phase-aware evaluation (opening rush vs city race vs endgame denial) with ~2k extra params. FiLM: Pérez et al. 2018 |
| bf16 autocast (CPU/GPU training) | ✓ training only | Inference stays fp32 for calibration stability |
| OneCycle LR + cosine decay + grad clip | ✓ always | The reliable 2020s defaults; see <https://arxiv.org/abs/1706.10290> lineage |
| TD(λ) eligibility traces / n-step returns | ✓ | Sutton & Barto classics — best label variance reduction per CPU cycle for sequential value learning; combine with §0.3's VP_DISCOUNTED_RETURN |
| Temperature/Platt calibration of v(s) on held-out games | ✓ always | Decisions and endgame thresholds consume PROBABILITIES; an uncalibrated value net wastes them |

**The honest SOTA judgment for this build:** the truly cutting-edge move here is not importing
an attention architecture — it is (a) the **reanalysis loop** (§8.2) maximizing learning per
CPU-hour, (b) **self-supervised hidden-state modeling** (§8.4) aimed at the game's actual
information asymmetry, and (c) **phase-conditioned dynamic gating** (FiLM) as the lightweight
descendant of "dynamic computation" research. Deep Blue beat Kasparov with 1997 search;
Logistello beat the Othello world champion with a logistic eval. Under a 12.5 ms budget, the
2026 crop mostly points at which *training-side* tricks transfer — and those are EMA, reanalysis,
SIL weighting, calibration, and optionally Muon.

## 11. Dataset plan (full spec in `DATASET_CONTEXT.md`)

External Catan datasets (Kaggle records, StacSettlers/STAC, JSettlers logs) are ALL mismatched
to our 2-player no-trade variant and lack per-candidate scores + hidden-card ground truth →
**we generate everything.** Full spec — harness, the D1–D15 corpus catalog (incl. zoo of
archetype bots, placement book, endgame oversampling, world-variance pairs, rare-event
generators, reanalysis relabels), row/label schema with the observability rule (features from
the deciding seat's masked view ONLY), seed-ledger splits, weighting/versioning, storage and
acceptance gates — lives in `DATASET_CONTEXT.md`, written for the coding assistant executing
the build. Rough volume: 8 workers ≈ 30–70k games/hour ≈ 3–8M labeled positions/hour.

## 7. Full source list

**Catan-specific:** Szita et al. 2009 (spronck.net/pubs/ACG12Szita.pdf) · Dobre & Lascarides
AAAI-20 (cdn.aaai.org/ojs/13014/13014-52-16531-1-2-20201228.pdf) · settlers-rl.github.io ·
uu.nl actor-critic thesis 2020 · Stanford AA228 final128.pdf · sidhardhburre.com/final ·
docs.catanatron.com · github.com/bcollazo/catanatron · "5 Ways NOT to Build a Catan AI"
(medium.com/@bcollazo2010) · Colonist starting strategies (blog.colonist.io) ·
kingofcatan.net/initial-settlement-placements-production
**Search:** Ballard *-minimax (organizers' list) · Lanctot MC *-Minimax IJCAI-13 ·
en.wikipedia.org/wiki/Expectiminimax · CS188 expectimax notes · arXiv:2003.05988
**Learning:** ExIt / Thinking Fast & Slow 2017 · BRExIt IJCAI-23 · Student of Games
(SciAdv 2023) · arXiv:2605.28863 (2026) · alpha-zero-general · muzero-general · AI Singapore
Connect4 · AppliedDS Keras AlphaZero
**Robustness:** arXiv:2208.05083 (PBT exploitability) · MDPI PBT survey 2023 · PMLR mixture
policies 2022
**Tuning:** chessprogramming.org/Automated_Tuning (Texel) · jamesswaffield.dev walkthrough ·
texel-tuner repo · chessprogramming.org/SPSA · Kocsis & Szepesvári ACG-05
**CPU inference:** PyTorch performance checklist · Lei Mao dynamic quantization · PyTorch
quantization blog
**Self-improvement (2024–2026):** EfficientZero (alignmentforum explainer) · ReZero arXiv:2404.16364 ·
M0RV (IEEE 2024) · Mastering Go with Self-play Experience Replay (arXiv 2601.03306, Jan 2026) ·
Self-Imitation Learning (Oh et al., ICML 2018) · Match-or-Replay (OpenReview 2025) · Episodic SIL
(arXiv:2011.13467) · Self-play and Autocurricula in the Age of Agents (Amplify, Nov 2025) ·
Recursive Self-Improvement of AI Research Agents (arXiv:2609.26457, 2026)
**Memory management:** Hayes & Kanan ExStream (arXiv:1809.05922) · Fedus et al. Revisiting
Fundamentals of Experience Replay (ICML 2020) · labml.ai PER reference · Map-based Experience
Replay (2023)
**Training mechanics (2025–2026):** Muon — kellerjordan.github.io/posts/muon · Fantastic
Pretraining Optimizers (arXiv:2509.02046) · Moonlight/Muon FLOP results · Gated DeltaNet
(OpenReview) · Gated DeltaNet-2 (arXiv:2605.22791, NVIDIA 2026) · Raschka's DeltaNet chapter ·
DeltaNet explained (sustcsonglin.github.io) · FiLM (Pérez et al. 2018) · mean-teacher EMA lineage
