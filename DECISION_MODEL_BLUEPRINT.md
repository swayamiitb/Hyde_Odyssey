# Decision-Model Blueprint — Hyde Catan Hackathon

> Companion to `GAME_DAY_CONTEXT.md`. This document is the complete build spec for the learned
> decision model: what it is, the exact compatibility contract across all three machines, the
> architecture, the self-learning pipeline, the adversarial-maximization strategy, and the
> quality gates every parameter must pass. Written to be executable by anyone on the team.

---

## 1. What the decision model is (and is not)

**It is NOT an LLM.** No Perplexity/frontier-style model can run in the bot: services are banned
at play time (rule 11.3), no network on the scoring box, no `transformers`/`onnxruntime` (only
stdlib + numpy/scipy/sklearn/torch exist there), 2 GB RAM, and a ~12.5 ms/decision CPU budget.

**It IS the game-playing lineage the organizers' own reading list names**, shrunk to one CPU core:

| Ancestor | Idea we take | Where it lands in our bot |
|---|---|---|
| TD-Gammon (Tesauro) | neural **value function** over positions, learned by self-play | `v(s)` scores the world *after* each candidate move |
| Logistello (Buro) | linear/shallow eval fitted to outcomes beats hand-tuning | distillation targets for the net's early iterations |
| Deep Blue | handcrafted eval + deep search | our heuristic fallback + budgeted search layer |
| AlphaZero (Silver) | net + tree search, iterated self-play improvement | v-net + expectimax, policy-improvement loop |
| Thinking Fast & Slow (Anthony et al.) | distill a slow deep search into a fast net | teacher (deep search) → student (fast net) |
| DeepStack/Libratus | value nets over *information sets* for hidden info | average decisions over `obs.sample_world()` draws |

**Core design decision: a state-VALUE network, not an action-classification policy.** The legal
action list changes size and content every decision (1 to ~200+ actions); a value net sidesteps
that entirely (score each successor, argmax) and plugs into search. This is the TD-Gammon lesson.

**Deployment shape:** for each decision → sample k worlds consistent with the observation →
for each legal action: execute on world copy → `v(successor)` → average over worlds → argmax
(deterministic tie-break). Optionally one opponent-reply level (2-ply) on high-leverage
decisions. Net inference is ~50–150 µs/candidate on one core (verify locally) → all candidates
× several worlds fit comfortably inside the budget.

## 2. The compatibility contract — laptop → team machine → AWS scoring ring

The model is *trained* under no clock on laptop raw compute; it must *play* inside the ring.
These rules are non-negotiable and apply from the first line of code:

### 2.1 Environment parity (both dev machines)
- **Python 3.11.x exactly** (scoring interpreter). Fallback: `uv python install 3.11` + `uv venv --python 3.11 .venv`.
- Install order: `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu` FIRST, then `pip install -r requirements.txt` (pinned catanatron @ git `d3f4ad0`; never PyPI — different action space).
- `python -m catan_kit.selfcheck` must pass 100% on laptop AND team machine before every milestone. It pins engine version, determinism (seed 4256 → RED, 111 turns), and hash seed.

### 2.2 Import surface (the entire whitelist)
`stdlib + numpy 2.4.6 + scipy 1.17.1 + scikit-learn 1.9.1 + torch 2.8.0`. Nothing else exists on
the ring. Any other import crashes the agent → random moves → >2% faults → **disqualification**.

### 2.3 Artifact hygiene (zip rules)
- Folder: `agent.py` at zip top level + helper `.py` files + data files. ≤100 MB total.
- **Pure Python + data only.** No `.so`/`.pyd`/`.exe`/ONNX/binaries — upload auto-rejected.
- Weights: save `state_dict`, load with `torch.load(..., map_location="cpu", weights_only=True)`.
  Same major torch version on all three machines (2.8.0 pinned) → no pickle/version drift.
- No `__pycache__` in the zip. No network at runtime, no subprocesses, no `multiprocessing`.

### 2.4 Runtime discipline (the ring enforces)
- **Single thread:** `torch.set_num_threads(1)` in `__init__`. Host also forces OMP/MKL=1.
  Threads share one core in the ring — they buy nothing and burn clock.
- **Load once, warm everything** in `__init__` (≤60 s, uncharged): import net, run 10 warmup
  inferences (first torch call pays JIT/allocation), precompute board-derived tables.
- **Filesystem:** agent dir is READ-ONLY on the ring. Load via
  `os.path.join(os.path.dirname(__file__), ...)`. Scratch only in `tempfile.gettempdir()`.
  Linux paths: `os.path.join`, never `\`; filenames case-sensitive.
- **Errors:** any exception/illegal index = fault → random move that turn. Hard ceiling 2% of
  decisions = DQ. Every `decide()` wrapped so it can never raise; fall back to F-eval argmax →
  first legal action, never crash.
- **Return** a genuine `int` (`int(np.argmax(...))`).

### 2.5 Clock budgets (the laptop is ~1.5–2× faster per core than the ring's EPYC 9R14)

| Parameter | Ring limit | Our local target (2× margin) |
|---|---|---|
| CPU per game | 300 ms + 12.5 ms/decision, ≤1.5 s | **≤ 0.5 s mean, ≤ 0.9 s worst** |
| Mean per decision | 12.5 ms increment | **≤ 5 ms** |
| Single decision wall | 3 s (silent = killed + restarted) | ≤ 1 s hard internal guard |
| Games over clock | leaderboard column | **0 in every 1000-game run** |
| Load/init | 60 s | ≤ 5 s |
| Faults | 2% DQ threshold | **0 in every run** |

Budget manager: read `obs.cpu_left` every decision; track own spend with `time.process_time()`
deltas; degrade gracefully: k worlds → 1 world → heuristic-only → cached argmax. A budget-aware
bot NEVER plays random from overdraw — overdraw randoms are how the library search bots died at
0.001.

### 2.6 Determinism & reproducibility
- Seed everything (torch, numpy, `random`); the harness reseeds agent `random` per game.
- Custom driver scripts: call `catan_kit._patch.ensure_hash_seed()` first line, code under
  `if __name__ == "__main__":`, or run with `PYTHONHASHSEED=0`.
- Every experiment logged with its seeds → the report can reproduce every number (they may re-run).

### 2.7 Workflow
Develop laptop-first (unlimited); sync milestones to the team machine (scp/git); on the team
machine: `selfcheck` → `evaluate --games 1000` on a **fresh seed range** → zip → `~/upload.sh`.
Uploads exist ONLY on the team machine. Keep `--seed0` ranges disjoint: TRAIN (e.g. 1xxxx),
EVAL (e.g. 990000+, the default), never mix.

## 3. Architecture spec

### 3.1 State encoding (v1: ~180 compact engineered features — fast, explainable, report-friendly)
Per player (both seats, from my perspective): effective production per resource (robber-aware),
total production per resource, production reachable at 0/1 roads, hand counts ×5, hand size,
distance-to-settlement/city/city-upgrade (fractional), public VPs, actual VPs (mine), dev cards
in hand (mine exact, theirs = count), played devs ×5, knights played + army holder flag, longest
road length + holder flag, buildable node count, roads/settlements/cities remaining, port
ownership (2:1 per resource ×5 + 3:1), bank ×5 + dev deck size, robber-on-my-best-tile flag,
turn number, pending robber/discard prompt flags. Normalize continuous features (÷19, ÷prob
max). Rationale: F's blind spots are our features (ports, bank, hidden-VP proxy). The raw 614
vector (full graph one-hot) is a v2 option — richer but needs a bigger net and more data.

### 3.2 Network
`ValueNet: Linear(≈180→256) → GELU → Linear(256→256) → GELU → Linear(256→1) → sigmoid`
(~110k params ≈ 0.5 MB). Init in `__init__`, `eval()` mode, inference under `torch.inference_mode()`.
Batch all candidate successors into ONE forward pass (tensor stack) — ~1 ms for 100 candidates.
v2 (only if v1 saturates): + policy head (probability over candidate set via scored successors),
or tiny transformer over per-tile embeddings. Ship only what measurably raises win rate.

### 3.3 Decision procedure (the shipped bot)
1. Forced move (len==1) never reaches us. Parse action types by `ActionType`.
2. Cheap-path decisions (END_TURN when nothing buildable, single-candidate types): heuristic
   shortcut, ~0 cost.
3. Real decisions: k = 1 world (opening placement & robber: k=2–3 while budget rich).
   Execute every candidate on world copies → encode successors → batch `v()` → mean over
   worlds → argmax (tie: lowest index — deterministic).
4. Selective 2-ply (placement, robber target, city-vs-settlement-vs-dev when close to win/deny):
   my action → opponent best greedy reply (F-eval, fast) → v(). Gated by budget manager.
5. Endgame overrides (see §5.3) applied as bonus/penalty terms on top of v — keeps the net
   simple and the overrides explainable.

## 4. Self-learning & improvement pipeline (the day's engine)

**Teacher→student→iterate, all on laptop raw compute; training is unclocked, only play is clocked.**

- **P0 — Heuristic teacher H (F-fixed):** 1-ply greedy over corrected eval (ports, LA/LR pricing,
  robber by marginal enemy production, hidden-VP-aware endgame). Beats F; becomes the label
  generator, the fallback, and a sparring partner forever.
- **P1 — Data generation:** parallel self-play (one process per game, one thread each,
  `--seed0` in TRAIN range): H-vs-H, H-vs-F, F-vs-F. Log every decision: (my-state-encoding,
  chosen action encoding, search scores of all candidates, final outcome z). Yield ≈ 2×~56
  labeled positions/game; thousands of games/hour across cores → 10⁵–10⁶ samples in the first
  hour. Log to sharded `.npz`.
- **P2 — Distill v1:** train ValueNet on outcomes z (MSE/BCE) with TD-style weighting (later
  positions matter more) + teacher-search-score regression (Logistello move). CPU minutes/epoch.
  Early-stop on held-out TRAIN-range games.
- **P3 — Policy improvement rounds (AlphaZero-lite):** deploy v1+search as a stronger teacher →
  regenerate labels (deeper search verdicts) → retrain → repeat 2–3 rounds while win rate rises.
  This is the self-improvement loop; measure each round vs the same eval seeds.
- **P4 — Scalar knob tuning:** remaining hand-tuned weights (build priorities, endgame
  thresholds) tuned by random search/CMA-ES directly on 400+ game win rate vs the portfolio
  (§5.1). Accept only changes outside the 95% CI.
- **P5 — Online match adaptation (legal & unique edge):** one process plays many games vs the
  SAME opponent. Maintain opponent stats from public history (dev-buy rate, robber pattern,
  expansion greed) via `obs` + action records of sampled worlds; shift 2–3 eval knobs within
  bounded ranges. In the 1000-game group stage this compounds; nothing persists across restarts.

**Checkpointing:** every artifact named `v{N}_{data}_{seed}_{wrF}.pt` + one row in
`experiments.md` (change, seeds, games, wr, CI, CPU/game, faults). This log IS the report.

## 5. Adversarial maximization (winning in every scenario)

### 5.1 Optimize the actual metric, not the fantasy
Group advancement is **robustness = mean win rate vs the worst half of your 9 group opponents**;
strength = 0.5·wr_F + 0.5·wr_O on private boards; leaderboard ≤25% of total. Therefore:
- **Portfolio evaluation for every decision:** {F, random, F-family (weight-perturbed F clones —
  cheap diverse opponents), our own older versions, an aggressive-roader, a dev-card-spammer}.
  Selection criterion: raise the WORST-HALF mean, not the F mean.
- Never select a change that raises wr_F but drops portfolio-worst-half.

### 5.2 Anti-overfit discipline
Disjoint train/eval seeds; boards are randomized anyway (no board memorization possible); net
regularization (weight decay, early stop); no change accepted inside noise. The group stage
plays brand-new boards — assume nothing transfers except skill.

### 5.3 Adversarial play baked into the decision layer
- **Hidden-VP endgame:** opponent's effective VPs = public + E[VP cards | dev count bought,
  deck composition seen]. When their effective VP ≥ 8: shift to denial — robber on their best
  tile, contest LA/LR (4-VP swings = the duel-winning fights), buy dev cards to thin the deck's
  VP cards out of their reach.
- **Robber as a weapon (7 is the most likely roll, ~17%):** target by marginal enemy production
  loss × their hand size (steal value); knights timed for tempo and army theft.
- **Resource denial:** hold the port corners (2-player economy = bank/ports — port corners are
  disproportionately valuable), cut their roads, drain the bank of a resource they need when
  near their build window.
- **Clock aggression is impossible for them against us** (our budget manager), and our bot never
  depends on opponent behavior — no exploitable predictability in ties (deterministic but
  board-dependent).

### 5.4 Failure-mode immunity (robustness includes engineering)
Zero-fault design: catch-all wrapper in `decide()`; watchdog budget manager; fallback chain
net → heuristic → first-legal. Validate: 1000-game runs ×3 seed ranges, 0 errors, 0 over-clock,
variance within CI. A submission that never faults beats a brilliant one that crashes 3% of games.

## 6. Quality gates — every parameter, checked per candidate submission

| Gate | Target |
|---|---|
| selfcheck | 100% pass on laptop + team machine |
| wr vs F (1000 games, EVAL seeds) | report with CI; gate for keeping a change |
| Portfolio worst-half wr | the primary selection metric |
| Mean / worst CPU per game | ≤0.5 s / ≤0.9 s local (ring ≈ 2×) |
| Games over clock / faults / timeouts | 0 / 0 / 0 |
| Load time | ≤5 s |
| Zip | ≤100 MB, no binaries, `agent.py` top-level, loads from clean extraction |
| Reproducibility | same seeds → same games; every report number has a logged run |

## 7. Timeline (IST, from now to 17:00 hard stop)

| Window | Deliverable |
|---|---|
| now–11:30 | Team machine: password, selfcheck, upload F-copy as submission #1 (pipeline proof) |
| 11:30–13:00 | H teacher + generator harness + first data flowing; laptop env parity done |
| 13:00–14:30 | v-net v1 distilled; bot = net-1-ply; first `evaluate` vs F (submission #2) |
| 14:30–16:00 | P3 improvement rounds; selective 2-ply; portfolio + robustness tuning (submission #3–4) |
| 16:00–16:45 | FREEZE. 3×1000-game validations, fresh seeds; final upload; zip hygiene re-test |
| 16:45–17:30 | Report from `experiments.md` (template kept live all day); PDF upload by 17:30 |
| 18:00+ | Presentation: method, journey, honest failures; drill the Q&A ("what would you do with 10× compute?") |

## 8. Do-NOT list (each has killed someone before)
- Do NOT `pip install catanatron` from PyPI (wrong action space).
- Do NOT ship a full pickled model object / any binary / `__pycache__`.
- Do NOT let `decide()` raise, return numpy scalar un-cast, bool, or float.
- Do NOT use multiprocessing/threads in the agent; do NOT write next to `agent.py`.
- Do NOT trust one leaderboard reading: CI ±3% at 1000 games; don't chase noise.
- Do NOT overfit F: the funnel's robustness metric punishes it and O is half the strength score.
- Do NOT burn uploads: 7 scored / 10 total / 2 queued; always `evaluate` locally first.
- Do NOT debug on the ring; reproduce locally with the logged seed.

*Sources: `catan-hackathon-agent/` (PROBLEM_STATEMENT.md, README.md, catan_kit/*), pinned
catanatron commit `d3f4ad0` sources in `.engine_ref/` (value.py, features.py, apply_action.py,
state_functions.py, match/host mechanics), organizers' reading list.*
