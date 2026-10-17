# Full Context Download — Hyde × WnCC Catan Agent Hackathon

> THE master document, compiled 17 Oct 2026 (~11:35 IST). Supersedes nothing — it *contains* the
> full context of the entire team conversation from kickoff prep through the current build phase,
> including every question asked and the answer settled on. Self-contained: a new teammate, or a
> fresh AI session, can read only this file and be fully up to speed. Companion files:
> `GAME_DAY_CONTEXT.md` (deep repo/engine reference), `DECISION_MODEL_BLUEPRINT.md` (build spec),
> `HYDE_CATAN_HACKATHON_BRIEF.md` (pre-kickoff prep, superseded where it conflicts).

---

## PART A — The ground truth (established from the official repo + pinned engine)

### A1. The event
- **Hyde × WnCC one-day AI Builder Hackathon, Sat 17 Oct 2026, LA001 IIT Bombay.**
- Task: build a program that plays **2-player Catan** vs their two anchor bots (**F** and **O**)
  and every other team's bot. ~250 teams.
- Schedule (IST): 10:00 kickoff → **11:00 submissions open + leaderboard live** → 15:30 checkpoint
  (no cut) → **17:00 LAST UPLOAD** → 17:30 report due (top 30) → 17:50 top-30 + group results →
  18:00 six finalists present (10 min each, questions weigh most) → 19:30 end.
- Prizes ₹2.5L pool; Hyde (RL post-training lab, DeepMind/Optiver eng, GC+BCV backed) is scouting
  hires — the explanation track is a hiring screen.

### A2. The game as the engine implements it
- Engine = **catanatron pinned to git commit `d3f4ad0`** (NOT PyPI 3.2.1 — different action space).
- Standard base Catan, official spiral board, randomized seeds, **first to 10 VP**, except:
  **2 players** and **no player-to-player trading** (bank 4:1, generic port 3:1, specialty 2:1).
- Dev deck = **full standard 25 cards: 14 Knight, 5 VP, 2 Road Building, 2 Year of Plenty,
  2 Monopoly** (the prep-brief guess that Monopoly was out was WRONG). Can't play a dev the turn
  it's bought; one dev per turn.
- Setup (verified in engine source): snake placement P0, P1, P1, P0; **first placer also takes
  the first turn**; second settlement pays starting resources immediately.
- Robber on 7 (~17%): >7 cards → discard half (floor); roller blocks a hex + steals 1 random card.
- Longest Road (5+) and Largest Army (3 knights) = stealable 2-VP bonuses; steal requires
  STRICTLY longer/more (ties keep the holder).
- 1000-turn cap; no winner = **loss for both**.

### A3. The agent contract
```python
class Agent:
    def __init__(self):     # once per PROCESS, ≤60 s, uncharged — load+warm everything here
    def decide(self, obs, playable_actions) -> int   # return an INDEX into the list
```
- One folder: `agent.py` at zip top level + helpers + data files, ≤100 MB, **Python only** —
  no compiled binaries, no subprocesses, no network at play time, no multiprocessing (harness
  refuses); threads share the single core → **single-threaded** (`torch.set_num_threads(1)`).
- Allowed imports ONLY: stdlib + numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, torch 2.8.0 CPU.
  **Python 3.11** (scoring interpreter).
- Agent dir read-only at scoring; scratch in `tempfile.gettempdir()`; Linux paths
  (`os.path.join`, case-sensitive names); load weights via `os.path.dirname(__file__)`.
- Faults: exception or illegal index → random move played + fault recorded; **>2% of decisions
  → disqualified**. Must return a genuine `int`.
- Forced moves (exactly 1 legal action — 47% of decisions) never reach the agent, cost no clock.
- One process plays MANY games vs the same opponent → **online adaptation between games is
  explicitly legal**; nothing persists across restarts.

### A4. What the agent sees
Everything except **which dev cards the opponent holds unplayed** (hence their hidden VP cards).
Visible: full board (tiles/numbers/ports/robber/buildings/roads), own exact hands, own VPs
(public + actual), bank counts, **opponent's exact resource hand** (derivable anyway),
**count** of opponent's unplayed devs, opponent public VPs, all devs played by both, turn number,
full move+dice history, `obs.me`, `obs.cpu_left`.
`obs.sample_world(rng)` → a complete engine `Game` consistent with all of that (opponent's
unseen devs drawn from the unseen pile; deck reshuffled; opponent BUY records blanked). Sample
several and average to act well under uncertainty. Engine random draws inside search replay
exactly (agent's `random` is reseeded per game).

### A5. The clock (the true opponent)
```
allowance after k decisions = min(300 ms + 12.5 ms × k, 1500 ms)   # agent's own CPU per game
```
~56 decisions/game → **~1 s CPU/game**, ceiling 1.5 s. Overdraw → harness plays UNIFORM RANDOM
moves for the rest of the game ("games over the clock" on the leaderboard). Also: 3 s max per
decision (silent agent killed+restarted), 3 s/game wall slack, 60 s load. Observation delivery
(<1 ms/decision) charged. Their box (AWS c7a.48xlarge, AMD EPYC 9R14, 1 core, 2 GB): random bot
11 ms/game, **F 158 ms/game (2.9 ms/decision)** — we get ~6× F's compute. Their core is ~1.5–2×
slower than a recent laptop → **plan ~2× margin**; the leaderboard clock column is ground truth.

### A6. The scoring harness
`python -m catan_kit.evaluate --a my_team/agent.py --b examples/f_agent.py --games 1000` IS the
scoring script (on the day only board seeds, dice key, opponents change). Each board played
twice, seats alternated, same dice; dice keyed by HMAC (secret on the day). Kit applies three
determinism fixes (sorted maritime trades, PYTHONHASHSEED=0 re-exec, feature-ordering reseed
guard) → any game replays exactly from its record. `python -m catan_kit.selfcheck` verifies the
install (seed 4256 must give RED win in 111 turns).

### A7. F — fully decoded (source ships in the kit)
F = catanatron `ValueFunctionPlayer(base_fn, DEFAULT_WEIGHTS)`: **1-ply greedy** — copy one
sampled world, execute each candidate, evaluate the successor, deterministic argmax. Exact
weights: public_vps **3e14**; own effective production +1e8 (with variety bonus: each distinct
resource ≈ 4 production points); enemy effective production −1e8; production reachable with
**0 roads = 0 (ignored)**; 1-road-reachable 1e4; buildable nodes 1e3; hand synergy 1e2;
longest road 10 but **0.1 while buildable nodes exist**; hand resources 1; discard penalty −5;
hand devs 10; army 10.1.
**Blind spots to exploit:** ports valued at zero, bank ignored, opponent hidden VPs invisible,
1-ply only, single world sample, LR suppressed early, tie-breaking fixed.
**O** = Hyde's own trained bot, "beats F by some margin", code withheld.
Library search bots (MCTS-25, AlphaBeta-d3, SameTurnAB-d2) all score ≈0.001–0.08 — they die by
clock-out. *"A search that fits the allowance is the point of the exercise."*
Engine's 614-number `create_sample_vector` = raw board state only (full graph one-hots, tiles,
ports, bank) — contains **none** of F's derived features; we compute those ourselves.

### A8. The funnel and scoring formula
- Uploads: `~/upload.sh` (team machine only). **7 scored submissions, 10 uploads total, max 2
  queued**; failed-to-load uploads are returned and don't consume a scored slot; each scored
  upload = 1000 games vs F + 1000 vs O on public boards AND private boards.
- Portal: `https://main.d2sgimrfzis7m5.amplifyapp.com/` (team name + leader's full phone).
- Top 30 by mean of private win rates (best submission) → 3 groups of 10 → **1000 games vs each
  of 9 group opponents on fresh boards** → top 2 per group by **ROBUSTNESS = mean win rate vs
  the half of your group opponents you do WORST against** → present.
- Weights: robustness + strength (`0.5·wr_F + 0.5·wr_O`) together ≈ report + presentation
  together; leaderboard ≤25% of total. Report ≤4 pages A4 ≥11pt (method clarity, design
  justification, journey insights, experiment quality). Presentation 6+4 min, **questions weigh
  most**. "High win rate without good reasoning only fetches marks in the first two components."
- Fair play: all our own work built TODAY (public libs/papers/code allowed if named in report);
  LLMs may write code but the agent may never call one or any service; no reaching for hidden
  info/seeds/dice key; no collusion; every report number must come from a run we made.

---

## PART B — The full conversation ledger (every question, every settled answer)

### B1. "Gain the context of this hackathon thoroughly; go through lyric-hyde/catan-hackathon-agent with depth and detail"
**Settled:** Repo cloned to `D:\Odyssey\catan-hackathon-agent` and read in full (~2,150 lines:
PROBLEM_STATEMENT.md, README.md, catan_kit/{obs,match,host,evaluate,sandbox,selfcheck,_patch}.py,
examples/{f_agent,random_agent}.py, requirements.txt, NOTICE). Pinned catanatron engine sources
(`value.py`, `features.py`, `apply_action.py`, `state.py`, `state_functions.py`, `actions.py`,
`game.py`, `enums.py`) fetched and cached at `D:\Odyssey\.engine_ref` — F's eval, the 614-vector,
setup order, LA/LR semantics all verified in source, not assumed. Deliverable written:
`GAME_DAY_CONTEXT.md`. **Corrections vs prep brief:** Monopoly IS in; no player trading at all;
learned component NOT mandatory (submitting a copy of F is explicitly legal); engine/versions/
funnel/judging all now precise (Part A above).

### B2. "Detail it here in the interface, not just the file"
**Settled:** full detailed briefing delivered in-chat (condensed into Part A of this document).

### B3. "What do I do first? How do I make the agent? Laptop vs their sandbox/AWS 2 GB machine?"
**Settled — the three-machine picture:**

| Machine | What it is | Role | Limits |
|---|---|---|---|
| Their AWS c7a.48xlarge | Scoring ring — never log in; runs uploaded zips vs F/O | constraint to FIT | 1 core, 2 GB, ≤1.5 s CPU/game, py3.11 + pinned pkgs |
| Team machine (`ssh team<name>@32.190.233.160`) | Their box for us: 8 cores/16 GB, kit installed | primary dev + ONLY upload path (`~/upload.sh`) | practice unlimited; clock only inside evaluate |
| Our laptops / Colab / Kaggle | Our raw compute | dev + parallel self-play/training, explicitly encouraged | none for practice |

SSH: initial password = teamname + leader's last-4 phone digits twice (e.g. `blanc12341234`);
fingerprint `SHA256:Z8tsuVPiJ3dyI2vOQt2wV8hcBQxyaZOLpB2H4rnvvBE`; `passwd` immediately; one
command at a time.
**First 30 minutes (settled):** SSH in → passwd → `cd ~/catan-agent-hackathon && source
.venv/bin/activate` → `python -m catan_kit.selfcheck` → make `my_team/agent.py` = copy of
`examples/f_agent.py` → 40-game sanity evaluate → zip → `~/upload.sh` → **submission #1 =
F-copy** (legal, validates the whole pipeline, ~50% baseline on the board before any real work).
**Build order settled:** F-copy → F-fixed heuristic (ports, LA/LR, robber, hidden-VP endgame) →
speed+budget engineering → depth where it matters (2-ply/multi-world) → optional distilled net →
freeze + report. Two discipline rules: `experiments.md` log from hour one; never submit without
0 errors / 0 timeouts / 0 over-clock.

### B4. "I must win among 250 teams. Is the team machine my laptop's compute? Can I put a small open-source model or decision model in the bot?"
**Settled:** Team machine is THEIR remote box (not the laptop); laptop is ours and encouraged.
**Neural nets: yes, explicitly allowed and anticipated** (torch/sklearn/scipy/numpy are
pre-installed on the ring — a deliberate signal). Rules: any model that loads with exactly those
packages and ships in the zip; open-source CODE allowed with attribution (fair-play rule 2);
**LLM at play time banned** (rule 11.3, no network anyway); no `transformers`/`onnx`/etc.;
no compiled artifacts; weights must ship in the ≤100 MB zip; nothing pretrained-before-today
(gray zone + nothing compatible exists for this pinned engine anyway). **Honesty clause (kept
throughout): "sure win at any case" doesn't exist vs 250 teams and unknown O; leaderboard ≤25%
of weight; our commitment = maximum-probability build on their actual formula.**

### B5. "Do I need to start on the team machine, or can I work/train laptop-first and transition later?"
**Settled: laptop-first is sound and encouraged.** Team machine strictly needed only for (a)
password/SSH validation + (b) uploads — so touch it ONCE early (F-copy submission) to prove the
path, then develop locally all day and sync per milestone. **The compatibility contract**
(baked into §2 of the blueprint): Python 3.11 exact (`uv` fallback); install torch-CPU then
`-r requirements.txt` (pinned commit, never PyPI catanatron); selfcheck green on both machines;
imports limited to the whitelist; state_dict weights with
`torch.load(map_location="cpu", weights_only=True)`; single-thread; Linux-proof paths; read-only
agent dir; timing margin ~2× (local ≤0.5 s/game); disjoint TRAIN/EVAL `--seed0` ranges;
determinism via `ensure_hash_seed()` in custom drivers.

### B6. "So I can't put ANY open-source or decision model in?"
**Settled precisely:**
- ✅ Our own models built/trained today (torch MLP/tiny transformer/sklearn/numpy tables).
- ✅ Open-source code/libraries/algorithms ported today + named in the report.
- ✅ Data/weights files inside the zip.
- ❌ LLMs or any service at play time; anything needing packages beyond the four; compiled
  binaries; runtime downloads; pretrained-before-today weights.
- The practical shape that fits the interface (variable action list, ~12.5 ms): a **state-value
  net** — score each candidate successor, argmax (TD-Gammon pattern); a hand-written small
  transformer in raw torch is legal if it fits the clock.

### B7. "How do I make a decision model like the leading ones (Perplexity / 'Jev'), and is a decision model an absolute edge?"
**Settled — category fix:** Perplexity-style models are LLMs = language models, not game
decision models; physically impossible here (banned, no network, no package, 12.5 ms, 2 GB).
The ACTUAL leading decision-model lineage for this setting (and the organizers' own reading
list): **TD-Gammon** (neural value fn, self-play, dice game), **Logistello/Deep Blue**
(fitted/handcrafted eval + search), **AlphaZero/MuZero** (net + tree search, iterated
self-play), **DeepStack/Libratus** (value nets over information sets — matches
`obs.sample_world` exactly), **Thinking Fast & Slow** (distill slow deep search into fast net).
**Concrete pipeline settled:** (1) generator = parallel self-play on laptop logging
(encoding, chosen move, search scores, outcome); (2) value net v1 distilled from outcomes +
teacher search; (3) deployed as sample-worlds → execute candidates → batch v() → argmax, with
selective 2-ply; (4) policy-improvement rounds where deployed bot becomes the new teacher.
**Edge assessment settled:** a trained model is TABLE STAKES vs O (Hyde's own trained bot), not
an auto-win; most of the field will ship F-clones or clock-out search; the winning combination
= net+search clock-green + robustness-first selection + best-documented journey; full
AlphaZero-from-scratch won't converge on CPU in 6 h, but distillation-based AlphaZero-lite will.

### B8. "Make the context download for the decision model; ensure laptop output is fully compatible with team machine + AWS ring; maximize every parameter, especially self-learning and winning in any adversarial scenario"
**Settled:** `DECISION_MODEL_BLUEPRINT.md` written — the executable spec:
- **Architecture:** ~180-feature compact encoding (both seats' effective/total/reachable
  production, hands, distances-to-build, VPs, buildable counts, LA/LR state, ports, bank, dev
  counts, robber flags, turn) → 256×256 GELU MLP → sigmoid win-prob; batch all candidates in one
  forward; k=1 worlds normally, 2–3 for placement/robber; selective 2-ply; endgame overrides as
  transparent bonus terms.
- **Compatibility contract (§2):** full parity checklist + clock table (local ≤0.5 s/game,
  ≤5 ms/decision, 0 over-clock, 0 faults, load ≤5 s) + zip hygiene + determinism + workflow.
- **Self-learning loop (§4):** heuristic teacher H → mass parallel self-play data → distill
  v-net → 2–3 AlphaZero-lite improvement rounds → CMA-ES on remaining knobs → **legal in-match
  online adaptation** (opponent stats from public history, bounded eval-knob shifts; compounds
  over 1000-game group stage). Checkpoints named with config+seeds+wr; `experiments.md` row per
  run (the log IS the report).
- **Adversarial maximization (§5):** select changes on **portfolio worst-half win rate**
  ({F, random, F-family clones, our old versions, archetype bots}) — the actual advancement
  metric; anti-overfit (disjoint seeds, regularization, CI discipline); hidden-VP endgame
  denial (effective-VP model of opponent, robber/LA/LR denial, dev-deck VP starvation);
  robber-as-weapon targeting; port-corner economics; zero-fault engineering (fallback chain
  net → heuristic → first-legal; 2% fault DQ untouchable).
- **Quality gates (§6)** per submission; **timeline (§7)** now→17:00 freeze; **do-NOT list
  (§8)** of every known way to die.

### B9. "Include the full context of this whole chat in the context download"
**Settled:** this document (`FULL_CONTEXT_DOWNLOAD.md`).

### B10. "We can surely train the neural net on the laptop GPU to maximise training gains?"
**Settled — yes, legal and portable, with one honest caveat about where the gains come from:**
- **Legality:** training hardware is completely unconstrained — the rules govern the shipped
  artifact, not how weights were produced (laptops/Colab/Kaggle explicitly encouraged).
- **Compatibility bridge (mandatory discipline):** train anywhere, but SAVE with
  `torch.save(model.state_dict(), ...)` and the bot LOADS with
  `torch.load(..., map_location="cpu", weights_only=True)`, constructs on CPU, never `.cuda()`
  in shipped code. Keep torch **2.8.0 both sides** (CUDA build for training, CPU build in the
  parity venv) — same version = same serialization; device never persists in a state_dict.
  **GPU training happens in a SEPARATE venv** so the parity venv (CPU torch + pinned kit +
  honest `evaluate` timings + selfcheck) stays pristine. Bridge sanity check before committing:
  1 GPU epoch → load weights in CPU parity venv → one inference → identical outputs.
- **Where GPU helps:** tiny net (~110k params) → epochs go minutes→seconds, which in remaining
  hours = **more AlphaZero-lite improvement rounds**; equally important, GPU training frees all
  CPU cores to run self-play 100% of the time. The real gain is this parallelism.
- **Where it doesn't:** the pipeline bottleneck is **data generation** (self-play = pure-Python
  catanatron engine, CPU-bound, one process per game) — GPU does nothing there; the lever is
  more CPU workers. And the ring is CPU-only, so inference gains nothing from GPU (batched
  forward is already ~1 ms/candidate batch on one core).
- **Decision tree (settled):** (1) `nvidia-smi` — no NVIDIA GPU (Intel/AMD iGPU does NOT count;
  torch CUDA needs NVIDIA) → do NOT chase CUDA mid-hackathon; CPU training is minutes/epoch and
  the gains lever is self-play throughput + more rounds. (2) NVIDIA present → separate venv,
  `pip install torch==2.8.0` (default CUDA wheel), state_dict bridge as above. (3) No local GPU
  but CPU epochs become a MEASURED bottleneck → Colab/Kaggle free T4 (legal), cost = shuttling
  `.npz` datasets up and weights back.

---

## PART C — Locked decisions (the team's plan of record)

1. **Goal framing:** maximize WIN PROBABILITY on the organizers' actual formula (robustness +
   strength + report + presentation), not leaderboard vanity; no guaranteed-win fantasies —
   maximum-probability build.
2. **Workflow:** laptop-first development + training (raw compute, no clock); team machine for
   the one early pipeline-proof upload and per-milestone validation + submission; never debug on
   the ring — reproduce locally from logged seeds.
3. **Submission #1 = F-copy**, fired early to validate folder→zip→upload→leaderboard.
4. **Bot architecture:** handcrafted F-fixed heuristic teacher → distilled state-value net →
   sample-world 1-ply + selective 2-ply search → budget manager (never clock-out) → endgame
   hidden-VP/denial overrides → optional online adaptation. Ship only what measurably raises
   portfolio worst-half win rate.
5. **Measurement religion:** 400+ games tuning / 1000 pre-submit, same-seed comparisons,
   disjoint TRAIN/EVAL seeds, CI-aware acceptance, `experiments.md` from hour one, report
   drafted continuously.
6. **Engineering bar:** 0 faults / 0 timeouts / 0 over-clock in every 1000-game run ×3 seed
   ranges; load ≤5 s; single-thread; catch-all wrappers everywhere.
7. **Training hardware policy:** GPU if an NVIDIA GPU is already working on the laptop
   (separate venv, state_dict bridge via `map_location="cpu"`, torch 2.8.0 both sides); CPU
   training otherwise (tiny net → minutes/epoch, not the bottleneck); Colab/Kaggle T4 only if
   CPU epochs become a measured bottleneck. Never let GPU setup eat hackathon time; the gains
   lever is rounds of generate→distill→redeploy, which is CPU-self-play-bound. Inference path
   in the shipped bot is CPU-only, single-threaded, forever.

## PART D — Workspace index

| Path | Contents |
|---|---|
| `D:\Odyssey\FULL_CONTEXT_DOWNLOAD.md` | THIS master document (conversation-complete) |
| `D:\Odyssey\GAME_DAY_CONTEXT.md` | Deep repo/engine reference (compact truth tables) |
| `D:\Odyssey\DECISION_MODEL_BLUEPRINT.md` | Executable build spec (architecture, learning loop, gates) |
| `D:\Odyssey\HYDE_CATAN_HACKATHON_BRIEF.md` | Pre-kickoff prep brief (superseded on conflicts) |
| `D:\Odyssey\catan-hackathon-agent\` | Official kit clone (problem statement, harness, examples) |
| `D:\Odyssey\.engine_ref\` | Pinned catanatron engine sources (F's brain, features, mechanics) |
| Team portal | https://main.d2sgimrfzis7m5.amplifyapp.com/ |
| SSH | `team<name>@32.190.233.160` · fp `SHA256:Z8tsuVPiJ3dyI2vOQt2wV8hcBQxyaZOLpB2H4rnvvBE` |

## PART E — Immediate next actions (as of writing)
1. Team machine: passwd, selfcheck, F-copy upload (submission #1) — if not already done.
2. Laptop: Python 3.11 venv + pinned installs + selfcheck (parity). Run `nvidia-smi`; if an
   NVIDIA GPU is present, create the SEPARATE GPU-training venv + state_dict bridge sanity
   check; otherwise stay CPU and prioritize self-play worker count.
3. Build heuristic teacher H (F-fixed eval) + parallel self-play generator harness.
4. First `evaluate` of H vs F at 1000 games → submission #2 if ≥0.60.
5. Data generation → v-net v1 distillation → net-1-ply bot → submission #3.
6. Keep `experiments.md` live; report skeleton from the first hour.
