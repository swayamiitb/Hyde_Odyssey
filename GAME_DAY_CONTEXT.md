# Game-Day Context Download — Hyde × WnCC Catan Agent Hackathon

> Compiled 17 Oct 2026, ~10:40 IST, from the official repo
> `github.com/lyric-hyde/catan-hackathon-agent` (cloned at `D:\Odyssey\catan-hackathon-agent`)
> plus the pinned catanatron engine commit `d3f4ad0` (key sources cached at `D:\Odyssey\.engine_ref`).
> Supersedes `HYDE_CATAN_HACKATHON_BRIEF.md` wherever they disagree. Single source of truth for today.

---

## 0. Corrections to the prep brief (ground truth from the problem statement)

| Prep brief said | Reality |
|---|---|
| "Monopoly probably absent (deck lists 4 dev types)" | **Monopoly IS in.** Engine dev deck = standard 25 cards: 14 Knight, 5 VP, 2 Road Building, 2 Year of Plenty, 2 Monopoly. Rulebook-vs-engine conflicts resolve in the engine's favor. |
| "Player trades maybe in protocol" | **No player-to-player trading at all.** Bank 4:1, generic port 3:1, specialty port 2:1 only. |
| "Is a learned component mandatory? Resolve at kickoff" | **Not mandatory.** "You may use any language model to write code"; submitting a copy of F is explicitly legal. Judging = robustness + strength + report + presentation. |
| "Tournament ~18:00–20:00 vs every team" | Precise funnel below: 7 scored submissions → top 30 by private-set win rate → 3 groups of 10, 1000 games round-robin → top 2 per group by ROBUSTNESS present. Ends 19:30. |
| Unknown engine | **catanatron** pinned to git commit `d3f4ad0` (v3.3.0 internals, NOT PyPI 3.2.1 — action space differs). Python 3.11. Allowed imports: stdlib + numpy, scipy, scikit-learn, torch 2.8.0 CPU. Nothing else. |

Confirmed from prep: 2 players, first to 10 VP, official spiral board, randomized seeds. Event in LA001 IITB, 10:00–19:30.

---

## 1. The contract your agent must satisfy

```python
class Agent:
    def __init__(self):
        """Load weights/tables here. Once per process, up to 60 s, NOT charged to the clock."""
    def decide(self, obs, playable_actions) -> int:
        """Return an INDEX into playable_actions."""
```

- One folder with `agent.py` at zip top level + any data files (≤100 MB total). **Python only, no compiled code, no spawning programs, no network.**
- One process plays MANY games in a row vs the same opponent → **online adaptation across games is allowed** (nothing persists across restarts). `Agent.__init__` runs once per process, not per game.
- Runs on Linux: case-sensitive filenames, use `os.path.join`, load weights via `os.path.dirname(__file__)`. Agent dir is **read-only** at eval; write temp files to `tempfile.gettempdir()` only.
- Errors: exception or illegal index → harness plays a random legal move and counts a fault. **>2% of decisions faulty → submission disqualified.** Return type must be a genuine int (bool/float/str/numpy are rejected or error); wrap-up: return `int(...)`.
- Agent's own `random` module is **reseeded per game** from the game seed (deterministic replay). Forced moves (exactly 1 legal action — 47% of decisions!) never reach the agent and don't consume clock.
- `multiprocessing` refuses inside the harness; threads share one core — **single-threaded design only** (`torch.set_num_threads(1)`; OMP/MKL already forced to 1 by the host).

## 2. The clock (the central constraint)

Per game, agent's own CPU (all threads+children, read from outside by the harness):

```
allowance after k decisions = min(300 ms + 12.5 ms × k, 1500 ms)
```

- ~56 decisions/game average → **~1 s CPU per game**, ceiling 1.5 s (reached at 96 decisions). Observation delivery (<1 ms/decision) is charged.
- Overdraw → harness plays **uniform random** moves for the rest of the game (leaderboard shows "games over the clock").
- Also: 3 s max per single decision (silent agent killed+restarted, random moves finish game); 3 s wall-slack beyond CPU allowance; 60 s load.
- Reference costs on their machine (AWS c7a.48xlarge, AMD EPYC 9R14, 1 core, 2 GB): random agent 11 ms/game, **F 158 ms/game (2.9 ms/decision)**. We get ~6× F's compute. **Our laptops are ~1.5–2× faster per core than their EPYC — plan for their machine to charge ~2× what local runs report.**
- `obs.cpu_left` (seconds, this decision's increment included) → budget searches adaptively. The 300 ms base is best spent on opening placement (most legal moves, highest leverage).

## 3. What the agent sees (`catan_kit/obs.py`)

Full: board (tiles/numbers/ports/robber/buildings/roads), exact own hand + own dev cards, own public & actual VPs, bank counts, opponent's exact resource hand (derivable anyway: 19−bank−mine), **count** of opponent's unplayed dev cards, opponent public VPs, all dev cards played by both, turn number, full move+dice history, own color, cpu_left.

Hidden (only real hidden info): **which** dev cards opponent holds → their hidden VP cards. Opponent can win a turn that looks safe.

`obs.sample_world(rng)` → a full catanatron `Game` consistent with everything visible; opponent's unplayed devs drawn uniformly from the unseen pile (deck order also reshuffled; opponent BUY records blanked; pile sorted-before-shuffle so no info leaks). **Deterministic world-copy: engine random draws (e.g. robber steal) replay exactly** (agent process's `random` is seeded per game). Sample multiple worlds and average for robust decisions — costs CPU, trade-off is ours.

## 4. Engine ground truth (verified at commit d3f4ad0)

- **Setup snake order, verified in `apply_action.py`:** P0 places S+R, P1 places S+R, P1 again, P0 again; **P0 (first placer) takes the first turn.** Second settlement immediately yields 1 of each adjacent resource.
- Turn limit 1000 turns; no winner = loss for BOTH (matters vs slow strategies).
- Roll 7: players with >7 cards discard half (floor); robber moved by roller; steal 1 random card from a victim on the target hex.
- Dev cards: can't play the turn you buy (`OWNED_AT_START` check); at most one played per turn. VP cards never "played", revealed at win.
- **Largest Army:** first to 3 knights takes +2; steal requires **strictly more** knights (ties keep the holder). **Longest Road:** 5+ to hold, standard steal on strictly longer.
- Actions (engine never generates player trades): ROLL, END_TURN, BUILD_ROAD (edge tuple), BUILD_SETTLEMENT/BUILD_CITY (node id), BUY_DEVELOPMENT_CARD, PLAY_KNIGHT_CARD, PLAY_ROAD_BUILDING, PLAY_YEAR_OF_PLENTY (2 resources), PLAY_MONOPOLY (resource), MOVE_ROBBER ((coord, victim_or_None)), DISCARD_RESOURCE (resource), MARITIME_TRADE (give-give-give-give/get tuple, None-padded).
- `create_sample_vector(game, color)` = **614 numbers**: player VPs/flags/pieces-left/longest-road-length, own hand counts, played-dev counts, tile one-hots + probabilities + robber, port one-hots, full graph one-hot (every node×player×building, every edge×player), bank counts, prompt flags. It does **NOT** contain production/reachability/expansion features (those extractors are commented out in `feature_extractors`) — i.e., none of F's derived features; you'd compute those yourself or learn them.

## 5. F — the anchor bot we must beat (code ships: `examples/f_agent.py`)

F = catanatron `ValueFunctionPlayer(base_fn, DEFAULT_WEIGHTS)`. **1-ply greedy**: for each legal action, execute on ONE sampled world, evaluate resulting position, argmax (deterministic on ties). Its exact eval (weights × features):

| Feature | Weight | Meaning |
|---|---|---|
| public_vps | **3e14** | winning is everything |
| production | +1e8 | own EFFECTIVE production (robber-aware) + variety bonus: each distinct resource produced counts as +4 production points (≈+0.111 prob-sum) |
| enemy production | −1e8 | opponent's effective production |
| reachable_production_0 | **0** | production buildable with 0 roads — **ignored** |
| reachable_production_1 | 1e4 | production reachable with 1 road |
| buildable_nodes | 1e3 | expansion spots |
| hand_synergy | 1e2 | closeness of hand to city/settlement costs |
| longest_road | 10 (or **0.1** while buildable nodes exist) | only values LR when boxed in |
| hand_resources | 1 | raw card count |
| discard_penalty | −5 | when hand >7 |
| hand_devs | 10 | dev cards in hand |
| army_size | 10.1 | knights played |

F's blind spots (exploitable): **ports never valued**, bank never valued, opponent hidden VPs invisible to it, no lookahead past 1 ply, deterministic argmax (tie-breaking fixed by the sorted-trades patch), single world sample per decision, reachable-at-0 production ignored, LR actively undervalued early. F spends 2.9 ms/decision on their box.

Reference ladder vs F (1000 games, same clock): all library bots ≈ 0.001 — the three search bots (MCTS-25, AlphaBeta-d3, SameTurnAlphaBeta-d2 at 0.080) lose by **running out of clock** and finishing on random moves. "A search that fits the allowance is the point of the exercise."

**O** = Hyde's own trained bot, "beats F by some margin". Unknown method. Strength = 0.5·wr_vs_F + 0.5·wr_vs_O. We never see O's code.

## 6. The funnel and scoring (all times IST)

- **11:00** submissions open (leaderboard live) — ~20 min after this doc. **17:00** last upload. **17:30** report due (top 30). **17:50** top-30 + group results. **18:00** six teams present (10 min each). **19:30** end.
- Each upload: must load + play 2 test games without error, else returned (doesn't consume a scored slot). **7 scored submissions per team, 10 uploads total, max 2 in queue.** Practice locally = unlimited.
- Every submission scored twice: public boards (leaderboard) + private boards. Group stage uses fresh boards.
- Top 30 by mean of private win rates vs F and O (best submission). Groups of 10 (snake fill A,B,C,C,B,A,…), **1000 games vs each of 9 group opponents, seats alternated**.
- Top 2 per group advance by **ROBUSTNESS = mean win rate against the half of your group opponents you do WORST against** → a narrow bot overfit to F loses to a well-rounded one.
- Weights: robustness+strength together ≈ report+presentation together. Leaderboard strength ≤ 25% of total. Report ≤ 4 pages A4 11pt, PDF: method clarity, design justification, journey insights, experiment quality. Presentation 6 min + 4 min questions; **questions carry the most weight**. High win rate without good reasoning caps your score.
- Fair play: no team-to-team sharing, everything built today (public libs/papers OK if named), agent may not call any service while playing, no extracting hidden info / seeds / dice key, no collusion. Every number in the report must come from a run we made (they may re-run our agent).

## 7. Ops runbook

- **Team machine** (8 cores/16 GB, kit installed): `ssh team<teamname>@32.190.233.160`, initial password = teamname + last-4-digits of leader's phone, twice (e.g. `blanc12341234`). Fingerprint must be `SHA256:Z8tsuVPiJ3dyI2vOQt2wV8hcBQxyaZOLpB2H4rnvvBE`. `passwd` immediately (≥8 chars, not palindrome). Kit at `~/catan-agent-hackathon`, `cd ~/catan-agent-hackathon && source .venv/bin/activate`.
- Verify env: `python -m catan_kit.selfcheck` (checks pinned versions, determinism — seed 4256 must give RED win in 111 turns — and f_agent beats random 20/20).
- Play matches: `python -m catan_kit.evaluate --a my_team/agent.py --b examples/f_agent.py --games 1000` (this IS the scoring script; on the day only seeds/dice key/opponents change). Seats alternate; each board played twice with the same dice; `--seed0` controls the set — **use disjoint seed ranges for train vs eval**.
- Submit: zip with `agent.py` at top level → `~/upload.sh my_team.zip`; report → `~/upload.sh report.pdf`. Portal: `https://main.d2sgimrfzis7m5.amplifyapp.com/` (team name + leader's full phone).
- Free LLM assistant (OpenCode + GLM-5.3 Flash via `inference.hyde.tech`, API key from team leader; see README §7).
- Local Windows dev (this laptop): Python 3.11 + git; `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu` then `pip install -r requirements.txt` from the kit root. Use `uv python install 3.11` if needed. Local timings ≈ ÷2 vs their EPYC.
- Determinism for custom scripts: call `catan_kit._patch.ensure_hash_seed()` first, guard with `if __name__ == "__main__":`, or run with `PYTHONHASHSEED=0`.

## 8. Strategic read (implications, ranked)

1. **The exercise is search-that-fits-~12.5 ms/decision.** Naive MCTS/alphabeta (library versions) all clock out and die at 0.001. Anything we build must be budget-aware via `obs.cpu_left`.
2. **F is fully known.** Cheapest big edge: a better handcrafted eval + 1-ply (beat F at its own game, fix its blind spots: ports, hidden-VP endgame awareness, robber targeting, LA/LR correctly priced) — then spend remaining budget on 2-ply expectimax over opponent reply only where it matters, or distill a deeper search into a tiny MLP (the reading list points exactly there: Buro's Othello linear eval, TD-Gammon, distillation, DAgger).
3. **Opening placement is the highest-leverage decision** and the 300 ms base exists for it: precompute-free EV scoring of corners (number probability × yield, diversity, port access, expansion room) — worth spending most of the base budget there.
4. **Don't overfit to F.** Robustness (worst-half win rate) + O in the strength score + fresh boards mean we need a genuinely good Catan player, not an F-exploit. Verify vs random AND vs our own past versions (portfolio of sparring partners).
5. **Online adaptation across games in one process is legal** — e.g., simple opponent modeling from played_devs/action history. Cheap, safe edge for the 1000-game group stage.
6. **Win-rate noise:** 1000 games gives ±~3% at 95%; differences inside the CI are noise (advice §5: ≥400 games during tuning, 1000 before submitting; same seeds when comparing versions).
7. **Report as you go** (experiment log table, failures included) — it's worth as much as strength and can't be written at 17:05 from scratch.
8. Immediate sequence once building starts: selfcheck → copy random_agent into `my_team/agent.py` → heuristic 1-ply F-clone runs → upload submission #1 early (tests the whole path; we have 7) → iterate eval + search + (maybe) tiny net, keeping the clock column green.

## 9. File map

- `D:\Odyssey\catan-hackathon-agent\` — the kit (README, PROBLEM_STATEMENT, catan_kit/, examples/).
- `D:\Odyssey\.engine_ref\` — cached engine sources at the pinned commit: `value.py` (F's brain), `features.py` (614-vector), `apply_action.py`/`state.py` (turn logic), `state_functions.py` (LA/LR, dev rules), `actions.py`, `game.py`, `enums.py`.
- `D:\Odyssey\HYDE_CATAN_HACKATHON_BRIEF.md` — prep-time brief (superseded on conflicts).
- Team portal: https://main.d2sgimrfzis7m5.amplifyapp.com/
