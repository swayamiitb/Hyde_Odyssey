# Dataset Context Download — Self-Generated Training Data Plan

> Compiled 17 Oct 2026 (~12:45 IST) for the coding assistant executing the build. Companion to
> `DECISION_MODEL_BLUEPRINT.md` (architecture/learning loop), `RESEARCH_SOTA_DOSSIER.md`
> (techniques + sources, §8–§11), `FULL_CONTEXT_DOWNLOAD.md` (master event/engine/clock facts).
> Self-contained: everything needed to build the data pipeline is here.

---

## 1. Verdict on external datasets: NONE are usable — we generate everything

The internet was swept; what exists is 3–4-player human games with trading (Kaggle records,
STAC/StacSettlers corpus, JSettlers/BSettlers logs, Settlers Game Notation). Our game is a
**2-player, no-trading variant on the pinned catanatron commit** with snake setup, Monopoly in,
1000-turn cap. External data fails on three independent axes: wrong distribution, no converter
into our feature encoding, and missing labels (per-candidate search scores + ground-truth hidden
dev cards exist only in self-generated games). Self-play on the exact engine is definitionally
on-distribution for the scoring funnel. External items go in the report's related-work only
(Kaggle records; Guhe & Liszka CIG 2014; StacSettlers).

**Consequence: the dataset pipeline is a first-class deliverable, not a side effect.** Budgeted
volume: ~0.5–1 s/game per worker pair → 8 workers ≈ **30–70k games/hour ≈ 3–8M labeled
positions/hour** (~112 decisions/game). 5M rows × ~200 feats × 4 B ≈ 4 GB → memmap shards (§6).

## 2. Generation harness spec (build this first)

1. Custom driver script (NOT `evaluate`, we need per-decision logging): top of file
   `from catan_kit._patch import ensure_hash_seed; ensure_hash_seed()`, code under
   `if __name__ == "__main__":`. `PYTHONHASHSEED=0` is mandatory for reproducibility.
2. Workers: `multiprocessing.Pool`, ONE thread each (`torch.set_num_threads(1)`,
   OMP/MKL=1). Match laptop cores + team machine cores; stop-generation criterion below.
3. Per worker: run `match.play_chunk`-style games between two bot files over a seed range;
   hook a `Seat`-like wrapper that, at EVERY real decision (len(actions) > 1), records one row
   from THAT seat's masked view (§4 observability rule).
4. Seed ledger (sacred): TRAIN range A (e.g. 100000–499999), TRAIN range B for later rounds,
   EVAL range (990000+, the `evaluate` default = local benchmark), HOLDOUT (never trained on).
   Disjoint forever. Every experiment logs its ranges.
5. Bots are plain `agent.py`-style classes loaded in-process (no pipes needed offline); the
   teacher H, F (copy `examples/f_agent.py`), and the zoo (§3 D10) are all importable modules.
6. Output: per-worker shard dir `data/D{n}_r{round}/{worker}.npz` — append-only; a manifest
   JSON per shard (round, matchup, seeds, bot versions, catan_kit commit).

## 3. Corpus catalog — D1…D15 (the extensive list)

| ID | Corpus | Matchup / filter | Why it exists |
|---|---|---|---|
| D1 | Anchor | F vs F (~5k games) | Baseline distribution; F-imitation sanity check |
| D2 | Main | Teacher H vs H | Bulk supervision for value + policy |
| D3 | Cross | H vs F, both seats | Adversarial mix; F-zone robustness |
| D4 | League | H vs past checkpoints + perturbed-F weights | Population robustness (worst-half metric); anti-forgetting mix ≤30% |
| D5 | Placement book | EVERY legal opening placement on every generated board, scored by deep teacher lookahead | Dense supervision for the highest-leverage decision; effectively a learned opening book |
| D6 | Endgame oversample | combined public VP ≥ 14, or turn > median | The rare decisive region (oversampling per PER/SIL logic) |
| D7 | Decision slices | robber moves, discards, maritime trades | Conditional fix-ups where generic nets are weakest |
| D8 | Reanalysis relabels | D2–D4 re-scored by each new net (no new games) | Free epochs; the §8.2 self-improvement loop of the dossier |
| D9 | Seat-pair corpus | same board, both seats (evaluate already pairs them; keep both rows) | First-mover advantage + per-seat calibration |
| D10 | **Zoo corpus** | scripted archetype bots: road-rusher, city-builder, dev-spammer, robber-abuser, port-monopolist, greedy-blocker, ε-random(H) | Off-policy coverage of "weird" states a greedy teacher never reaches — this IS the robustness portfolio (dual use, §7) |
| D11 | World-variance pairs | same decision, k=4 `sample_world` draws recorded | Robustness to hidden-state uncertainty; feeds the uncertainty head (§4) |
| D12 | Curriculum ladder | temperature-softmaxed teacher (τ high→low) + ε-exploration mixes | Wide early coverage → sharp later data; classic curriculum |
| D13 | Rare-event generators | scripted near-endgame starts (8–9 VPs), robber-war scenarios, 7-heavy reseeded dice | Rare dynamics coverage; endgame denial training |
| D14 | Placement table (offline analysis) | board hash → best placement + margins | Report figures + placement-net validation set |
| D15 | Game-level meta | checkpoints at setup/mid/end of every game | Tournament-level stats, drift monitoring, report tables |

Priority if time-starved: D2 → D10 → D5 → D6 → D8 → D3 → D12 → rest.

## 4. Row & label schema (and THE critical correctness rule)

**Observability rule (load-bearing):** every feature written to disk must be computable from the
DECIDING seat's information set — `obs` fields + a sampled world — NEVER from true hidden state
or future. Rationale: at play time the net sees exactly this view; training on god-view features
= silent distribution shift that only shows up as unexplained win-rate loss. The ONE exception:
auxiliary hidden-card LABELS (below), which deliberately regress public-view features → true
hidden composition (that asymmetry is the point of the masked-modeling task).

Per real decision, record:
- `feat` float32[~200]: my encoding of the sampled world BEFORE the move (blueprint §3.1) —
  identical function to the bot's runtime encoder. One function, two callers. No divergence.
- `cand_score` float32[C_max] + `cand_mask` bool[C_max] + `cand_type` uint8[C_max]:
  teacher/search score per legal candidate (padded, masked). C_max ≈ 256; opening placement
  may exceed → cap at top-200 by teacher prescore (never cap the legal set at runtime).
- `chosen` int16, `best_idx` int16 (teacher argmax — they differ when exploring/τ>0).
- Labels (multi-target): `z` (win 0/1), `vpdr` (VP_DISCOUNTED_RETURN = 0.9999^turns × final VPs),
  `nstep2/4/8` returns, computed offline from the game record; `opp_dev_truth` uint8[5] (the
  opponent's true unplayed dev composition — auxiliary head only); `world_var` float32 (variance
  of teacher score across D11 draws — uncertainty label).
- Meta: `turn`, `board_seed`, `seat`, `decision_type` (placement/build/robber/discard/trade/dev/
  end), `round`, `matchup_id`. dtypes fixed forever (fp32 features, no fp16 on disk).
- Opponent-model label (optional, BRExIt-lite): opponent's realized next action type.

## 5. Splits, weighting, sampling, versioning

- **Splits by board-seed family** (group-wise): train / val (5%) / test-frozen (5%) — no board
  in two splits; the group stage plays fresh boards, so this is the honest estimate.
- Frozen evaluation slices: fix `val` at round 1 and never retrain on it; per-decision-type
  accuracy (placement acc, robber acc, build acc) + win rate of the deployed bot = the model
  selection signal.
- Weighting policy: SIL advantage weight (outcome − v̂), endgame ×2–3 cap, decision-type cap
  ≤30% share per epoch, D4 mix ≤30%. Implement as a probability side-array (§9.1 dossier).
- Versioning: every ExIt round appends shards under a new `round` tag; `experiments.md` row per
  version with row counts + val metrics + downstream 400-game win rate.
- Stop-generation criterion: two consecutive rounds with val-loss improvement < ε → freeze data,
  spend time on search/tuning instead.

## 6. Storage & compute budget

- Sharded `np.memmap` (float32 .npy) + per-shard manifest; concatenation by index arithmetic;
  loader yields shuffled batches without resident copies.
- Ring discipline per worker (fixed shard count, overwrite oldest) for streaming rounds.
- Hardware split: laptop cores + team machine (leave 1 core for the interactive dev loop;
  laptop battery → plug in). Colab/Kaggle optional for extra workers (ship shards back).
- Checkpoints: state_dict only (~0.5 MB) per round, named `v{r}_{rows}_{wrF}.pt`.

## 7. Dual-use: the zoo IS the portfolio

D10's archetype bots and D4's old checkpoints are simultaneously (a) training opponents,
(b) the local robustness portfolio for the blueprint's worst-half selection metric, and
(c) report material (per-archetype win-rate heat map). One implementation, three jobs —
write them once as simple `agent.py` classes.

## 8. Pitfalls (each one silently costs win rate)

1. Training features from the god-view (violates §4 observability) — the #1 silent bug.
2. Seed leakage into EVAL/HOLDOUT ranges — inflates local numbers, crashes on fresh boards.
3. Teacher collapse (all data from one greedy policy) — always keep ≥20% zoo/cross games.
4. Seat imbalance — record both seats (D9); the value net must see both perspectives.
5. fp16 on disk / dtype drift between generator and runtime encoder — one encoder module,
   fp32, imported by both paths; unit test: `encoder(obs_from_masked_world) == stored feat`.
6. Forgetting across ExIt rounds — keep D4 old-checkpoint games mixed in.
7. Unlabeled candidate sets (storing only chosen action) — candidate scores are what make
   distillation and reanalysis (D8) possible; store them from day one.
8. Placement rows overwhelming the corpus (every board enumerates ~50+ spots in D5) — keep D5
   in its own shard family with its own sampling cap.

## 9. Acceptance gates for "dataset ready"

- [ ] Harness reproduces identical games from a logged seed (determinism check).
- [ ] ≥1M rows in D2+D10 by first training run; manifest complete.
- [ ] Encoder unit test passes (§8.5) on 1k random rows.
- [ ] val slice frozen and untouched; per-decision-type metrics reporting live.
- [ ] First value net trained: val AUC > 0.75 (weak signal gate) before wiring into the bot.
- [ ] experiments.md updated with dataset versions + downstream win rates.

## 10. External sources (context only, no training use)

Kaggle Catan Game Records & Statistics · StacSettlers (STAC corpus) · JSettlers/BSettlers logs ·
Settlers Game Notation (JSON spec) · Guhe & Liszka CIG 2014 · game-datasets list · settlers-rl ·
catan-rl — full URLs in `RESEARCH_SOTA_DOSSIER.md` §7 and the conversation record.
