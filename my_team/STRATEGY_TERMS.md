# STRATEGY_TERMS.md — book-derived evaluation/search adjustments

**Variant:** 2-player, no trading (bank 4:1 / ports only), robber on 7, 10 VP to win.
**Bot:** weighted eval + 1-ply search + gated same-turn depth-2 (`my_team/agent.py`).
**Current win rate vs anchor F:** 0.523 (from the task brief).

---

## 0. Sources actually read (and their limits)

| Source | What it is | Usable for strategy? |
|---|---|---|
| `books/book1.txt` (Galland, *Catan Puzzle Book*) — read in full, lines 1–2328 | Word searches, logic riddles, lateral-thinking puzzles. The only game-mechanics content is the build-cost/VP list (lines 124–151: road = 1 brick + 1 lumber; settlement = B+L+W+G; city = 3 ore + 2 grain; knight = O+W+G; 10 VP to win) and a "coastal happiness" scoring gimmick inside puzzle 53 (lines 1846–1858). | **No.** It contains no placement, robber, or build-order doctrine. Nothing below is drawn from book1. |
| `books/book2.txt` (Tetrault & Tetrault, *Winning Strategies and Tactics*) — read in full, lines 1–2752 | Real strategy doctrine: general principles, winning patterns, robber impact, full placement phase (settlements + roads), turn process, "who is most likely to win". | **Yes, but truncated.** The file ends mid-example in the early "who is leading" analysis (last lines 2749–2752). The TOC's Middle Game (p.258), End Game (p.320), Social/Interaction (p.383) and Appendices (p.402) are **not present in the extraction**. The book title's robber chapter itself is not in the file; robber doctrine below comes from the "Impact of Robber" section (lines 374–417) and the targeting guidance in "How to determine player who is most likely to win" (lines 2451–2530). |

All book doctrine in this document is written for **3–4 player, trading-enabled** Catan. Every term below states its 2-player no-trade adaptation explicitly. All line references are to the files above; code references are to `my_team/agent.py` unless noted.

## 0.1 Current eval facts the formulas build on (verified in code)

From `agent.py:39-58` and `agent.py:73-150`:

- VP terms: `public_vps = 3e14`, `enemy_public_vps = -3e14` (`agent.py:40-41`) — the ceiling nothing may approach.
- `production = 1e8` per probability unit (`agent.py:42`); production features are probabilities, and `proba_point = 2.778/100` (`agent.py:74`), so **1 pip = 2.778e6 eval units**.
- Self reachability: `reachable_production_0 = 8e3`, `_1 = 1e4` (`agent.py:46-47`) — these matter a lot (v3 experiment, `agent.py:2-5`).
- Ports: `port_specialty = 2.5e3`, `port_generic = 1.2e3` (`agent.py:56-57`) = **0.0009 pips** — four orders of magnitude below a single pip of production.
- Enemy has **no** reachability term and **no** buildable-node term (only `enemy_production` and `enemy_army_size`, `agent.py:43,55`).
- Placement is scored by the *same* `evaluate()` (`agent.py:196-217`), so any eval-term change automatically feeds placement too.
- Robber moves are scored by 1-ply eval; because production features are "EFFECTIVE" (robber-blocked hex excluded), the current bot already greedily blocks max enemy pips. There is no resource-quality or steal component.
- Dev-card incentives are near-zero tie-breakers: `hand_devs = 10`, `army_size = 10.1`, `knight_in_hand = 5` (`agent.py:49,53-54`).

Scale discipline (verified by computation this session): every proposed term below has maximum magnitude ≤ ~2.2 pips-equivalent (≤ 6e6), i.e. **≤ 2e-8 of the 3e14 VP term** and below one settlement's production swing (~3e7). The robber tilt is ≤ 0.15 pips. The dev gate is a tie-breaker (1.5e3).

---

## 1. THE SIX ADJUSTMENTS (ranked by expected impact)

### 1. Port value: raise ~3 orders of magnitude for the bank-only economy
**Type:** additive eval term (weight change + production scaling). Feeds placement automatically.

**Book doctrine:** "3:1 ports are often underestimated… with a 3:1 port there are many more situations where you will be able to do things" (book2 lines 562–566, Quick Tip #22); "Great Ports: 1. 2:1 ports which once built upon you will have good production of that resource (5+ pips); 2. 3:1 ports with production of 7 or more pips" (lines 663–667, Quick Tip #23); "Do not undervalue 3:1 ports, they are generally much more valuable than 2:1 ports, especially in the early game" (lines 2670–2673); the scarcity chapter repeats that ports mitigate scarcity precisely because they replace trading (lines 852–860, Quick Tip #31). The book's own placement table awards "Built on Port: 2" points on a ~12–13 pip base (lines 958–959, 968–981) — i.e. a good port is worth ~2 pips.

**2p no-trade adaptation (the core of this change):** every "trade it away" option in the book is gone. The *only* conversion mechanisms are the 4:1 bank and ports, so a 3:1 port saves 25% of every conversion and a matching 2:1 saves 50%; port value should sit at the *top* of the book's range, not the bottom. The current weight (2.5e3 = 0.001 pips) makes ports invisible next to production (1e8); the bot is effectively playing bank-4:1 forever.

**Formula** (replaces the `port_bonus` block at `agent.py:122-130`):

```
for each port (resource r or None) with owned node:
    if r is None:                       # 3:1 generic
        bonus += 2.5e6
    else:                               # 2:1 specialty
        prod_r = EFFECTIVE_P0_{r}_PRODUCTION
        bonus += 4e6 * clamp(prod_r / 0.139, 0.0, 1.2)   # 0.139 = 5 pips, book's threshold
```

Max ~4.8e6 for a matched 2:1 ≈ 1.7 pips; generic 3:1 ≈ 0.9 pips — inside the book's "2 points" placement value. Unmatched 2:1 keeps the existing small residual (0.4 factor → 1.6e6·clamp). Verify no double counting: `port_generic`/`port_specialty` weights are removed from `WEIGHTS` and replaced by these absolute numbers.

**Expected effect:** the bot finally prices port settlements/roads (placement + build orders) and converts dead hands through 3:1 instead of stalling at 8 cards; in no-trade duels liquidity is usually the binding constraint, so this should be the single biggest win-rate mover of the six.

---

### 2. Denial: enemy reachability + enemy buildable terms (the 2p zero-sum term)
**Type:** additive eval term (mirror of existing self terms).

**Book doctrine:** "Between 2 equal spots choose the one that will hurt your opponents the most" (book2 lines 1979–1980, Quick Tip #43); placing "directly across from player on that key hex… shut-out players 3 and 4 from Ore" (lines 1282–1290); "building in a certain spot prevents building to all spots that are one away" (lines 1937–1939); taking a spot can deny a rival's perfect complement (lines 1971–1977).

**2p no-trade adaptation (this is the biggest doctrinal flip):** Quick Tip #44 warns "Don't go out of your way to hurt another player… you still have to beat the other two players" (lines 1981–1983) and Quick Tip #48 prices the "piss-off factor" of retaliation (lines 2243–2244). In a duel there are **no other two players and no diplomacy**: every spot, pip and port the enemy secures is one you permanently cannot. Denial is not a side-effect, it is half the game. The current eval has self-reachability (`agent.py:91-93`) but zero enemy-side expansion awareness, so it will happily let the opponent walk to the last good intersection.

**Formula** (add to `evaluate`, mirroring `agent.py:91-93` and `agent.py:119`):

```
enemy_reach_sample = reachability_features(game, enemy_color, 2)
er0 = sum(enemy_reach_sample[f"P0_0_ROAD_REACHABLE_{r}"] for r in RESOURCES)
er1 = sum(enemy_reach_sample[f"P0_1_ROAD_REACHABLE_{r}"] for r in RESOURCES)
enemy_buildable = len(st.board.buildable_node_ids(enemy_color))

eval += er0 * (-3e3) + er1 * (-4e3) + enemy_buildable * (-1e3)
```

Weights are deliberately ~40% below the self terms (8e3/1e4/1e3) so the bot prefers building itself over pure spite when both are possible, per the book's own "don't take a substantially inferior position to hurt someone" caveat (book2 lines 1981–1983) — which survives into 2p as "deny at equal value, not at a loss".

**Expected effect:** road/settlement choices start contesting chokepoints and the last ore/wheat clusters instead of only optimizing own expansion; should convert some of the near-50/50 racing losses (the book devotes its whole "To Race or Not to Race" section, lines 2180–2244, to exactly these spots).

---

### 3. Robber-targeting rule: quality-weighted blocking + steal sizing
**Type:** robber-targeting rule (small additive tilt at the MOVE_ROBBER decision).

**Book doctrine:** the robber is "of central importance to any strategy" — blocking a 15-pip exposure is a "Net 30 pips" swing and mid-game robber placement is "the deciding factor on which player is favored" (book2 lines 374–411); "Don't just count points, look at the quality of the production" when choosing whom to block (lines 2525–2530); "when in doubt put it on the good player" (lines 2489–2491).

**2p no-trade adaptation:** "which player" collapses to the single opponent (no king-making, no retaliation politics — the book's piss-off factor is void). But *which hex* becomes sharper in two ways: (a) with no trading, an opponent missing a blocked resource cannot trade around it — blocking their *only* source of a build-critical resource (ore for cities, wheat for everything) is closer to a full lockout than in multiplayer; (b) the steal matters relatively more because hands are the only buffer against the 4:1 bank. The book's robber-magnet warning (lines 1296–1301: a sole producer of a scarce hex "will be a robber magnet… invest in a few DCs") is the defensive corollary — see term 6.

**Formula** (applied only while comparing MOVE_ROBBER successors; the heavy lifting stays with the existing `-1e8 × enemy_production` EFFECTIVE-production channel):

```
board_pips(r)  = sum of number-pips over all hexes of resource r
scarce(r)      = clamp((median_r' board_pips(r') - board_pips(r)) / median, 0, 1)
rw(r)          = 1 + 0.5 * scarce(r)                        # 1.0 … 1.5
tilt(h)        = 3e5 * (rw(resource_of_h) - 1)              # scarcity quality tilt
                + 1e5 * min(enemy_hand, 4)/4                # steal sizing, random victim card
# enemy_pips_on(h) itself is already priced by the -1e8 enemy_production channel
choose argmax over hexes h (non-desert) of: evaluate(succ_h) + tilt(h)
```

i.e. among near-equal blocks, prefer the scarce/city-critical resource and prefer timing the steal when the enemy hand is fat. Max tilt 4e5 = 0.14 pips — a pure tie-breaker that never overrides a genuinely bigger block.

**Expected effect:** fixes systematic robber misallocation on "big but redundant" hexes (e.g. blocking the enemy's 3rd lumber source instead of their only ore); ~10 sevens per 60-roll game (book2 line 184: ~60 rolls per game) each worth up to 15 blocked pips (lines 405–410), so even small per-7 improvements compound.

---

### 4. Scarcity-weighted production (own and enemy)
**Type:** additive eval term + placement-scoring bonus (same term; placement reuses evaluate).

**Book doctrine:** "The scarcer a resource will be during a game the more valuable that resource will be" (book2 lines 848–849, Quick Tip #29); scarcity "most often applies to Ore, and secondly to Brick… only 3 hexes of each" and ore because "in the end game we need a lot of Ore" (lines 822–828); the anomaly story where the only sheep producer won easily (lines 791–810); the placement table scores "Scarcity: Connected to Scarce Resource: 1" (line 950) and spot H at 8 pips out-scoring 10-pip spots because "Ore and Wheat are both so rare on this board" (lines 1063–1064).

**2p no-trade adaptation:** in multiplayer a scarce resource can be *traded* to the producer's benefit (the whole anecdote at lines 791–810); with no trading, scarcity is purely about self-sufficiency and denial — owning the scarce pips matters even more (you cannot buy your way out at better than 4:1), and the enemy owning them is worse. Ports partially mitigate (lines 852–860), which is why this term is ranked below term 1.

**Formula:**

```
scarce_pips(player) = sum over resources r of effective_pips(player, r) * scarce(r)   # scarce() as in term 3
eval += 1.5e7 * (scarce_pips(me) - scarce_pips(enemy))
```

Max ≈ 3.3e6 ≈ 1.2 pips. This is a deliberate *re-weighting* of existing pips (scarce pips count up to 1.5×), matching Quick Tip #29's intent without inflating raw production.

**Expected effect:** placement picks the 11-pip ore/wheat spot over the 12-pip sheep-heavy spot (exactly the book's spot-A-vs-B comparisons, lines 1099–1104); robber (term 3) reads the same weights, keeping the two consistent.

---

### 5. Complementary-ratio balance term ("what you can do with them")
**Type:** additive eval term.

**Book doctrine:** Principle #2 — "It is not the # of resources that you produce that matters it is what you do with them… the right mix" (book2 lines 198–202); complementary ratios: "Wheat and Ore… 3:2 ratio to build upgrades… Lumber & Brick… 1:1 ratio for building roads… Wheat, Ore & Sheep… approximately 3:3:1" (lines 723–732, Quick Tip #27); the synergy table grading 21–22 pip setups "As bad as it gets!" when ratios are wrong (lines 1704–1800); "Always sacrifice a few pips to get all resources" (line 2556).

**2p no-trade adaptation:** the book's remedy for a lopsided hand is trading (lines 522–533) — unavailable. A 2:1-wrong ratio must be repaired through ports/bank at a 25–75% card tax, or by new placement. So ratio-balance should rank *higher* than in multiplayer, but its absolute scale stays modest because the existing `hand_synergy` term (`agent.py:95-106`, weight 1e2) already covers the current hand; this term covers the *production profile* the hand will keep drawing from.

**Formula:**

```
W,O,S,L,B = own EFFECTIVE production per resource (probability units)
city_kits = min(W / 2, O / 3)      # cities cost 3 ore + 2 wheat
road_kits = min(L, B)              # roads cost 1 lumber + 1 brick
dc_kits   = min(W, O, S)           # dev cards cost O + W + S
balance   = clamp(city_kits + road_kits + 0.5 * dc_kits, 0, 0.6)
eval += 1e7 * balance
```

Max 6e6 ≈ 2.2 pips: a fully complementary setup beats a same-size broken one by ~2 pips of equivalent value — the book's "sacrifice a few pips" (line 2556) made exact, capped so it can never override a genuine production gap. Optional tie-breaker from the numbers-diversity note (lines 1855–1866: prefer 6,5,3 / 8,9,12 over doubled numbers): `+ 2e5 * max(0, distinct_numbers_covered - 3)` — the book itself calls this "a small factor" (line 1865).

**Expected effect:** upgrades the W/O and L/B pairings during placement and city-vs-settlement build choices; reduces the "full hand, nothing buildable" stall states the book calls wasted resources (lines 1787–1801, Quick Tip #39) which, with no trading, directly become 7-discards.

---

### 6. Development-card gate: soldiers once the mid-game engine exists
**Type:** build-priority tweak (gated re-weighting).

**Book doctrine:** "There comes a point where getting soldiers and being able to control the robber is more important than increasing production… Proxy General Rule: Once you have at least one city and 4 points in production: get soldiers" (book2 lines 405–417, Quick Tip #11); winning patterns #1/#2 are city-then-DC sequences where soldiers "ensure the robber does not stay on you" (lines 280–311); the robber-magnet passage recommends "invest in a few DCs before getting a city on that spot" when you monopolize a scarce hex (lines 1296–1301); "Buy your development cards first… one at a time. Re-Evaluate" (lines 2447–2449, Quick Tip #51 — the depth-2 expansion at `agent.py:232-247` already approximates the one-at-a-time re-evaluation).

**2p no-trade adaptation:** Largest Army is a 2-VP swing contested by exactly one opponent, and every soldier also *is* robber control (move it off your key hex / onto theirs) — in a duel the defensive half of soldier value roughly doubles, because there is no third player's robber to hide behind. Conversely VP-card density per purchase is unchanged, so the gate stays conservative.

**Formula** (replace the flat `hand_devs`/`knight_in_hand`/`army_size` contributions when the gate is open):

```
gate = (cities >= 1) and (production_points >= 4)      # book's proxy rule verbatim; production_points = settlements + 2*cities
if gate:
    eval += 1.5e3 * unplayed_dev_cards + 1.0e3 * knights_in_hand + 30.0 * army_size
else:
    eval += 10 * unplayed_dev_cards + 5 * knights_in_hand + 10.1 * army_size   # unchanged (agent.py:49,53-54)
```

Gated bonuses are 3 orders below one pip of production — they only reorder near-equal build choices (settlement #5 vs first dev-card buys), which is precisely the decision the book's proxy rule addresses. Keep `enemy_army_size = -12` as is (`agent.py:55`).

**Expected effect:** the bot stops ignoring the dev deck once its engine exists; expects to convert some losses where the opponent compounds robber pressure / takes Largest Army uncontested (the anchor F already fields `enemy_army_size`, `agent.py:55`, suggesting it fights this race).

---

## 2. Explicit 3–4 player → 2-player no-trade adaptation summary

| Book doctrine (3–4p, trading) | 2p no-trade duel version | Used in |
|---|---|---|
| Robber: "put it on the good player" / block the likely winner among several (book2 2489–2530) | Only one opponent — target selection collapses to hex choice; no king-making calculus | Term 3 |
| Piss-off factor / retaliation pricing (2243–2244) | Void — no diplomacy to protect; always block maximally | Term 3 |
| Robbed of a resource → trade with other players (522–533) | Cannot trade around a block; only 4:1 bank or ports → blocking sole sources is stronger; own scarcity hurts more | Terms 1, 3, 4 |
| Denial caveat "you still have to beat the other two players" (1981–1983) | No other two players; board is zero-sum → denial term added, sized below self-growth | Term 2 |
| Scarcity profits via trading your scarce surplus (791–810) | Scarcity = self-sufficiency + denial only; ports are the only mitigation (852–860) | Terms 1, 4 |
| Hand > 7 risk: 51.8% a 7 lands before your next turn in 4p (529–531; verified 1−(5/6)⁴ = 0.5177) | One opponent roll between your turns: 1/6 ≈ 16.7% — **keep the discard penalty small** (current flat −5 at `agent.py:52` is correctly sized; no change recommended) | note |
| Longest Road: focus production, steal LR at the end (328–337, Quick Tip #7); initial roads are for expansion, not LR (1994–1995, Quick Tip #46) | Unchanged and already structurally true in the eval (LR weight is a tie-breaker, `agent.py:48,120`); no new term | note |
| 2nd settlement on Lumber+Brick to win races / free cards (1867–1926, Quick Tips #41–42) | Even stronger with one racer; covered by term 2's denial pressure rather than a separate term | Term 2 |

## 3. Not found / not done

- **Book1 contributed nothing** beyond confirming build costs; it is a puzzle book.
- **Book2's extraction is truncated** (ends line 2752): no middle-game, end-game, trading, or appendix content exists in the file, so no terms are derived from them. In particular the book's dedicated robber-placement chapter (implied by its title and TOC) is absent; robber doctrine above is reconstructed from the "Impact of Robber" and "most likely to win" sections only.
- Neither book discusses 2-player or no-trading variants at all; every 2p adaptation above is mine, flagged as such.
- No agent code was written or modified; no games were run (the ask names no check to run — all numbers above are arithmetic verified in-session: pip conversions, 1−(5/6)⁴, 1/6, and each term's pips/VP-ratio).
