"""v11: v4 plus the four highest-impact terms from my_team/STRATEGY_TERMS.md.

STRATEGY_TERMS.md section 1 ranks its six adjustments by expected impact; the
top four are implemented here (terms 5 and 6 of the doc — complementary-ratio
balance and the dev-card gate — are deliberately NOT included, per the
"at most 4 terms" limit). Search structure, expansion gates and every other
weight are v4 verbatim.

Terms added (all additive, all conservative scale — each is <= ~4.8e6, i.e.
<= ~1.7 pips of production-equivalent, seven orders below the 3e14 VP term):

  1. Port value for the bank-only economy (doc term 1). The only conversion
     mechanisms are the 4:1 bank and ports, so ports are priced at the top of
     the book's range instead of invisible (old 2.5e3 = 0.001 pips):
       3:1 generic port                + 2.5e6          (~0.9 pips)
       2:1 specialty port on resource  + 4e6 * clamp(prod_r / 0.139, 0, 1.2)
     (0.139 = 5 pips, the book's threshold; max 4.8e6 ~ 1.7 pips). The old
     port_specialty / port_generic weights are removed, so no double count.

  2. Denial: enemy reachability + enemy buildable (doc term 2). In a 2p
     zero-sum duel every spot/pip the enemy secures is one we cannot; the
     eval gains the mirror of the self expansion terms, sized ~40% below
     them so the bot still prefers building itself over pure spite:
       er0 * -3e3 + er1 * -4e3 + enemy_buildable * -1e3

  3. Robber-targeting rule (doc term 3), applied only while comparing
     MOVE_ROBBER successors. The -1e8 enemy_production channel still does the
     heavy lifting; among near-equal blocks, prefer the scarce resource and
     prefer a steal when the enemy hand is fat:
       tilt(h) = 3e5 * (1 + 0.5*scarce(res_h) - 1) + 1e5 * min(enemy_hand,4)/4
     Max 2.5e5 (< the doc's 4e5 = 0.14 pips ceiling): a pure tie-breaker.

  4. Scarcity-weighted production, own minus enemy (doc term 4):
       1.5e7 * (scarce_pips(me) - scarce_pips(enemy))
     with scarce(r) = clamp((median board pips - pips(r)) / median, 0, 1),
     the same scarcity weights term 3 reads. Max swing ~3.3e6 (~1.2 pips).

Safety: decide() is wrapped so the decision path can never raise — on any
unexpected error it falls back to index 0.
"""
import random
import time

from catanatron.models.enums import ActionType, RESOURCES, SETTLEMENT, CITY
from catanatron.state_functions import (
    get_longest_road_length,
    get_played_dev_cards,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)
from catanatron.features import (
    build_production_features,
    reachability_features,
    resource_hand_features,
)

TRANSLATE_VARIETY = 4

WEIGHTS = {
    "public_vps": 3e14,
    "enemy_public_vps": -3e14,
    "production": 1e8,
    "enemy_production": -1e8,
    "num_tiles": 1,
    "reachable_production_0": 8e3,
    "reachable_production_1": 1e4,
    "buildable_nodes": 1e3,
    "longest_road": 10.0,
    "knight_in_hand": 5.0,
    "hand_synergy": 1e2,
    "hand_resources": 1,
    "discard_penalty": -5,
    "hand_devs": 10,
    "army_size": 10.1,
    "enemy_army_size": -12.0,
    # --- v11 term 2: denial (enemy expansion, 2p zero-sum) ---
    "enemy_reachable_production_0": -3e3,
    "enemy_reachable_production_1": -4e3,
    "enemy_buildable_nodes": -1e3,
    # --- v11 term 4: scarcity-weighted production (own minus enemy) ---
    "scarcity_production": 1.5e7,
}

# --- v11 term 1: port values for the bank-only economy (absolute eval units) ---
PORT_GENERIC_3TO1 = 2.5e3 * 1e3        # 2.5e6: ~0.9 pips
PORT_SPECIALTY_2TO1 = 4e6              # scaled by clamp(prod_r / 0.139, 0, 1.2)
PORT_PROD_FULL = 0.139                 # 5 pips of that resource's production

# --- v11 terms 3/4: board scarcity ---
PIPS_BY_NUMBER = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
ROBBER_SCARCITY_TILT = 3e5             # times (rw - 1) = 0.5 * scarce(r)
ROBBER_STEAL_TILT = 1e5                # times min(enemy_hand, 4) / 4

_PRODUCTION = build_production_features(True)

_EXPANDABLE = {
    ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY, ActionType.BUILD_ROAD,
    ActionType.BUY_DEVELOPMENT_CARD, ActionType.PLAY_ROAD_BUILDING,
    ActionType.PLAY_YEAR_OF_PLENTY, ActionType.PLAY_MONOPOLY,
}
_FOLLOWUP = {
    ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY, ActionType.BUILD_ROAD,
    ActionType.BUY_DEVELOPMENT_CARD,
}


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def board_scarce(st):
    """scarce(r) in [0, 1] per resource: how far below the median the board's
    pip count for r sits (STRATEGY_TERMS.md terms 3 and 4). Desert / None tiles
    and numbers without pips contribute nothing. Pure-python median of 5."""
    try:
        pips = {r: 0.0 for r in RESOURCES}
        for tile in st.board.map.land_tiles.values():
            res = tile.resource
            if res in pips:
                pips[res] += PIPS_BY_NUMBER.get(tile.number, 0)
        vals = sorted(pips.values())
        median = vals[len(vals) // 2]
        if median <= 0:
            return {r: 0.0 for r in RESOURCES}
        return {r: _clamp((median - pips[r]) / median, 0.0, 1.0) for r in RESOURCES}
    except Exception:
        return {r: 0.0 for r in RESOURCES}


def robber_tilt(st, me, action):
    """v11 term 3: small additive tilt for one MOVE_ROBBER candidate.

    Scarcity-weighted hex quality (block the enemy's scarce resources, not
    their redundant ones) plus steal sizing (prefer hexes that also rob a fat
    enemy hand). Max 2.5e5 = 0.09 pips: never overrides a bigger block, which
    the -1e8 enemy_production channel still decides. Never raises.
    """
    try:
        tilt = 0.0
        payload = action.value
        coord = payload[0]
        robbed = payload[1] if len(payload) > 1 else None
        tile = st.board.map.land_tiles.get(coord)
        if tile is not None and tile.resource in RESOURCES:
            scarce = board_scarce(st)
            tilt += ROBBER_SCARCITY_TILT * 0.5 * scarce[tile.resource]
        enemy = enemy_key_color(st, me)
        if robbed == enemy:
            hand = player_num_resource_cards(st, enemy)
            tilt += ROBBER_STEAL_TILT * min(hand, 4) / 4.0
        return tilt
    except Exception:
        return 0.0


def value_production(sample, player_name, include_variety=True):
    proba_point = 2.778 / 100
    features = [f"EFFECTIVE_{player_name}_{r}_PRODUCTION" for r in RESOURCES]
    prod_sum = sum(sample[f] for f in features)
    prod_variety = sum(sample[f] != 0 for f in features) * TRANSLATE_VARIETY * proba_point
    return prod_sum + (prod_variety if include_variety else 0)


def evaluate(game, p0_color):
    w = WEIGHTS
    st = game.state
    key = player_key(st, p0_color)
    enemy = enemy_key_color(st, p0_color)
    enemy_key = player_key(st, enemy)

    production_sample = _PRODUCTION(game, p0_color)
    production = value_production(production_sample, "P0")
    enemy_production = value_production(production_sample, "P1", False)

    reachability_sample = reachability_features(game, p0_color, 2)
    r0 = sum(reachability_sample[f"P0_0_ROAD_REACHABLE_{r}"] for r in RESOURCES)
    r1 = sum(reachability_sample[f"P0_1_ROAD_REACHABLE_{r}"] for r in RESOURCES)

    # v11 term 2: enemy expansion awareness (denial). Same features, enemy color;
    # reachability_features names the passed color "P0".
    enemy_reach_sample = reachability_features(game, enemy, 2)
    er0 = sum(enemy_reach_sample[f"P0_0_ROAD_REACHABLE_{r}"] for r in RESOURCES)
    er1 = sum(enemy_reach_sample[f"P0_1_ROAD_REACHABLE_{r}"] for r in RESOURCES)

    # v11 term 4: scarcity-weighted production, own minus enemy.
    scarce = board_scarce(st)
    my_scarce_pips = sum(
        production_sample[f"EFFECTIVE_P0_{r}_PRODUCTION"] * scarce[r] for r in RESOURCES
    )
    enemy_scarce_pips = sum(
        production_sample[f"EFFECTIVE_P1_{r}_PRODUCTION"] * scarce[r] for r in RESOURCES
    )

    hand_sample = resource_hand_features(game, p0_color)
    distance_to_city = (
        max(2 - hand_sample["P0_WHEAT_IN_HAND"], 0)
        + max(3 - hand_sample["P0_ORE_IN_HAND"], 0)
    ) / 5.0
    distance_to_settlement = (
        max(1 - hand_sample["P0_WHEAT_IN_HAND"], 0)
        + max(1 - hand_sample["P0_SHEEP_IN_HAND"], 0)
        + max(1 - hand_sample["P0_BRICK_IN_HAND"], 0)
        + max(1 - hand_sample["P0_WOOD_IN_HAND"], 0)
    ) / 4.0
    hand_synergy = (2 - distance_to_city - distance_to_settlement) / 2

    num_in_hand = player_num_resource_cards(st, p0_color)
    discard_penalty = w["discard_penalty"] if num_in_hand > 7 else 0

    buildings = st.buildings_by_color[p0_color]
    owned_nodes = set(buildings[SETTLEMENT] + buildings[CITY])

    owned_tiles = set()
    for n in owned_nodes:
        owned_tiles.update(st.board.map.adjacent_tiles[n])
    num_tiles = len(owned_tiles)

    num_buildable = len(st.board.buildable_node_ids(p0_color))
    enemy_buildable = len(st.board.buildable_node_ids(enemy))
    longest_road_factor = w["longest_road"] if num_buildable == 0 else 0.1

    # v11 term 1: port value for the bank-only economy. The generic / specialty
    # WEIGHTS entries are gone, so these absolute numbers replace them outright.
    port_bonus = 0.0
    for resource, nodes in st.board.map.port_nodes.items():
        if not (owned_nodes & nodes):
            continue
        if resource is None:
            port_bonus += PORT_GENERIC_3TO1
        else:
            prod = production_sample[f"EFFECTIVE_P0_{resource}_PRODUCTION"]
            port_bonus += PORT_SPECIALTY_2TO1 * _clamp(prod / PORT_PROD_FULL, 0.0, 1.2)

    return float(
        st.player_state[f"{key}_VICTORY_POINTS"] * w["public_vps"]
        + st.player_state[f"{enemy_key}_VICTORY_POINTS"] * w["enemy_public_vps"]
        + production * w["production"]
        + enemy_production * w["enemy_production"]
        + r0 * w["reachable_production_0"]
        + r1 * w["reachable_production_1"]
        + er0 * w["enemy_reachable_production_0"]
        + er1 * w["enemy_reachable_production_1"]
        + enemy_buildable * w["enemy_buildable_nodes"]
        + w["scarcity_production"] * (my_scarce_pips - enemy_scarce_pips)
        + hand_synergy * w["hand_synergy"]
        + num_buildable * w["buildable_nodes"]
        + num_tiles * w["num_tiles"]
        + num_in_hand * w["hand_resources"]
        + discard_penalty
        + get_longest_road_length(st, p0_color) * longest_road_factor
        + player_num_dev_cards(st, p0_color) * w["hand_devs"]
        + get_played_dev_cards(st, p0_color, "KNIGHT") * w["army_size"]
        + get_played_dev_cards(st, enemy, "KNIGHT") * w["enemy_army_size"]
        + st.player_state[f"{key}_KNIGHT_IN_HAND"] * w["knight_in_hand"]
        + port_bonus
    )


def enemy_key_color(st, p0_color):
    return st.colors[1 - st.color_to_index[p0_color]]


def _successor(world, action):
    succ = world.copy()
    try:
        succ.execute(action)
    except Exception:
        return None
    return succ


def _best_followup(succ, me, deadline, value_now):
    """Max eval over affordable follow-up actions (bounded by wall deadline)."""
    best = value_now
    for action in succ.playable_actions:
        if action.action_type not in _FOLLOWUP:
            continue
        gsucc = _successor(succ, action)
        if gsucc is None:
            continue
        v = evaluate(gsucc, me)
        if v > best:
            best = v
        if time.perf_counter() > deadline:
            break
    return best


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        try:
            world = obs.sample_world(self.rng)
            types = {a.action_type for a in playable_actions}
            budget = obs.cpu_left

            if world.state.is_initial_build_phase and ActionType.BUILD_SETTLEMENT in types:
                return self._initial_settlement(world, obs.me, playable_actions, budget)
            return self._turn(world, obs.me, playable_actions, budget)
        except Exception:
            # The decision path must never raise: fall back to a legal index.
            return 0

    def _initial_settlement(self, world, me, actions, budget):
        scored = []
        for i, action in enumerate(actions):
            succ = _successor(world, action)
            if succ is None:
                continue
            scored.append((evaluate(succ, me), i, succ))
        if not scored:
            return 0
        scored.sort(key=lambda t: (-t[0], t[1]))
        # Joint with best follow-up road for the top candidates.
        if budget is None or budget > 0.15:
            top = scored[:12]
            plan_limit = (time.perf_counter() + max(0.5 * (budget or 1.0), 0.05))
            recheck = []
            for v1, i, succ in top:
                v = _best_followup_road(succ, me, plan_limit, v1)
                recheck.append((v, i))
            recheck.sort(key=lambda t: (-t[0], t[1]))
            return recheck[0][1]
        best = scored[0]
        return best[1]

    def _turn(self, world, me, actions, budget):
        # 1-ply every action (F behaviour), remember successors.
        values = {}
        succs = {}
        for i, action in enumerate(actions):
            succ = _successor(world, action)
            if succ is None:
                continue
            values[i] = evaluate(succ, me)
            # v11 term 3: robber-targeting tilt, only on MOVE_ROBBER comparisons.
            if action.action_type == ActionType.MOVE_ROBBER:
                values[i] += robber_tilt(world.state, me, action)
            succs[i] = succ
        if not values:
            return 0

        # Same-turn depth-2 on the top few expandable actions, clock permitting.
        if budget is None or budget > 0.35:
            spend = 0.018
        elif budget > 0.12:
            spend = 0.008
        else:
            return min(values, key=lambda i: (-values[i], i))
        deadline = time.perf_counter() + spend
        expanded = 0
        for i in sorted(values, key=lambda i: (-values[i], i)):
            if actions[i].action_type not in _EXPANDABLE:
                continue
            values[i] = _best_followup(succs[i], me, deadline, values[i])
            expanded += 1
            if expanded >= 8 or time.perf_counter() > deadline:
                break
        return min(values, key=lambda i: (-values[i], i))


def _best_followup_road(succ, me, deadline, value_now):
    """Best eval after the best road following a fresh initial settlement."""
    best = value_now
    for action in succ.playable_actions:
        if action.action_type != ActionType.BUILD_ROAD:
            continue
        gsucc = _successor(succ, action)
        if gsucc is None:
            continue
        v = evaluate(gsucc, me)
        if v > best:
            best = v
        if time.perf_counter() > deadline:
            break
    return best
