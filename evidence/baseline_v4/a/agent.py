"""v4: v2 with wider same-turn expansion (top-8) and tighter spend gates.

v3 (fast eval without reachability terms) measured 0.2300 vs F: dropping
reachable-production signals breaks expansion decisions. Node cost is
copy-dominated anyway, so full-fidelity eval everywhere is correct.

Changes vs v1a:
  A. enemy_public_vps at -3e14: F has NO enemy VP term — it cannot tell it is
     losing and never prices stealing Longest Road / Largest Army (a 4-VP
     swing). Constant across most successors, so it only changes the ranking
     of LR/LA-flipping moves — exactly the intent.
  B. enemy knights played at -12: race pressure toward Largest Army parity.
  C. Joint placement search: 1-ply-score all settlement candidates, then
     expand the top ones with their best follow-up road (F picks settlement
     and road greedily, one at a time).
  D. Same-turn depth-2 on build/dev decisions: 1-ply every action, then
     expand the top-K actions with their affordable follow-ups and back up
     the max. Gated on obs.cpu_left so the clock can never run out.
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
    "port_specialty": 2.5e3,
    "port_generic": 1.2e3,
}

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
    enemy_key = player_key(st, st.colors[1 - st.color_to_index[p0_color]])

    production_sample = _PRODUCTION(game, p0_color)
    production = value_production(production_sample, "P0")
    enemy_production = value_production(production_sample, "P1", False)

    reachability_sample = reachability_features(game, p0_color, 2)
    r0 = sum(reachability_sample[f"P0_0_ROAD_REACHABLE_{r}"] for r in RESOURCES)
    r1 = sum(reachability_sample[f"P0_1_ROAD_REACHABLE_{r}"] for r in RESOURCES)

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
    longest_road_factor = w["longest_road"] if num_buildable == 0 else 0.1

    port_bonus = 0.0
    for resource, nodes in st.board.map.port_nodes.items():
        if not (owned_nodes & nodes):
            continue
        if resource is None:
            port_bonus += w["port_generic"]
        else:
            prod = production_sample[f"EFFECTIVE_P0_{resource}_PRODUCTION"]
            port_bonus += w["port_specialty"] * (1.0 if prod > 0.11 else 0.4)

    return float(
        st.player_state[f"{key}_VICTORY_POINTS"] * w["public_vps"]
        + st.player_state[f"{enemy_key}_VICTORY_POINTS"] * w["enemy_public_vps"]
        + production * w["production"]
        + enemy_production * w["enemy_production"]
        + r0 * w["reachable_production_0"]
        + r1 * w["reachable_production_1"]
        + hand_synergy * w["hand_synergy"]
        + num_buildable * w["buildable_nodes"]
        + num_tiles * w["num_tiles"]
        + num_in_hand * w["hand_resources"]
        + discard_penalty
        + get_longest_road_length(st, p0_color) * longest_road_factor
        + player_num_dev_cards(st, p0_color) * w["hand_devs"]
        + get_played_dev_cards(st, p0_color, "KNIGHT") * w["army_size"]
        + get_played_dev_cards(st, enemy_key_color(st, p0_color), "KNIGHT") * w["enemy_army_size"]
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
        world = obs.sample_world(self.rng)
        types = {a.action_type for a in playable_actions}
        budget = obs.cpu_left

        if world.state.is_initial_build_phase and ActionType.BUILD_SETTLEMENT in types:
            return self._initial_settlement(world, obs.me, playable_actions, budget)
        return self._turn(world, obs.me, playable_actions, budget)

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
