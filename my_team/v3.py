"""v3: fast direct-read evaluation + wider budget-aware search.

The v2 evaluation called catanatron's feature functions per search node
(reachability BFS, production walks, buildable-node scans). v3 reads the
player_state and board directly with per-map caches:

  - fast_eval: production/variety/robber adjustment, hand synergy, ports,
    longest-road length (engine-maintained), knights, VPs. No reachability,
    no buildable count. ~5-10x cheaper per node.
  - full_eval (v2's): used ONLY for the 4 initial placement decisions, where
    reachability matters and the 300 ms base pays for it.
  - MOVE_ROBBER candidates are evaluated by mutating robber_coordinate and
    restoring it: no game copy needed (~50x faster than copy+execute).

Search: 1-ply everything with fast_eval; depth-2 expand top-8 with follow-ups;
joint settlement+road placement on top-16 candidates. Clock-gated throughout.
"""
import random
import time

from catanatron.models.enums import ActionType, RESOURCES, SETTLEMENT, CITY
from catanatron.models.map import number_probability
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
_DEV_TYPES = ("KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY")

# --- per-map caches (robber-adjusted production needs them per node) ---

_MAP_CACHES = {}   # id(map) -> (map_ref, cache); map_ref pins the id


def _map_cache(catan_map):
    hit = _MAP_CACHES.get(id(catan_map))
    if hit is not None:
        return hit[1]
    node_prod = {}      # node -> [5 production probabilities]
    node_tile_ids = {}  # node -> set of tile ids
    for node, tiles in catan_map.adjacent_tiles.items():
        probs = [0.0] * 5
        ids = set()
        for t in tiles:
            ids.add(t.id)
            if getattr(t, "number", None) is not None and t.resource is not None:
                probs[RESOURCES.index(t.resource)] += number_probability(t.number)
        node_prod[node] = probs
        node_tile_ids[node] = ids
    tile_info = {}      # tile id -> (resource index, probability); land tiles only
    for t in catan_map.tiles.values():
        if getattr(t, "number", None) is not None and t.resource is not None:
            tile_info[t.id] = (RESOURCES.index(t.resource), number_probability(t.number))
    port_nodes = {res: set(nodes) for res, nodes in catan_map.port_nodes.items()}
    cache = dict(node_prod=node_prod, node_tile_ids=node_tile_ids,
                 tile_info=tile_info, port_nodes=port_nodes)
    if len(_MAP_CACHES) > 4:   # keep memory flat across many games
        _MAP_CACHES.clear()
    _MAP_CACHES[id(catan_map)] = (catan_map, cache)
    return cache


def _production_probs(st, cache, color, robber_tid):
    probs = [0.0] * 5
    buildings = st.buildings_by_color[color]
    rinfo = cache["tile_info"].get(robber_tid)   # None when the robber is on desert
    for node in buildings[SETTLEMENT]:
        np_ = cache["node_prod"][node]
        for k in range(5):
            probs[k] += np_[k]
        if rinfo is not None and robber_tid in cache["node_tile_ids"][node]:
            ridx, prob = rinfo
            probs[ridx] -= prob
    for node in buildings[CITY]:
        np_ = cache["node_prod"][node]
        for k in range(5):
            probs[k] += 2 * np_[k]
        if rinfo is not None and robber_tid in cache["node_tile_ids"][node]:
            ridx, prob = rinfo
            probs[ridx] -= 2 * prob
    return probs


def fast_eval(st, me, w=WEIGHTS):
    cache = _map_cache(st.board.map)
    enemy = st.colors[1 - st.color_to_index[me]]
    i = st.color_to_index[me]
    ps = st.player_state
    key, ekey = f"P{i}", f"P{1 - i}"
    robber_tid = st.board.map.tiles[st.board.robber_coordinate].id

    probs = _production_probs(st, cache, me, robber_tid)
    proba_point = 2.778 / 100
    production = sum(probs) + sum(p > 1e-9 for p in probs) * TRANSLATE_VARIETY * proba_point
    eprobs = _production_probs(st, cache, enemy, robber_tid)
    enemy_production = sum(eprobs)

    wheat, ore = ps[f"{key}_WHEAT_IN_HAND"], ps[f"{key}_ORE_IN_HAND"]
    sheep, brick, wood = ps[f"{key}_SHEEP_IN_HAND"], ps[f"{key}_BRICK_IN_HAND"], ps[f"{key}_WOOD_IN_HAND"]
    distance_to_city = (max(2 - wheat, 0) + max(3 - ore, 0)) / 5.0
    distance_to_settlement = (max(1 - wheat, 0) + max(1 - sheep, 0)
                              + max(1 - brick, 0) + max(1 - wood, 0)) / 4.0
    hand_synergy = (2 - distance_to_city - distance_to_settlement) / 2

    num_in_hand = wheat + ore + sheep + brick + wood
    discard_penalty = w["discard_penalty"] if num_in_hand > 7 else 0

    buildings = st.buildings_by_color[me]
    owned_nodes = buildings[SETTLEMENT] + buildings[CITY]
    owned_tile_ids = set()
    for n in owned_nodes:
        owned_tile_ids.update(cache["node_tile_ids"][n])

    port_bonus = 0.0
    owned_set = set(owned_nodes)
    for resource, nodes in cache["port_nodes"].items():
        if not (owned_set & nodes):
            continue
        if resource is None:
            port_bonus += w["port_generic"]
        else:
            ridx = RESOURCES.index(resource)
            port_bonus += w["port_specialty"] * (1.0 if probs[ridx] > 0.11 else 0.4)

    num_devs = sum(ps[f"{key}_{d}_IN_HAND"] for d in _DEV_TYPES)

    return (
        ps[f"{key}_VICTORY_POINTS"] * w["public_vps"]
        + ps[f"{ekey}_VICTORY_POINTS"] * w["enemy_public_vps"]
        + production * w["production"]
        + enemy_production * w["enemy_production"]
        + hand_synergy * w["hand_synergy"]
        + len(owned_tile_ids) * w["num_tiles"]
        + num_in_hand * w["hand_resources"]
        + discard_penalty
        + ps[f"{key}_LONGEST_ROAD_LENGTH"] * 0.1
        + num_devs * w["hand_devs"]
        + ps[f"{key}_PLAYED_KNIGHT"] * w["army_size"]
        + ps[f"{ekey}_PLAYED_KNIGHT"] * w["enemy_army_size"]
        + ps[f"{key}_KNIGHT_IN_HAND"] * w["knight_in_hand"]
        + port_bonus
    )


# --- full evaluation (v2's), used only for initial placement ---

def value_production(sample, player_name, include_variety=True):
    proba_point = 2.778 / 100
    features = [f"EFFECTIVE_{player_name}_{r}_PRODUCTION" for r in RESOURCES]
    prod_sum = sum(sample[f] for f in features)
    prod_variety = sum(sample[f] != 0 for f in features) * TRANSLATE_VARIETY * proba_point
    return prod_sum + (prod_variety if include_variety else 0)


def full_eval(game, p0_color):
    w = WEIGHTS
    st = game.state
    key = player_key(st, p0_color)
    enemy = st.colors[1 - st.color_to_index[p0_color]]
    enemy_key = player_key(st, enemy)

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
        + get_played_dev_cards(st, enemy, "KNIGHT") * w["enemy_army_size"]
        + st.player_state[f"{key}_KNIGHT_IN_HAND"] * w["knight_in_hand"]
        + port_bonus
    )


def _successor(world, action):
    succ = world.copy()
    try:
        succ.execute(action)
    except Exception:
        return None
    return succ


def _eval_succ(succ, me):
    return fast_eval(succ.state, me)


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        world = obs.sample_world(self.rng)
        types = {a.action_type for a in playable_actions}
        budget = obs.cpu_left

        if world.state.is_initial_build_phase and ActionType.BUILD_SETTLEMENT in types:
            return self._initial_settlement(world, obs.me, playable_actions, budget)
        if ActionType.MOVE_ROBBER in types and len(types) <= 2:
            return self._robber(world, obs.me, playable_actions)
        return self._turn(world, obs.me, playable_actions, budget)

    def _initial_settlement(self, world, me, actions, budget):
        scored = []
        for i, action in enumerate(actions):
            succ = _successor(world, action)
            if succ is None:
                continue
            scored.append((full_eval(succ, me), i, succ))
        if not scored:
            return 0
        scored.sort(key=lambda t: (-t[0], t[1]))
        if budget is None or budget > 0.15:
            deadline = time.perf_counter() + max(0.5 * (budget or 1.0), 0.05)
            recheck = []
            for v1, i, succ in scored[:16]:
                best = v1
                for action in succ.playable_actions:
                    if action.action_type != ActionType.BUILD_ROAD:
                        continue
                    gsucc = _successor(succ, action)
                    if gsucc is None:
                        continue
                    v = full_eval(gsucc, me)
                    if v > best:
                        best = v
                    if time.perf_counter() > deadline:
                        break
                recheck.append((best, i))
            recheck.sort(key=lambda t: (-t[0], t[1]))
            return recheck[0][1]
        return scored[0][1]

    def _robber(self, world, me, actions):
        """Evaluate robber destinations by moving the robber on the shared world
        and restoring it: no copy, no execute. Blocking a hex hurts whoever
        builds on it, us included."""
        st = world.state
        cache = _map_cache(st.board.map)
        enemy = st.colors[1 - st.color_to_index[me]]
        original = st.board.robber_coordinate
        ps = st.player_state
        key = f"P{st.color_to_index[me]}"
        ekey = f"P{st.color_to_index[enemy]}"
        best_i, best_v = 0, None
        for i, action in enumerate(actions):
            value = action.value
            coord = value[0] if isinstance(value, tuple) else value
            st.board.robber_coordinate = coord
            robber_tid = st.board.map.tiles[coord].id
            own = _production_probs(st, cache, me, robber_tid)
            eprobs = _production_probs(st, cache, enemy, robber_tid)
            v = (ps[f"{key}_VICTORY_POINTS"] * WEIGHTS["public_vps"]
                 + ps[f"{ekey}_VICTORY_POINTS"] * WEIGHTS["enemy_public_vps"]
                 + sum(own) * 1e8 - sum(eprobs) * 1e8)
            # prefer victims with fatter hands (the steal)
            victim = value[1] if isinstance(value, tuple) and len(value) > 1 else None
            if victim is not None and victim != me:
                v += 2.0 * sum(ps[f"{ekey}_{r}_IN_HAND"] for r in RESOURCES)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        st.board.robber_coordinate = original
        return best_i

    def _turn(self, world, me, actions, budget):
        values = {}
        succs = {}
        for i, action in enumerate(actions):
            succ = _successor(world, action)
            if succ is None:
                continue
            values[i] = _eval_succ(succ, me)
            succs[i] = succ
        if not values:
            return 0

        if budget is None or budget > 0.35:
            spend = 0.025
        elif budget > 0.12:
            spend = 0.010
        else:
            return min(values, key=lambda i: (-values[i], i))
        deadline = time.perf_counter() + spend
        expanded = 0
        for i in sorted(values, key=lambda i: (-values[i], i)):
            if actions[i].action_type not in _EXPANDABLE:
                continue
            best = values[i]
            for action in succs[i].playable_actions:
                if action.action_type not in _FOLLOWUP:
                    continue
                gsucc = _successor(succs[i], action)
                if gsucc is None:
                    continue
                v = _eval_succ(gsucc, me)
                if v > best:
                    best = v
                if time.perf_counter() > deadline:
                    break
            values[i] = best
            expanded += 1
            if expanded >= 8 or time.perf_counter() > deadline:
                break
        return min(values, key=lambda i: (-values[i], i))
