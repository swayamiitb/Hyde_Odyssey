"""v5: v4 + win-now sequence search + trade-enables-build follow-ups.

New decision layers, in order:
  1. WIN-NOW SEARCH: when our actual VPs >= 7, DFS over this turn's action
     sequences (builds, dev plays, and maritime trades that unafford a build)
     looking for any line that reaches 10 VP. Re-run every decide(), so the
     returned first action self-repairs as the turn unfolds. Forced moves
     inside the line (MOVE_ROBBER, DISCARD) are played with their first
     option: they cannot gain VP themselves.
  2. Same-turn depth-2 as v4, but follow-ups now include maritime trades
     that enable a currently-unaffordable build (port-assisted builds).

Everything else (joint placement, full-fidelity eval, clock gates) as v4.
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

COSTS = [  # (wood, brick, sheep, wheat, ore)
    (1, 1, 1, 1, 0),   # settlement
    (0, 0, 0, 2, 3),   # city
    (1, 1, 0, 0, 0),   # road
    (0, 0, 1, 1, 1),   # dev card
]


def _hand(st, key):
    ps = st.player_state
    return tuple(ps[f"{key}_{r}_IN_HAND"] for r in RESOURCES)


def _affords(hand):
    return any(all(hand[k] >= c[k] for k in range(5)) for c in COSTS)


def _trade_enables(st, key, action):
    """True when this maritime trade makes some build newly affordable."""
    value = action.value
    get = value[-1]
    gives = [v for v in value[:-1] if v is not None]
    hand = list(_hand(st, key))
    for g in gives:
        hand[RESOURCES.index(g)] -= 1
    hand[RESOURCES.index(get)] += 1
    if _affords(tuple(hand)):
        return True
    return not _affords(tuple(_hand(st, key))) and False


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    # ---------- decision entry ----------

    def decide(self, obs, playable_actions) -> int:
        world = obs.sample_world(self.rng)
        types = {a.action_type for a in playable_actions}
        budget = obs.cpu_left

        if world.state.is_initial_build_phase and ActionType.BUILD_SETTLEMENT in types:
            return self._initial_settlement(world, obs.me, playable_actions, budget)

        i = self._win_now(world, obs.me, playable_actions)
        if i is not None:
            return i
        return self._turn(world, obs.me, playable_actions, budget)

    # ---------- 1. win-now sequence search ----------

    def _win_now(self, world, me, actions):
        st = world.state
        key = player_key(st, me)
        if st.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] < 7:
            return None
        deadline = time.perf_counter() + 0.040

        def can_win(game, depth):
            if game.winning_color() is not None:
                return game.winning_color() == me
            if depth == 0 or time.perf_counter() > deadline:
                return False
            tried_forced = False
            for act in game.playable_actions:
                t = act.action_type
                if t == ActionType.END_TURN:
                    continue
                if t in (ActionType.MOVE_ROBBER, ActionType.DISCARD_RESOURCE):
                    if tried_forced:   # forced moves: first option is enough
                        continue
                    tried_forced = True
                elif t == ActionType.MARITIME_TRADE:
                    if not _trade_enables(game.state, key, act):
                        continue
                g2 = game.copy()
                try:
                    g2.execute(act)
                except Exception:
                    continue
                if can_win(g2, depth - 1):
                    return True
            return False

        for i, act in enumerate(actions):
            if act.action_type in (ActionType.END_TURN, ActionType.MOVE_ROBBER,
                                   ActionType.DISCARD_RESOURCE):
                continue
            if act.action_type == ActionType.MARITIME_TRADE \
                    and not _trade_enables(st, key, act):
                continue
            g2 = world.copy()
            try:
                g2.execute(act)
            except Exception:
                continue
            if g2.winning_color() == me or can_win(g2, 3):
                return i
        return None

    # ---------- 2. joint placement (as v2/v4) ----------

    def _initial_settlement(self, world, me, actions, budget):
        scored = []
        for i, action in enumerate(actions):
            succ = world.copy()
            try:
                succ.execute(action)
            except Exception:
                continue
            scored.append((evaluate(succ, me), i, succ))
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
                    gsucc = succ.copy()
                    try:
                        gsucc.execute(action)
                    except Exception:
                        continue
                    v = evaluate(gsucc, me)
                    if v > best:
                        best = v
                    if time.perf_counter() > deadline:
                        break
                recheck.append((best, i))
            recheck.sort(key=lambda t: (-t[0], t[1]))
            return recheck[0][1]
        return scored[0][1]

    # ---------- 3. same-turn depth-2 with trade follow-ups ----------

    def _turn(self, world, me, actions, budget):
        values = {}
        succs = {}
        for i, action in enumerate(actions):
            succ = world.copy()
            try:
                succ.execute(action)
            except Exception:
                continue
            values[i] = evaluate(succ, me)
            succs[i] = succ
        if not values:
            return 0

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
            best = values[i]
            key = player_key(succs[i].state, me)
            for action in succs[i].playable_actions:
                t = action.action_type
                if t not in _FOLLOWUP:
                    if t == ActionType.MARITIME_TRADE and \
                            _trade_enables(succs[i].state, key, action):
                        pass        # trade that enables a build: explore it
                    else:
                        continue
                gsucc = succs[i].copy()
                try:
                    gsucc.execute(action)
                except Exception:
                    continue
                v = evaluate(gsucc, me)
                if v > best:
                    best = v
                if time.perf_counter() > deadline:
                    break
            values[i] = best
            expanded += 1
            if expanded >= 8 or time.perf_counter() > deadline:
                break
        return min(values, key=lambda i: (-values[i], i))


# ---------- evaluation (identical to v2/v4) ----------

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
