"""orion: v4 UNCHANGED below, plus an exact-opponent cross-turn layer.

The one asymmetry never exploited: F's source is given. F is a deterministic
argmax over catanatron's base_fn on 1-ply successors. So for each of our top
candidate actions we can affordably play out what F would actually DO about
it — not what a self-greedy proxy guesses (every prior cross-turn attempt:
v4xct teacher, v14 DART, v16 re-rank modeled the opponent with our own eval,
which misevaluates F by construction — F cannot see enemy VPs, never prices
LR/LA denial, and greedily races production).

Layer 1 (mid-game): after v4's 1-ply + same-turn depth-2, take the top-K
  candidates and score each by "our action -> END_TURN -> sampled roll ->
  F's EXACT replies -> our eval", blended with the 1-ply value.
Layer 2 (first settlement only): for the top joint settlement+road plans,
  simulate F's next TWO placements exactly and keep the plan that is best
  after F's greedy grab — real denial, not the -1e8 production proxy alone.

Cost (measured, bench_orion.py): copy 34us, copy+execute 78us, F eval 71us,
full F reply chain ~3ms, F placement argmax ~7ms. Budget: cpu-gated, hard
wall deadlines inside every loop; any exception falls back to v4's decision;
top-level guard returns a legal index so faults stay at zero.
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
from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn

TRANSLATE_VARIETY = 4

# ---- orion layer knobs (screened) ----
XCT_W = 1.0          # blend: (1-W)*1ply + W*cross-turn
XCT_TOP = 4           # candidates re-ranked cross-turn (mid-game)
XCT_CHAINS = 2        # reply chains per candidate (roll samples)
XCT_PLACE_TOP = 5     # settlement plans re-ranked with F's reply
XCT_MIN_BUDGET = 0.45 # skip mid-game layer when cpu_left below this
XCT_PLACE_MIN_BUDGET = 0.20
XCT_MID_CAP_S = 0.030
XCT_ROBBER_CAP = 30   # max robber variants F evaluates in sim
XCT_TRADE_SKIP = 16   # skip F's maritime trades only above this action count

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
F_EXACT = get_value_fn("base_fn", DEFAULT_WEIGHTS)

_STATS = {"layer": 0, "chains": 0, "place": 0, "place_ok": 0}

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


# ================= orion: exact-opponent machinery =================

def _f_decide(world, enemy):
    """F's actual decision: argmax base_fn over 1-ply successors.

    Returns (successor_after_best, chosen_action). Iterates playable_actions
    in order with strict > — F's exact tie-break (first best wins).
    """
    pa = world.playable_actions
    best, best_v, best_a = None, None, None
    robber_seen = 0
    for action in pa:
        at = action.action_type
        if at == ActionType.MARITIME_TRADE and len(pa) > XCT_TRADE_SKIP:
            continue  # F almost never trades; cap cost on trade-heavy states
        if at == ActionType.MOVE_ROBBER:
            robber_seen += 1
            if robber_seen > XCT_ROBBER_CAP:
                continue
        succ = _successor(world, action)
        if succ is None:
            continue
        v = F_EXACT(succ, enemy)
        if best_v is None or v > best_v:
            best, best_v, best_a = succ, v, action
    return best, best_a


def _sim_discard(world, me, deadline):
    """Discard step inside a simulated chain (either seat): argmax by the
    seat's own eval over the playable discard combos, capped for time."""
    pa = world.playable_actions
    current = world.state.current_color
    fn = (lambda g: evaluate(g, current)) if current == me else (lambda g: F_EXACT(g, current))
    best = None
    for k, action in enumerate(pa):
        if k >= 80 or time.perf_counter() > deadline:
            break
        succ = _successor(world, action)
        if succ is None:
            continue
        if best is None:
            best = succ
        else:
            try:
                if fn(succ) > fn(best):
                    best = succ
            except Exception:
                continue
    return best


def _f_turn(world, me, enemy, deadline):
    """Play F's turn exactly (rolls, discards, robber, builds) from `world`.
    Stops when control returns to us. Returns the world or None on failure."""
    g = world
    for _step in range(14):
        if time.perf_counter() > deadline:
            break
        pa = g.playable_actions
        if not pa:
            break
        at = pa[0].action_type
        if at == ActionType.ROLL:
            nxt = _successor(g, pa[0])
            if nxt is None:
                return None
            g = nxt
            continue
        if at == ActionType.DISCARD_RESOURCE:
            nxt = _sim_discard(g, me, deadline)
            if nxt is None:
                return None
            g = nxt
            continue
        if g.state.current_color != enemy:
            break  # our decision again — stop before our roll
        if at == ActionType.END_TURN and len(pa) == 1:
            break  # F's turn done
        succ, chosen = _f_decide(g, enemy)
        if succ is None:
            return None
        g = succ
        if chosen is not None and chosen.action_type == ActionType.END_TURN:
            break
    return g


def _xct_value(succ, me, enemy, deadline, chains):
    """Mean over roll samples of: our action -> END_TURN -> roll -> F's exact
    turn -> our eval of the position as our roll comes up."""
    total, n = 0.0, 0
    for _ in range(chains):
        if time.perf_counter() > deadline:
            break
        g = succ.copy()
        try:
            end = next((a for a in g.playable_actions
                        if a.action_type == ActionType.END_TURN), None)
            if end is None:
                continue
            g.execute(end)
            g2 = _f_turn(g, me, enemy, deadline)
            if g2 is None:
                continue
            total += evaluate(g2, me)
            n += 1
        except Exception:
            continue
    _STATS["chains"] += n
    return total / n if n else None


# ================= v4 (unchanged logic, orion hooks in _turn/_initial) =================

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
            return self._decide(obs, playable_actions)
        except Exception:
            return 0

    def _decide(self, obs, playable_actions) -> int:
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
        our_buildings = (world.state.buildings_by_color[me][SETTLEMENT]
                         + world.state.buildings_by_color[me][CITY])
        first_placement = len(our_buildings) == 0
        do_fmodel = (first_placement and budget is not None
                     and budget > XCT_PLACE_MIN_BUDGET)
        if budget is None or budget > 0.15:
            plan_limit = (time.perf_counter() + max(0.5 * (budget or 1.0), 0.05))
            if do_fmodel:
                plan_limit = time.perf_counter() + 0.25 * budget
            recheck = []
            for v1, i, succ in scored[:12]:
                road_succ, v = None, v1
                for action in succ.playable_actions:
                    if action.action_type != ActionType.BUILD_ROAD:
                        continue
                    if time.perf_counter() > plan_limit:
                        break
                    gsucc = _successor(succ, action)
                    if gsucc is None:
                        continue
                    rv = evaluate(gsucc, me)
                    if rv > v:
                        v, road_succ = rv, gsucc
                recheck.append((v, i, road_succ))
            recheck.sort(key=lambda t: (-t[0], t[1]))
            if do_fmodel:
                _STATS["place"] += 1
                enemy = world.state.colors[1 - world.state.color_to_index[me]]
                f_limit = time.perf_counter() + 0.35 * budget
                denial = []
                for v_road, i, road_succ in recheck:
                    if road_succ is None or time.perf_counter() > f_limit:
                        denial.append((v_road, i))
                        continue
                    after_f = _f_turn(road_succ, me, enemy, f_limit)
                    if after_f is not None:
                        _STATS["place_ok"] += 1
                    v = v_road if after_f is None else (
                        (1.0 - XCT_W) * v_road + XCT_W * evaluate(after_f, me))
                    denial.append((v, i))
                denial.sort(key=lambda t: (-t[0], t[1]))
                return denial[0][1]
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

        # orion layer: exact-opponent cross-turn re-rank of the top candidates.
        if budget is None or budget > XCT_MIN_BUDGET:
            _STATS["layer"] += 1
            deadline = time.perf_counter() + min(XCT_MID_CAP_S, 0.07 * (budget or 0.5))
            enemy = world.state.colors[1 - world.state.color_to_index[me]]
            for i in sorted(values, key=lambda i: (-values[i], i))[:XCT_TOP]:
                if time.perf_counter() > deadline:
                    break
                x = _xct_value(succs[i], me, enemy, deadline, XCT_CHAINS)
                if x is not None:
                    values[i] = (1.0 - XCT_W) * values[i] + XCT_W * x
        return min(values, key=lambda i: (-values[i], i))
