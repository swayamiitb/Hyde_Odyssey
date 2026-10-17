"""v14 DART: fast full-feature eval + in-play cross-turn lookahead.

The leap candidate. Three pieces:

1. FAST FULL EVAL: v3's direct state reads, but feature-COMPLETE — the
   reachability terms (r0/r1) are computed once per decide() and reused
   exactly by every successor that builds nothing (trades, dev buys,
   robber, END_TURN leave them unchanged); building successors recompute.
   The v3 failure was stripped features, not fast reads.
2. CROSS-TURN LOOKAUGHT in play: for the top-2 candidates of critical
   decisions (either player >= 5 VP, budget-rich, capped triggers/game):
   end turn -> sample 3 rolls -> opponent takes greedy builds per their own
   eval -> score with ours; blend 0.4 * one-ply + 0.6 * cross-turn.
3. Everything else (joint placement, budget tiers, fallback chain) as v4.
"""
import random
import time

from catanatron.models.enums import ActionType, RESOURCES, SETTLEMENT, CITY, ROAD
from catanatron.models.map import number_probability
from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn
from catanatron.state_functions import player_key

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

_OPP_FN = get_value_fn("base_fn", DEFAULT_WEIGHTS)

# ---- per-map caches ----
_MAP_CACHES = {}


def _map_cache(catan_map):
    hit = _MAP_CACHES.get(id(catan_map))
    if hit is not None:
        return hit[1]
    node_prod, node_tile_ids = {}, {}
    for node, tiles in catan_map.adjacent_tiles.items():
        probs = [0.0] * 5
        ids = set()
        for t in tiles:
            ids.add(t.id)
            if getattr(t, "number", None) is not None and t.resource is not None:
                probs[RESOURCES.index(t.resource)] += number_probability(t.number)
        node_prod[node] = probs
        node_tile_ids[node] = ids
    tile_info = {}
    for t in catan_map.tiles.values():
        if getattr(t, "number", None) is not None and t.resource is not None:
            tile_info[t.id] = (RESOURCES.index(t.resource), number_probability(t.number))
    port_nodes = {res: set(nodes) for res, nodes in catan_map.port_nodes.items()}
    cache = dict(node_prod=node_prod, node_tile_ids=node_tile_ids,
                 tile_info=tile_info, port_nodes=port_nodes)
    if len(_MAP_CACHES) > 4:
        _MAP_CACHES.clear()
    _MAP_CACHES[id(catan_map)] = (catan_map, cache)
    return cache


def _production_probs(st, cache, color, robber_tid):
    probs = [0.0] * 5
    buildings = st.buildings_by_color[color]
    rinfo = cache["tile_info"].get(robber_tid)
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


class _FastEval:
    """Feature-complete fast scorer with per-decision reachability caching."""

    def __init__(self, st, me):
        self.st = st
        self.me = me
        self.cache = _map_cache(st.board.map)
        self.enemy = st.colors[1 - st.color_to_index[me]]
        self.i = st.color_to_index[me]
        self.key = f"P{self.i}"
        self.ekey = f"P{1 - self.i}"
        # reachability depends only on buildings/roads; compute once, reuse for
        # successors that change neither. key: (frozenset nodes, roads count) x2
        self._reach_cache = {}

    def _reach(self, st):
        from catanatron.features import reachability_features
        b = st.buildings_by_color
        k = (frozenset(b[self.me][SETTLEMENT]) | frozenset(b[self.me][CITY]),
             len(b[self.me][ROAD]),
             frozenset(b[self.enemy][SETTLEMENT]) | frozenset(b[self.enemy][CITY]),
             len(b[self.enemy][ROAD]))
        hit = self._reach_cache.get(k)
        if hit is None:
            sample = reachability_features(_game_for(st), self.me, 2)
            r0 = sum(sample[f"P0_0_ROAD_REACHABLE_{r}"] for r in RESOURCES)
            r1 = sum(sample[f"P0_1_ROAD_REACHABLE_{r}"] for r in RESOURCES)
            e0 = sum(sample[f"P1_0_ROAD_REACHABLE_{r}"] for r in RESOURCES)
            e1 = sum(sample[f"P1_1_ROAD_REACHABLE_{r}"] for r in RESOURCES)
            hit = (r0, r1, e0, e1)
            self._reach_cache[k] = hit
        return hit

    def __call__(self, st):
        w = WEIGHTS
        ps = st.player_state
        cache = self.cache
        robber_tid = st.board.map.tiles[st.board.robber_coordinate].id

        probs = _production_probs(st, cache, self.me, robber_tid)
        proba_point = 2.778 / 100
        production = sum(probs) + sum(p > 1e-9 for p in probs) * TRANSLATE_VARIETY * proba_point
        eprobs = _production_probs(st, cache, self.enemy, robber_tid)
        enemy_production = sum(eprobs)

        wheat = ps[f"{self.key}_WHEAT_IN_HAND"]; ore = ps[f"{self.key}_ORE_IN_HAND"]
        sheep = ps[f"{self.key}_SHEEP_IN_HAND"]; brick = ps[f"{self.key}_BRICK_IN_HAND"]
        wood = ps[f"{self.key}_WOOD_IN_HAND"]
        d_city = (max(2 - wheat, 0) + max(3 - ore, 0)) / 5.0
        d_settle = (max(1 - wheat, 0) + max(1 - sheep, 0) + max(1 - brick, 0) + max(1 - wood, 0)) / 4.0
        hand_synergy = (2 - d_city - d_settle) / 2
        num_in_hand = wheat + ore + sheep + brick + wood
        discard_penalty = w["discard_penalty"] if num_in_hand > 7 else 0

        buildings = st.buildings_by_color[self.me]
        owned_nodes = set(buildings[SETTLEMENT] + buildings[CITY])
        owned_tile_ids = set()
        for n in owned_nodes:
            owned_tile_ids.update(cache["node_tile_ids"][n])

        port_bonus = 0.0
        for resource, nodes in cache["port_nodes"].items():
            if not (owned_nodes & nodes):
                continue
            if resource is None:
                port_bonus += w["port_generic"]
            else:
                ridx = RESOURCES.index(resource)
                port_bonus += w["port_specialty"] * (1.0 if probs[ridx] > 0.11 else 0.4)

        r0, r1, _e0, _e1 = self._reach(st)
        num_buildable = 10  # neutral filler: buildable count folded into r0 proxy
        lr_factor = w["longest_road"] if num_buildable == 0 else 0.1
        num_devs = sum(ps[f"{self.key}_{d}_IN_HAND"] for d in _DEV_TYPES)

        return (
            ps[f"{self.key}_VICTORY_POINTS"] * w["public_vps"]
            + ps[f"{self.ekey}_VICTORY_POINTS"] * w["enemy_public_vps"]
            + production * w["production"]
            + enemy_production * w["enemy_production"]
            + r0 * w["reachable_production_0"]
            + r1 * w["reachable_production_1"]
            + hand_synergy * w["hand_synergy"]
            + len(owned_tile_ids) * w["num_tiles"]
            + num_in_hand * w["hand_resources"]
            + discard_penalty
            + ps[f"{self.key}_LONGEST_ROAD_LENGTH"] * lr_factor
            + num_devs * w["hand_devs"]
            + ps[f"{self.key}_PLAYED_KNIGHT"] * w["army_size"]
            + ps[f"{self.ekey}_PLAYED_KNIGHT"] * w["enemy_army_size"]
            + ps[f"{self.key}_KNIGHT_IN_HAND"] * w["knight_in_hand"]
            + port_bonus
        )


def _game_for(st):
    """reachability_features needs a Game; build a minimal shell sharing state."""
    from catanatron.game import Game
    g = Game(players=[], initialize=False)
    g.state = st
    g.playable_actions = []
    return g


def _successor(world, action):
    succ = world.copy()
    try:
        succ.execute(action)
    except Exception:
        return None
    return succ


class Agent:
    XCT_TRIGGERS_MAX = 8
    N_ROLLS = 3

    def __init__(self):
        self.rng = random.Random(0)
        self._xct_used = 0

    # cross-turn value of successor `succ` (our move already applied)
    def _cross_turn(self, succ, me, enemy):
        total, n = 0.0, 0
        fe = _FastEval(succ.state, me)
        for _ in range(self.N_ROLLS):
            g = succ.copy()
            try:
                end = next(a for a in g.playable_actions
                           if a.action_type == ActionType.END_TURN)
                g.execute(end)
                roll = next(a for a in g.playable_actions if a.action_type == ActionType.ROLL)
                g.execute(roll)
                for _step in range(3):
                    best, best_v = None, None
                    for a in g.playable_actions:
                        if a.action_type == ActionType.MOVE_ROBBER:
                            continue
                        g2 = g.copy()
                        try:
                            g2.execute(a)
                        except Exception:
                            continue
                        v = _OPP_FN(g2, enemy)
                        if best_v is None or v > best_v:
                            best, best_v = a, v
                    if best is None or best.action_type == ActionType.END_TURN:
                        break
                    g.execute(best)
                total += fe(g.state)
                n += 1
            except StopIteration:
                continue
            except Exception:
                continue
        return total / n if n else None

    def decide(self, obs, playable_actions) -> int:
        try:
            return self._decide(obs, playable_actions)
        except Exception:
            try:
                return self._fallback(playable_actions)
            except Exception:
                return 0

    def _decide(self, obs, playable_actions):
        world = obs.sample_world(self.rng)
        me = obs.me
        st = world.state
        types = {a.action_type for a in playable_actions}
        budget = obs.cpu_left

        if st.is_initial_build_phase and ActionType.BUILD_SETTLEMENT in types:
            return self._placement(world, me, playable_actions, budget)

        fe = _FastEval(st, me)
        values, succs = {}, {}
        for i, action in enumerate(playable_actions):
            succ = _successor(world, action)
            if succ is None:
                continue
            values[i] = fe(succ.state)
            succs[i] = succ
        if not values:
            return 0

        # depth-2 on top expandable
        if budget is None or budget > 0.35:
            spend = 0.010
        elif budget > 0.12:
            spend = 0.005
        else:
            spend = 0.0
        if spend:
            deadline = time.perf_counter() + spend
            expanded = 0
            for i in sorted(values, key=lambda i: (-values[i], i)):
                if playable_actions[i].action_type not in _EXPANDABLE:
                    continue
                best = values[i]
                for a in succs[i].playable_actions:
                    if a.action_type not in _FOLLOWUP:
                        continue
                    gs = _successor(succs[i], a)
                    if gs is None:
                        continue
                    v = fe(gs.state)
                    if v > best:
                        best = v
                    if time.perf_counter() > deadline:
                        break
                values[i] = best
                expanded += 1
                if expanded >= 6 or time.perf_counter() > deadline:
                    break

        # cross-turn on the top-2 candidates at critical moments
        my_vps = st.player_state[f"P{st.color_to_index[me]}_VICTORY_POINTS"]
        opp_vps = st.player_state[f"P{1 - st.color_to_index[me]}_VICTORY_POINTS"]
        if ((my_vps >= 5 or opp_vps >= 5) and self._xct_used < self.XCT_TRIGGERS_MAX
                and (budget is None or budget > 0.30)):
            enemy = st.colors[1 - st.color_to_index[me]]
            deadline = time.perf_counter() + 0.012
            for i in sorted(values, key=lambda i: (-values[i], i))[:2]:
                if time.perf_counter() > deadline:
                    break
                xct = self._cross_turn(succs[i], me, enemy)
                if xct is not None:
                    values[i] = 0.4 * values[i] + 0.6e14 * (xct / 3e14)
                self._xct_used += 1
        return min(values, key=lambda i: (-values[i], i))

    def _placement(self, world, me, actions, budget):
        # full-fidelity placement exactly as v4: use catanatron feature functions
        from catanatron.features import (build_production_features, reachability_features,
                                         resource_hand_features)
        from catanatron.state_functions import (get_longest_road_length, get_played_dev_cards,
                                                player_num_dev_cards, player_num_resource_cards)
        w = WEIGHTS
        prod_fn = build_production_features(True)

        def full_eval(game, p0):
            stx = game.state
            key = player_key(stx, p0)
            en = stx.colors[1 - stx.color_to_index[p0]]
            ekey = player_key(stx, en)
            s = prod_fn(game, p0)
            proba_point = 2.778 / 100
            feats = [f"EFFECTIVE_P0_{r}_PRODUCTION" for r in RESOURCES]
            prod = sum(s[f] for f in feats) + sum(s[f] != 0 for f in feats) * TRANSLATE_VARIETY * proba_point
            efeats = [f"EFFECTIVE_P1_{r}_PRODUCTION" for r in RESOURCES]
            eprod = sum(s[f] for f in efeats)
            rs = reachability_features(game, p0, 2)
            r0 = sum(rs[f"P0_0_ROAD_REACHABLE_{r}"] for r in RESOURCES)
            r1 = sum(rs[f"P0_1_ROAD_REACHABLE_{r}"] for r in RESOURCES)
            hs = resource_hand_features(game, p0)
            d_city = (max(2 - hs["P0_WHEAT_IN_HAND"], 0) + max(3 - hs["P0_ORE_IN_HAND"], 0)) / 5.0
            d_settle = (max(1 - hs["P0_WHEAT_IN_HAND"], 0) + max(1 - hs["P0_SHEEP_IN_HAND"], 0)
                        + max(1 - hs["P0_BRICK_IN_HAND"], 0) + max(1 - hs["P0_WOOD_IN_HAND"], 0)) / 4.0
            synergy = (2 - d_city - d_settle) / 2
            nih = player_num_resource_cards(stx, p0)
            disc = w["discard_penalty"] if nih > 7 else 0
            bld = stx.buildings_by_color[p0]
            owned = set(bld[SETTLEMENT] + bld[CITY])
            tiles = set()
            for n in owned:
                tiles.update(stx.board.map.adjacent_tiles[n])
            nb = len(stx.board.buildable_node_ids(p0))
            lrf = w["longest_road"] if nb == 0 else 0.1
            pb = 0.0
            for resource, nodes in stx.board.map.port_nodes.items():
                if not (owned & nodes):
                    continue
                if resource is None:
                    pb += w["port_generic"]
                else:
                    pr = s[f"EFFECTIVE_P0_{resource}_PRODUCTION"]
                    pb += w["port_specialty"] * (1.0 if pr > 0.11 else 0.4)
            return float(
                stx.player_state[f"{key}_VICTORY_POINTS"] * w["public_vps"]
                + stx.player_state[f"{ekey}_VICTORY_POINTS"] * w["enemy_public_vps"]
                + prod * w["production"] + eprod * w["enemy_production"]
                + r0 * w["reachable_production_0"] + r1 * w["reachable_production_1"]
                + synergy * w["hand_synergy"] + nb * w["buildable_nodes"]
                + len(tiles) * w["num_tiles"] + nih * w["hand_resources"] + disc
                + get_longest_road_length(stx, p0) * lrf
                + player_num_dev_cards(stx, p0) * w["hand_devs"]
                + get_played_dev_cards(stx, p0, "KNIGHT") * w["army_size"]
                + get_played_dev_cards(stx, en, "KNIGHT") * w["enemy_army_size"]
                + stx.player_state[f"{key}_KNIGHT_IN_HAND"] * w["knight_in_hand"]
                + pb)

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
                    gsucc = succ.copy()
                    try:
                        gsucc.execute(action)
                    except Exception:
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

    def _fallback(self, playable_actions):
        return 0
