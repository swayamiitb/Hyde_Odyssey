"""v9: round-2 value-net agent (ValueNet2, aux hidden-VP head). — net-scored 1-ply + depth-2, cheap win-now, fallback chain.

Decision stack:
  1. Initial settlement: joint settlement+road search with the proven full
     heuristic eval (placement is too important to trust a young net).
  2. Win-now DFS at >= 8 actual VP, 15 ms deadline (v5's lesson: the full-fat
     version at >= 7 VP / 40 ms burned the clock and self-destructed).
  3. Main decisions: execute every legal action on a sampled world, encode
     each successor, ONE batched net forward, argmax (tie: lowest index).
  4. Same-turn depth-2 on the top-6 expandable actions, net-scored.
  5. Clock gates exactly as v4 (0.018 s / 0.008 s / plain 1-ply tiers).

Fallback chain: any failure in net loading or decide() -> v4 heuristic path
-> first legal action. The agent can never raise.
"""
import os
import random
import time

import torch
import torch.nn as nn

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

import encoder as _enc

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
    (1, 1, 1, 1, 0), (0, 0, 0, 2, 3), (1, 1, 0, 0, 0), (0, 0, 1, 1, 1),
]


class _ValueNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
        )
        self.value = nn.Linear(256, 1)
        self.aux_vp = nn.Linear(256, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.value(h).squeeze(-1), self.aux_vp(h).squeeze(-1)


def _successor(world, action):
    succ = world.copy()
    try:
        succ.execute(action)
    except Exception:
        return None
    return succ


def _hand(st, key):
    ps = st.player_state
    return tuple(ps[f"{key}_{r}_IN_HAND"] for r in RESOURCES)


def _affords(hand):
    return any(all(hand[k] >= c[k] for k in range(5)) for c in COSTS)


def _trade_enables(st, key, action):
    value = action.value
    get = value[-1]
    gives = [v for v in value[:-1] if v is not None]
    hand = list(_hand(st, key))
    for g in gives:
        hand[RESOURCES.index(g)] -= 1
    hand[RESOURCES.index(get)] += 1
    return _affords(tuple(hand))


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


class Agent:
    def __init__(self):
        self.rng = random.Random(0)
        torch.set_num_threads(1)
        self.net = None
        try:
            import json
            here = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(here, "value_net_v3.json")) as fh:
                meta = json.load(fh)
            state = torch.load(os.path.join(here, "value_net_v3.pt"),
                               map_location="cpu", weights_only=True)
            net = _ValueNet(int(meta["dim"]))
            net.load_state_dict(state)
            net.eval()
            with torch.inference_mode():
                for _ in range(10):
                    net(torch.zeros(4, int(meta["dim"])))
            self.net = net
        except Exception:
            self.net = None

    # ---------- net helpers ----------

    def _net_values(self, succs, me):
        feats = [_enc.features(s, me) for s in succs]
        x = torch.tensor(feats, dtype=torch.float32)
        with torch.inference_mode():
            logit, aux = self.net(x)
            # opponent hidden VP cards make the position worse for us
            return (logit - 0.5 * aux).tolist()

    # ---------- decision entry ----------

    def decide(self, obs, playable_actions) -> int:
        try:
            return self._decide(obs, playable_actions)
        except Exception:
            try:
                return self._heuristic(obs, playable_actions)
            except Exception:
                return 0

    def _decide(self, obs, playable_actions):
        world = obs.sample_world(self.rng)
        types = {a.action_type for a in playable_actions}
        budget = obs.cpu_left

        if world.state.is_initial_build_phase and ActionType.BUILD_SETTLEMENT in types:
            return self._initial_settlement(world, obs.me, playable_actions, budget)

        i = self._win_now(world, obs.me, playable_actions)
        if i is not None:
            return i

        if self.net is not None:
            return self._net_turn(world, obs.me, playable_actions, budget)
        return self._heuristic_turn(world, obs.me, playable_actions, budget)

    # ---------- win-now (cheap version of v5's) ----------

    def _win_now(self, world, me, actions):
        st = world.state
        key = player_key(st, me)
        if st.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] < 8:
            return None
        deadline = time.perf_counter() + 0.015

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
                    if tried_forced:
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

    # ---------- placement (full eval, as v4) ----------

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

    # ---------- net-scored turn ----------

    def _net_turn(self, world, me, actions, budget):
        values = {}
        succs = {}
        live = []
        for i, action in enumerate(actions):
            succ = _successor(world, action)
            if succ is None:
                continue
            succs[i] = succ
            live.append(i)
        if not live:
            return 0
        vals = self._net_values([succs[i] for i in live], me)
        for i, v in zip(live, vals):
            values[i] = v

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
            follow = [a for a in succs[i].playable_actions
                      if a.action_type in _FOLLOWUP]
            if not follow:
                continue
            gs = [s for s in (_successor(succs[i], a) for a in follow) if s is not None]
            if gs:
                vals = self._net_values(gs, me)
                values[i] = max(values[i], max(vals))
            expanded += 1
            if expanded >= 6 or time.perf_counter() > deadline:
                break
        return min(values, key=lambda i: (-values[i], i))

    # ---------- heuristic fallbacks (v4 paths) ----------

    def _heuristic_turn(self, world, me, actions, budget):
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
            for action in succs[i].playable_actions:
                if action.action_type not in _FOLLOWUP:
                    continue
                gsucc = _successor(succs[i], action)
                if gsucc is None:
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

    def _heuristic(self, obs, playable_actions):
        world = obs.sample_world(self.rng)
        return self._heuristic_turn(world, obs.me, playable_actions, obs.cpu_left)
