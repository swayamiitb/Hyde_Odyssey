"""v13_2: v4 plus a ranker residual on IN-TURN scoring only (LAMBDA = 300).

Adds LAMBDA * ranker(features(successor, me)) to the values compared inside
Agent._turn — both the 1-ply successor values and the depth-2 follow-up
values. _initial_settlement (and its road follow-ups) are v4 verbatim, so
placement is untouched. The ranker is the pairwise preference net trained by
train_ranker.py (Ranker class copied in from there, state in ranker_v1.pt,
input dim in ranker_v1.json); features come from encoder.features, the exact
vector the net was trained on. This is distilled deep search as a small
additive residual: at 1e6 it stays orders below the 3e14 VP term and below
one pip of production (1e8 per probability unit).

For speed the residual is computed with one batched forward per decision
(single-item forwards cost ~50x more per successor than a batched one, which
pushed games over the CPU clock); the math per successor is unchanged.

Loading is defensive: torch.set_num_threads(1) and a warmup forward pass in
__init__; if anything fails, self.ranker = None, the residual contributes
0.0 everywhere, and every code path taken is v4's own (values bit-identical).
decide() can never raise (falls back to index 0).
"""
import json
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

LAMBDA = 300

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

_HERE = os.path.dirname(os.path.abspath(__file__))
_RANKER_PT = os.path.join(_HERE, "ranker_v1.pt")
_RANKER_JSON = os.path.join(_HERE, "ranker_v1.json")


class Ranker(nn.Module):
    """Copied in verbatim from train_ranker.py (the trained architecture)."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 192), nn.GELU(),
            nn.Linear(192, 192), nn.GELU(),
            nn.Linear(192, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


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


def _best_followup(succ, me, deadline, value_now, agent=None):
    """Max eval over affordable follow-up actions (bounded by wall deadline).

    v13: when `agent` is given and its ranker is live, follow-up values get
    the same LAMBDA * ranker residual as the 1-ply values (one batched
    forward). With agent=None or ranker=None this is v4's function verbatim."""
    if agent is None or agent.ranker is None:
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
    gsuccs = []
    for action in succ.playable_actions:
        if action.action_type not in _FOLLOWUP:
            continue
        gsucc = _successor(succ, action)
        if gsucc is None:
            continue
        gsuccs.append(gsucc)
        if time.perf_counter() > deadline:
            break
    if not gsuccs:
        return value_now
    best = value_now
    vals = [evaluate(g, me) for g in gsuccs]
    for v, r in zip(vals, agent._rank_batch(gsuccs, me, deadline)):
        w = v + LAMBDA * r
        if w > best:
            best = w
    return best


class Agent:
    def __init__(self):
        self.rng = random.Random(0)
        torch.set_num_threads(1)
        self.ranker = None
        try:
            with open(_RANKER_JSON) as fh:
                dim = int(json.load(fh)["dim"])
            state = torch.load(_RANKER_PT, map_location="cpu", weights_only=True)
            ranker = Ranker(dim)
            ranker.load_state_dict(state)
            ranker.eval()
            with torch.inference_mode():          # warmup
                for _ in range(3):
                    ranker(torch.zeros(8, dim))
            self.ranker = ranker
        except Exception:
            self.ranker = None

    def _rank_batch(self, games, me, deadline=None):
        """Ranker residual for each game, in ONE batched forward pass.

        Games whose features fail or that fall past the deadline score 0.0.
        With no ranker this is a list of zeros and nothing is computed, which
        keeps the fallback path exactly v4. Never raises."""
        out = [0.0] * len(games)
        if self.ranker is None:
            return out
        try:
            feats = []
            idx = []
            for k, g in enumerate(games):
                try:
                    feats.append(_enc.features(g, me))
                except Exception:
                    continue
                idx.append(k)
                if deadline is not None and time.perf_counter() > deadline:
                    break
            if feats:
                x = torch.tensor(feats, dtype=torch.float32)
                with torch.inference_mode():
                    scores = self.ranker(x).tolist()
                for k, s in zip(idx, scores):
                    out[k] = float(s)
        except Exception:
            pass
        return out

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
            succs[i] = succ
        if not values:
            return 0

        # v13: every 1-ply value gets the residual LAMBDA * ranker(successor),
        # computed in one batched forward.
        if self.ranker is not None:
            order = sorted(values)
            residuals = self._rank_batch([succs[i] for i in order], me)
            for i, r in zip(order, residuals):
                values[i] += LAMBDA * r

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
            values[i] = _best_followup(succs[i], me, deadline, values[i], self)
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
