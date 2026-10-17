"""v8b: v8 skeleton with the Texel weights refitted on round-2 data.

sigmoid(w.features + b) ~= win probability, fitted by logistic regression on
579k self-play (position, outcome) pairs. Same 1-ply + depth-2 + joint
placement structure as v4; only the leaf scorer changes.

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


import os as _os

import numpy as _np

import encoder as _enc

_texel = _np.load(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "texel_eval_r2.npz"))
_W, _B, _MU, _SD = _texel["w"], float(_texel["b"]), _texel["mean"], _texel["scale"]


def evaluate(game, p0_color):
    x = _np.asarray(_enc.features(game, p0_color), dtype=_np.float64)
    return float(((x - _MU) / _SD) @ _W + _B)


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
