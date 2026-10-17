"""Sparring partner: F's algorithm with catanatron's tuned CONTENDER_WEIGHTS.

Same 1-ply greedy search, different weight vector — a proxy for "another
team's decent bot" and a check that our agent is not overfit to F's exact
evaluation.
"""
import random

from catanatron.players.value import CONTENDER_WEIGHTS, get_value_fn

VALUE_FN = get_value_fn("contender_fn", CONTENDER_WEIGHTS)


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        world = obs.sample_world(self.rng)
        best_i, best_v = 0, None
        for i, action in enumerate(playable_actions):
            successor = world.copy()
            try:
                successor.execute(action)
            except Exception:
                continue
            v = VALUE_FN(successor, obs.me)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i
