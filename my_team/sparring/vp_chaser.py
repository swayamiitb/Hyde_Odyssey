"""Sparring partner: a pure victory-point chaser.

Argmax over successors by own public VP, ties broken toward more buildings.
A deliberately narrow bot: beats nobody good, but checks that our agent
punishes one-dimensional play (robustness sanity check).
"""
import random

from catanatron.models.enums import SETTLEMENT, CITY, ROAD


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def _score(self, game, color):
        st = game.state
        key = f"P{st.color_to_index[color]}"
        vps = st.player_state[f"{key}_VICTORY_POINTS"]
        buildings = st.buildings_by_color[color]
        return vps * 1000 + len(buildings[SETTLEMENT]) + 2 * len(buildings[CITY]) + 0.1 * len(buildings[ROAD])

    def decide(self, obs, playable_actions) -> int:
        world = obs.sample_world(self.rng)
        best_i, best_v = 0, None
        for i, action in enumerate(playable_actions):
            successor = world.copy()
            try:
                successor.execute(action)
            except Exception:
                continue
            v = self._score(successor, obs.me)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i
