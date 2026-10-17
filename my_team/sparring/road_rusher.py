"""Zoo bot: road rusher. Buys roads and settlements toward Longest Road,
preferring wood/brick spots. One-trick opponent for portfolio robustness."""
import random

from catanatron.models.enums import ActionType, RESOURCES


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        world = obs.sample_world(self.rng)
        best_i, best_rank = 0, -1
        for i, action in enumerate(playable_actions):
            t = action.action_type
            rank = 0
            if t == ActionType.BUILD_ROAD:
                rank = 5
            elif t == ActionType.PLAY_ROAD_BUILDING:
                rank = 6
            elif t == ActionType.BUILD_SETTLEMENT:
                rank = 4
            elif t == ActionType.BUY_DEVELOPMENT_CARD:
                rank = 1
            elif t == ActionType.MOVE_ROBBER:
                rank = 2
            elif t == ActionType.END_TURN:
                rank = 0
            else:
                rank = 1
            if rank > best_rank:
                best_i, best_rank = i, rank
        return best_i
