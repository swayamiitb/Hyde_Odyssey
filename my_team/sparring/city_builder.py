"""Zoo bot: city builder. Hoards wheat/ore, upgrades cities, never expands."""
import random

from catanatron.models.enums import ActionType


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        best_i, best_rank = 0, -1
        for i, action in enumerate(playable_actions):
            t = action.action_type
            if t == ActionType.BUILD_CITY:
                rank = 6
            elif t == ActionType.BUY_DEVELOPMENT_CARD:
                rank = 4
            elif t == ActionType.PLAY_KNIGHT_CARD:
                rank = 3
            elif t == ActionType.MOVE_ROBBER:
                rank = 2
            elif t == ActionType.BUILD_SETTLEMENT:
                rank = 1
            elif t == ActionType.END_TURN:
                rank = 0
            else:
                rank = 1
            if rank > best_rank:
                best_i, best_rank = i, rank
        return best_i
