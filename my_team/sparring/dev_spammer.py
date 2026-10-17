"""Zoo bot: dev spammer. Buys every development card it can, plays knights
for Largest Army, rides hidden VP cards."""
import random

from catanatron.models.enums import ActionType


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        best_i, best_rank = 0, -1
        for i, action in enumerate(playable_actions):
            t = action.action_type
            if t == ActionType.BUY_DEVELOPMENT_CARD:
                rank = 7
            elif t == ActionType.PLAY_KNIGHT_CARD:
                rank = 6
            elif t == ActionType.PLAY_YEAR_OF_PLENTY:
                rank = 5
            elif t == ActionType.PLAY_MONOPOLY:
                rank = 5
            elif t == ActionType.PLAY_ROAD_BUILDING:
                rank = 4
            elif t == ActionType.MOVE_ROBBER:
                rank = 3
            elif t == ActionType.BUILD_CITY:
                rank = 2
            elif t == ActionType.END_TURN:
                rank = 0
            else:
                rank = 1
            if rank > best_rank:
                best_i, best_rank = i, rank
        return best_i
