"""v4xct: OFFLINE cross-turn teacher — v4 plus opponent-reply lookahead.

For the top candidates of the 1-ply stage, plays END_TURN, rolls the dice
(3 sampled rolls), lets the opponent take their eval-greedy builds, and
scores the resulting position with our own evaluation. The mix of 1-ply and
cross-turn value is what makes this teacher disagree with plain v4 — the
preference signal the ranker distills. Offline only (way over the play clock).

Never shipped. Used by selfplay_gen3x.py to produce preference pairs.
"""
import random

from catanatron.models.enums import ActionType
from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn

import importlib.util as ilu
import os

_spec = ilu.spec_from_file_location(
    "v4base", os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4.py"))
_v4 = ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v4)

evaluate = _v4.evaluate
_successor = _v4._successor

_OPP_FN = get_value_fn("base_fn", DEFAULT_WEIGHTS)


class Agent(_v4.Agent):
    N_ROLLS = 3
    TOP = 6

    def _turn(self, world, me, actions, budget):
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
        order = sorted(values, key=lambda i: (-values[i], i))

        enemy = world.state.colors[1 - world.state.color_to_index[me]]
        for i in order[: self.TOP]:
            xct = self._cross_turn(succs[i], me, enemy)
            if xct is not None:
                values[i] = 0.5 * values[i] + 0.5e14 * (xct / 3e14)  # VP-scale blend
        return min(values, key=lambda i: (-values[i], i))

    def _cross_turn(self, succ, me, enemy):
        """END_TURN -> roll (sampled) -> opponent greedy builds -> our eval."""
        total, n = 0.0, 0
        for _ in range(self.N_ROLLS):
            g = succ.copy()
            try:
                # end our turn
                end = next(a for a in g.playable_actions
                           if a.action_type == ActionType.END_TURN)
                g.execute(end)
                # opponent rolls
                roll = next(a for a in g.playable_actions if a.action_type == ActionType.ROLL)
                g.execute(roll)
                # opponent takes eval-greedy builds (their own eval, like F)
                for _step in range(4):
                    best, best_v = None, None
                    for a in g.playable_actions:
                        if a.action_type in (ActionType.MOVE_ROBBER,):
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
                total += evaluate(g, me)
                n += 1
            except StopIteration:
                continue
            except Exception:
                continue
        return total / n if n else None
