"""Diagnose the round-2 fitted-eval collapse: rank correlation between the
deployed v8b scorer and the known-good v4 heuristic on LIVE successor sets,
plus a direct probe of the fitted model vs the deployed scorer on the same
live feature vectors. If the deployed scorer disagrees with the fitted model
on live inputs, the fit-to-deploy path is broken; if they agree but both rank
badly, the training data is the problem.

Usage: PYTHONHASHSEED=0 .venv/Scripts/python.exe diag_score.py
"""
import importlib.util as ilu
import os
import random
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "my_team"))

import catan_kit  # noqa: F401
from catan_kit.obs import Obs, masked_copy
from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer

import numpy as np


def load(path, name):
    spec = ilu.spec_from_file_location(name, path)
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


v4 = load("my_team/v4.py", "v4d")
v8b = load("my_team/v8b_texel.py", "v8bd")
enc = load("my_team/encoder.py", "encd")

tex = np.load("my_team/texel_eval_r2.npz")
W, B, MU, SD = tex["w"], float(tex["b"]), tex["mean"], tex["scale"]





def _spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def _summary(xs):
    if not xs:
        return "no data"
    a = np.asarray(xs)
    return f"mean {a.mean():+.3f} min {a.min():+.3f} max {a.max():+.3f} n {len(a)}"

agent = v4.Agent()
rng = random.Random(3)
corr_model_deploy = []   # deployed scorer vs direct model math on same features
corr_v8b_v4 = []         # v8b scorer vs v4 heuristic rankings
n_decisions = 0

for seed in range(4):
    random.seed(seed)
    g = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=seed, vps_to_win=10)
    random.seed(seed ^ 0xD1A6)

    def decide_fn(player, game, actions):
        global n_decisions
        color = player.color
        acts = list(actions)
        if len(acts) == 1:
            return acts[0]
        if color == Color.RED and n_decisions < 40:
            n_decisions += 1
            view = masked_copy(game, color, rng)
            view.state.players = [RandomPlayer(c) for c in view.state.colors]
            view.seed = 0
            obs = Obs(view, color)
            obs.cpu_left = None
            # score every successor with both scorers + raw model math
            v4_scores, v8b_scores, model_scores = [], [], []
            for a in acts:
                succ = view.copy()
                try:
                    succ.execute(a)
                except Exception:
                    continue
                v4_scores.append(v4.evaluate(succ, color))
                feats = enc.features(succ, color)
                x = np.asarray(feats, dtype=np.float64)
                model_scores.append(float(((x - MU) / SD) @ W + B))
                v8b_scores.append(v8b.evaluate(succ, color))
            if len(v4_scores) >= 5:
                corr_v8b_v4.append(_spearman(v8b_scores, v4_scores))
                corr_model_deploy.append(_spearman(model_scores, v8b_scores))
        if color == Color.RED:
            view = masked_copy(game, color, rng)
            view.state.players = [RandomPlayer(c) for c in view.state.colors]
            view.seed = 0
            obs = Obs(view, color)
            obs.cpu_left = 0.8
            return acts[agent.decide(obs, acts)]
        return RandomPlayer(Color.BLUE).decide(game, actions)

    ticks = 0
    while g.winning_color() is None and ticks < 400:
        g.play_tick(decide_fn=decide_fn)
        ticks += 1

print("decisions sampled:", n_decisions)
print("spearman(deployed v8b scorer, v4 heuristic):",
      _summary(corr_v8b_v4))
print("spearman(direct model math, deployed v8b scorer):",
      _summary(corr_model_deploy))

