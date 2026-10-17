"""P1: parallel self-play data generation for the value net.

Workers play teacher-vs-teacher (v4) and teacher-vs-F games on TRAIN-range
seeds and log, for every real decision, the encoded SUCCESSOR state after
the chosen action, the teacher's squashed evaluation of it, and the final
game outcome from that seat's perspective. Output: data/shard_*.npz.

Usage: python selfplay_gen.py [--games 3000] [--workers 12] [--seed0 1000]
Run from the repo root with the parity venv. This is training tooling: it
runs OUTSIDE the shipped agent, so multiprocessing is fine here.
"""
import argparse
import importlib.util
import multiprocessing as mp
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402


def _load_agent(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker_main(worker_id, games, seed0, out_dir):
    os.environ["PYTHONHASHSEED"] = "0"
    sys.path.insert(0, os.path.join(HERE, "my_team"))
    import catan_kit  # noqa: F401  determinism patch
    from catan_kit.obs import Obs, masked_copy
    from catanatron.game import Game
    from catanatron.models.player import Color, RandomPlayer

    v4 = _load_agent(os.path.join(HERE, "my_team", "v8_texel.py"), f"teacher_{worker_id}")
    fmod = _load_agent(os.path.join(HERE, "examples", "f_agent.py"), f"fagent_{worker_id}")
    enc = _load_agent(os.path.join(HERE, "my_team", "encoder.py"), f"encoder_{worker_id}")

    teacher = v4.Agent()
    f_agent = fmod.Agent()
    rng = random.Random(1000 + worker_id)

    X, Z, T, H, VPDR, TN = [], [], [], [], [], []
    done = 0
    t0 = time.time()
    while done < games:
        seed = seed0 + worker_id * 100000 + done
        use_f_opponent = (done % 2 == 0)   # half the games vs F for diversity
        random.seed(seed)
        game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
                    seed=seed, vps_to_win=10)
        random.seed(seed ^ 0xD1CE)

        agents = {Color.RED: teacher, Color.BLUE: (f_agent if use_f_opponent else teacher)}
        seats = {Color.RED: [], Color.BLUE: []}   # per-seat rows: (feats, t, hidden_opp_vp, turn)

        def decide_fn(player, g, actions):
            color = player.color
            acts = list(actions)
            if len(acts) == 1:
                return acts[0]
            view = masked_copy(g, color, rng)
            view.state.players = [RandomPlayer(c) for c in view.state.colors]
            view.seed = 0
            obs = Obs(view, color)
            obs.cpu_left = None
            idx = agents[color].decide(obs, acts)
            try:
                succ = view.copy()
                succ.execute(acts[idx])
                feats = enc.encode(succ, color)
                t_val = max(-6.0, min(6.0, v4.evaluate(succ, color) / 3e14 * 2))
                enemy = succ.state.colors[1 - succ.state.color_to_index[color]]
                ekey = f"P{succ.state.color_to_index[enemy]}"
                hidden = succ.state.player_state[f"{ekey}_VICTORY_POINT_IN_HAND"]
                seats[color].append((feats, t_val, hidden, succ.state.num_turns))
            except Exception:
                pass
            return acts[idx]

        try:
            winner = game.play(decide_fn=decide_fn)
        except Exception:
            done += 1
            continue
        for color, rows in seats.items():
            z = 0.5 if winner is None else (1.0 if winner == color else 0.0)
            ckey = f"P{game.state.color_to_index[color]}"
            final_vps = game.state.player_state[f"{ckey}_ACTUAL_VICTORY_POINTS"] if winner == color                 else game.state.player_state[f"{ckey}_VICTORY_POINTS"]
            for feats, t_val, hidden, turns in rows:
                X.append(feats)
                Z.append(z)
                T.append(t_val)
                H.append(hidden)
                VPDR.append((0.9999 ** turns) * final_vps / 10.0)
                TN.append(turns)
        done += 1
        if done % 25 == 0:
            rate = done / (time.time() - t0)
            print(f"[w{worker_id}] {done}/{games} games ({rate:.2f} g/s, "
                  f"{len(X)} rows)", flush=True)

    out = os.path.join(out_dir, f"shard_{worker_id:02d}.npz")
    np.savez_compressed(out,
                        X=np.asarray(X, dtype=np.float32),
                        Z=np.asarray(Z, dtype=np.float32),
                        T=np.asarray(T, dtype=np.float32),
                        H=np.asarray(H, dtype=np.float32),
                        VPDR=np.asarray(VPDR, dtype=np.float32),
                        TN=np.asarray(TN, dtype=np.float32))
    print(f"[w{worker_id}] wrote {out}: {len(X)} rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400, help="games per worker")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=1000)
    args = ap.parse_args()

    out_dir = os.path.join(HERE, "data")
    os.makedirs(out_dir, exist_ok=True)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker_main, args=(w, args.games, args.seed0, out_dir))
             for w in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    total = sum(len(np.load(os.path.join(out_dir, f), allow_pickle=False)["Z"])
                for f in os.listdir(out_dir) if f.startswith("shard_"))
    print(f"generation complete: {total} labeled rows in {out_dir}")


if __name__ == "__main__":
    main()
