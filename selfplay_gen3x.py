"""Round-3 generation: PREFERENCE PAIRS from a deep teacher.

At every real teacher decision, the deep teacher (v4deep: 10x search, offline
only) and the shallow v4 each pick a move. When they DISAGREE, we log the two
successor feature vectors — (deep_choice, shallow_choice) — a pairwise label
saying the deep choice is better. This is Expert Iteration done on
preferences, the fix the diagnostic pointed at: outcome labels carry no
per-candidate gradient signal at our data scale, teacher preferences do.

Output: data3/shard_XX.npz with P_deep [n,114], P_shallow [n,114], Z [n]
(game outcome from the mover's seat, reference only).

Usage: PYTHONHASHSEED=0 .venv/Scripts/python.exe selfplay_gen3.py \
          [--games 300] [--workers 10] [--seed0 3000]
"""
import argparse
import importlib.util as ilu
import multiprocessing as mp
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np  # noqa: E402


def _load(path, name):
    spec = ilu.spec_from_file_location(name, path)
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def worker_main(worker_id, games, seed0, out_dir):
    os.environ["PYTHONHASHSEED"] = "0"
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "my_team"))
    import catan_kit  # noqa: F401
    from catan_kit.obs import Obs, masked_copy
    from catanatron.game import Game
    from catanatron.models.player import Color, RandomPlayer

    deep_mod = _load(os.path.join(HERE, "my_team", "v4xct.py"), f"xct_{worker_id}")
    shallow_mod = _load(os.path.join(HERE, "my_team", "v4.py"), f"shallow_{worker_id}")
    fmod = _load(os.path.join(HERE, "examples", "f_agent.py"), f"fagent_{worker_id}")
    enc = _load(os.path.join(HERE, "my_team", "encoder.py"), f"encoder_{worker_id}")

    deep = deep_mod.Agent()
    shallow = shallow_mod.Agent()
    f_agent = fmod.Agent()
    rng = random.Random(3000 + worker_id)

    P_deep, P_shallow, Z = [], [], []
    done = 0
    t0 = time.time()
    while done < games:
        seed = seed0 + worker_id * 100000 + done
        use_f_opponent = (done % 2 == 0)
        random.seed(seed)
        game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
                    seed=seed, vps_to_win=10)
        random.seed(seed ^ 0x0FF3)
        movers = {Color.RED: (deep, shallow), Color.BLUE: (f_agent, None)}
        rows = {Color.RED: [], Color.BLUE: []}

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
            primary, secondary = movers[color]
            idx = primary.decide(obs, acts)
            if secondary is not None:
                try:
                    s_idx = secondary.decide(obs, acts)
                    if s_idx != idx:
                        d_succ = view.copy(); d_succ.execute(acts[idx])
                        s_succ = view.copy(); s_succ.execute(acts[s_idx])
                        rows[color].append((enc.encode(d_succ, color),
                                            enc.encode(s_succ, color)))
                except Exception:
                    pass
            return acts[idx]

        try:
            winner = game.play(decide_fn=decide_fn)
        except Exception:
            done += 1
            continue
        for color, pairs in rows.items():
            z = 0.5 if winner is None else (1.0 if winner == color else 0.0)
            for d_feats, s_feats in pairs:
                P_deep.append(d_feats)
                P_shallow.append(s_feats)
                Z.append(z)
        done += 1
        if done % 20 == 0:
            rate = done / (time.time() - t0)
            print(f"[w{worker_id}] {done}/{games} games ({rate:.2f} g/s, "
                  f"{len(P_deep)} pairs)", flush=True)

    out = os.path.join(out_dir, f"shard_{worker_id:02d}.npz")
    np.savez_compressed(out,
                        P_deep=np.asarray(P_deep, dtype=np.float32),
                        P_shallow=np.asarray(P_shallow, dtype=np.float32),
                        Z=np.asarray(Z, dtype=np.float32))
    print(f"[w{worker_id}] wrote {out}: {len(P_deep)} pairs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=3000)
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "data3")
    os.makedirs(out_dir, exist_ok=True)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker_main, args=(w, args.games, args.seed0, out_dir))
             for w in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    total = sum(len(np.load(os.path.join(out_dir, f))["Z"])
                for f in os.listdir(out_dir) if f.startswith("shard_"))
    print(f"generation complete: {total} preference pairs in {out_dir}")


if __name__ == "__main__":
    main()
