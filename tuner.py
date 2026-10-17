"""Parallel coordinate descent on the tunable eval weights.

Fitness = win rate vs F over a FIXED 200-game seed range (paired boards and
dice across candidates, so comparisons see only the agent difference).
Candidates run 8 at a time as evaluate.py subprocesses; each passes its
weights to tune_agent.py through the TUNE_WEIGHTS_JSON env var.

Usage:  python tuner.py [--cycles 4] [--games 200] [--workers 8] [--seed0 99000]
Logs one JSON line per evaluation to tune_log.jsonl; best weights to tune_best.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT = os.path.join(HERE, "my_team", "tune_agent.py")
OPPONENT = os.path.join(HERE, "examples", "f_agent.py")
LOG = os.path.join(HERE, "tune_log.jsonl")
BEST = os.path.join(HERE, "tune_best.json")

BASE_WEIGHTS = {
    "translate_variety": 4.0,
    "reachable_production_0": 8e3,
    "reachable_production_1": 1e4,
    "buildable_nodes": 1e3,
    "longest_road": 10.0,
    "knight_in_hand": 5.0,
    "hand_synergy": 1e2,
    "hand_resources": 1.0,
    "hand_devs": 10.0,
    "army_size": 10.1,
    "enemy_army_size": -12.0,
    "port_specialty": 2.5e3,
    "port_generic": 1.2e3,
    "num_tiles": 1.0,
    "discard_penalty": -5.0,
}
FACTORS = [0.25, 0.5, 2.0, 4.0]


def run_eval(weights, games, seed0, tag):
    """One evaluate.py subprocess; returns (win_rate, faults, over_clock) or None."""
    env = dict(os.environ)
    env["TUNE_WEIGHTS_JSON"] = json.dumps(weights)
    env["PYTHONHASHSEED"] = "0"
    cmd = [sys.executable, "-m", "catan_kit.evaluate",
           "--a", AGENT, "--b", OPPONENT,
           "--games", str(games), "--seed0", str(seed0)]
    try:
        out = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True,
                             text=True, timeout=3600).stdout
    except subprocess.TimeoutExpired:
        return None
    wr = faults = over = None
    for line in out.splitlines():
        if line.startswith("A win rate"):
            wr = float(line.split()[3])
        elif line.startswith("A errors"):
            faults = int(line.split()[2])
        elif line.startswith("A games over budget"):
            over = int(line.split()[4])
    if wr is None:
        return None
    rec = dict(tag=tag, t=time.time(), win_rate=wr, faults=faults, over_clock=over,
               games=games, seed0=seed0, weights=weights)
    with open(LOG, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def batch(cands, games, seed0, workers):
    """Run candidate dicts in parallel; yields (tag, record)."""
    pending = list(cands)
    running = []  # (tag, Popen, weights)
    results = []
    while pending or running:
        while pending and len(running) < workers:
            tag, weights = pending.pop(0)
            env = dict(os.environ)
            env["TUNE_WEIGHTS_JSON"] = json.dumps(weights)
            env["PYTHONHASHSEED"] = "0"
            cmd = [sys.executable, "-m", "catan_kit.evaluate",
                   "--a", AGENT, "--b", OPPONENT,
                   "--games", str(games), "--seed0", str(seed0)]
            p = subprocess.Popen(cmd, cwd=HERE, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 text=True)
            running.append((tag, p, weights))
        tag, p, weights = running.pop(0)
        out, _ = p.communicate(timeout=3600)
        wr = faults = over = None
        for line in out.splitlines():
            if line.startswith("A win rate"):
                wr = float(line.split()[3])
            elif line.startswith("A errors"):
                faults = int(line.split()[2])
            elif line.startswith("A games over budget"):
                over = int(line.split()[4])
        if wr is None:
            print(f"  [{tag}] FAILED to parse output", flush=True)
            results.append((tag, None))
            continue
        rec = dict(tag=tag, t=time.time(), win_rate=wr, faults=faults, over_clock=over,
                   games=games, seed0=seed0, weights=weights)
        with open(LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  [{tag}] wr={wr:.3f} faults={faults} over={over}", flush=True)
        results.append((tag, rec))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=99000)
    args = ap.parse_args()

    current = dict(BASE_WEIGHTS)
    print("baseline run...", flush=True)
    base = batch([("base", dict(current))], args.games, args.seed0, 1)[0][1]
    best_wr = base["win_rate"]
    print(f"baseline win rate {best_wr:.3f}", flush=True)

    for cycle in range(args.cycles):
        print(f"=== cycle {cycle + 1} (best {best_wr:.3f}) ===", flush=True)
        cands = []
        for name in current:
            for f in FACTORS:
                w = dict(current)
                w[name] = w[name] * f
                cands.append((f"{cycle + 1}:{name}:x{f}", w))
        results = batch(cands, args.games, args.seed0, args.workers)
        best_rec = None
        for tag, rec in results:
            if rec is None or (rec["faults"] or 0) != 0 or (rec["over_clock"] or 0) != 0:
                continue
            if rec["win_rate"] > best_wr + 0.005:
                if best_rec is None or rec["win_rate"] > best_rec["win_rate"]:
                    best_rec = rec
        if best_rec is not None:
            best_wr = best_rec["win_rate"]
            current = dict(best_rec["weights"])
            print(f"  ACCEPT {best_rec['tag']} -> {best_wr:.3f}", flush=True)
        with open(BEST, "w") as fh:
            json.dump(dict(win_rate=best_wr, weights=current), fh, indent=2)
        if best_rec is None:
            print("  no improvement this cycle; stopping", flush=True)
            break

    print(json.dumps(current, indent=2))
    print(f"best win rate {best_wr:.3f}; saved to {BEST}")


if __name__ == "__main__":
    main()
