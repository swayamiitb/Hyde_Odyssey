"""Auditable parallel driver for the UNMODIFIED, clocked official match harness.

Fixed 20-board batches keep game indices and agent lifetimes independent of worker
count. Every requested game must complete. Both seats' faults and clocks are saved.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(os.environ['CATAN_KIT_ROOT']).resolve()
sys.path.insert(0, str(ROOT))


def worker(job):
    from catan_kit.match import play_chunk
    a, b, start, count, seed0, key = job
    indices = list(range(start, start + count))
    records, info = play_chunk(a, b, [seed0 + i // 2 for i in indices], indices,
                               key=key.encode())
    return records, info


def wilson(wins, n):
    p = wins / n
    z2 = 1.96 ** 2
    c = (p + z2 / (2 * n)) / (1 + z2 / n)
    d = 1.96 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [round(c - d, 6), round(c + d, 6)]


def summary(records):
    from catan_kit.evaluate import summarise
    result = {s: summarise(records, s) for s in ('a', 'b')}
    for s in ('a', 'b'):
        r = result[s]
        r['wilson95'] = wilson(r['wins'], r['games'])
        pairs = {}
        for g in records:
            pairs.setdefault(g['game'] // 2, []).append(int(g['winner'] == s))
        values = [sum(p) / 2 for p in pairs.values() if len(p) == 2]
        if len(values) > 1:
            avg = sum(values) / len(values)
            se = math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1) / len(values))
            r['board_cluster95'] = [max(0, avg - 1.96 * se), min(1, avg + 1.96 * se)]
        r['wins_by_seat'] = {str(first): sum(g['winner'] == s for g in records
                                            if g['a_first'] == first) for first in (True, False)}
    return result


def main():
    from catan_kit._patch import ensure_hash_seed
    ensure_hash_seed()
    p = argparse.ArgumentParser()
    p.add_argument('--a', required=True)
    p.add_argument('--b', default=str(ROOT / 'examples/f_agent.py'))
    p.add_argument('--games', type=int, default=400)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seed0', type=int, default=710000)
    p.add_argument('--key', default='codex-catan-development-20261017')
    p.add_argument('--name', required=True)
    args = p.parse_args()
    if args.games <= 0 or args.games % 2 or args.workers <= 0:
        p.error('games must be positive and even; workers must be positive')
    out = Path(__file__).resolve().parents[1] / 'local_runs' / args.name
    out.mkdir(parents=True, exist_ok=False)
    paths = []
    for side, name in [('a', args.a), ('b', args.b)]:
        src = Path(name).resolve()
        dst = out / side
        dst.mkdir()
        for f in src.parent.glob('*.py'):
            shutil.copy2(f, dst / f.name)
        paths.append(str((dst / src.name).resolve()))
    manifest = vars(args).copy()
    manifest['python'] = sys.version
    manifest['source_hashes'] = {s: hashlib.sha256(Path(f).read_bytes()).hexdigest()
                                  for s, f in zip(('a', 'b'), paths)}
    manifest['harness_hashes'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in (ROOT / 'catan_kit').glob('*.py')}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    jobs = [(*paths, i, min(40, args.games - i), args.seed0, args.key)
            for i in range(0, args.games, 40)]
    results = []
    t0 = time.monotonic()
    print(json.dumps({'event': 'start', 'name': args.name, 'games': args.games,
                      'workers': args.workers, 'out': str(out)}), flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = [pool.submit(worker, j) for j in jobs]
        with (out / 'games.jsonl').open('w') as log:
            for future in concurrent.futures.as_completed(futures):
                records, info = future.result()
                results.extend(records)
                for r in records:
                    log.write(json.dumps(r) + '\n')
                log.flush()
                s = summary(results)
                (out / 'progress.json').write_text(json.dumps(s, indent=2))
                print(json.dumps({'event': 'progress', 'games': len(results),
                    'wins': s['a']['wins'], 'rate': round(s['a']['win_rate'], 4),
                    'a_errors': s['a']['errors'], 'b_errors': s['b']['errors'],
                    'a_over': s['a']['games_over_budget'], 'b_over': s['b']['games_over_budget'],
                    'cpu_ms': round(s['a']['mean_cpu_s'] * 1000, 1),
                    'wall_s': round(time.monotonic() - t0, 1)}), flush=True)
    assert len(results) == args.games
    assert len({r['game'] for r in results}) == args.games
    results.sort(key=lambda r: r['game'])
    s = summary(results)
    s['wall_s'] = time.monotonic() - t0
    s['complete'] = True
    (out / 'summary.json').write_text(json.dumps(s, indent=2))
    print(json.dumps({'event': 'complete', **s}), flush=True)


if __name__ == '__main__':
    main()
