"""Round-3 trainer: pairwise preference ranker (margin-ranking loss).

Learns s(x) such that s(deep_choice) > s(shallow_choice) + margin on every
logged disagreement pair. Deployed as a small residual on top of v4's
evaluation, this is distilled deep search — the ExIt student.

Usage: PYTHONHASHSEED=0 .venv/Scripts/python.exe train_ranker.py
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util as ilu
_spec = ilu.spec_from_file_location("enc", os.path.join(HERE, "my_team", "encoder.py"))
enc = ilu.module_from_spec(_spec)
_spec.loader.exec_module(enc)


class Ranker(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 192), nn.GELU(),
            nn.Linear(192, 192), nn.GELU(),
            nn.Linear(192, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "my_team", "ranker_v1.pt"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))

    D, S = [], []
    for f in sorted(glob.glob(os.path.join(HERE, "data3", "shard_*.npz"))):
        d = np.load(f)
        D.append(d["P_deep"])
        S.append(d["P_shallow"])
    Pd = np.concatenate(D)
    Ps = np.concatenate(S)
    Ad = np.stack([enc.augment(r) for r in Pd]).astype(np.float32)
    As = np.stack([enc.augment(r) for r in Ps]).astype(np.float32)
    Xd = np.concatenate([Pd, Ad], axis=1)
    Xs = np.concatenate([Ps, As], axis=1)
    n, dim = Xd.shape
    perm = rng.permutation(n)
    Xd, Xs = Xd[perm], Xs[perm]
    n_val = n // 20
    print(f"{n} pairs, dim {dim}")

    t = lambda a: torch.from_numpy(a)
    Xd_t, Xs_t = t(Xd), t(Xs)

    model = Ranker(dim)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        perm_t = torch.randperm(n - n_val)
        tot, correct = 0.0, 0
        for k in range(0, len(perm_t), args.batch):
            idx = perm_t[k:k + args.batch]
            sd = model(Xd_t[idx])
            ss = model(Xs_t[idx])
            loss = nn.functional.relu(1.0 - (sd - ss)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            correct += int((sd > ss).sum())
        sched.step()
        model.eval()
        with torch.inference_mode():
            vi = torch.arange(n - n_val, n)
            acc = float((model(Xd_t[vi]) > model(Xs_t[vi])).float().mean())
        print(f"epoch {epoch + 1}/{args.epochs} hinge {tot / (n - n_val):.4f} "
              f"train_pair_acc {correct / (n - n_val):.3f} val_pair_acc {acc:.3f} "
              f"({time.time() - t0:.1f}s)", flush=True)

    torch.save(model.state_dict(), args.out)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(dict(dim=dim, pairs=int(n), arch="ranker"), fh)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
