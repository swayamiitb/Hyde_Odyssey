"""P2: distill the ValueNet from self-play data.

Trains on (features, outcome z, teacher squash t): BCE on the win outcome plus
a mild regression toward the teacher's VP-dominant squash (Logistello-style
stabilizer). Saves a state_dict plus a meta json (feature dim, normalization).
CPU training is fine at this size: run under the parity venv.

Usage: python train_value_net.py [--epochs 6] [--out my_team/value_net_v1.pt]
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


class ValueNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "my_team", "value_net_v1.pt"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))

    Xs, Zs, Ts = [], [], []
    for f in sorted(glob.glob(os.path.join(args.data, "shard_*.npz"))):
        d = np.load(f)
        Xs.append(d["X"]); Zs.append(d["Z"]); Ts.append(d["T"])
    X = np.concatenate(Xs); Z = np.concatenate(Zs); T = np.concatenate(Ts)
    n, dim = X.shape
    perm = rng.permutation(n)
    X, Z, T = X[perm], Z[perm], T[perm]
    n_val = max(1, n // 20)
    Xtr, Ztr, Ttr = X[n_val:], Z[n_val:], T[n_val:]
    Xva, Zva, Tva = X[:n_val], Z[:n_val], T[:n_val]
    print(f"{n} rows, dim {dim}, train {len(Xtr)} val {len(Xva)}")

    dev = "cpu"
    tens = lambda a: torch.from_numpy(a).to(dev)
    Xtr_t, Ztr_t, Ttr_t = tens(Xtr), tens(Ztr), tens(Ttr)
    Xva_t, Zva_t, Tva_t = tens(Xva), tens(Zva), tens(Tva)

    model = ValueNet(dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        perm_t = torch.randperm(len(Xtr_t))
        tot = 0.0
        for k in range(0, len(Xtr_t), args.batch):
            idx = perm_t[k:k + args.batch]
            xb, zb, tb = Xtr_t[idx], Ztr_t[idx], Ttr_t[idx]
            opt.zero_grad()
            logit = model(xb)
            loss = bce(logit, zb) + 0.25 * mse(logit, tb.clamp(-6, 6))
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        model.eval()
        with torch.inference_mode():
            lv = model(Xva_t)
            val_bce = float(nn.functional.binary_cross_entropy_with_logits(lv, Zva_t))
            pred = (lv > 0).float()
            acc = float((pred == (Zva_t > 0.5).float()).float().mean())
            # accuracy on decided games only
            mask = (Zva_t != 0.5)
            acc_dec = float((pred[mask] == (Zva_t[mask] > 0.5).float()).float().mean())
        print(f"epoch {epoch + 1}/{args.epochs} train {tot / len(Xtr_t):.4f} "
              f"val_bce {val_bce:.4f} val_acc {acc:.3f} decided_acc {acc_dec:.3f} "
              f"({time.time() - t0:.1f}s)", flush=True)

    torch.save(model.state_dict(), args.out)
    meta = dict(dim=dim, epochs=args.epochs, rows=int(n), seed=args.seed)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(meta, fh)
    print(f"saved {args.out} meta {meta}")


if __name__ == "__main__":
    main()
