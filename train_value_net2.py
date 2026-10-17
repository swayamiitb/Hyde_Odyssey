"""Round-2 ValueNet training: VPDR-shaped targets + hidden-VP aux head + EMA.

Target: y = 0.35*z + 0.65*VPDR  (VP_DISCOUNTED_RETURN carries speed-to-win;
z is the pure outcome). Sample weight grows with game progress (later
positions have less noisy outcomes). Aux head predicts the opponent's
unplayed VP-card count from public features (self-play ground truth).
Ships EMA weights.

Usage: PYTHONHASHSEED=0 .venv/Scripts/python.exe train_value_net2.py [--round2]
Reads data/ shards (H, VPDR, TN columns required).
"""
import argparse
import copy
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("enc", os.path.join(HERE, "my_team", "encoder.py"))
enc = _ilu.module_from_spec(_spec); _spec.loader.exec_module(enc)


class ValueNet2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
        )
        self.value = nn.Linear(256, 1)
        self.aux_vp = nn.Linear(256, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.value(h).squeeze(-1), self.aux_vp(h).squeeze(-1)


def load(round2):
    Xs, Zs, Ts, Hs, Vs, TNs = [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "shard_*.npz"))):
        d = np.load(f)
        Xs.append(d["X"]); Zs.append(d["Z"]); Ts.append(d["T"])
        Hs.append(d["H"] if "H" in d else np.zeros(len(d["X"]), dtype=np.float32))
        Vs.append(d["VPDR"] if "VPDR" in d else d["Z"])
        TNs.append(d["TN"] if "TN" in d else np.full(len(d["X"]), 60.0, dtype=np.float32))
    return (np.concatenate(Xs), np.concatenate(Zs), np.concatenate(Ts),
            np.concatenate(Hs), np.concatenate(Vs), np.concatenate(TNs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "my_team", "value_net_v2.pt"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))

    X, Z, T, H, VPDR, TN = load(args)
    A = np.stack([enc.augment(row) for row in X], axis=0).astype(np.float32)
    XA = np.concatenate([X, A], axis=1)
    y = 0.35 * Z + 0.65 * VPDR
    w = 0.5 + 0.5 * np.minimum(TN / 60.0, 1.0)

    n, dim = XA.shape
    perm = rng.permutation(n)
    XA, y, w, Z, H = XA[perm], y[perm], w[perm], Z[perm], H[perm]
    n_val = n // 20
    splits = dict(Xtr=XA[n_val:], ytr=y[n_val:], wtr=w[n_val:], Ztr=Z[n_val:], Htr=H[n_val:],
                  Xva=XA[:n_val], yva=y[:n_val], Zva=Z[:n_val], Hva=H[:n_val])
    print(f"{n} rows, dim {dim} (114 + {A.shape[1]})")

    dev = "cpu"
    t = lambda a: torch.from_numpy(a).to(dev)
    Xtr, ytr, wtr = t(splits["Xtr"]), t(splits["ytr"]).float(), t(splits["wtr"]).float()
    Ztr_t, Htr_t = t(splits["Ztr"]).float(), t(splits["Htr"]).float()
    Xva, yva, Zva_t, Hva_t = t(splits["Xva"]), t(splits["yva"]).float(), t(splits["Zva"]).float(), t(splits["Hva"]).float()

    model = ValueNet2(dim).to(dev)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mse = nn.MSELoss(reduction="none")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        perm_t = torch.randperm(len(Xtr))
        tot = 0.0
        for k in range(0, len(Xtr), args.batch):
            idx = perm_t[k:k + args.batch]
            xb, yb, wb = Xtr[idx], ytr[idx], wtr[idx]
            zb, hb, tb = Ztr_t[idx], Htr_t[idx], None
            opt.zero_grad()
            logit, aux = model(xb)
            loss_v = (bce(logit, yb) * wb).mean()
            loss_aux = mse(aux, hb).mean()
            loss = loss_v + 0.3 * loss_aux
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        with torch.no_grad():   # EMA update
            d_ema = ema.state_dict()
            b_ = 0.995
            for k_, v_ in model.state_dict().items():
                d_ema[k_].mul_(b_).add_(v_, alpha=1 - b_)
        model.eval()
        ema.eval()
        with torch.inference_mode():
            for tag, m in (("train-model", model), ("ema", ema)):
                lv, auxv = m(Xva)
                p = torch.sigmoid(lv)
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(splits["Zva"], p.numpy())
                mae = float((auxv - Hva_t).abs().mean())
                if tag == "ema":
                    print(f"epoch {epoch + 1}/{args.epochs} loss {tot / len(Xtr):.4f} "
                          f"[ema] val_AUC {auc:.4f} auxVP_MAE {mae:.3f} ({time.time() - t0:.1f}s)",
                          flush=True)

    torch.save(ema.state_dict(), args.out)
    meta = dict(dim=dim, epochs=args.epochs, rows=int(n), seed=args.seed, arch="v2_aux")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(meta, fh)
    print(f"saved {args.out} meta {meta}")


if __name__ == "__main__":
    main()
