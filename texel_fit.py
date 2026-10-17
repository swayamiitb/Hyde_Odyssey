"""Texel-style eval fitting: sigmoid(linear(features)) ~= win probability.

Fits sklearn LogisticRegression on stored self-play (features, outcome) pairs
with derived interaction features from the shared encoder module (single
source of truth with the deployed agent). Saves weights + normalization.

Usage: PYTHONHASHSEED=0 python texel_fit.py [--out my_team/texel_eval.npz]
"""
import argparse
import glob
import importlib.util as ilu
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))

_spec = ilu.spec_from_file_location("enc", os.path.join(HERE, "my_team", "encoder.py"))
enc = ilu.module_from_spec(_spec)
_spec.loader.exec_module(enc)


def augment(X):
    return np.stack([enc.augment(row) for row in X], axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "my_team", "texel_eval.npz"))
    args = ap.parse_args()

    Xs, Zs = [], []
    for f in sorted(glob.glob(os.path.join(args.data, "shard_*.npz"))):
        d = np.load(f)
        Xs.append(d["X"])
        Zs.append(d["Z"])
    X = np.concatenate(Xs)
    Z = np.concatenate(Zs)
    A = augment(X)
    XA = np.concatenate([X, A], axis=1)
    rng = np.random.default_rng(11)
    perm = rng.permutation(len(XA))
    XA, Z = XA[perm], Z[perm]
    n_val = len(XA) // 20
    Xtr, Ztr, Xva, Zva = XA[n_val:], Z[n_val:], XA[:n_val], Z[:n_val]
    print(f"{len(XA)} rows, {XA.shape[1]} features (114 raw + {A.shape[1]} derived)")

    sc = StandardScaler().fit(Xtr)
    model = LogisticRegression(max_iter=400, C=0.5, solver="lbfgs")
    model.fit(sc.transform(Xtr), Ztr)
    p = model.predict_proba(sc.transform(Xva))[:, 1]
    auc = roc_auc_score(Zva, p)
    mask = Zva != 0.5
    acc_dec = float(((p[mask] > 0.5) == (Zva[mask] > 0.5)).mean())
    print(f"val AUC {auc:.4f} decided-acc {acc_dec:.3f}")

    np.savez(args.out,
             w=model.coef_[0].astype(np.float64),
             b=float(model.intercept_[0]),
             mean=sc.mean_.astype(np.float64),
             scale=sc.scale_.astype(np.float64),
             n_raw=114)
    top = np.argsort(-np.abs(model.coef_[0]))[:12]
    print("top |w| features:", [(int(i), round(float(model.coef_[0][i]), 2)) for i in top])


if __name__ == "__main__":
    main()
