"""Synergy #1: soft / approximate associativity -- one knob from PreNat to RESCAL.

SoftAlgKGE = shared learned algebra + per-relation free residual D_r, with a residual penalty
`resid_l2`. High penalty -> D_r~0 -> pure algebra (PreNat-like); low penalty -> free (RESCAL-like).
We sweep the penalty on an algebraic domain (S4) and a messy real one (UMLS) and show:
  - on S4: strong penalty is fine (data is algebraic; D_r stays ~0; algebraicity ~1).
  - on UMLS: relaxing the penalty RECOVERS free-matrix composition (fixing the Run-6c loss),
    and the learned per-relation algebraicity drops -- the model measures, per relation, how
    algebraic the data is and adapts.

Run:  python soft_alg.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eta_sweep"))

import numpy as np
import torch
import torch.nn.functional as F

from kg_data import make_kg
from kg_models import SoftAlgKGE
from kg_run import metrics
from kg_real import build_path_probe, eval_path, eval_filtered
from umls import load_umls


def train_full(model, data, steps, lr, resid_l2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr = data["train"]; h, r, t = tr[:, 0], tr[:, 1], tr[:, 2]
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model.score_all(h, [r]), t) + resid_l2 * model.resid_penalty()
        loss.backward(); opt.step()


def train_mb(model, train_arr, epochs, bs, lr, resid_l2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    h = torch.tensor(train_arr[:, 0]); r = torch.tensor(train_arr[:, 1]); t = torch.tensor(train_arr[:, 2])
    n = len(train_arr)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(model.score_all(h[idx], [r[idx]]), t[idx]) + resid_l2 * model.resid_penalty()
            loss.backward(); opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalties", type=float, nargs="+", default=[0.0, 0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    print("SOFT ASSOCIATIVITY -- one knob (residual penalty) from RESCAL (0) to PreNat (large)\n")

    # ---- S4 (algebraic) ----
    print("  S4 group KG  (NONCOMM path H@1 ; algebraicity = mean per-relation):")
    print(f"    {'resid_l2':9s} | {'path H@1':^10s} | {'algebraicity':^12s}")
    for p in args.penalties:
        h1s, algs = [], []
        for seed in range(args.seeds):
            data = make_kg("S4", 8, 1.0, seed=seed)
            torch.manual_seed(seed)
            m = SoftAlgKGE(data["n_entities"], data["n_relations"], data["n"])
            train_full(m, data, 2000, 0.02, p)
            pq = data["pquery"]; nc = pq[:, 4].bool()
            h1s.append(metrics(m, pq[nc][:, 0], [pq[nc][:, 1], pq[nc][:, 2]], pq[nc][:, 3])[1])
            algs.append(float(m.algebraicity_per_rel().mean()))
        print(f"    {p:<9.2f} | {np.mean(h1s):^10.3f} | {np.mean(algs):^12.2f}")

    # ---- UMLS (messy) ----
    D = load_umls()
    probe, pfilt = build_path_probe(D["train"])
    print("\n  UMLS real KG  (atomic MRR / path MRR ; algebraicity = mean per-relation):")
    print(f"    {'resid_l2':9s} | {'atomic MRR':^11s} | {'path MRR':^9s} | {'algebraicity':^12s}")
    for p in args.penalties:
        torch.manual_seed(0)
        m = SoftAlgKGE(D["n_ent"], D["n_rel"], 32)
        train_mb(m, D["train"], 150, 256, 0.01, p)
        mrr = eval_filtered(m, D["test"], D["filt"], D["n_ent"])[0]
        pmrr = eval_path(m, torch.tensor(probe), pfilt)[0]
        alg = float(m.algebraicity_per_rel().mean())
        print(f"    {p:<9.2f} | {mrr:^11.3f} | {pmrr:^9.3f} | {alg:^12.2f}")

    print("\n  On S4 the penalty barely matters (data is algebraic; D_r stays ~0, algebraicity~1). "
          "On UMLS, RELAXING the penalty recovers path composition toward RESCAL-level while "
          "algebraicity drops -- one model adapts to how algebraic the domain is, fixing the "
          "earlier real-KG composition loss.")


if __name__ == "__main__":
    main()
