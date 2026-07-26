"""Learn a NOVEL algebra (remove the candidate-library requirement).

Run 4 discovers the algebra by selecting from a candidate library. Here we instead try to
*learn* the structure constants gamma from scratch (PreNatLearnedAlgebra), with associativity
enforced as a loss. If it works, PreNat needs neither the true group nor a candidate set.

Compared on the non-abelian S4 KG (non-commuting path H@1):
  oracle (fixed)          : PreNat with the TRUE group algebra (upper bound).
  RESCAL (free)           : free per-relation matrices (no algebra; data-hungry).
  learned-algebra (warm)  : learn gamma, warm-started from a cyclic (associative) algebra.
  learned-algebra (random): learn gamma from random init (the Run-2 hard case).

This is the hardest experiment; an honest negative (learning-from-scratch stays hard, selection
remains the practical answer) is a valid and useful outcome.

Run:  python kg_learn_algebra.py
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
from kg_models import build, PreNatLearnedAlgebra
from kg_run import train, metrics, n_params


def train_la(model, data, steps, lr, lam_assoc=1.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr = data["train"]
    h, r, t = tr[:, 0], tr[:, 1], tr[:, 2]
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model.score_all(h, [r]), t) + lam_assoc * model.assoc_loss()
        loss.backward()
        opt.step()
    return float(model.assoc_loss().detach())


def noncomm_h1(model, data):
    pq = data["pquery"]
    nc = pq[:, 4].bool()
    return metrics(model, pq[nc][:, 0], [pq[nc][:, 1], pq[nc][:, 2]], pq[nc][:, 3])[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S4")
    ap.add_argument("--n_relations", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.5])
    args = ap.parse_args()

    print(f"Learn-the-algebra on {args.group} KG  (non-commuting path H@1; assoc = final "
          f"associativity residual of learned gamma)  seeds={args.seeds}\n")
    for frac in args.fracs:
        agg = {k: {"h1": [], "assoc": [], "params": []} for k in
               ("oracle (fixed)", "RESCAL (free)", "learned (warm)", "learned (random)")}
        for seed in range(args.seeds):
            data = make_kg(args.group, args.n_relations, frac, seed=seed)
            ne, nr, n = data["n_entities"], data["n_relations"], data["n"]

            torch.manual_seed(seed)
            m = build("PreNat", data); train(m, data, args.steps, args.lr)
            agg["oracle (fixed)"]["h1"].append(noncomm_h1(m, data))
            agg["oracle (fixed)"]["params"].append(n_params(m))

            torch.manual_seed(seed)
            m = build("RESCAL", data); train(m, data, args.steps, args.lr)
            agg["RESCAL (free)"]["h1"].append(noncomm_h1(m, data))
            agg["RESCAL (free)"]["params"].append(n_params(m))

            for label, init in (("learned (warm)", "cyclic"), ("learned (random)", "random")):
                torch.manual_seed(seed)
                m = PreNatLearnedAlgebra(ne, nr, n, init=init)
                ar = train_la(m, data, args.steps, args.lr)
                agg[label]["h1"].append(noncomm_h1(m, data))
                agg[label]["assoc"].append(ar)
                agg[label]["params"].append(n_params(m))

        print(f"  train_frac = {frac}")
        print(f"    {'model':22s} | {'NONCOMM H@1':^13s} | {'assoc resid':^12s} | params")
        print("    " + "-" * 60)
        for k in agg:
            h1 = np.array(agg[k]["h1"]); pr = int(np.mean(agg[k]["params"]))
            assoc = (f"{np.mean(agg[k]['assoc']):.2e}" if agg[k]["assoc"] else "   (exact)")
            print(f"    {k:22s} | {h1.mean():5.3f}+/-{h1.std():4.3f} | {assoc:^12s} | {pr}")
        print()

    print("  Reading: does 'learned' approach 'oracle' and beat 'RESCAL'? If learned-from-random "
          "fails (low H@1) while warm-start works, a good associative init is the key. If both "
          "fail, learning the algebra from scratch remains open and SELECTION (Run 4) is the "
          "practical route.")


if __name__ == "__main__":
    main()
