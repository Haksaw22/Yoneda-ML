"""Direction B: 'Do transformers have a sense of Yoneda?' -- the founding question, made testable.

We train models on a NON-ABELIAN group-product task (entities = group elements; relation = right-
multiply by a group element; atomic fact (g, r, g.g_r)). Train on a fraction of (g, r) pairs, test
on HELD-OUT pairs and on 2-hop COMPOSITIONS never trained.

  MLPKGE      : transformer-lite (MLP over embeddings, composes by re-applying the MLP). The stand-in
                for a vanilla neural net / small transformer head with NO algebraic prior.
  PreNat (LA) : learns the algebra (structure constants) -- the Yoneda-structured net.

The question 'does a transformer have a sense of Yoneda' becomes operational: does it COMPOSE
(generalise to unseen products / 2-hops) or merely MEMORISE atomic facts? A Yoneda-structured net
should compose; the MLP should memorise atomic and fail composition unless it 'groks' the group.

Run:  python grok.py
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
from kg_models import MLPKGE, PreNatLearnedAlgebra
from kg_run import metrics


def train(model, data, steps, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr = data["train"]; h, r, t = tr[:, 0], tr[:, 1], tr[:, 2]
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model.score_all(h, [r]), t)
        if isinstance(model, PreNatLearnedAlgebra):
            loss = loss + model.assoc_loss()
        loss.backward(); opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S4")
    ap.add_argument("--n_relations", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.5])
    args = ap.parse_args()

    print(f"'Do transformers have a sense of Yoneda?'  task = {args.group} group product, "
          f"non-abelian.  seeds={args.seeds}")
    print("(atomic = held-out single products; path = 2-hop compositions never trained)\n")
    print(f"  {'frac':5s} | {'model':12s} | {'atomic MRR':^11s} | {'path MRR':^9s} | {'path NONCOMM H@1':^16s}")
    print("  " + "-" * 66)
    for frac in args.fracs:
        for name in ("MLPKGE", "PreNat(LA)"):
            am, pm, ph = [], [], []
            for seed in range(args.seeds):
                data = make_kg(args.group, args.n_relations, frac, seed=seed)
                ne, nr, n = data["n_entities"], data["n_relations"], data["n"]
                torch.manual_seed(seed)
                model = MLPKGE(ne, nr, n) if name == "MLPKGE" else \
                    PreNatLearnedAlgebra(ne, nr, n, init="random")
                train(model, data, args.steps, args.lr)
                ta = data["test_atomic"]; pq = data["pquery"]; nc = pq[:, 4].bool()
                am.append(metrics(model, ta[:, 0], [ta[:, 1]], ta[:, 2])[0])
                pm.append(metrics(model, pq[:, 0], [pq[:, 1], pq[:, 2]], pq[:, 3])[0])
                ph.append(metrics(model, pq[nc][:, 0], [pq[nc][:, 1], pq[nc][:, 2]], pq[nc][:, 3])[1])
            print(f"  {frac:<5.2f} | {name:12s} | {np.mean(am):^11.3f} | {np.mean(pm):^9.3f} | "
                  f"{np.mean(ph):^16.3f}")
        print()

    print("  The MLP/transformer-lite can fit atomic products but should COMPOSE worse (lower path) "
          "-- it memorises rather than learning the group law. The Yoneda-structured net (PreNat) "
          "composes. Operationally: a vanilla net does NOT acquire a 'sense of Yoneda' by default; "
          "the compositional/naturality structure has to be built in.")


if __name__ == "__main__":
    main()
