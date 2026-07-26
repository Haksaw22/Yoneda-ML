"""Attack the ONE blocker: low-data code-locking -- WITHOUT naturality (which failed 3x).

Run 6b/Run 7: the learned algebra recovers oracle at full data but COLLAPSES at 30-50% (0.07-0.10),
and naturality self-supervision does not rescue it. The diagnosis was entity/state under-
determination + over-parameterisation. So we attack those directly:

  baseline      : learn E + full gamma            (the collapsing model)
  fix-entity    : freeze E to the canonical basis (removes the entity gauge freedom)
  low-rank      : CP-rank structure constants      (fewer algebra params)
  code-sparsity : entropy penalty pushing relation codes toward one-hot (lock onto basis)
  combined      : fix-entity + low-rank + sparsity

Target = the FIXED-algebra oracle at the same data (the data-efficiency we want to recover).

Run:  python codelock.py
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
from kg_models import PreNatLearnedAlgebra, build
from kg_run import train as train_fixed, metrics

VARIANTS = {
    "baseline":      dict(fix_entity=False, rank=None, sparsity=0.0),
    "fix-entity":    dict(fix_entity=True,  rank=None, sparsity=0.0),
    "low-rank":      dict(fix_entity=False, rank=8,    sparsity=0.0),
    "code-sparsity": dict(fix_entity=False, rank=None, sparsity=0.3),
    "combined":      dict(fix_entity=True,  rank=8,    sparsity=0.3),
}


def train_la(model, data, steps, lr, sparsity):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr = data["train"]; h, r, t = tr[:, 0], tr[:, 1], tr[:, 2]
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model.score_all(h, [r]), t) + model.assoc_loss()
        if sparsity:
            loss = loss + sparsity * model.code_entropy_loss()
        loss.backward(); opt.step()


def noncomm_h1(model, data):
    pq = data["pquery"]; nc = pq[:, 4].bool()
    return metrics(model, pq[nc][:, 0], [pq[nc][:, 1], pq[nc][:, 2]], pq[nc][:, 3])[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.5, 0.3])
    args = ap.parse_args()

    print(f"CODE-LOCKING ATTACK on {args.group} KG (NONCOMM path H@1; target = fixed-algebra oracle)"
          f"  seeds={args.seeds}\n")
    cols = ["oracle (fixed)"] + list(VARIANTS)
    print(f"  {'frac':6s} | " + " | ".join(f"{c:^13s}" for c in cols))
    print("  " + "-" * (9 + 16 * len(cols)))
    for frac in args.fracs:
        res = {c: [] for c in cols}
        for seed in range(args.seeds):
            data = make_kg(args.group, 8, frac, seed=seed)
            ne, nr, n = data["n_entities"], data["n_relations"], data["n"]
            torch.manual_seed(seed)
            mo = build("PreNat", data); train_fixed(mo, data, args.steps, args.lr)
            res["oracle (fixed)"].append(noncomm_h1(mo, data))
            for name, cfg in VARIANTS.items():
                torch.manual_seed(seed)
                m = PreNatLearnedAlgebra(ne, nr, n, init="random",
                                         fix_entity=cfg["fix_entity"], rank=cfg["rank"])
                train_la(m, data, args.steps, args.lr, cfg["sparsity"])
                res[name].append(noncomm_h1(m, data))
        cells = [f"{np.mean(res[c]):.3f}+/-{np.std(res[c]):.2f}" for c in cols]
        print(f"  {frac:0.2f}   | " + " | ".join(f"{x:^13s}" for x in cells))

    print("\n  Does any structural fix (esp. fix-entity, which removes the entity gauge) lift the "
          "learned algebra toward the fixed-oracle at low data, where naturality could not? That "
          "would crack the low-data code-locking blocker (and unblock LIMN prereq #1).")


if __name__ == "__main__":
    main()
