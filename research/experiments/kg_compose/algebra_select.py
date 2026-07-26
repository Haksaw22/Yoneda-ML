"""Discovering the algebra: turn "PreNat wins IF you know the group" into "PreNat picks it".

Run 3 showed PreNat's win is conditional on being handed the correct non-abelian algebra.
Here we remove that crutch: we DO NOT tell PreNat the group. We enumerate the COMPLETE set of
candidate algebras of the right order -- all five groups of order 8 (C8, C4xC2, C2x2x2 abelian;
D4, Q8 non-abelian) -- train PreNat with each, and SELECT by validation path-query fit.

Selection signal = path-query MRR on a VALIDATION split (disjoint from test). This is standard
model selection; the signal is the training graph's own compositional redundancy (2-hop paths),
not extra supervision. The right algebra composes the graph consistently; wrong ones do not.

Success = the selected algebra is the true data group, and selected-PreNat's TEST performance
matches the oracle while abelian candidates are rejected. Default data group D4; try Q8 to show
it distinguishes the two *non-abelian* groups (not just abelian vs non-abelian).

Run:  python algebra_select.py                 (data = D4)
      python algebra_select.py --data_group Q8
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eta_sweep"))

import numpy as np
import torch

from groups import GROUPS, ORDER8
from kg_data import make_kg
from kg_models import build
from kg_run import train, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_group", default="D4", choices=["D4", "Q8"])
    ap.add_argument("--n_relations", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.02)
    args = ap.parse_args()

    cands = ORDER8
    val_mrr = {c: [] for c in cands}
    test_nc = {c: [] for c in cands}
    atom = {c: [] for c in cands}
    selected = []

    for seed in range(args.seeds):
        data = make_kg(args.data_group, args.n_relations, train_frac=1.0, seed=seed)
        pq = data["pquery"]
        nval = len(pq) // 2
        val, test = pq[:nval], pq[nval:]
        ta = data["test_atomic"]
        ncm = test[:, 4].bool()
        per_seed_val = {}
        for c in cands:
            Gc = GROUPS[c]()
            algebra = {"P": torch.tensor(Gc.regular_rep()), "Tc": torch.tensor(Gc.struct_const())}
            torch.manual_seed(seed)
            model = build("PreNat", data, algebra=algebra)
            train(model, data, args.steps, args.lr)
            vm = metrics(model, val[:, 0], [val[:, 1], val[:, 2]], val[:, 3])[0]   # val path MRR
            per_seed_val[c] = vm
            val_mrr[c].append(vm)
            test_nc[c].append(metrics(model, test[ncm][:, 0], [test[ncm][:, 1], test[ncm][:, 2]],
                                      test[ncm][:, 3])[1])
            atom[c].append(metrics(model, ta[:, 0], [ta[:, 1]], ta[:, 2])[0])
        selected.append(max(per_seed_val, key=per_seed_val.get))

    n_correct = sum(s == args.data_group for s in selected)
    print(f"Algebra discovery  |  data group = {args.data_group} (hidden from the model)")
    print(f"Candidates = all 5 groups of order 8.  Selection = validation path-query MRR.  "
          f"seeds={args.seeds}\n")
    print(f"  {'candidate':10s} {'kind':12s} | {'val path MRR':^14s} | {'test NONCOMM H@1':^16s} | "
          f"{'atomic MRR':^11s}")
    print("  " + "-" * 70)
    for c in cands:
        Gc = GROUPS[c]()
        kind = "abelian" if Gc.is_abelian() else "NON-abelian"
        star = " *TRUE*" if c == args.data_group else ""
        vm, vs = np.mean(val_mrr[c]), np.std(val_mrr[c])
        print(f"  {c:10s} {kind:12s} | {vm:6.3f}+/-{vs:4.3f}  | {np.mean(test_nc[c]):16.3f} | "
              f"{np.mean(atom[c]):11.3f}{star}")
    print(f"\n  Selected by validation: {selected}  ->  correct {n_correct}/{args.seeds}")
    if n_correct == args.seeds:
        print(f"  SUCCESS: validation always picks the true algebra ({args.data_group}); PreNat "
              "discovers the group it was never told.")
    print("\n  Note: atomic MRR barely separates the candidates (single edges don't reveal "
          "non-commutativity); the validation path-query signal is what discriminates.")


if __name__ == "__main__":
    main()
