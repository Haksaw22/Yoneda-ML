"""Algebra-MISMATCH control: PreNat's win is conditional on the RIGHT non-abelian prior.

PreNat scored perfectly on the main benchmark partly because it was handed the exact group that
generated the data (a "home advantage"). This control shows that advantage is not circular magic:
give PreNat the WRONG algebra and it fails like an abelian model.

Data: D4 (dihedral, order 8, NON-abelian).
  - PreNat-correct : composes in D4's group algebra            -> should ace non-abelian composition.
  - PreNat-wrong   : composes in C8's group algebra (ABELIAN)  -> should FAIL it, like RotatE.
  - RotatE         : abelian reference.

If PreNat-wrong collapses to RotatE-level on NONCOMM H@1 while PreNat-correct stays high, the
contribution is precisely the framework for imposing the RIGHT non-abelian structure -- and the
open problem (selecting / learning that algebra) is the learned-rho experiment in ../eta_sweep/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eta_sweep"))

import numpy as np
import torch

from groups import cyclic_group
from kg_data import make_kg
from kg_models import build
from kg_run import train, metrics, n_params

SEEDS, STEPS, LR = 3, 3000, 0.02


def main():
    # wrong (abelian) algebra of the same order as D4 (=8): C8 regular rep + structure constants
    C8 = cyclic_group(8)
    wrong = dict(P=torch.tensor(C8.regular_rep()), Tc=torch.tensor(C8.struct_const()))

    variants = ["RotatE", "PreNat", "PreNat-wrong"]
    agg = {v: {"nc_h1": [], "c_h1": [], "atom_mrr": []} for v in variants}
    meta = {}
    for seed in range(SEEDS):
        data = make_kg("D4", n_relations=6, train_frac=1.0, seed=seed)
        meta = data
        pq = data["pquery"]; ph, pr1, pr2, pt, nc = (pq[:, i] for i in range(5))
        ncm = nc.bool()
        ta = data["test_atomic"]
        for v in variants:
            torch.manual_seed(seed)
            algebra = wrong if v == "PreNat-wrong" else None
            model = build("PreNat" if v.startswith("PreNat") else v, data, algebra=algebra)
            train(model, data, STEPS, LR)
            agg[v]["atom_mrr"].append(metrics(model, ta[:, 0], [ta[:, 1]], ta[:, 2])[0])
            agg[v]["nc_h1"].append(metrics(model, ph[ncm], [pr1[ncm], pr2[ncm]], pt[ncm])[1])
            agg[v]["c_h1"].append(metrics(model, ph[~ncm], [pr1[~ncm], pr2[~ncm]], pt[~ncm])[1])

    print(f"Algebra-mismatch control  |  data group = D4 (non-abelian, order 8)  "
          f"non-commuting path frac={meta['frac_noncomm']:.2f}  seeds={SEEDS}")
    print(f"  {'variant':14s} | {'atomic MRR':^12s} | {'NONCOMM H@1':^13s} | {'comm H@1':^11s}")
    print("  " + "-" * 56)
    for v in variants:
        am = np.mean(agg[v]["atom_mrr"]); ncm = np.mean(agg[v]["nc_h1"]); cm = np.mean(agg[v]["c_h1"])
        print(f"  {v:14s} | {am:12.3f} | {ncm:13.3f} | {cm:11.3f}")
    print("\nExpected: PreNat (correct D4 algebra) high on NONCOMM; PreNat-wrong (C8/abelian) and "
          "RotatE near-floor on NONCOMM. Confirms the win requires the right non-abelian prior.")


if __name__ == "__main__":
    main()
