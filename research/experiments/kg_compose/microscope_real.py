"""Microscope as a PREDICTIVE instrument across several real KGs (UMLS, Kinship, Nations).

Claim: the cheap structural profile (esp. algebraicity) PREDICTS whether the algebraic prior helps
composition. We profile each real KG, then confirm with the soft-associativity model: the 'algebraic
benefit' = (path MRR with strong algebra penalty) - (path MRR with no penalty / free). If the
microscope's algebraicity ranks the datasets the same way the measured benefit does, the instrument
has real predictive value -- you can profile a dataset cheaply and know whether PreNat will help.

Kinship's relations literally compose (mother's mother = grandmother), so we expect it MORE algebraic
than UMLS; the microscope should say so, and the algebraic prior should help (or hurt less) there.

Run:  python microscope_real.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eta_sweep"))

import numpy as np
import torch

from umls import load_kg
from kg_models import SoftAlgKGE
from kg_real import build_path_probe, eval_path
from soft_alg import train_mb
from microscope import functionality, invertibility, non_abelianness, algebraicity


def soft_alg_benefit(D, probe, pfilt, d=32, seed=0):
    """path MRR with strong algebra penalty (algebraic) vs no penalty (free)."""
    out = {}
    for tag, pen in (("free(p=0)", 0.0), ("algebraic(p=10)", 10.0)):
        torch.manual_seed(seed)
        m = SoftAlgKGE(D["n_ent"], D["n_rel"], d)
        train_mb(m, D["train"], 120, 256, 0.01, pen)
        out[tag] = eval_path(m, torch.tensor(probe), pfilt)[0]
    return out["free(p=0)"], out["algebraic(p=10)"]


def main():
    print("MICROSCOPE AS A PREDICTIVE INSTRUMENT (real KGs)\n")
    print(f"  {'dataset':9s} | {'func':5s} {'inv':5s} {'nonab':6s} | {'algebraicity':^12s} | "
          f"{'soft-alg: free->alg path MRR':^28s} | benefit")
    print("  " + "-" * 92)
    rows = []
    for name in ("Nations", "Kinship", "UMLS"):
        D = load_kg(name)
        tr = D["train"]
        func = functionality(tr); inv = invertibility(tr, D["n_rel"]); nonab = non_abelianness(tr, D["n_rel"])
        probe, pfilt = build_path_probe(tr)
        _, _, ratio = algebraicity(torch.tensor(tr), D["n_ent"], D["n_rel"], 32,
                                   torch.tensor(probe), pfilt, steps=1000, lr=0.01, seed=0)
        free_p, alg_p = soft_alg_benefit(D, probe, pfilt)
        benefit = alg_p - free_p
        rows.append((name, ratio, benefit))
        na = f"{nonab:.2f}" if nonab == nonab else "n/a"
        print(f"  {name:9s} | {func:4.2f} {inv:4.2f} {na:>6s} | {ratio:^12.2f} | "
              f"{free_p:.3f} -> {alg_p:.3f}{'':18s}| {benefit:+.3f}")

    # predictiveness: does algebraicity rank agree with benefit rank?
    rows.sort(key=lambda x: x[1])
    print(f"\n  datasets by ascending algebraicity: {[r[0] for r in rows]}")
    print(f"  their algebraic benefits (alg-free):  {[round(r[2], 3) for r in rows]}")
    print("  If benefit rises with algebraicity, the cheap microscope profile PREDICTS whether the "
          "algebraic prior helps -- the instrument's payoff. (Kinship, whose relations compose, "
          "should profile MORE algebraic than UMLS.)")


if __name__ == "__main__":
    main()
