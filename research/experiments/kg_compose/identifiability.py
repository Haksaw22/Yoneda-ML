"""Direction C: identifiability -- atomic data cannot identify the algebra; composition can.

Run 4 hinted at it; here we make it crisp. Train every candidate algebra on ATOMIC triples only,
then measure how well atomic vs composition (path) performance DISCRIMINATES the true algebra.

Claim (informal identifiability theorem): the hom-functor restricted to single morphisms (atomic
link prediction) does NOT determine the composition law -- many algebras fit the atomic data
equally. Composition (2-hops) is what pins it down. This is a Yoneda statement: structure is
revealed by relationships UNDER COMPOSITION, not pointwise.

We report, per data group, the spread (max - min across candidates) of atomic MRR vs path MRR.
A small atomic spread + large path spread = "atomic non-identifiable, composition identifiable."

Run:  python identifiability.py
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
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.02)
    args = ap.parse_args()

    print("IDENTIFIABILITY: can ATOMIC-only data identify the algebra?  (candidates = 5 order-8 "
          f"groups)  seeds={args.seeds}\n")
    print(f"  {'data group':10s} | {'atomic MRR (per candidate)':^40s} | spread")
    print("  " + "-" * 70)
    for dg in ("D4", "Q8"):
        atom_spreads, path_spreads = [], []
        atom_rows = {c: [] for c in ORDER8}
        path_rows = {c: [] for c in ORDER8}
        for seed in range(args.seeds):
            data = make_kg(dg, 6, 1.0, seed=seed)
            ta = data["test_atomic"]; pq = data["pquery"]; nc = pq[:, 4].bool()
            atoms, paths = {}, {}
            for c in ORDER8:
                Gc = GROUPS[c]()
                alg = {"P": torch.tensor(Gc.regular_rep()), "Tc": torch.tensor(Gc.struct_const())}
                torch.manual_seed(seed)
                m = build("PreNat", data, algebra=alg)
                train(m, data, args.steps, args.lr)
                atoms[c] = metrics(m, ta[:, 0], [ta[:, 1]], ta[:, 2])[0]
                paths[c] = metrics(m, pq[nc][:, 0], [pq[nc][:, 1], pq[nc][:, 2]], pq[nc][:, 3])[1]
                atom_rows[c].append(atoms[c]); path_rows[c].append(paths[c])
            atom_spreads.append(max(atoms.values()) - min(atoms.values()))
            path_spreads.append(max(paths.values()) - min(paths.values()))
        amean = {c: np.mean(atom_rows[c]) for c in ORDER8}
        astr = " ".join(f"{c.split('x')[0]}:{amean[c]:.2f}" for c in ORDER8)
        print(f"  {dg:10s} | {astr:40s} | atomic {np.mean(atom_spreads):.2f}  "
              f"path {np.mean(path_spreads):.2f}")

    print("\n  If atomic spread << path spread, atomic-only data is (near-)non-identifiable for the "
          "algebra while composition identifies it -- the Yoneda point, made quantitative. (Note "
          "the true group's atomic MRR is high but so are several wrong candidates'; only the path "
          "column singles out the truth.)")


if __name__ == "__main__":
    main()
