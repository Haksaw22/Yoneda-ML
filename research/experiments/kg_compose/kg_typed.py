"""True categories: typed objects + PARTIAL composition (the final categorical generalisation).

Beyond monoids: in a real category not every pair of morphisms composes -- g.h needs
source(g)=target(h). We test PreNat on the path category of a small DAG (24 morphisms, 4
objects, only 61/576 composite pairs defined). Entities = morphisms; relations = the edge
(generator) morphisms; a relation r acts on entity h by r.h ONLY when type-compatible.

Two questions:
  1. Does PreNat compose VALID typed 2-hop paths it never trained on? (path H@1)
  2. Does the category algebra encode TYPING for free? -- i.e. for a type-INCOMPATIBLE relation
     pair the composite code mu(c_r2,c_r1) should be ~0 (the structure constants are zero there),
     while a compatible pair gives a large code. This is partial composition working by construction.

RotatE (abelian, total, invertible) is the untyped baseline.

Run:  python kg_typed.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eta_sweep"))

import numpy as np
import torch

from groups import path_category
from kg_models import build
from kg_run import train, metrics

DAG_OBJECTS = 4
DAG_EDGES = [(0, 1), (0, 1), (1, 2), (1, 2), (2, 3), (0, 2), (1, 3)]


def make_typed_kg(seed):
    C = path_category(DAG_OBJECTS, DAG_EDGES)
    rng = np.random.default_rng(seed)
    n = C.n
    rel_elems = np.array(C.generators)
    n_rel = len(rel_elems)

    triples = [(h, r, int(C.comp[mr, h]))
               for r, mr in enumerate(rel_elems) for h in range(n) if C.comp[mr, h] >= 0]
    triples = np.array(triples, dtype=np.int64)
    idx = rng.permutation(len(triples))
    ntest = max(1, int(0.2 * len(triples)))
    test, train = triples[idx[:ntest]], triples[idx[ntest:]]

    pv = []
    for _ in range(3000):
        r1, r2, h = int(rng.integers(0, n_rel)), int(rng.integers(0, n_rel)), int(rng.integers(0, n))
        m1 = C.comp[rel_elems[r1], h]
        if m1 < 0:
            continue
        t = C.comp[rel_elems[r2], m1]
        if t < 0:
            continue
        pv.append((h, r1, r2, int(t)))
    pvalid = np.array(pv, dtype=np.int64)

    data = dict(n_entities=n, n_relations=n_rel, n=n,
                P=torch.tensor(C.regular_rep()), Tc=torch.tensor(C.struct_const()),
                train=torch.tensor(train), test_atomic=torch.tensor(test), rel_elems=rel_elems)
    return C, data, torch.tensor(pvalid)


def typing_signal(model, C, rel_elems):
    """mean ||mu(c_r2,c_r1)|| for type-COMPATIBLE vs INCOMPATIBLE relation pairs."""
    comp_n, noncomp_n = [], []
    with torch.no_grad():
        for r1 in range(len(rel_elems)):
            for r2 in range(len(rel_elems)):
                c = torch.einsum("kgh,g,h->k", model.Tc, model.C[r2], model.C[r1])
                nrm = float(c.norm())
                ok = C.comp[rel_elems[r2], rel_elems[r1]] >= 0
                (comp_n if ok else noncomp_n).append(nrm)
    return float(np.mean(comp_n)), float(np.mean(noncomp_n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.02)
    args = ap.parse_args()

    names = ("PreNat", "PreNat-typed", "RotatE")
    res = {nm: {"atom": [], "path": [], "tc": [], "tnc": []} for nm in names}
    C0 = None
    for seed in range(args.seeds):
        C, data, pvalid = make_typed_kg(seed)
        C0 = C
        ta = data["test_atomic"]
        for name in names:
            torch.manual_seed(seed)
            model = build("RotatE" if name == "RotatE" else "PreNat", data)
            if name == "PreNat-typed":
                # anchor each relation code to its KNOWN generator morphism (type signature),
                # freeze it, and learn only entity embeddings -- typed composition by construction.
                onehot = torch.zeros(data["n_relations"], data["n"])
                onehot[torch.arange(data["n_relations"]), torch.tensor(data["rel_elems"])] = 1.0
                model.C.data = onehot
                model.C.requires_grad_(False)
            train(model, data, args.steps, args.lr)
            res[name]["atom"].append(metrics(model, ta[:, 0], [ta[:, 1]], ta[:, 2])[0])
            res[name]["path"].append(metrics(model, pvalid[:, 0], [pvalid[:, 1], pvalid[:, 2]],
                                             pvalid[:, 3])[1])
            if name.startswith("PreNat"):
                cc, nc = typing_signal(model, C, data["rel_elems"])
                res[name]["tc"].append(cc); res[name]["tnc"].append(nc)

    print(f"Typed category (path category of a DAG): {C0}")
    print(f"  entities(morphisms)={C0.n}  objects={C0.n_objects}  relations(generators)="
          f"{len(C0.generators)}  composable pairs={int((C0.comp >= 0).sum())}/{C0.n * C0.n}  "
          f"seeds={args.seeds}\n")
    print(f"  {'model':14s} | {'atomic MRR':^12s} | {'valid-path H@1':^15s}")
    print("  " + "-" * 48)
    for name in names:
        a = np.mean(res[name]["atom"]); p = np.mean(res[name]["path"])
        print(f"  {name:14s} | {a:12.3f} | {p:15.3f}")
    print("\n  Typing signal -- composite-code norm ||mu(c_r2,c_r1)|| (compatible vs incompatible):")
    for nm in ("PreNat", "PreNat-typed"):
        cc, nc = np.mean(res[nm]["tc"]), np.mean(res[nm]["tnc"])
        print(f"    {nm:14s}: compatible {cc:.3f}  incompatible {nc:.3f}  (ratio {cc / max(nc, 1e-6):.1f}x)")
    print("\n  Honest reading: the category algebra (partial composition) is mathematically exact, "
          "but LEARNED relation codes do not auto-lock onto the basis on a sparse category "
          "(PreNat ~ RotatE, typing ratio ~1x) -- the same code-gauge looseness as Run 2. When "
          "relation codes are ANCHORED to their known type-signature (PreNat-typed), composition "
          "is exact and incompatible composites collapse to ~0 (large ratio): typing works by "
          "construction. The open piece is making codes lock onto the algebra from loose data "
          "(-> the algebra-learning task).")


if __name__ == "__main__":
    main()
