"""Non-abelian path-query knowledge graph (external-validation benchmark).

This lifts the eta-sweep idea into the standard KG link-prediction / path-query protocol
(Guu et al. 2015, "Traversing Knowledge Graphs in Vector Space") on a graph whose relation
algebra is a genuine NON-ABELIAN group, so that relation composition is order-sensitive.

World
-----
- A finite group G (default S4, |G|=24). Entities = the group elements (the regular G-set);
  the action of an element g is the permutation h |-> g.h.
- Relations = a chosen subset of `n_relations` non-identity group elements. The atomic triple
  (h, r, t) holds iff t = g_r . h.
- TRAIN on atomic triples only (a `train_frac` subsample of them).
- EVALUATE on:
    (a) atomic link prediction (held-out atomic triples) -- competitiveness sanity check;
    (b) length-2 PATH queries (h, [r1, r2], ?), true tail = g_{r2}.g_{r1}.h -- composition,
        never trained directly;
    (c) the ORDER-SENSITIVE subset of path queries where g_{r2}g_{r1} != g_{r1}g_{r2}.
        Abelian models (TransE/RotatE) cannot distinguish the two orders and must fail here.

Each (h, rel-sequence) has a UNIQUE true tail, so ranking is clean (filtered == raw).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eta_sweep"))

import numpy as np
import torch

from groups import GROUPS


def make_kg(group_name="S4", n_relations=8, train_frac=1.0, n_path=3000, seed=0, device="cpu",
            rel_elems=None):
    G = GROUPS[group_name]()
    rng = np.random.default_rng(seed)
    n = G.n
    entities = np.arange(n)

    # m is a unit (invertible) iff y |-> m.y is a bijection (works for groups and monoids)
    is_unit = np.array([len(set(G.mult[m].tolist())) == n for m in range(n)])

    if rel_elems is None:
        nonid = [g for g in range(n) if g != G.e]
        n_relations = min(n_relations, len(nonid))
        rel_elems = rng.choice(nonid, size=n_relations, replace=False)   # relation -> element
    else:
        rel_elems = np.asarray(rel_elems)
        n_relations = len(rel_elems)

    # atomic triples (h, r, t = g_r . h)
    triples = np.array([(h, r, G.mult[g, h])
                        for r, g in enumerate(rel_elems) for h in entities], dtype=np.int64)
    idx = rng.permutation(len(triples))
    ntest = max(1, int(0.15 * len(triples)))
    test_atomic = triples[idx[:ntest]]
    rest = idx[ntest:]
    ntr = max(1, int(train_frac * len(rest)))
    train = triples[rest[:ntr]]

    # length-2 path queries; composite relation is NEVER a trained atomic relation
    pq, pq_comp_unit = [], []
    for _ in range(n_path):
        r1, r2 = int(rng.integers(0, n_relations)), int(rng.integers(0, n_relations))
        h = int(rng.integers(0, n))
        g12 = G.mult[rel_elems[r2], rel_elems[r1]]          # apply r1 then r2
        t = G.mult[g12, h]
        noncomm = int(G.mult[rel_elems[r1], rel_elems[r2]] != G.mult[rel_elems[r2], rel_elems[r1]])
        pq.append((h, r1, r2, t, noncomm))
        pq_comp_unit.append(bool(is_unit[g12]))             # is the COMPOSITE invertible?
    pquery = np.array(pq, dtype=np.int64)

    rel_units = is_unit[rel_elems]                          # per-relation invertibility
    test_atomic_units = is_unit[rel_elems[test_atomic[:, 1]]]

    def T(x):
        return torch.tensor(x, device=device, dtype=torch.long)

    return dict(
        group=G, group_name=group_name, n_entities=n, n_relations=n_relations,
        rel_elems=rel_elems,
        train=T(train), test_atomic=T(test_atomic), pquery=T(pquery),
        P=torch.tensor(G.regular_rep(), device=device),
        Tc=torch.tensor(G.struct_const(), device=device),
        n=n, device=device,
        frac_noncomm=float(pquery[:, 4].mean()),
        rel_units=np.asarray(rel_units, dtype=bool),
        test_atomic_units=np.asarray(test_atomic_units, dtype=bool),
        pquery_comp_units=np.asarray(pq_comp_unit, dtype=bool),
        n_units=int(is_unit.sum()),
    )
