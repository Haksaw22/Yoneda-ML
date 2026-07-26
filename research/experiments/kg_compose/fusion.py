"""Fusion: learn the algebra + enforce NATURALITY (composition-consistency) jointly.

Run 6b showed the learned algebra recovers oracle at full data but COLLAPSES at 50% data
(0.096) -- the codes don't lock on from scarce data. Naturality is free self-supervision: the
training graph's own 2-hop paths (h-r1->m-r2->t, both edges observed) must compose correctly.
We add a composition-consistency loss on those training paths (disjoint from the held-out test
composites) and ask: does it let the learned algebra lock on with less data?

This directly attacks the code-locking problem -- the #1 prerequisite for the LIMN moonshot.

Run:  python fusion.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eta_sweep"))

import numpy as np
import torch
import torch.nn.functional as F

from kg_data import make_kg
from kg_models import PreNatLearnedAlgebra
from kg_run import metrics


def make_train_paths(train_arr, exclude, max_p, rng):
    out_edges = defaultdict(list)
    for h, r, t in train_arr:
        out_edges[int(h)].append((int(r), int(t)))
    paths = []
    for h, r1, m in train_arr:
        for r2, t in out_edges[int(m)]:
            if (int(h), int(r1), int(r2)) in exclude:        # no leakage from test composites
                continue
            paths.append((int(h), int(r1), int(r2), int(t)))
    paths = np.array(paths, dtype=np.int64)
    if len(paths) > max_p:
        paths = paths[rng.choice(len(paths), max_p, replace=False)]
    return paths


def train(model, data, train_paths, steps, lr, use_paths):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr = data["train"]; h, r, t = tr[:, 0], tr[:, 1], tr[:, 2]
    if use_paths and len(train_paths):
        tp = torch.tensor(train_paths)
        ph, pr1, pr2, pt = tp[:, 0], tp[:, 1], tp[:, 2], tp[:, 3]
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model.score_all(h, [r]), t) + model.assoc_loss()
        if use_paths and len(train_paths):
            loss = loss + F.cross_entropy(model.score_all(ph, [pr1, pr2]), pt)
        loss.backward()
        opt.step()


def noncomm_h1(model, data):
    pq = data["pquery"]; nc = pq[:, 4].bool()
    return metrics(model, pq[nc][:, 0], [pq[nc][:, 1], pq[nc][:, 2]], pq[nc][:, 3])[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.5, 0.3])
    args = ap.parse_args()

    print(f"Fusion (learn algebra +/- naturality/path-consistency) on {args.group} KG  "
          f"NONCOMM H@1  seeds={args.seeds}\n")
    print(f"  {'frac':6s} | {'atomic-only':^15s} | {'+ naturality':^15s} | n_train_paths")
    print("  " + "-" * 56)
    for frac in args.fracs:
        a_only, a_nat, npaths = [], [], []
        for seed in range(args.seeds):
            data = make_kg(args.group, 8, frac, seed=seed)
            ne, nr, n = data["n_entities"], data["n_relations"], data["n"]
            rng = np.random.default_rng(100 + seed)
            exclude = {(int(x[0]), int(x[1]), int(x[2])) for x in data["pquery"]}
            tpaths = make_train_paths(data["train"].numpy(), exclude, 2000, rng)
            npaths.append(len(tpaths))
            for use_paths, bucket in ((False, a_only), (True, a_nat)):
                torch.manual_seed(seed)
                m = PreNatLearnedAlgebra(ne, nr, n, init="random")
                train(m, data, tpaths, args.steps, args.lr, use_paths)
                bucket.append(noncomm_h1(m, data))
        a_only, a_nat = np.array(a_only), np.array(a_nat)
        print(f"  {frac:0.2f}   | {a_only.mean():5.3f}+/-{a_only.std():4.3f} | "
              f"{a_nat.mean():5.3f}+/-{a_nat.std():4.3f} | {int(np.mean(npaths))}")

    print("\n  Question: does naturality (composition-consistency self-supervision) let the "
          "learned algebra lock on with less data, esp. at the 0.3-0.5 fracs where atomic-only "
          "collapses? If '+naturality' >> 'atomic-only' at low frac, code-locking is helped -- "
          "the key prerequisite for LIMN.")


if __name__ == "__main__":
    main()
