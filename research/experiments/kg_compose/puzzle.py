"""End-state B: an algebraic domain where exact composition is genuinely the right prior.

A permutation puzzle = a Cayley graph: states are group elements, moves are a few generators, the
next state is `state . move`. The task is LONG-HORIZON outcome prediction: given a start state and a
move sequence of length h, predict the end state. This is the regime that exposes the real value of
EXACT composition: a model whose composition is algebraically exact has ZERO drift over arbitrarily
long horizons, while a free-matrix model accumulates per-step error that compounds.

  TransE   : additive (abelian)            -> fails immediately (non-abelian moves).
  RESCAL   : free learned matrices         -> good single-step, DRIFTS over long horizons.
  PreNat   : fixed group algebra (exact mu) -> exact composition, NO drift at any horizon.

This is the clean 'algebra is real' win, and it matters for planning / long-horizon reasoning.

Run:  python puzzle.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eta_sweep"))

import numpy as np
import torch

from groups import GROUPS
from kg_data import make_kg
from kg_models import build
from kg_run import train


def rollout_acc(model, world_group, gens, horizon, n_q, seed):
    rng = np.random.default_rng(7000 + seed * 97 + horizon)
    G = world_group
    s = rng.integers(0, G.n, n_q)
    moves = rng.integers(0, len(gens), (n_q, horizon))          # indices into gens
    true = s.copy()
    for k in range(horizon):
        true = G.mult[gens[moves[:, k]], true]                  # generator . state (LEFT, matches make_kg)
    sh = torch.tensor(s)
    rel_list = [torch.tensor(moves[:, k]) for k in range(horizon)]
    with torch.no_grad():
        pred = model.score_all(sh, rel_list).argmax(1).numpy()
    return float((pred == true).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S4")
    ap.add_argument("--n_gens", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 5, 10, 20, 40])
    args = ap.parse_args()

    G = GROUPS[args.group]()
    rng = np.random.default_rng(0)
    gens = rng.choice([g for g in range(G.n) if g != G.e], args.n_gens, replace=False)

    print(f"Permutation puzzle (Cayley graph of {args.group}, {G.n} states, {args.n_gens} moves)")
    print(f"Long-horizon outcome accuracy (predict end state after h moves)  seeds={args.seeds}\n")
    print("  model    | " + " ".join(f"h={h:<5d}" for h in args.horizons))
    print("  " + "-" * (11 + 7 * len(args.horizons)))
    for name in ("TransE", "RESCAL", "PreNat"):
        accs = {h: [] for h in args.horizons}
        for seed in range(args.seeds):
            data = make_kg(args.group, args.n_gens, 1.0, seed=seed, rel_elems=gens)
            torch.manual_seed(seed)
            model = build(name, data)
            train(model, data, args.steps, args.lr)
            for h in args.horizons:
                accs[h].append(rollout_acc(model, G, gens, h, 1000, seed))
        row = " ".join(f"{np.mean(accs[h]):.2f}  " for h in args.horizons)
        print(f"  {name:8s} | {row}")
    print("\n  PreNat (exact group-algebra composition) should hold ~1.0 at ALL horizons; RESCAL "
          "(free matrices) decays as per-step error compounds; TransE (additive) fails on the "
          "non-abelian moves. Exact composition = drift-free long-horizon reasoning.")


if __name__ == "__main__":
    main()
