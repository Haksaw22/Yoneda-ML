"""LIMN seed: the uniqueness-of-mediator certificate (universal-property concept minting).

The long-horizon moonshot is minting NEW latent objects defined by a universal property. Its
defensible novel atom is the **uniqueness-of-mediator certificate** that distinguishes a TRUE
limit from a WEAK one. This is a focused prototype of exactly that mechanism (not the full system).

Setup (product object A x B). A = R^2, B = R^2. A 'cone' is an object X with maps f: X->A, g: X->B,
observed as the pair (f_x, g_x). The product L = A x B is the universal object: every cone factors
through L by a UNIQUE mediator u with pi_A(u)=f, pi_B(u)=g.

We test three candidate apexes by LEARNING their projections to factor the cones, then certifying:
  - existence : factorization residual (can a mediator reproduce every cone?).
  - uniqueness: smallest eigenvalue of P^T P where P = [pi_A; pi_B]  (a null direction => two
                different mediators give the same (f,g) => NOT a true limit, only a weak one).

Expected:
  d_L = 4 (= dimA+dimB) : low residual + high uniqueness margin  -> ACCEPT (true product).
  d_L = 6 (too big)     : low residual + ~0 uniqueness margin    -> REJECT (weak limit, non-unique).
  d_L = 3 (too small)   : high residual                          -> REJECT (no mediator exists).

Run:  python limn.py
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn


def fit_candidate(d_L, cones, steps=3000, lr=0.05, seed=0):
    torch.manual_seed(seed)
    N = cones.shape[0]
    piA = nn.Parameter(0.3 * torch.randn(2, d_L))
    piB = nn.Parameter(0.3 * torch.randn(2, d_L))
    U = nn.Parameter(0.3 * torch.randn(N, d_L))               # per-cone mediator
    opt = torch.optim.Adam([piA, piB, U], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        fa = U @ piA.t()                                       # pi_A(u)  [N,2]
        gb = U @ piB.t()                                       # pi_B(u)  [N,2]
        resid = ((fa - cones[:, :2]) ** 2 + (gb - cones[:, 2:]) ** 2).mean()
        loss = resid + 1e-3 * (U ** 2).mean()                 # min-norm mediator (uniqueness pressure)
        loss.backward(); opt.step()
    with torch.no_grad():
        P = torch.cat([piA, piB], dim=0)                      # [4, d_L]
        eig = torch.linalg.eigvalsh(P.t() @ P)               # d_L eigenvalues
        uniq_margin = float(eig.min().clamp(min=0).sqrt())   # 0 if a null direction exists
        final_resid = float(((U @ piA.t() - cones[:, :2]) ** 2
                             + (U @ piB.t() - cones[:, 2:]) ** 2).mean())
    return final_resid, uniq_margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cones", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cones = torch.tensor(rng.standard_normal((args.n_cones, 4)), dtype=torch.float32)

    print("LIMN uniqueness-of-mediator certificate  (product A x B, dimA=dimB=2 -> true apex dim 4)\n")
    print(f"  {'candidate apex':22s} | {'factor residual':^16s} | {'uniqueness margin':^17s} | verdict")
    print("  " + "-" * 78)
    cases = [(4, "TRUE product (dim 4)"), (6, "too big (dim 6)"), (3, "too small (dim 3)")]
    for d_L, label in cases:
        resid, margin = fit_candidate(d_L, cones, seed=args.seed)
        exists = resid < 1e-2
        unique = margin > 1e-2
        verdict = ("ACCEPT (true limit)" if exists and unique
                   else "REJECT: weak limit (non-unique)" if exists and not unique
                   else "REJECT: no mediator")
        print(f"  {label:22s} | {resid:^16.2e} | {margin:^17.3f} | {verdict}")

    print("\n  The certificate accepts the true product (mediator exists AND is unique) and rejects "
          "both a too-big apex (mediator exists but is non-unique = a WEAK limit) and a too-small "
          "one (no mediator). This uniqueness test -- not mere factorization -- is the novel atom "
          "LIMN would use to mint objects by universal property. (Seed prototype, linear case.)")


if __name__ == "__main__":
    main()
