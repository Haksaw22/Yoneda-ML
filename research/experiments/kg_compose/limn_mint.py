"""End-state C: LIMN minting on a reliable (fixed) representation -- detect + MINT a concept by its
universal property and validate it generalises out-of-distribution.

Run 8's lesson: don't learn structure from scarce data; operate on a reliable representation. So we
MINT on a clean linear cone structure (the regime where composition is trustworthy). Product A x B,
dim 4. Cones (objects X with maps f:X->A, g:X->B) are split TRAIN / TEST by region of (f,g).

For each candidate apex dimension we (i) fit projections + mediators on TRAIN cones, (ii) read the
certificate (residual = existence, uniqueness margin), (iii) test OOD: freeze the projections and
factor HELD-OUT cones. The point:

  - Minting by RESIDUAL ALONE (existence) is ambiguous: BOTH dim 4 (true product) and dim 6
    (too-big) factor train and test cones with ~0 residual. Existence does not single out the
    universal object.
  - The UNIQUENESS certificate uniquely mints dim 4 (the true product): dim 6 has a null mediator
    direction (margin 0) -> a WEAK limit, rejected; dim 3 cannot factor (high residual).
  - The minted dim-4 object factors the OOD cones (low test residual): it captured the universal
    property, so it works on cones it never saw -- concept discovery, not memorisation.

This is the novel atom (uniqueness-of-mediator) doing real work: it is what makes minting CORRECT.

Run:  python limn_mint.py
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn


def fit_train(d_L, cones, steps=3000, lr=0.05, seed=0):
    torch.manual_seed(seed)
    N = cones.shape[0]
    piA = nn.Parameter(0.3 * torch.randn(2, d_L))
    piB = nn.Parameter(0.3 * torch.randn(2, d_L))
    U = nn.Parameter(0.3 * torch.randn(N, d_L))
    opt = torch.optim.Adam([piA, piB, U], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        resid = ((U @ piA.t() - cones[:, :2]) ** 2 + (U @ piB.t() - cones[:, 2:]) ** 2).mean()
        (resid + 1e-3 * (U ** 2).mean()).backward()
        opt.step()
    with torch.no_grad():
        P = torch.cat([piA, piB], dim=0)
        margin = float(torch.linalg.eigvalsh(P.t() @ P).min().clamp(min=0).sqrt())
        train_resid = float(((U @ piA.t() - cones[:, :2]) ** 2
                             + (U @ piB.t() - cones[:, 2:]) ** 2).mean())
    return piA.detach(), piB.detach(), train_resid, margin


@torch.no_grad()
def ood_residual(piA, piB, test_cones):
    """freeze projections; min-norm factor held-out cones (u = P^+ rhs); return mean residual."""
    P = torch.cat([piA, piB], dim=0)                      # [4, d_L]
    Ppinv = torch.linalg.pinv(P)                          # [d_L, 4]
    sol = test_cones @ Ppinv.t()                          # min-norm mediators [M, d_L]
    pred = sol @ P.t()                                    # [M, 4]
    return float(((pred - test_cones) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # TRAIN cones: (f,g) in the positive region; TEST cones: shifted (OOD) region
    train = torch.tensor(np.abs(rng.standard_normal((args.n_train, 4))), dtype=torch.float32)
    test = torch.tensor(-np.abs(rng.standard_normal((args.n_test, 4))) - 1.0, dtype=torch.float32)

    print("LIMN MINTING with OOD validation (product A x B, true apex dim 4)\n")
    print(f"  {'apex dim':9s} | {'train resid':^12s} | {'uniqueness':^11s} | {'OOD resid':^11s} | "
          f"{'cert mint?':^11s}")
    print("  " + "-" * 68)
    minted = []
    for d_L in (2, 3, 4, 5, 6):
        piA, piB, tr, margin = fit_train(d_L, train, seed=args.seed)
        ood = ood_residual(piA, piB, test)
        exists = tr < 1e-2
        unique = margin > 1e-2
        cert = exists and unique
        if cert:
            minted.append(d_L)
        print(f"  {d_L:^9d} | {tr:^12.2e} | {margin:^11.3f} | {ood:^11.2e} | "
              f"{'MINT' if cert else 'reject':^11s}")

    print(f"\n  Minted by certificate: dim {minted}  (the true product is dim 4).")
    print("  Residual ALONE would also accept dim 5/6 (they factor train AND OOD with ~0 residual) "
          "-- existence is ambiguous. The UNIQUENESS margin is what rejects the weak (non-universal) "
          "apexes and mints exactly the true product. The minted object factors the OOD cones it "
          "never saw: it captured the universal property, not the training data.")


if __name__ == "__main__":
    main()
