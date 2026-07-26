"""The near-functorial eta-sweep synthetic world.

The decisive question for PreNat is: does a *soft, learned* naturality penalty beat
a *hard-wired* one? That has a determinate answer only when the data is APPROXIMATELY
but not EXACTLY functorial -- because on exactly-functorial data hard-wiring is
correctly specified and should tie. So this world has a functoriality-violation knob
`eta`.

The world (ground truth)
------------------------
- A finite (non-abelian) group G with faithful regular rep P_g (n x n).
- `n_probes` probe objects b_i, each carrying a fixed group element p_i (p_0 = identity)
  and a fixed functor value F_i in R^n.
- `n_objects` data objects A, each with a latent group element a_A.
- The TRUE hom element from probe i into object A is  g(i, A) = a_A . p_i.
  Hence the whole hom-profile of A is determined by the single element a_A plus the
  fixed probe morphisms -- this is exact functoriality / naturality:
        g(j, A) = g(i, A) . (p_i^{-1} p_j)   for all A.        (NAT)
- The known probe-to-probe transport code is  t_{ij} = onehot(p_i^{-1} p_j).

What the model sees (deliberately under-determined)
---------------------------------------------------
- An observation at (i, A) is  o(i, A) = W . P_{g(i,A)} . F_i  in R^{d_obs}, with
  `d_obs < n`. A SINGLE probe is therefore insufficient to recover g(i, A); the model
  must fuse evidence across probes -- which is exactly the work naturality does.
- eta-corruption: an `eta` fraction of OBSERVED entries have g(i, A) replaced by a
  random element before rendering, so they are inconsistent with any single a_A. The
  model is not told which are corrupt.

Splits
------
- Each object is observed on `n_obs_probes` random probes.
- The held-out test set is the (k, A) pairs whose probe k was NOT observed for A;
  predicting them requires composing a seen hom-code with a known transport (NON-abelian).
- Test targets use the CLEAN (uncorrupted) g(k, A): we test recovery of true structure.
"""

from __future__ import annotations

import numpy as np
import torch


def make_world(group, n_objects, n_probes, d_obs, seed, device="cpu"):
    rng = np.random.default_rng(seed)
    n = group.n
    P = group.regular_rep()            # [n, n, n]
    T = group.struct_const()           # [n, n, n]

    p_elem = rng.integers(0, n, size=n_probes)
    p_elem[0] = group.e                # reference probe carries the identity
    a_elem = rng.integers(0, n, size=n_objects)

    # g_true[i, A] = a_A . p_i
    g_true = group.mult[a_elem[None, :], p_elem[:, None]]      # [n_probes, n_objects]

    F = rng.standard_normal((n_probes, n)).astype(np.float32)
    W = (rng.standard_normal((d_obs, n)) / np.sqrt(n)).astype(np.float32)

    # transport codes t_{ij} = onehot(p_i^{-1} p_j)   (known categorical scaffold)
    pinv = group.inv[p_elem]                                   # [P]
    t_elem = group.mult[pinv[:, None], p_elem[None, :]]        # [P, P]
    Tcode = np.zeros((n_probes, n_probes, n), dtype=np.float32)
    for i in range(n_probes):
        for j in range(n_probes):
            Tcode[i, j, t_elem[i, j]] = 1.0

    Ponehot = np.zeros((n_probes, n), dtype=np.float32)        # onehot(p_i)
    Ponehot[np.arange(n_probes), p_elem] = 1.0

    def t(x):
        return torch.tensor(x, device=device)

    return dict(
        group=group, n=n, n_probes=n_probes, n_objects=n_objects, d_obs=d_obs,
        device=device,
        P=t(P), T=t(T), F=t(F), W=t(W), Tcode=t(Tcode), Ponehot=t(Ponehot),
        p_elem=p_elem, a_elem=a_elem, g_true=g_true, e=group.e,
    )


def make_splits(world, n_obs_probes, seed):
    rng = np.random.default_rng(seed + 1)
    P, N = world["n_probes"], world["n_objects"]
    assert 2 <= n_obs_probes < P, "need >=2 observed probes and >=1 held-out probe"
    observed = np.zeros((P, N), dtype=bool)
    for A in range(N):
        probes = rng.choice(P, size=n_obs_probes, replace=False)
        observed[probes, A] = True

    test_pairs = [(k, A) for A in range(N) for k in range(P) if not observed[k, A]]

    # cycle triples: a held-out (k, A) reachable from two distinct observed probes.
    # L_cycle forces the two transport paths to agree -- supervision in the held-out
    # region itself, computable without test labels.
    cycle_triples = []
    for (k, A) in test_pairs:
        obs_i = np.where(observed[:, A])[0]
        if len(obs_i) >= 2:
            i1, i2 = rng.choice(obs_i, size=2, replace=False)
            cycle_triples.append((A, k, int(i1), int(i2)))

    return dict(observed=observed, test_pairs=test_pairs, cycle_triples=cycle_triples)


def _render(world, g_grid):
    """O[i, A] = W . P_{g} . F_i  for an element grid g_grid [P, N]."""
    P, N, n = world["n_probes"], world["n_objects"], world["n"]
    g = torch.tensor(np.asarray(g_grid), dtype=torch.long, device=world["device"])
    Pg = world["P"][g]                                   # [P, N, n, n]
    PgF = torch.einsum("pAxy,py->pAx", Pg, world["F"])   # P_g F_i  -> [P, N, n]
    return torch.einsum("dx,pAx->pAd", world["W"], PgF)  # [P, N, d_obs]


def make_observations(world, splits, eta, seed):
    rng = np.random.default_rng(seed + 7)
    P, N, n = world["n_probes"], world["n_objects"], world["n"]
    observed = splits["observed"]
    g_obs = world["g_true"].copy()
    corrupt = np.zeros((P, N), dtype=bool)
    if eta > 0:
        draw = rng.random((P, N))
        rand_elem = rng.integers(0, n, size=(P, N))
        flip = observed & (draw < eta)
        g_obs[flip] = rand_elem[flip]
        corrupt = flip
    return dict(
        O_obs=_render(world, g_obs),                 # corrupted training targets
        O_clean=_render(world, world["g_true"]),     # clean targets for held-out eval
        corrupt=corrupt,
        n_corrupt=int(corrupt.sum()),
    )
