"""Models compared in the eta-sweep.

All models share the SAME faithful decoder  o_hat = W . rho(code) . F_i , so they all
have equal capacity to fit any individual observation. They differ only in HOW a
hom-code is composed/transported, which is exactly the variable under test:

  PreNat (soft)  : per-(probe, object) free codes; composition = the fixed non-abelian
                   group algebra (exact `mu`); naturality is a SOFT learned penalty.
  PreNat (hard)  : ONE reference code per object; all probe codes are DERIVED by exact
                   composition -> naturality holds by construction (the sheaf/CENN-style
                   hard-wired baseline). Correctly specified at eta=0.
  Abelian        : faithful decoder, but transport is ADDITIVE (commutative, TransE/RotatE
                   class). Cannot represent non-abelian transport consistently across
                   objects -> a fair "abelian fails" baseline.

`no-nat` is not a class: it is PreNat(soft) trained with lambda_nat = lambda_cycle = 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---- group-algebra ops (exact, associative, differentiable) -------------------

def rho(codes, P):
    """rho(c) = sum_g c_g P_g.  codes [..., n], P [n, n, n] -> [..., n, n]."""
    return torch.einsum("...g,gxy->...xy", codes, P)


def compose(a, b, T):
    """Group-algebra product mu(a, b)_k = sum_{g.h=k} a_g b_h.  [..., n] x [..., n]."""
    return torch.einsum("kgh,...g,...h->...k", T, a, b)


def _decode(codes, F_rows, world, rep=None):
    """o_hat = W . rho(codes) . F_rows.  codes [..., n], F_rows broadcastable [..., n].

    `rep` is the n x n x n representation tensor; defaults to the fixed regular rep
    world["P"]. The learned-rho model passes its own learned R."""
    R = rho(codes, world["P"] if rep is None else rep)  # [..., n, n]
    RF = torch.einsum("...xy,...y->...x", R, F_rows)    # [..., n]
    return torch.einsum("dx,...x->...d", world["W"], RF)


def robust(sq, kind="l2", delta=0.5):
    """Per-element robust transform of a squared residual `sq` (>= 0).

    Corrupted (eta) observations create large naturality residuals; an L2 penalty
    lets those outliers drag the whole object's codes. A robust estimator caps their
    influence so the clean majority dominates.
      l2      : identity (squared error)            -- baseline.
      huber   : quadratic <= delta, linear beyond   -- bounded gradient.
      welsch  : c^2/2 * (1 - exp(-sq/c^2)), c=delta -- redescending (outliers ~ignored).
    """
    if kind == "l2":
        return sq
    if kind == "huber":
        r = torch.sqrt(sq + 1e-12)
        return torch.where(r <= delta, 0.5 * sq, delta * (r - 0.5 * delta))
    if kind == "welsch":
        c2 = delta * delta
        return 0.5 * c2 * (1.0 - torch.exp(-sq / c2))
    raise ValueError(kind)


# ---- models -------------------------------------------------------------------

class PreNat(nn.Module):
    def __init__(self, world, mode="soft", init_scale=0.1, nat_robust="l2", nat_delta=0.5):
        super().__init__()
        assert mode in ("soft", "hard")
        self.world = world
        self.mode = mode
        self.nat_robust = nat_robust          # "l2" | "huber" | "welsch"
        self.nat_delta = nat_delta
        P, N, n = world["n_probes"], world["n_objects"], world["n"]
        if mode == "soft":
            self.H = nn.Parameter(init_scale * torch.randn(P, N, n))
        else:  # hard: only a reference (probe-0 / identity) code per object
            self.A = nn.Parameter(init_scale * torch.randn(N, n))

    def codes(self):
        """All hom-codes H[i, A], shape [P, N, n]."""
        if self.mode == "soft":
            return self.H
        P, N, n = self.world["n_probes"], self.world["n_objects"], self.world["n"]
        a = self.A[None, :, :].expand(P, N, n)                  # reference code a_A
        b = self.world["Ponehot"][:, None, :].expand(P, N, n)   # onehot(p_i)
        return compose(a, b, self.world["T"])                   # exact: code of a_A . p_i

    def predict_grid(self):
        """Predicted observation at every (i, A): [P, N, d_obs]."""
        H = self.codes()
        return _decode(H, self.world["F"][:, None, :], self.world)

    def transport_predict(self, A_idx, k_idx, i_idx):
        """Predict o(k, A) by transporting a code from observed probe i to held-out k.

        h_k = mu(h_{i,A}, t_{i->k})   (exact non-abelian composition).
        Returns (o_hat [M, d_obs], h_k [M, n])."""
        H = self.codes()
        h_i = H[i_idx, A_idx]                                   # [M, n]
        tcode = self.world["Tcode"][i_idx, k_idx]              # [M, n]
        h_k = compose(h_i, tcode, self.world["T"])             # [M, n]
        o = _decode(h_k, self.world["F"][k_idx], self.world)   # [M, d_obs]
        return o, h_k

    def nat_loss(self, observed):
        """Soft (NAT): for observed (i, A) and (j, A), mu(h_i, t_{ij}) == h_j.

        Vectorised, diagonal (i==j) excluded, with a robust estimator on the per-pair
        squared residual so eta-corrupted pairs cannot drag the clean consensus."""
        if self.mode == "hard":
            return torch.zeros((), device=self.world["F"].device)
        H = self.codes()                                       # [P, N, n]
        T, Tcode = self.world["T"], self.world["Tcode"]        # Tcode [P, P, n]
        obs = torch.as_tensor(observed, device=H.device)       # [P, N] bool
        P = self.world["n_probes"]
        # trans[i, j, A, k] = compose(H[i, A], t_{ij})[k]
        trans = torch.einsum("kgh,iAg,ijh->ijAk", T, H, Tcode)  # [P, P, N, n]
        sq = ((trans - H[None, :, :, :]) ** 2).sum(-1)          # vs H[j, A]; [P, P, N]
        mask = obs[:, None, :] & obs[None, :, :]                # both i, j observed
        eye = torch.eye(P, dtype=torch.bool, device=H.device)
        mask = mask & ~eye[:, :, None]                          # drop i == j
        loss = robust(sq, self.nat_robust, self.nat_delta)
        return (loss * mask).sum() / mask.sum().clamp(min=1)

    def cycle_loss(self, cycle_triples):
        """Two distinct observed transport paths to a held-out (k, A) must agree."""
        if not cycle_triples or self.mode == "hard":
            return torch.zeros((), device=self.world["F"].device)
        dev = self.world["F"].device
        A = torch.tensor([t[0] for t in cycle_triples], device=dev)
        k = torch.tensor([t[1] for t in cycle_triples], device=dev)
        i1 = torch.tensor([t[2] for t in cycle_triples], device=dev)
        i2 = torch.tensor([t[3] for t in cycle_triples], device=dev)
        _, h1 = self.transport_predict(A, k, i1)
        _, h2 = self.transport_predict(A, k, i2)
        sq = ((h1 - h2) ** 2).sum(-1)                          # [M]
        return robust(sq, self.nat_robust, self.nat_delta).mean()


class Abelian(nn.Module):
    """Faithful decoder + additive (commutative) transport -- TransE/RotatE-class."""

    def __init__(self, world, init_scale=0.1):
        super().__init__()
        self.world = world
        self.mode = "abelian"
        P, N, n = world["n_probes"], world["n_objects"], world["n"]
        self.H = nn.Parameter(init_scale * torch.randn(P, N, n))   # codes (faithful decode)
        self.r = nn.Parameter(init_scale * torch.randn(P, P, n))   # learned additive transports

    def codes(self):
        return self.H

    def predict_grid(self):
        return _decode(self.H, self.world["F"][:, None, :], self.world)

    def transport_predict(self, A_idx, k_idx, i_idx):
        h_i = self.H[i_idx, A_idx]                 # [M, n]
        h_k = h_i + self.r[i_idx, k_idx]           # commutative transport
        o = _decode(h_k, self.world["F"][k_idx], self.world)
        return o, h_k

    def nat_loss(self, observed):
        H = self.H
        P = self.world["n_probes"]
        obs = torch.as_tensor(observed, device=H.device)
        total = H.new_zeros(())
        count = 0
        for i in range(P):
            trans = H[i][None] + self.r[i][:, None, :]    # [P, N, n]
            mask = (obs[i][None, :] & obs)
            total = total + (((trans - H) ** 2) * mask.unsqueeze(-1)).sum()
            count += int(mask.sum().item())
        return total / max(count, 1)

    def cycle_loss(self, cycle_triples):
        if not cycle_triples:
            return torch.zeros((), device=self.world["F"].device)
        dev = self.world["F"].device
        A = torch.tensor([t[0] for t in cycle_triples], device=dev)
        k = torch.tensor([t[1] for t in cycle_triples], device=dev)
        i1 = torch.tensor([t[2] for t in cycle_triples], device=dev)
        i2 = torch.tensor([t[3] for t in cycle_triples], device=dev)
        _, h1 = self.transport_predict(A, k, i1)
        _, h2 = self.transport_predict(A, k, i2)
        return ((h1 - h2) ** 2).mean()


class PreNatLearnedRho(nn.Module):
    """PreNat with a LEARNED representation rho (the R_g matrices) instead of the fixed
    regular rep. The abstract algebra (structure constants T) is still fixed, so code-space
    composition `mu` stays exact -- but the model must LEARN a faithful matrix representation
    of it. This re-introduces the collapse failure modes the fixed-rho toy excluded:

      L_homo : rho must be an algebra homomorphism, R_g R_h = R_{g.h}  (else transporting in
               code space does not agree with decoding).
      L_comm : non-commutativity floor -- without it the model can fit observations with an
               abelian / non-faithful rho and silently lose composition (the categorical
               analogue of posterior collapse).
      L_recon: Yoneda autoencoding -- reconstruct a held-out probe's response from another
               probe via the learned rho; a metric-free representability signal that a
               collapsed (constant / abelian) rho cannot satisfy.
    """

    def __init__(self, world, init_scale=0.1, rep_init="random",
                 nat_robust="l2", nat_delta=0.5, comm_tau=None):
        super().__init__()
        self.world = world
        self.mode = "learned-rho"
        P, N, n = world["n_probes"], world["n_objects"], world["n"]
        self.H = nn.Parameter(init_scale * torch.randn(P, N, n))
        if rep_init == "perturbed":              # start near the true rep (easy mode)
            R0 = world["P"].clone() + 0.3 * torch.randn(n, n, n)
        else:                                    # "random": generic start, NOT the answer
            R0 = torch.randn(n, n, n) / (n ** 0.5)
        self.R = nn.Parameter(R0)
        self.nat_robust, self.nat_delta = nat_robust, nat_delta
        self._mult = torch.as_tensor(world["group"].mult, device=world["F"].device)
        # commutativity floor: a fraction of the TRUE regular rep's scale (a hyperparameter)
        if comm_tau is None:
            Pm = world["P"]
            comm_true = torch.linalg.matrix_norm(
                torch.einsum("gxy,hyz->ghxz", Pm, Pm)
                - torch.einsum("hxy,gyz->ghxz", Pm, Pm)).mean()
            comm_tau = 0.5 * float(comm_true)
        self.comm_tau = comm_tau

    def codes(self):
        return self.H

    def predict_grid(self):
        return _decode(self.H, self.world["F"][:, None, :], self.world, rep=self.R)

    def transport_predict(self, A_idx, k_idx, i_idx):
        h_i = self.H[i_idx, A_idx]
        tcode = self.world["Tcode"][i_idx, k_idx]
        h_k = compose(h_i, tcode, self.world["T"])
        o = _decode(h_k, self.world["F"][k_idx], self.world, rep=self.R)
        return o, h_k

    def nat_loss(self, observed):
        H = self.codes()
        T, Tcode = self.world["T"], self.world["Tcode"]
        obs = torch.as_tensor(observed, device=H.device)
        P = self.world["n_probes"]
        trans = torch.einsum("kgh,iAg,ijh->ijAk", T, H, Tcode)
        sq = ((trans - H[None]) ** 2).sum(-1)
        mask = obs[:, None, :] & obs[None, :, :]
        eye = torch.eye(P, dtype=torch.bool, device=H.device)
        mask = mask & ~eye[:, :, None]
        return (robust(sq, self.nat_robust, self.nat_delta) * mask).sum() / mask.sum().clamp(min=1)

    def cycle_loss(self, cycle_triples):
        if not cycle_triples:
            return torch.zeros((), device=self.world["F"].device)
        dev = self.world["F"].device
        A = torch.tensor([t[0] for t in cycle_triples], device=dev)
        k = torch.tensor([t[1] for t in cycle_triples], device=dev)
        i1 = torch.tensor([t[2] for t in cycle_triples], device=dev)
        i2 = torch.tensor([t[3] for t in cycle_triples], device=dev)
        _, h1 = self.transport_predict(A, k, i1)
        _, h2 = self.transport_predict(A, k, i2)
        sq = ((h1 - h2) ** 2).sum(-1)
        return robust(sq, self.nat_robust, self.nat_delta).mean()

    # ---- learned-rho-specific structural losses ----
    def homo_loss(self):
        """R_g R_h == R_{g.h}: rho is an algebra homomorphism (makes transport meaningful)."""
        R = self.R
        prod = torch.einsum("gxy,hyz->ghxz", R, R)
        return ((prod - R[self._mult]) ** 2).mean()

    def comm_loss(self):
        """Hinge floor on mean non-commutativity -- blocks abelian / non-faithful collapse."""
        R = self.R
        comm = torch.einsum("gxy,hyz->ghxz", R, R) - torch.einsum("hxy,gyz->ghxz", R, R)
        c = torch.linalg.matrix_norm(comm).mean()
        return torch.clamp(torch.as_tensor(self.comm_tau, device=R.device) - c, min=0.0)

    def recon_loss(self, recon, O_obs):
        """Yoneda autoencoding: predict probe `tgt`'s response by transporting from `src`."""
        o_pred, _ = self.transport_predict(recon["A"], recon["tgt"], recon["src"])
        return ((o_pred - O_obs[recon["tgt"], recon["A"]]) ** 2).mean()

    @torch.no_grad()
    def diagnostics(self):
        R = self.R
        comm = torch.einsum("gxy,hyz->ghxz", R, R) - torch.einsum("hxy,gyz->ghxz", R, R)
        homo = torch.einsum("gxy,hyz->ghxz", R, R) - R[self._mult]
        n = R.shape[0]
        pd = torch.linalg.matrix_norm(R[:, None] - R[None, :]) + torch.eye(n, device=R.device) * 1e9
        return dict(
            comm_use=float(torch.linalg.matrix_norm(comm).mean()),
            comm_tau=float(self.comm_tau),
            homo_err=float((homo ** 2).mean()),
            faith_min=float(pd.min()),       # min distance between distinct R_g (0 => collapsed)
        )


# variant -> robust kind for the naturality/cycle losses
_VARIANT_ROBUST = {"soft": "l2", "no-nat": "l2", "soft-huber": "huber", "soft-welsch": "welsch"}


def build_model(variant, world, cfg):
    init_scale = cfg.get("init_scale", 0.1)
    delta = cfg.get("nat_delta", 0.5)
    if variant in _VARIANT_ROBUST:
        return PreNat(world, mode="soft", init_scale=init_scale,
                      nat_robust=_VARIANT_ROBUST[variant], nat_delta=delta)
    if variant == "hard":
        return PreNat(world, mode="hard", init_scale=init_scale)
    if variant == "abelian":
        return Abelian(world, init_scale=init_scale)
    raise ValueError(variant)
