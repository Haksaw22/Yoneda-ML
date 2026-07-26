"""KG scorers compared on non-abelian path composition.

All models share the SAME interface and the SAME final scoring (negative L2 distance from a
composed query embedding to every entity embedding), so the only thing that varies is HOW a
relation sequence is composed onto the head -- which is exactly what we are testing:

  TransE   : relation = translation; compose = ADD            (commutative) -> fails non-abelian.
  RotatE   : relation = rotation (phase); compose = ADD phase  (commutative) -> fails non-abelian.
  RESCAL   : relation = FREE matrix; compose = matmul          (non-abelian capable, many params).
  PreNat   : relation = code in a FIXED group algebra; compose = exact group-algebra product mu;
             rho fixed to the regular rep                      (non-abelian, few params, exact).

`score_all(h_idx, rel_list)` returns [B, n_entities]; rel_list is a list of [B] relation-index
tensors (one per hop). Atomic query: rel_list=[r]. Path query: rel_list=[r1, r2].
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _neg_dist_to_all(q, E):
    """ -||q - E_t|| for every entity t.  q [B,d], E [n_ent,d] -> [B,n_ent]."""
    return -torch.cdist(q, E)


class TransE(nn.Module):
    def __init__(self, n_ent, n_rel, d, **_):
        super().__init__()
        self.E = nn.Embedding(n_ent, d)
        self.R = nn.Embedding(n_rel, d)
        nn.init.normal_(self.E.weight, std=0.1)
        nn.init.normal_(self.R.weight, std=0.1)

    def query(self, h_idx, rel_list):
        q = self.E(h_idx)
        for r in rel_list:
            q = q + self.R(r)
        return q

    def score_all(self, h_idx, rel_list):
        return _neg_dist_to_all(self.query(h_idx, rel_list), self.E.weight)


class RotatE(nn.Module):
    def __init__(self, n_ent, n_rel, d, **_):
        super().__init__()
        assert d % 2 == 0
        self.dh = d // 2
        self.E = nn.Embedding(n_ent, d)                     # [re | im]
        self.phase = nn.Embedding(n_rel, self.dh)
        nn.init.normal_(self.E.weight, std=0.1)
        nn.init.uniform_(self.phase.weight, 0, 6.2831853)

    def _q(self, h_idx, rel_list):
        e = self.E(h_idx)
        re, im = e[..., :self.dh], e[..., self.dh:]
        for r in rel_list:
            ph = self.phase(r)
            c, s = torch.cos(ph), torch.sin(ph)
            re, im = re * c - im * s, re * s + im * c
        return torch.cat([re, im], dim=-1)

    def score_all(self, h_idx, rel_list):
        return _neg_dist_to_all(self._q(h_idx, rel_list), self.E.weight)


class RESCAL(nn.Module):
    """Free per-relation matrix (non-abelian capable; d*d params per relation)."""

    def __init__(self, n_ent, n_rel, d, **_):
        super().__init__()
        self.E = nn.Embedding(n_ent, d)
        M = torch.eye(d).unsqueeze(0).repeat(n_rel, 1, 1) + 0.05 * torch.randn(n_rel, d, d)
        self.M = nn.Parameter(M)
        nn.init.normal_(self.E.weight, std=0.1)

    def query(self, h_idx, rel_list):
        q = self.E(h_idx)
        for r in rel_list:
            q = torch.einsum("bij,bj->bi", self.M[r], q)
        return q

    def score_all(self, h_idx, rel_list):
        return _neg_dist_to_all(self.query(h_idx, rel_list), self.E.weight)


class PreNatKGE(nn.Module):
    """Relation = code in a FIXED non-abelian group algebra; composition is EXACT mu;
    rho fixed to the regular rep. d == |G| == n."""

    def __init__(self, n_ent, n_rel, d, P=None, Tc=None, **_):
        super().__init__()
        n = P.shape[0]
        assert d == n, "PreNatKGE uses d = |G|"
        self.E = nn.Embedding(n_ent, n)
        self.C = nn.Parameter(0.1 * torch.randn(n_rel, n))   # relation codes (group-algebra)
        self.register_buffer("P", P)                          # [n,n,n] regular rep
        self.register_buffer("Tc", Tc)                        # [n,n,n] structure constants
        nn.init.normal_(self.E.weight, std=0.1)

    def _compose_codes(self, rel_list):
        # rel_list = [r1, r2, ...] applied in order -> combined code mu(c_last, mu(..., c_1))
        c = self.C[rel_list[0]]
        for r in rel_list[1:]:
            c = torch.einsum("kgh,bg,bh->bk", self.Tc, self.C[r], c)   # mu(c_r, c_accum)
        return c

    def query(self, h_idx, rel_list):
        c = self._compose_codes(rel_list)                     # [B, n]
        R = torch.einsum("bg,gxy->bxy", c, self.P)            # rho(c) [B,n,n]
        return torch.einsum("bxy,by->bx", R, self.E(h_idx))

    def score_all(self, h_idx, rel_list):
        return _neg_dist_to_all(self.query(h_idx, rel_list), self.E.weight)


class PreNatLearnedAlgebra(nn.Module):
    """LEARN the algebra (structure constants gamma) instead of selecting it from a library.

    The model parameterises a d_alg-dimensional associative algebra by its structure constants
    gamma[p,k,l] (e_k . e_l = sum_p gamma[p,k,l] e_p). Composition mu and the regular
    representation rho both derive from gamma:
        mu(a,b)_p = sum_{k,l} gamma[p,k,l] a_k b_l
        rho(a)_{p,l} = sum_k a_k gamma[p,k,l]            (left-multiplication matrix)
    Relations are codes a_r; entities are vectors e. Score = -||rho(a_r) e_h - e_t||.
    Associativity of gamma is enforced as a loss; everything else is the same PreNat scoring.
    This removes the candidate-library requirement IF the algebra can be learned from data.

    init: 'random' (hard, from scratch) or 'cyclic' (warm-start from an associative C_{d_alg}
    algebra, then deform)."""

    def __init__(self, n_ent, n_rel, d_alg, init="cyclic", fix_entity=False, rank=None, **_):
        super().__init__()
        self.d_alg = d_alg
        self.fix_entity = fix_entity
        self.rank = rank
        # entity embeddings: fixed canonical basis removes the entity gauge freedom (code-locking)
        if fix_entity:
            self.register_buffer("Efix", torch.eye(n_ent, d_alg))
        else:
            self.E = nn.Embedding(n_ent, d_alg)
            nn.init.normal_(self.E.weight, std=0.1)
        self.A = nn.Parameter(0.1 * torch.randn(n_rel, d_alg))
        if rank is not None:                                # low-rank (CP) structure constants
            self.gU = nn.Parameter(0.1 * torch.randn(d_alg, rank))
            self.gK = nn.Parameter(0.1 * torch.randn(d_alg, rank))
            self.gL = nn.Parameter(0.1 * torch.randn(d_alg, rank))
        elif init == "cyclic":
            g = torch.zeros(d_alg, d_alg, d_alg)
            for k in range(d_alg):
                for l in range(d_alg):
                    g[(k + l) % d_alg, k, l] = 1.0          # C_{d_alg} structure constants
            self.gamma = nn.Parameter(g + 0.05 * torch.randn(d_alg, d_alg, d_alg))
        else:
            self.gamma = nn.Parameter(0.1 * torch.randn(d_alg, d_alg, d_alg))

    def _gamma(self):
        if self.rank is not None:
            return torch.einsum("pr,kr,lr->pkl", self.gU, self.gK, self.gL)
        return self.gamma

    def ent_emb(self):
        return self.Efix if self.fix_entity else self.E.weight

    def _mu(self, a, b):
        return torch.einsum("pkl,...k,...l->...p", self._gamma(), a, b)

    def query(self, h_idx, rel_list):
        a = self.A[rel_list[0]]
        for r in rel_list[1:]:
            a = self._mu(self.A[r], a)                      # mu(a_r, a_accum): r after accum
        R = torch.einsum("bk,pkl->bpl", a, self._gamma())   # rho(a)
        return torch.einsum("bpl,bl->bp", R, self.ent_emb()[h_idx])

    def score_all(self, h_idx, rel_list):
        return _neg_dist_to_all(self.query(h_idx, rel_list), self.ent_emb())

    def assoc_loss(self, n=128):
        g = self._gamma()
        z = torch.randn(n, 3, self.d_alg, device=g.device)
        mu = lambda a, b: torch.einsum("pkl,bk,bl->bp", g, a, b)
        return ((mu(mu(z[:, 0], z[:, 1]), z[:, 2]) - mu(z[:, 0], mu(z[:, 1], z[:, 2]))) ** 2).mean()

    def code_entropy_loss(self):
        """Push relation codes toward sparse/one-hot (lock onto basis directions)."""
        p = torch.softmax(self.A.abs(), dim=-1)
        return -(p * (p + 1e-9).log()).sum(-1).mean()


class SoftAlgKGE(nn.Module):
    """Soft / approximate associativity: a SHARED learned algebra plus a PER-RELATION free residual.

        M_r = (shared algebra part)  +  D_r
            = sum_k a_{r,k} B_k       +  D_r

    A large residual penalty `resid_l2` drives D_r -> 0 (pure shared algebra ~ PreNat); a small one
    lets D_r dominate (free per-relation matrices ~ RESCAL). So one knob interpolates the two
    extremes, and ||D_r|| / ||M_r|| per relation is a measured *degree of non-algebraicity* -- the
    fix for the real-KG loss (relax associativity exactly where the data demands it)."""

    def __init__(self, n_ent, n_rel, d, d_alg=None, **_):
        super().__init__()
        d_alg = d if d_alg is None else d_alg
        self.E = nn.Embedding(n_ent, d)
        self.B = nn.Parameter(0.1 * torch.randn(d_alg, d, d))     # shared algebra basis
        self.A = nn.Parameter(0.1 * torch.randn(n_rel, d_alg))    # relation codes
        self.Dr = nn.Parameter(0.01 * torch.randn(n_rel, d, d))   # per-relation free residual
        nn.init.normal_(self.E.weight, std=0.1)

    def M(self, r):
        return torch.einsum("bk,kxy->bxy", self.A[r], self.B) + self.Dr[r]

    def query(self, h_idx, rel_list):
        q = self.E(h_idx)
        for r in rel_list:
            q = torch.einsum("bxy,by->bx", self.M(r), q)
        return q

    def score_all(self, h_idx, rel_list):
        return _neg_dist_to_all(self.query(h_idx, rel_list), self.E.weight)

    def resid_penalty(self):
        return (self.Dr ** 2).mean()

    @torch.no_grad()
    def algebraicity_per_rel(self):
        shared = torch.einsum("rk,kxy->rxy", self.A, self.B)
        dn = self.Dr.flatten(1).norm(dim=1)
        mn = (shared + self.Dr).flatten(1).norm(dim=1)
        return (1.0 - dn / (mn + 1e-6)).clamp(0, 1)               # 1 = fully algebraic relation


class MLPKGE(nn.Module):
    """Transformer-lite baseline: predict the tail from (head, relation) with an MLP over
    embeddings, composing a path by applying the MLP per hop. It has NO algebraic/compositional
    prior -- a stand-in for what a vanilla neural net (or a small transformer head) does. The test
    'does a transformer have a sense of Yoneda?' becomes: does this *compose* (generalise to unseen
    products) or merely memorise atomic facts?"""

    def __init__(self, n_ent, n_rel, d, hidden=128, **_):
        super().__init__()
        self.E = nn.Embedding(n_ent, d)
        self.R = nn.Embedding(n_rel, d)
        self.mlp = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, d))
        nn.init.normal_(self.E.weight, std=0.1)
        nn.init.normal_(self.R.weight, std=0.1)

    def query(self, h_idx, rel_list):
        q = self.E(h_idx)
        for r in rel_list:
            q = self.mlp(torch.cat([q, self.R(r)], dim=-1))
        return q

    def score_all(self, h_idx, rel_list):
        return _neg_dist_to_all(self.query(h_idx, rel_list), self.E.weight)


def build(name, data, d=None, algebra=None):
    """`algebra`, if given, is a dict {P, Tc} overriding the group PreNat composes in
    (used for the algebra-MISMATCH control: a wrong/abelian algebra on non-abelian data)."""
    n_ent, n_rel, n = data["n_entities"], data["n_relations"], data["n"]
    d = n if d is None else d
    if name == "TransE":
        return TransE(n_ent, n_rel, d)
    if name == "RotatE":
        return RotatE(n_ent, n_rel, d if d % 2 == 0 else d + 1)
    if name == "RESCAL":
        return RESCAL(n_ent, n_rel, d)
    if name in ("PreNat", "PreNat-wrong"):
        alg = algebra or data
        return PreNatKGE(n_ent, n_rel, n, P=alg["P"], Tc=alg["Tc"])
    raise ValueError(name)
