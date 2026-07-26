"""Finite groups -> the exact, non-abelian associative algebra PreNat composes in.

A finite group G gives us, for free, a *genuine* associative (and for S_k>=3,
non-abelian) algebra: its group algebra R[G] under the regular representation.
PreNat fixes its composition `mu` to this algebra, so associativity / identity /
the rho-homomorphism law hold as algebraic identities, never as losses.

Conventions
-----------
- Elements are indexed 0..n-1; `mult[a, b]` is the index of a . b (a after b).
- `regular_rep()[g]` is the n x n permutation matrix P_g with P_g e_y = e_{g.y},
  i.e. P_g[x, y] = 1 iff g.y = x. Then P_g P_h = P_{g.h} (a homomorphism).
- `struct_const()[k, g, h] = 1 iff g.h = k`; this is the tensor that makes the
  group-algebra product a single einsum.
"""

from __future__ import annotations

import itertools

import numpy as np


class FiniteGroup:
    def __init__(self, mult, inv, labels, name):
        self.mult = np.asarray(mult, dtype=np.int64)  # [n, n]
        self.inv = np.asarray(inv, dtype=np.int64)    # [n]
        self.labels = list(labels)
        self.name = name
        self.n = int(self.mult.shape[0])
        assert self.mult.shape == (self.n, self.n)
        # identity: the g with g.h = h for all h
        self.e = next(
            g for g in range(self.n)
            if all(self.mult[g, h] == h for h in range(self.n))
        )

    def is_abelian(self) -> bool:
        return bool(np.array_equal(self.mult, self.mult.T))

    def regular_rep(self) -> np.ndarray:
        """P[g][x, y] = 1 iff g.y = x. Faithful, exact, (non-)abelian as G is."""
        n = self.n
        P = np.zeros((n, n, n), dtype=np.float32)
        for g in range(n):
            for y in range(n):
                P[g, self.mult[g, y], y] = 1.0
        return P

    def struct_const(self) -> np.ndarray:
        """T[k, g, h] = 1 iff g.h = k. Group-algebra product = einsum over this."""
        n = self.n
        T = np.zeros((n, n, n), dtype=np.float32)
        for g in range(n):
            for h in range(n):
                T[self.mult[g, h], g, h] = 1.0
        return T

    def __repr__(self):
        return f"FiniteGroup({self.name}, n={self.n}, abelian={self.is_abelian()})"


class FiniteMonoid:
    """A finite monoid (associative, two-sided identity) -- elements need NOT be invertible.

    Same regular-rep / structure-constant machinery as FiniteGroup, but P_m can be a
    non-invertible (collapsing) 0/1 matrix when m has no inverse. This is the genuinely
    categorical generalisation: relations like is-a / part-of / causes are non-invertible."""

    def __init__(self, mult, labels, name):
        self.mult = np.asarray(mult, dtype=np.int64)
        self.n = int(self.mult.shape[0])
        self.labels = list(labels)
        self.name = name
        self.e = next(g for g in range(self.n)
                      if all(self.mult[g, h] == h and self.mult[h, g] == h for h in range(self.n)))
        # m is a unit iff y |-> m.y is a bijection (surjective <=> bijective, finite)
        self.is_unit = np.array([len(set(self.mult[m].tolist())) == self.n for m in range(self.n)])

    def is_abelian(self) -> bool:
        return bool(np.array_equal(self.mult, self.mult.T))

    def regular_rep(self) -> np.ndarray:
        n = self.n
        P = np.zeros((n, n, n), dtype=np.float32)
        for g in range(n):
            for y in range(n):
                P[g, self.mult[g, y], y] = 1.0      # P_g e_y = e_{g.y}; columns may collide
        return P

    def struct_const(self) -> np.ndarray:
        n = self.n
        T = np.zeros((n, n, n), dtype=np.float32)
        for g in range(n):
            for h in range(n):
                T[self.mult[g, h], g, h] = 1.0
        return T

    def __repr__(self):
        u = int(self.is_unit.sum())
        return (f"FiniteMonoid({self.name}, n={self.n}, units={u}, "
                f"non-invertible={self.n - u}, abelian={self.is_abelian()})")


def full_transformation_monoid(k: int) -> FiniteMonoid:
    """T_k: all functions [k]->[k] under composition (f.g)(x)=f(g(x)). |T_k| = k^k.

    The canonical non-invertible, non-abelian monoid; its units are exactly S_k. T3 has 27
    elements (6 invertible = S3, 21 non-invertible)."""
    funcs = list(itertools.product(range(k), repeat=k))
    index = {f: i for i, f in enumerate(funcs)}
    n = len(funcs)
    mult = np.zeros((n, n), dtype=np.int64)
    for i, f in enumerate(funcs):
        for j, g in enumerate(funcs):
            mult[i, j] = index[tuple(f[g[x]] for x in range(k))]
    labels = ["".join(map(str, f)) for f in funcs]
    return FiniteMonoid(mult, labels, f"T{k}")


class FiniteCategory:
    """Finite category: morphisms with PARTIAL composition (typed). `comp[g, h]` is the index
    of g.h (g after h) or -1 when not composable (source(g) != target(h)). Its category algebra
    has structure constants that are ZERO where composition is undefined -- this *partial* /
    typed composition is the genuinely categorical feature beyond a monoid. Same regular-rep /
    structure-constant machinery (P_g can now have ZERO columns: 'this relation does not apply')."""

    def __init__(self, comp, src, tgt, identities, labels, name, generators=None):
        self.comp = np.asarray(comp, dtype=np.int64)       # [n, n], -1 = undefined
        self.src = np.asarray(src, dtype=np.int64)         # source object of each morphism
        self.tgt = np.asarray(tgt, dtype=np.int64)         # target object of each morphism
        self.identities = list(identities)
        self.generators = list(generators) if generators is not None else []
        self.n = int(self.comp.shape[0])
        self.labels = list(labels)
        self.name = name
        self.e = identities[0]                             # a representative identity (not global)
        self.n_objects = len(identities)
        # a morphism is a "unit" iff invertible; in a path category only identities are
        self.is_unit = np.array([m in self.identities for m in range(self.n)])

    def is_abelian(self) -> bool:
        return False

    def regular_rep(self) -> np.ndarray:
        n = self.n
        P = np.zeros((n, n, n), dtype=np.float32)
        for g in range(n):
            for y in range(n):
                k = self.comp[g, y]
                if k >= 0:
                    P[g, k, y] = 1.0                       # column y is ALL ZERO if g.y undefined
        return P

    def struct_const(self) -> np.ndarray:
        n = self.n
        T = np.zeros((n, n, n), dtype=np.float32)
        for g in range(n):
            for h in range(n):
                k = self.comp[g, h]
                if k >= 0:
                    T[k, g, h] = 1.0
        return T

    def __repr__(self):
        defined = int((self.comp >= 0).sum())
        return (f"FiniteCategory({self.name}, morphisms={self.n}, objects={self.n_objects}, "
                f"composable_pairs={defined}/{self.n * self.n})")


def path_category(n_objects, edges, name="Cat") -> FiniteCategory:
    """Free category on a finite DAG: morphisms = directed paths (incl. length-0 identities),
    composition = path concatenation (defined iff endpoints match). `edges` is a list of
    (src, tgt) object pairs. Must be acyclic so the category is finite."""
    edges = [tuple(e) for e in edges]
    morphisms = [(o, ()) for o in range(n_objects)]        # identities first
    max_len = n_objects                                    # a DAG path visits each object once

    def dfs(start, cur, path):
        for ei, (s, t) in enumerate(edges):
            if s == cur and len(path) < max_len:
                morphisms.append((start, path + (ei,)))
                dfs(start, t, path + (ei,))

    for o in range(n_objects):
        dfs(o, o, ())
    morphisms = list(dict.fromkeys(morphisms))
    index = {m: i for i, m in enumerate(morphisms)}
    n = len(morphisms)

    def s_of(m):
        return m[0]

    def t_of(m):
        return m[0] if not m[1] else edges[m[1][-1]][1]

    comp = -np.ones((n, n), dtype=np.int64)
    for gi, g in enumerate(morphisms):
        for hi, h in enumerate(morphisms):
            if s_of(g) == t_of(h):                         # g after h: need src(g) = tgt(h)
                m = (s_of(h), h[1] + g[1])                 # traverse h, then g
                if m in index:
                    comp[gi, hi] = index[m]
    src = np.array([s_of(m) for m in morphisms], dtype=np.int64)
    tgt = np.array([t_of(m) for m in morphisms], dtype=np.int64)
    identities = [index[(o, ())] for o in range(n_objects)]
    generators = [index[(edges[ei][0], (ei,))] for ei in range(len(edges))]   # length-1 morphisms
    labels = [f"{m[0]}>{'.'.join(map(str, m[1])) if m[1] else 'id'}" for m in morphisms]
    return FiniteCategory(comp, src, tgt, identities, labels, name, generators=generators)


def _from_permutations(perms, name) -> FiniteGroup:
    """Build a group from a closed list of permutations of range(k).

    Composition convention: (a . b)(x) = a[b[x]]  (apply b, then a)."""
    perms = [tuple(p) for p in perms]
    index = {p: i for i, p in enumerate(perms)}
    n = len(perms)
    k = len(perms[0])
    mult = np.zeros((n, n), dtype=np.int64)
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            comp = tuple(a[b[x]] for x in range(k))
            mult[i, j] = index[comp]
    inv = np.zeros(n, dtype=np.int64)
    for i, a in enumerate(perms):
        ainv = [0] * k
        for x in range(k):
            ainv[a[x]] = x
        inv[i] = index[tuple(ainv)]
    labels = ["".join(map(str, p)) for p in perms]
    return FiniteGroup(mult, inv, labels, name)


def symmetric_group(k: int) -> FiniteGroup:
    """S_k. Non-abelian for k >= 3. Smallest interesting case: S3 (n=6)."""
    return _from_permutations(itertools.permutations(range(k)), f"S{k}")


def cyclic_group(k: int) -> FiniteGroup:
    """C_k. ABELIAN control group (sanity check that abelian transport suffices)."""
    perms = [tuple((x + s) % k for x in range(k)) for s in range(k)]
    return _from_permutations(perms, f"C{k}")


def dihedral_group(k: int) -> FiniteGroup:
    """D_k, order 2k, acting on the k vertices of a k-gon. Non-abelian for k >= 3."""
    rot = [tuple((x + s) % k for x in range(k)) for s in range(k)]
    ref = [tuple((s - x) % k for x in range(k)) for s in range(k)]
    return _from_permutations(rot + ref, f"D{k}")


def direct_product(G1: FiniteGroup, G2: FiniteGroup, name=None) -> FiniteGroup:
    """G1 x G2 with componentwise multiplication. Abelian iff both factors are."""
    n1, n2, n = G1.n, G2.n, G1.n * G2.n
    pair = lambda a, b: a * n2 + b
    mult = np.zeros((n, n), dtype=np.int64)
    inv = np.zeros(n, dtype=np.int64)
    for a1 in range(n1):
        for b1 in range(n2):
            inv[pair(a1, b1)] = pair(G1.inv[a1], G2.inv[b1])
            for a2 in range(n1):
                for b2 in range(n2):
                    mult[pair(a1, b1), pair(a2, b2)] = pair(G1.mult[a1, a2], G2.mult[b1, b2])
    labels = [f"{G1.labels[a]}|{G2.labels[b]}" for a in range(n1) for b in range(n2)]
    return FiniteGroup(mult, inv, labels, name or f"{G1.name}x{G2.name}")


def quaternion_group() -> FiniteGroup:
    """Q8 = {+-1, +-i, +-j, +-k}. The other non-abelian group of order 8 (besides D4)."""
    units = ["1", "i", "j", "k"]
    elems = [(s, u) for u in units for s in (1, -1)]          # 8 elements
    base = {  # u1*u2 -> (sign, unit)
        ("1", "1"): (1, "1"), ("1", "i"): (1, "i"), ("1", "j"): (1, "j"), ("1", "k"): (1, "k"),
        ("i", "1"): (1, "i"), ("j", "1"): (1, "j"), ("k", "1"): (1, "k"),
        ("i", "i"): (-1, "1"), ("j", "j"): (-1, "1"), ("k", "k"): (-1, "1"),
        ("i", "j"): (1, "k"), ("j", "k"): (1, "i"), ("k", "i"): (1, "j"),
        ("j", "i"): (-1, "k"), ("k", "j"): (-1, "i"), ("i", "k"): (-1, "j"),
    }
    mul = lambda e1, e2: (lambda sb: (e1[0] * e2[0] * sb[0], sb[1]))(base[(e1[1], e2[1])])
    index = {e: i for i, e in enumerate(elems)}
    n = 8
    mult = np.array([[index[mul(e1, e2)] for e2 in elems] for e1 in elems], dtype=np.int64)
    inv = np.array([next(j for j, e2 in enumerate(elems) if mul(e1, e2) == (1, "1"))
                    for e1 in elems], dtype=np.int64)
    labels = [f"{'+' if s > 0 else '-'}{u}" for (s, u) in elems]
    return FiniteGroup(mult, inv, labels, "Q8")


GROUPS = {
    "S3": lambda: symmetric_group(3),
    "S4": lambda: symmetric_group(4),
    "D4": lambda: dihedral_group(4),
    "C6": lambda: cyclic_group(6),
    # the five groups of order 8 (complete candidate set for algebra selection)
    "C8": lambda: cyclic_group(8),
    "C4xC2": lambda: direct_product(cyclic_group(4), cyclic_group(2)),
    "C2x2x2": lambda: direct_product(direct_product(cyclic_group(2), cyclic_group(2)),
                                     cyclic_group(2), name="C2x2x2"),
    "Q8": lambda: quaternion_group(),
    # order-27 algebras (for the monoid vs group discovery): 3 abelian groups + the T3 monoid
    "C27": lambda: cyclic_group(27),
    "C9xC3": lambda: direct_product(cyclic_group(9), cyclic_group(3)),
    "C3x3x3": lambda: direct_product(direct_product(cyclic_group(3), cyclic_group(3)),
                                     cyclic_group(3), name="C3x3x3"),
    "T3": lambda: full_transformation_monoid(3),     # MONOID (non-invertible), |T3| = 27
}

# the complete list of groups of order 8 (3 abelian, 2 non-abelian)
ORDER8 = ["C8", "C4xC2", "C2x2x2", "D4", "Q8"]
# order-27 candidates: 3 abelian groups (all invertible) vs the T3 monoid (non-invertible)
ORDER27 = ["C27", "C9xC3", "C3x3x3", "T3"]


if __name__ == "__main__":
    for name, factory in GROUPS.items():
        G = factory()
        P = G.regular_rep()
        T = G.struct_const()
        # sanity: P is a homomorphism, T reproduces mult
        ok_hom = True
        for g in range(G.n):
            for h in range(G.n):
                if not np.allclose(P[g] @ P[h], P[G.mult[g, h]]):
                    ok_hom = False
        print(f"{G!s:40s} hom_ok={ok_hom} e={G.e}")
