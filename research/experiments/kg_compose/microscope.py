"""The relational structure microscope.

Reframe: instead of competing as one more KGE, USE this project's machinery to PROFILE the
compositional-algebraic structure of any relational dataset and report its signature:

  functionality   : avg #tails per (head,relation). 1.0 = functional (algebra-friendly); >1 relational.
  invertibility    : fraction of relations whose action is injective on heads (no collapse).
                     low => monoid-like (is-a / causes); high => group-like.
  non-abelianness  : fraction of observed (r1,r2) start-node pairs where order changes the endpoint
                     (measured from the training graph; clean only for functional data).
  algebraicity     : path-composition skill of an ASSOCIATIVITY-CONSTRAINED model relative to a FREE
                     model (PreNatLearnedAlgebra / RESCAL). ~1 or >1 => composition is genuinely
                     algebraic; <1 => associativity hurts (messy / non-associative composition).

This turns "we lose to RESCAL on UMLS composition" into "we MEASURE that UMLS composition is only
~60% algebraic and which relations are non-invertible" -- a structure-discovery instrument.

Run:  python microscope.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eta_sweep"))

import numpy as np
import torch

from kg_data import make_kg
from kg_models import RESCAL, PreNatLearnedAlgebra
from kg_run import train, metrics
from kg_real import build_path_probe, eval_path
from umls import load_umls


# ---- data-only structural measures ----

def functionality(train):
    tails = defaultdict(set)
    for h, r, t in train:
        tails[(int(h), int(r))].add(int(t))
    return float(np.mean([len(v) for v in tails.values()]))


def invertibility(train, n_rel):
    """fraction of relations whose head->tail map is injective (no two heads share a tail)."""
    inj = 0
    for r in range(n_rel):
        pairs = [(int(h), int(t)) for h, rr, t in train if int(rr) == r]
        tails = [t for _, t in pairs]
        if pairs and len(set(tails)) == len(tails):
            inj += 1
    return inj / max(n_rel, 1)


def non_abelianness(train, n_rel, n_samples=4000, seed=0):
    """fraction of sampled (h,r1,r2) where the two orders reach different endpoints (functional)."""
    rng = np.random.default_rng(seed)
    nxt = defaultdict(dict)                                  # functional next: nxt[h][r] = t (last wins)
    for h, r, t in train:
        nxt[int(h)][int(r)] = int(t)
    heads = list(nxt.keys())
    diff = tot = 0
    for _ in range(n_samples):
        h = heads[rng.integers(0, len(heads))]
        r1, r2 = int(rng.integers(0, n_rel)), int(rng.integers(0, n_rel))
        m1 = nxt[h].get(r1); m2 = nxt[h].get(r2)
        if m1 is None or m2 is None:
            continue
        t12 = nxt[m1].get(r2); t21 = nxt[m2].get(r1)         # r2 after r1  vs  r1 after r2
        if t12 is None or t21 is None:
            continue
        tot += 1
        diff += (t12 != t21)
    return (diff / tot) if tot else float("nan")


# ---- model-based algebraicity ----

def algebraicity(train_t, n_ent, n_rel, d, path_q, path_filt, steps, lr, seed):
    """path-MRR(associativity-constrained) / path-MRR(free).  ~1+ => algebraic; <1 => not."""
    def fit(model):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        import torch.nn.functional as F
        h, r, t = train_t[:, 0], train_t[:, 1], train_t[:, 2]
        n = len(train_t); bs = min(512, n)
        for _ in range(steps):
            idx = torch.randperm(n)[:bs]
            opt.zero_grad()
            loss = F.cross_entropy(model.score_all(h[idx], [r[idx]]), t[idx])
            if isinstance(model, PreNatLearnedAlgebra):
                loss = loss + model.assoc_loss()
            loss.backward(); opt.step()
        return eval_path(model, path_q, path_filt)[0]
    torch.manual_seed(seed)
    free = fit(RESCAL(n_ent, n_rel, d))
    torch.manual_seed(seed)
    alg = fit(PreNatLearnedAlgebra(n_ent, n_rel, d, init="random"))
    return alg, free, (alg / free if free > 1e-6 else float("nan"))


def classify(func, inv, nonab, algeb):
    if algeb is not None and algeb < 0.8:
        return "messy / non-associative (free matrices win)"
    if inv > 0.95:
        return "group-like (invertible, algebraic)" + (" non-abelian" if nonab and nonab > 0.05 else " ~abelian")
    return "monoid/category-like (non-invertible, algebraic)"


def profile(name, train_t, n_ent, n_rel, d, path_q, path_filt, steps, lr, seed=0):
    train_np = train_t.numpy()
    func = functionality(train_np)
    inv = invertibility(train_np, n_rel)
    nonab = non_abelianness(train_np, n_rel)
    alg, free, ratio = algebraicity(train_t, n_ent, n_rel, d, path_q, path_filt, steps, lr, seed)
    print(f"  {name:18s} | func {func:4.2f} | inv {inv:4.2f} | non-abelian "
          f"{('%.2f' % nonab) if nonab == nonab else ' n/a'} | "
          f"algebraicity {ratio:4.2f} (alg {alg:.2f} / free {free:.2f})")
    print(f"  {'':18s} -> SIGNATURE: {classify(func, inv, nonab, ratio)}")
    return dict(func=func, inv=inv, nonab=nonab, algebraicity=ratio)


def main():
    print("RELATIONAL STRUCTURE MICROSCOPE\n")
    print("  dataset            | functionality | invertibility | non-abelianness | algebraicity")
    print("  " + "-" * 90)
    # synthetic: a non-abelian group, a non-invertible monoid
    for nm, g in (("S4 (group)", "S4"), ("T3 (monoid)", "T3")):
        data = make_kg(g, 8, 1.0, seed=0)
        pf = defaultdict(set)
        for x in data["pquery"]:
            pf[(int(x[0]), int(x[1]), int(x[2]))].add(int(x[3]))
        pq = data["pquery"][:, [0, 1, 2, 3]]
        profile(nm, data["train"], data["n_entities"], data["n_relations"], data["n"],
                pq, pf, steps=1500, lr=0.02)
    # real KG
    D = load_umls()
    probe, pfilt = build_path_probe(D["train"])
    profile("UMLS (real)", torch.tensor(D["train"]), D["n_ent"], D["n_rel"], 32,
            torch.tensor(probe), pfilt, steps=1200, lr=0.01)
    print("\n  The microscope reports a dataset's compositional-algebraic signature -- a "
          "structure-discovery output no KGE/GNN provides. Algebraicity<0.8 flags where PreNat's "
          "exact-composition prior should be RELAXED (soft associativity).")


if __name__ == "__main__":
    main()
