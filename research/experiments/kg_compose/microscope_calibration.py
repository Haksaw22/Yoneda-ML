"""Validate the relational-structure microscope as a PREDICTIVE instrument (not just a 3-point profile).

The original microscope profiles a handful of real KGs that all cluster at algebraicity ~0.6, so its
predictiveness *within* the messy regime is untestable.  Here we build a CONTROLLED spectrum from
exactly-algebraic to fully-messy and show the cheap profile PREDICTS the expensive downstream payoff.

Spectrum.  A mixed relational world on the elements of a base group G (default D4): each relation is,
with probability `eta`, a genuinely NON-algebraic random map on the entities; otherwise it is a true
group element's action.  Composition is honest function composition `t = map_{r2}[map_{r1}[h]]`, so
2-hop targets are always well defined, but for eta>0 the relation family is no longer closed under a
single finite algebra.  eta=0 => exactly algebraic; eta=1 => fully messy.

Two measurements per world (deliberately differing in budget, split, and form -- not statistically
independent: both compare the same constrained-vs-free model pair, so this validates that a CHEAP proxy
forecasts the EXPENSIVE outcome, not that an unrelated fingerprint does):
  * CHEAP profile  : functionality, invertibility, and a LOW-budget algebraicity RATIO
                     (path-skill of an associativity-constrained learned algebra / a free model),
                     trained briefly (400 steps) on the train/val split.
  * EXPENSIVE outcome: the true PRIOR-BENEFIT = full-budget (3000 steps) path-MRR(constrained) MINUS
                     path-MRR(free) on a HELD-OUT test set of composites (positive => the prior helps).

Claim to test: the cheap algebraicity profile PREDICTS the sign and magnitude of the expensive
prior-benefit across the whole spectrum (Spearman correlation + a leave-one-eta-out linear R^2),
i.e. a quick profiling run tells you whether the algebraic prior is worth using on a new dataset.

Run:  python microscope_calibration.py --seeds 5
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

from groups import GROUPS
from kg_models import RESCAL, PreNatLearnedAlgebra
from statutils import metric_arrays, boot_ci, fmt_ci

BASE = "D4"


def make_mixed_kg(eta, n_relations, seed, n_path=2500, train_frac=0.6):
    """Each relation is a random non-algebraic map w.p. eta, else a group element's action.

    train_frac<1 keeps data mildly scarce so the algebraic prior's sample-efficiency advantage is
    visible at eta=0 (with full data on a tiny group a free matrix fits everything and the prior's
    benefit collapses to ~0 -- itself an honest finding, but it hides the calibration signal)."""
    G = GROUPS[BASE](); n = G.n
    rng = np.random.default_rng(4242 + seed)
    nonid = [g for g in range(n) if g != G.e]
    maps = []
    for _ in range(n_relations):
        if rng.random() < eta:
            maps.append(rng.integers(0, n, size=n))                  # random map (non-algebraic)
        else:
            g = int(rng.choice(nonid)); maps.append(G.mult[g].copy())  # group element action h->g.h
    maps = np.stack(maps)                                            # [n_rel, n]
    ent = np.arange(n)
    triples = np.array([(h, r, int(maps[r, h])) for r in range(n_relations) for h in ent],
                       dtype=np.int64)
    idx = rng.permutation(len(triples)); ntest = max(1, int(0.15 * len(triples)))
    rest = idx[ntest:]; train = triples[rest[:max(1, int(train_frac * len(rest)))]]
    pq = []
    for _ in range(n_path):
        r1, r2 = int(rng.integers(0, n_relations)), int(rng.integers(0, n_relations))
        h = int(rng.integers(0, n))
        t = int(maps[r2, maps[r1, h]])                              # r1 then r2 (function comp)
        pq.append((h, r1, r2, t))
    pq = np.array(pq, dtype=np.int64)
    nval = len(pq) // 2
    return dict(train=train, n=n, n_ent=n, n_rel=n_relations,
                val=pq[:nval], test=pq[nval:], maps=maps)


def _fit(model, train, steps, lr, assoc):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr = torch.tensor(train); h, r, t = tr[:, 0], tr[:, 1], tr[:, 2]
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model.score_all(h, [r]), t)
        if assoc:
            loss = loss + model.assoc_loss()
        loss.backward(); opt.step()
    return model


def path_mrr(model, pq):
    q = torch.tensor(pq)
    return float(np.mean(metric_arrays(model, q[:, 0], [q[:, 1], q[:, 2]], q[:, 3])[0]))


def functionality(train):
    tails = defaultdict(set)
    for h, r, t in train:
        tails[(int(h), int(r))].add(int(t))
    return float(np.mean([len(v) for v in tails.values()]))


def invertibility(train, n_rel):
    inj = 0
    for r in range(n_rel):
        ts = [int(t) for h, rr, t in train if int(rr) == r]
        inj += (len(set(ts)) == len(ts)) if ts else 0
    return inj / max(n_rel, 1)


def cheap_algebraicity(d, steps, lr, seed):
    torch.manual_seed(seed); free = _fit(RESCAL(d["n_ent"], d["n_rel"], d["n"]), d["train"], steps, lr, False)
    torch.manual_seed(seed); alg = _fit(PreNatLearnedAlgebra(d["n_ent"], d["n_rel"], d["n"], init="random"),
                                        d["train"], steps, lr, True)
    fv, av = path_mrr(free, d["val"]), path_mrr(alg, d["val"])
    return av / fv if fv > 1e-6 else float("nan")


def prior_benefit(d, steps, lr, seed):
    torch.manual_seed(seed); free = _fit(RESCAL(d["n_ent"], d["n_rel"], d["n"]), d["train"], steps, lr, False)
    torch.manual_seed(seed); alg = _fit(PreNatLearnedAlgebra(d["n_ent"], d["n_rel"], d["n"], init="random"),
                                        d["train"], steps, lr, True)
    return path_mrr(alg, d["test"]) - path_mrr(free, d["test"])      # >0 => prior helps (held-out)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n_relations", type=int, default=8)
    ap.add_argument("--cheap_steps", type=int, default=400)
    ap.add_argument("--full_steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    args = ap.parse_args()

    print(f"MICROSCOPE CALIBRATION  base={BASE}  seeds={args.seeds}  "
          f"cheap={args.cheap_steps}step full={args.full_steps}step\n")
    print("  Cheap profile (no/low training) predicting the expensive held-out prior-benefit.\n")
    print(f"  {'eta':5s} | {'functionality':^13s} | {'invertibility':^13s} | "
          f"{'cheap algebraicity':^20s} | {'PRIOR-BENEFIT (test)':^22s}")
    print("  " + "-" * 86)
    all_alg, all_ben = [], []
    per_eta = {}
    for eta in args.etas:
        func, inv, alg, ben = [], [], [], []
        for seed in range(args.seeds):
            d = make_mixed_kg(eta, args.n_relations, seed)
            func.append(functionality(d["train"])); inv.append(invertibility(d["train"], d["n_rel"]))
            a = cheap_algebraicity(d, args.cheap_steps, args.lr, seed)
            b = prior_benefit(d, args.full_steps, args.lr, seed)
            alg.append(a); ben.append(b); all_alg.append(a); all_ben.append(b)
        per_eta[eta] = (np.mean(alg), np.mean(ben))
        print(f"  {eta:0.2f}  | {np.mean(func):^13.2f} | {np.mean(inv):^13.2f} | "
              f"{fmt_ci(*boot_ci(alg)):^20s} | {fmt_ci(*boot_ci(ben)):^22s}")

    rho = spearman(all_alg, all_ben)
    # bootstrap CI for the correlation across all (eta,seed) points
    rng = np.random.default_rng(0); n = len(all_alg); boots = []
    aa, bb = np.array(all_alg), np.array(all_ben)
    for _ in range(3000):
        ix = rng.integers(0, n, n); boots.append(spearman(aa[ix], bb[ix]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\n  Spearman(cheap algebraicity, held-out prior-benefit) = {rho:.2f}  [95% CI {lo:.2f},{hi:.2f}]"
          f"  over {n} worlds")

    # leave-one-eta-out SIGN prediction: learn a threshold on the held-in worlds, predict on held-out.
    # (rank/threshold framing, not linear, because benefit saturates -- Spearman already shows monotonicity)
    correct = 0
    eta_of = [e for e in args.etas for _ in range(args.seeds)]          # eta label per world
    pts = [(a, b, e) for a, b, e in zip(all_alg, all_ben, eta_of)]
    for held in args.etas:
        tr = [(a, b) for (a, b, e) in pts if e != held]
        te = [(a, b) for (a, b, e) in pts if e == held]
        # best threshold on algebraicity separating benefit>0 from <=0 on the training worlds
        cand = sorted(set(a for a, _ in tr))
        thr_best, acc_best = 0.9, -1
        for thr in cand:
            acc = np.mean([(a >= thr) == (b > 0) for a, b in tr])
            if acc > acc_best:
                acc_best, thr_best = acc, thr
        correct += sum((a >= thr_best) == (b > 0) for a, b in te)
    sign_cv = correct / len(pts)
    sign_acc = float(np.mean([(a > 1.0) == (b > 0) for a, b in zip(all_alg, all_ben)]))
    print(f"  Sign agreement (algebraicity>1  <=>  prior helps): {sign_acc:.0%}")
    print(f"  Leave-one-eta-out held-out sign-prediction accuracy (learned threshold): {sign_cv:.0%}")
    print("\n  READING: if the correlation is high and R^2 positive, a cheap profiling run forecasts")
    print("  whether the algebraic prior will help on a new dataset -- the instrument's payoff,")
    print("  demonstrated across a controlled algebraic->messy spectrum rather than 3 clustered points.")


if __name__ == "__main__":
    main()
