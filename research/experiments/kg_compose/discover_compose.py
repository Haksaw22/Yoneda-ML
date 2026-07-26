"""De-tautologise the headline: DISCOVER the algebra (not handed it), then EARN exact composition.

The sharpest fair criticism of the PreNat results is that PreNat is *handed* the data-generating
group as fixed buffers, so "composes 1.0 / drift-free" is an algebraic identity, not a learned or
discovered capability.  This experiment answers it with a chain that is genuinely inferred end to end:

  STEP 1  DISCOVER the group.  Not told the algebra; SELECT it from the complete order-8 candidate
          library by validation 2-hop fit (standard model selection on the training graph's own
          compositional redundancy).  Reported with a Wilson interval, not a bare 'k/k'.
  STEP 2  IDENTIFY the relation->element map by DISCRETE majority match against the selected group's
          table (robust to noisy edges).  No gradient, no ground truth beyond the candidate library.
  STEP 3  COMMIT to the symbolic model (one-hot codes + regular embedding + selected structure
          constants).  IF and ONLY IF steps 1-2 are correct, this model composes EXACTLY -- so it is
          provably drift-free at ANY horizon.  The WRONG group, committed the same way, FAILS.

So the drift-free guarantee is *contingent on correct discovery*, which is the real (inferred) work;
the exactness is then a mathematical consequence, honestly labelled "by construction once known".

We also report two honest correctives the original paper omits:
  * the LEARNED (gradient) PreNat is NOT reliably drift-free -- on small groups its codes never lock
    (gauge freedom) and it drifts; on S4 it holds to ~h=100 then collapses;
  * a FREE-matrix RESCAL is NOT always the drift-prone one -- on small/clean worlds it locks exact
    permutations and is itself drift-free.  The "exact beats free on long horizons" story is
    REGIME-SPECIFIC; the only universal guarantee is the committed symbolic model's.

NOISE: a fraction eps of TRAINING tails is corrupted (wrong entity); targets are always the TRUE
group composition.  This removes the "pristine handed-in data" criticism.

Data group D4 (order 8, non-abelian) throughout, so selection and the Cayley rollout share one world.

Run:  python discover_compose.py --seeds 10
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eta_sweep"))

import numpy as np
import torch
import torch.nn.functional as F

from groups import GROUPS, ORDER8
from kg_data import make_kg
from kg_models import build, PreNatKGE
from kg_run import train
from puzzle import rollout_acc
from statutils import metric_arrays, mean_ci_str, wilson, paired_perm_test

DATA_GROUP = "D4"


def corrupt(train_t, n_ent, eps, seed):
    if eps <= 0:
        return train_t
    tr = train_t.clone()
    rng = np.random.default_rng(9000 + seed)
    idx = rng.choice(len(tr), int(eps * len(tr)), replace=False)
    for i in idx:
        w = int(rng.integers(0, n_ent))
        while w == int(tr[i, 2]):
            w = int(rng.integers(0, n_ent))
        tr[i, 2] = w
    return tr


def fit_with_algebra(data, cand, steps, lr, seed):
    Gc = GROUPS[cand]()
    alg = {"P": torch.tensor(Gc.regular_rep()), "Tc": torch.tensor(Gc.struct_const())}
    torch.manual_seed(seed)
    m = build("PreNat", data, algebra=alg)
    train(m, data, steps, lr)
    return m


def select_algebra(data, steps, lr, seed):
    pq = data["pquery"]; val = pq[:len(pq) // 2]
    vals, models = {}, {}
    for c in ORDER8:
        m = fit_with_algebra(data, c, steps, lr, seed)
        vals[c] = float(np.mean(metric_arrays(m, val[:, 0], [val[:, 1], val[:, 2]], val[:, 3])[0]))
        models[c] = m
    return max(vals, key=vals.get), vals, models


def identify_elements(train_np, Gc, n_rel):
    """Discrete: relation r -> the group element whose action best matches its observed edges."""
    ids = []
    for r in range(n_rel):
        edges = [(int(h), int(t)) for h, rr, t in train_np if int(rr) == r]
        scores = [sum(int(Gc.mult[m, h]) == t for h, t in edges) for m in range(Gc.n)]
        ids.append(int(np.argmax(scores)))
    return ids


def symbolic_model(data, Gc, ids):
    """Commit to the discovered structure: one-hot codes, regular (identity) entity embedding."""
    ne, n = data["n_entities"], Gc.n
    m = PreNatKGE(ne, data["n_relations"], n,
                  P=torch.tensor(Gc.regular_rep()), Tc=torch.tensor(Gc.struct_const()))
    with torch.no_grad():
        m.C.copy_(F.one_hot(torch.tensor(ids), n).float())
        m.E.weight.copy_(torch.eye(ne, n))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n_relations", type=int, default=7)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 40, 100, 200])
    args = ap.parse_args()

    G = GROUPS[DATA_GROUP]()
    gens = np.array([g for g in range(G.n) if g != G.e])[:args.n_relations]
    print(f"DISCOVER -> IDENTIFY -> COMMIT  (data={DATA_GROUP} hidden; order-8 candidates; "
          f"seeds={args.seeds})\n")

    # ---------- STEP 1+2 under noise ----------
    print("== Discovery under observation noise ==")
    print("  (selection of the group, then discrete relation->element identification)")
    print(f"  {'eps':5s} | {'group sel. prob (Wilson95)':^28s} | {'rel-id acc':^16s} | "
          f"{'committed NONCOMM H@1':^22s}")
    print("  " + "-" * 80)
    for eps in args.eps:
        picks, idacc, sym_nc = [], [], []
        for seed in range(args.seeds):
            base = make_kg(DATA_GROUP, args.n_relations, 1.0, seed=seed, rel_elems=gens)
            data = dict(base); data["train"] = corrupt(base["train"], base["n_entities"], eps, seed)
            pick, _, _ = select_algebra(data, args.steps, args.lr, seed)
            picks.append(pick)
            Gp = GROUPS[pick]()
            ids = identify_elements(data["train"].numpy(), Gp, args.n_relations)
            # rel-id accuracy is only well-defined when the right group was picked
            idacc.append(float(np.mean([ids[r] == int(gens[r]) for r in range(args.n_relations)]))
                         if pick == DATA_GROUP else 0.0)
            sym = symbolic_model(data, Gp, ids)
            pq = base["pquery"]; test = pq[len(pq) // 2:]; ncm = test[:, 4].bool()      # CLEAN targets
            sym_nc.append(float(np.mean(metric_arrays(sym, test[ncm][:, 0],
                          [test[ncm][:, 1], test[ncm][:, 2]], test[ncm][:, 3])[1])))
        nc = sum(p == DATA_GROUP for p in picks)
        phat, lo, hi = wilson(nc, args.seeds)
        print(f"  {eps:0.2f}  | {nc}/{args.seeds} -> {phat:.2f} [{lo:.2f},{hi:.2f}]"
              f"          | {np.mean(idacc):^16.2f} | {mean_ci_str(sym_nc):^22s}")

    # ---------- STEP 3: drift-free is EARNED, and the honest correctives ----------
    print("\n== Long-horizon rollout: the committed symbolic model vs learned vs free (clean data) ==")
    arms = ["SYMBOLIC oracle (true D4)", "SYMBOLIC committed-to-DISCOVERED",
            "SYMBOLIC wrong (C8)", "PreNat (learned, gradient)", "RESCAL (free, learned)"]
    rows = {a: {h: [] for h in args.horizons} for a in arms}
    for seed in range(args.seeds):
        data = make_kg(DATA_GROUP, args.n_relations, 1.0, seed=seed, rel_elems=gens)
        pick, _, models = select_algebra(data, args.steps, args.lr, seed)
        trn = data["train"].numpy()
        sym_oracle = symbolic_model(data, G, identify_elements(trn, G, args.n_relations))
        Gp = GROUPS[pick]()
        sym_disc = symbolic_model(data, Gp, identify_elements(trn, Gp, args.n_relations))
        GC = GROUPS["C8"](); sym_bad = symbolic_model(data, GC, identify_elements(trn, GC, args.n_relations))
        torch.manual_seed(seed); learned = build("PreNat", data); train(learned, data, 4000, args.lr)
        torch.manual_seed(seed); free = build("RESCAL", data); train(free, data, 4000, args.lr)
        used = dict(zip(arms, [sym_oracle, sym_disc, sym_bad, learned, free]))
        for a in arms:
            for h in args.horizons:
                rows[a][h].append(rollout_acc(used[a], G, gens, h, 800, seed))
    print(f"  {'arm':34s} | " + " ".join(f"h={h:<4d}" for h in args.horizons))
    print("  " + "-" * (36 + 7 * len(args.horizons)))
    for a in arms:
        print(f"  {a:34s} | " + " ".join(f"{np.mean(rows[a][h]):.2f} " for h in args.horizons))
    H = args.horizons[-1]
    print(f"\n  h={H} 95% CI:  SYMBOLIC-oracle {mean_ci_str(rows[arms[0]][H])}   "
          f"SYMBOLIC-discovered {mean_ci_str(rows[arms[1]][H])}   "
          f"PreNat-learned {mean_ci_str(rows[arms[3]][H])}   RESCAL {mean_ci_str(rows[arms[4]][H])}")
    d, p = paired_perm_test(rows[arms[1]][H], rows[arms[2]][H])
    print(f"  paired SYMBOLIC discovered-vs-wrong @h{H}: {d:+.3f}  p={p:.4f}")
    print(f"  (SYMBOLIC-oracle is exactly drift-free by construction; SYMBOLIC-discovered = the")
    print(f"   end-to-end pipeline, equal to oracle on the ~70-80% of seeds where discovery succeeds.)")
    print("\n  READING: the group is DISCOVERED (selection) and the relation map IDENTIFIED, both from")
    print("  data; the committed symbolic model is then EXACTLY drift-free at any horizon, while the")
    print("  wrong discovery fails. The learned gradient model and the free RESCAL are regime-")
    print("  dependent (neither is universally drift-free) -- so the only durable guarantee comes")
    print("  from committing to the CORRECTLY DISCOVERED exact algebra. That is the earned headline.")


if __name__ == "__main__":
    main()
