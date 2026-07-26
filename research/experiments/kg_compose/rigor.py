"""Credible re-run of the PreNat headline claims, with the rigor the original harness lacked:

  * seeds = 10 (was 3), bootstrap 95% CIs (was mean +/- std over 3 points),
  * TIE-AWARE mid-rank metrics (statutils; was strict '>' which flattered the exact model),
  * Wilson interval on the selection probability (was bare '3/3'),
  * paired sign-flip permutation tests for 'A beats B'.

It deliberately reuses the *same* data generators, model builders and training loop as the original
experiments (kg_data.make_kg, kg_models.build, kg_run.train) -- only the evaluation is upgraded --
so any change vs the original numbers is attributable to rigor, not a different setup.

Sections (run a subset with --sections A B ...):
  A  S4 non-abelian composition (full + 50% data)         claim: PreNat 1.0, RotatE ~0.42, RESCAL ~0.82
  B  T3 monoid non-invertible composition                 claim: PreNat 0.875, RotatE 0.40
  C  S4 Cayley long-horizon drift                          claim: PreNat 1.0 @h=40, RESCAL ->chance
  D  algebra selection (order-8), Wilson interval          claim: selects D4 / Q8
  E  identifiability with MATCHED metrics (atomic vs path MRR)
  F  learning the algebra (corrects the '0.999 recovers oracle' claim)

Run:  python rigor.py                 (all sections, seeds=10)
      python rigor.py --sections A F --seeds 10
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
from kg_models import build, PreNatLearnedAlgebra, RESCAL, PreNatKGE
from kg_run import train, n_params
from kg_monoid import choose_relations
from statutils import metric_arrays, boot_ci, fmt_ci, wilson, paired_perm_test, mean_ci_str
from puzzle import rollout_acc


def seed_means(model, h, rel_list, t, which=0):
    """per-seed scalar = mean over queries of the tie-aware metric (0=rr,1=h1,2=h3,3=h10)."""
    arr = metric_arrays(model, h, rel_list, t)[which]
    return float(np.mean(arr)) if len(arr) else float("nan")


# --------------------------------------------------------------------------------------------------

def section_A(seeds, steps, lr):
    print("\n=== A. S4 non-abelian composition  (tie-aware NONCOMM Hits@1, seeds=%d) ===" % seeds)
    MODELS = ["TransE", "RotatE", "RESCAL", "PreNat"]
    for frac in (1.0, 0.5):
        per = {m: [] for m in MODELS}
        comm = {m: [] for m in MODELS}
        params = {}
        for seed in range(seeds):
            data = make_kg("S4", 8, frac, seed=seed)
            pq = data["pquery"]; nc = pq[:, 4].bool()
            for name in MODELS:
                torch.manual_seed(seed)
                model = build(name, data)
                train(model, data, steps, lr)
                per[name].append(seed_means(model, pq[nc][:, 0], [pq[nc][:, 1], pq[nc][:, 2]],
                                            pq[nc][:, 3], which=1))
                comm[name].append(seed_means(model, pq[~nc][:, 0], [pq[~nc][:, 1], pq[~nc][:, 2]],
                                             pq[~nc][:, 3], which=1))
                params[name] = n_params(model)
        print(f"\n  train_frac={frac}  (non-commuting path frac={float(nc.float().mean()):.2f})")
        print(f"  {'model':8s} | {'NONCOMM H@1 [95% CI]':^26s} | {'comm H@1':^10s} | params")
        print("  " + "-" * 60)
        for name in MODELS:
            print(f"  {name:8s} | {mean_ci_str(per[name]):^26s} | "
                  f"{np.mean(comm[name]):^10.3f} | {params[name]}")
        d, p = paired_perm_test(per["PreNat"], per["RESCAL"])
        print(f"  paired PreNat-RESCAL: dmean={d:+.3f}  p={p:.4f}  (n={seeds} seeds)")
        d2, p2 = paired_perm_test(comm["RotatE"], per["RotatE"])
        print(f"  RotatE comm-vs-noncomm gap: {d2:+.3f}  p={p2:.4f}  (the abelian signature)")


def section_B(seeds, steps, lr):
    print("\n=== B. T3 monoid non-invertible composition  (tie-aware path NON-INV Hits@1) ===")
    MODELS = ["TransE", "RotatE", "RESCAL", "PreNat"]
    per = {m: [] for m in MODELS}; params = {}
    for seed in range(seeds):
        rel = choose_relations(seed, 4, 4)
        data = make_kg("T3", train_frac=1.0, seed=seed, rel_elems=rel)
        pq = data["pquery"]; comp_u = torch.tensor(data["pquery_comp_units"])
        ni = ~comp_u
        for name in MODELS:
            torch.manual_seed(seed)
            model = build(name, data); train(model, data, steps, lr)
            per[name].append(seed_means(model, pq[ni][:, 0], [pq[ni][:, 1], pq[ni][:, 2]],
                                        pq[ni][:, 3], which=1))
            params[name] = n_params(model)
    print(f"  {'model':8s} | {'path NON-INV H@1 [95% CI]':^28s} | params")
    print("  " + "-" * 50)
    for name in MODELS:
        print(f"  {name:8s} | {mean_ci_str(per[name]):^28s} | {params[name]}")
    d, p = paired_perm_test(per["PreNat"], per["RotatE"])
    print(f"  paired PreNat-RotatE: dmean={d:+.3f}  p={p:.4f}")


def section_C(seeds, steps, lr):
    print("\n=== C. S4 Cayley long-horizon drift  (outcome accuracy, seeds=%d) ===" % seeds)
    print("  Corrects the paper's '1.00 at any horizon': only the EXACT SYMBOLIC model (one-hot")
    print("  codes + regular embedding, built by construction) is truly drift-free; the LEARNED")
    print("  PreNat drifts at long horizon (its codes never lock exactly).")
    horizons = [1, 5, 10, 40, 100, 200]
    G = GROUPS["S4"](); rng = np.random.default_rng(0)
    gens = rng.choice([g for g in range(G.n) if g != G.e], 10, replace=False)
    # exact symbolic S4 model: E=I, C=one-hot(true elements), true structure constants
    sym = PreNatKGE(G.n, len(gens), G.n, P=torch.tensor(G.regular_rep()), Tc=torch.tensor(G.struct_const()))
    with torch.no_grad():
        sym.C.copy_(torch.nn.functional.one_hot(torch.tensor(gens.tolist()), G.n).float())
        sym.E.weight.copy_(torch.eye(G.n, G.n))
    rows = {"SYMBOLIC (exact, by construction)": {h: [rollout_acc(sym, G, gens, h, 1000, 0)] for h in horizons}}
    for name in ("TransE", "RESCAL", "PreNat (learned)"):
        accs = {h: [] for h in horizons}
        builder = "PreNat" if name.startswith("PreNat") else name
        for seed in range(seeds):
            data = make_kg("S4", 10, 1.0, seed=seed, rel_elems=gens)
            torch.manual_seed(seed); model = build(builder, data); train(model, data, steps, lr)
            for h in horizons:
                accs[h].append(rollout_acc(model, G, gens, h, 1000, seed))
        rows[name] = accs
    print(f"  {'model':34s} |" + " ".join(f"h={h:<4d}" for h in horizons))
    for name in rows:
        cells = " ".join(f"{np.mean(rows[name][h]):.2f} " for h in horizons)
        print(f"  {name:34s} | {cells}")
    print("  (per-horizon 95% CI at h=200):")
    print(f"    PreNat-learned h200: {mean_ci_str(rows['PreNat (learned)'][200])}   "
          f"RESCAL h200: {mean_ci_str(rows['RESCAL'][200])}   "
          f"SYMBOLIC h200: {rows['SYMBOLIC (exact, by construction)'][200][0]:.2f}")
    print("  HONEST READING: exactness/drift-freeness is a property of the SYMBOLIC algebra (a")
    print("  hand/discovery-committed lookup), NOT of the gradient-learned model; see")
    print("  discover_compose.py for the discovery->commit chain that earns it.")


def _select_once(data_group, n_rel, seed, steps, lr):
    data = make_kg(data_group, n_rel, train_frac=1.0, seed=seed)
    pq = data["pquery"]; nval = len(pq) // 2
    val, test = pq[:nval], pq[nval:]; ncm = test[:, 4].bool()
    vals = {}; testnc = {}
    for c in ORDER8:
        Gc = GROUPS[c]()
        alg = {"P": torch.tensor(Gc.regular_rep()), "Tc": torch.tensor(Gc.struct_const())}
        torch.manual_seed(seed); model = build("PreNat", data, algebra=alg); train(model, data, steps, lr)
        vals[c] = seed_means(model, val[:, 0], [val[:, 1], val[:, 2]], val[:, 3], which=0)
        testnc[c] = seed_means(model, test[ncm][:, 0], [test[ncm][:, 1], test[ncm][:, 2]],
                               test[ncm][:, 3], which=1)
    pick = max(vals, key=vals.get)
    return pick, vals, testnc


def section_D(seeds, steps, lr):
    print("\n=== D. Algebra selection (order-8), Wilson interval on selection probability ===")
    for data_group in ("D4", "Q8"):
        picks = []; valtab = {c: [] for c in ORDER8}
        for seed in range(seeds):
            pick, vals, _ = _select_once(data_group, 6, seed, steps, lr)
            picks.append(pick)
            for c in ORDER8:
                valtab[c].append(vals[c])
        ncorrect = sum(p == data_group for p in picks)
        phat, lo, hi = wilson(ncorrect, seeds)
        print(f"\n  data={data_group}: selected correctly {ncorrect}/{seeds}  "
              f"-> selection prob {phat:.2f}  Wilson95 [{lo:.2f},{hi:.2f}]")
        order = sorted(ORDER8, key=lambda c: -np.mean(valtab[c]))
        for c in order:
            tag = " *TRUE*" if c == data_group else ""
            kind = "abelian" if GROUPS[c]().is_abelian() else "non-abelian"
            print(f"    {c:8s} {kind:12s} val-path-MRR {mean_ci_str(valtab[c])}{tag}")
        true_vals = valtab[data_group]
        best_wrong = max((c for c in ORDER8 if c != data_group), key=lambda c: np.mean(valtab[c]))
        d, p = paired_perm_test(true_vals, valtab[best_wrong])
        print(f"    margin true-vs-best-wrong({best_wrong}): {d:+.3f}  p={p:.4f}")


def section_E(seeds, steps, lr):
    print("\n=== E. Identifiability with MATCHED metrics (atomic MRR vs path MRR, both tie-aware) ===")
    for data_group in ("D4", "Q8"):
        atom = {c: [] for c in ORDER8}; path = {c: [] for c in ORDER8}
        for seed in range(seeds):
            data = make_kg(data_group, 6, train_frac=1.0, seed=seed)
            ta = data["test_atomic"]; pq = data["pquery"]; ncm = pq[:, 4].bool()
            for c in ORDER8:
                Gc = GROUPS[c]()
                alg = {"P": torch.tensor(Gc.regular_rep()), "Tc": torch.tensor(Gc.struct_const())}
                torch.manual_seed(seed); m = build("PreNat", data, algebra=alg); train(m, data, steps, lr)
                atom[c].append(seed_means(m, ta[:, 0], [ta[:, 1]], ta[:, 2], which=0))
                # MATCHED: path MRR (not Hits@1) on the SAME (non-comm) discriminating subset
                path[c].append(seed_means(m, pq[ncm][:, 0], [pq[ncm][:, 1], pq[ncm][:, 2]],
                                          pq[ncm][:, 3], which=0))
        amean = {c: np.mean(atom[c]) for c in ORDER8}
        pmean = {c: np.mean(path[c]) for c in ORDER8}
        a_spread = max(amean.values()) - min(amean.values())
        p_spread = max(pmean.values()) - min(pmean.values())
        # does the true group WIN the atomic column, or is it tied by a wrong candidate?
        atop = max(amean, key=amean.get); ptop = max(pmean, key=pmean.get)
        print(f"\n  data={data_group}: atomic-MRR spread {a_spread:.2f} (top={atop}), "
              f"path-MRR spread {p_spread:.2f} (top={ptop})  [both MRR; matched]")
        print(f"    atomic argmax correct? {atop==data_group}   path argmax correct? {ptop==data_group}")
        for c in ORDER8:
            tag = " *TRUE*" if c == data_group else ""
            print(f"    {c:8s} atomic {amean[c]:.3f}  path {pmean[c]:.3f}{tag}")


def _train_la(data, init, steps, lr, fix_entity=False, rank=None):
    ne, nr, n = data["n_entities"], data["n_relations"], data["n"]
    m = PreNatLearnedAlgebra(ne, nr, n, init=init, fix_entity=fix_entity, rank=rank)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    tr = data["train"]; h, r, t = tr[:, 0], tr[:, 1], tr[:, 2]
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(m.score_all(h, [r]), t) + m.assoc_loss()
        loss.backward(); opt.step()
    return m


def section_F(seeds, steps, lr):
    print("\n=== F. Learning the algebra  (CORRECTS '0.999 recovers oracle'; tie-aware NONCOMM H@1) ===")
    for frac in (1.0, 0.5):
        res = {k: [] for k in ("oracle", "RESCAL", "learned-random", "learned-warm")}
        assoc = []
        for seed in range(seeds):
            data = make_kg("S4", 8, frac, seed=seed)
            pq = data["pquery"]; nc = pq[:, 4].bool()
            ev = lambda m: seed_means(m, pq[nc][:, 0], [pq[nc][:, 1], pq[nc][:, 2]], pq[nc][:, 3], which=1)
            torch.manual_seed(seed); res["oracle"].append(ev(_fit_fixed(data, "PreNat", steps, lr)))
            torch.manual_seed(seed); res["RESCAL"].append(ev(_fit_fixed(data, "RESCAL", steps, lr)))
            torch.manual_seed(seed); mr = _train_la(data, "random", steps, lr); res["learned-random"].append(ev(mr))
            assoc.append(float(mr.assoc_loss().detach()))
            torch.manual_seed(seed); res["learned-warm"].append(ev(_train_la(data, "cyclic", steps, lr)))
        print(f"\n  train_frac={frac}")
        for k in ("oracle", "RESCAL", "learned-random", "learned-warm"):
            print(f"    {k:16s} NONCOMM H@1 {mean_ci_str(res[k])}")
        print(f"    learned-random assoc-residual mean={np.mean(assoc):.1e}")
        d, p = paired_perm_test(res["learned-random"], res["oracle"])
        print(f"    paired learned-random - oracle: {d:+.3f}  p={p:.4f}  "
              f"(tests the 'recovers oracle' claim)")


def _fit_fixed(data, name, steps, lr):
    m = build(name, data); train(m, data, steps, lr); return m   # caller already seeded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--sections", nargs="+", default=list("ABCDEF"))
    args = ap.parse_args()
    print(f"RIGOR RE-RUN  seeds={args.seeds} steps={args.steps} lr={args.lr}  "
          f"sections={''.join(args.sections)}")
    fns = dict(A=section_A, B=section_B, C=section_C, D=section_D, E=section_E, F=section_F)
    for s in args.sections:
        fns[s](args.seeds, args.steps, args.lr)


if __name__ == "__main__":
    main()
