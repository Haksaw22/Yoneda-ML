"""Selection-at-scarcity probe. Pre-registered in PROBE-PREREG.md (same directory) BEFORE
running. Tests the 'select' branch of the published recipe ('fix or select; don't learn at
low data') at the scarce data fractions where the wall bites — which the record never did.

Reuses rigor.py's trainer/metrics verbatim (imported from the read-only source folder);
the only new degree of freedom is train_frac, plus a learned-random contrast arm per seed.
"""

import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "experiments", "kg_compose")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "..", "eta_sweep"))

import numpy as np
import torch

from groups import GROUPS, ORDER8
from kg_data import make_kg
from kg_models import build, PreNatLearnedAlgebra
from kg_run import train
from statutils import metric_arrays, wilson, paired_perm_test, mean_ci_str
import torch.nn.functional as F

SEEDS, STEPS, LR = 10, 2500, 0.02


def seed_means(model, h, rel_list, t, which=0):
    arr = metric_arrays(model, h, rel_list, t)[which]
    return float(np.mean(arr)) if len(arr) else float("nan")


def train_learned_random(data, steps, lr):
    ne, nr, n = data["n_entities"], data["n_relations"], data["n"]
    m = PreNatLearnedAlgebra(ne, nr, n, init="random")
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    tr = data["train"]
    h, r, t = tr[:, 0], tr[:, 1], tr[:, 2]
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(m.score_all(h, [r]), t) + m.assoc_loss()
        loss.backward()
        opt.step()
    return m


def select_once(data_group, n_rel, frac, seed):
    data = make_kg(data_group, n_rel, train_frac=frac, seed=seed)
    pq = data["pquery"]
    nval = len(pq) // 2
    val, test = pq[:nval], pq[nval:]
    ncm = test[:, 4].bool()
    vals, testnc = {}, {}
    for c in ORDER8:
        Gc = GROUPS[c]()
        alg = {"P": torch.tensor(Gc.regular_rep()), "Tc": torch.tensor(Gc.struct_const())}
        torch.manual_seed(seed)
        model = build("PreNat", data, algebra=alg)
        train(model, data, STEPS, LR)
        vals[c] = seed_means(model, val[:, 0], [val[:, 1], val[:, 2]], val[:, 3], which=0)
        testnc[c] = seed_means(model, test[ncm][:, 0], [test[ncm][:, 1], test[ncm][:, 2]],
                               test[ncm][:, 3], which=1)
    torch.manual_seed(seed)
    ml = train_learned_random(data, STEPS, LR)
    learned = seed_means(ml, test[ncm][:, 0], [test[ncm][:, 1], test[ncm][:, 2]],
                         test[ncm][:, 3], which=1)
    pick = max(vals, key=vals.get)
    return pick, vals, testnc, learned


def main():
    print(f"SELECTION-AT-SCARCITY PROBE  seeds={SEEDS} steps={STEPS} lr={LR}")
    print("Pre-registered: PROBE-PREREG.md (predictions + kill criterion written first).\n")
    for data_group in ("D4", "Q8"):
        for frac in (1.0, 0.5, 0.3):
            picks, valtab = [], {c: [] for c in ORDER8}
            sel_test, oracle_test, learned_test = [], [], []
            for seed in range(SEEDS):
                pick, vals, testnc, learned = select_once(data_group, 6, frac, seed)
                picks.append(pick)
                for c in ORDER8:
                    valtab[c].append(vals[c])
                sel_test.append(testnc[pick])
                oracle_test.append(testnc[data_group])
                learned_test.append(learned)
            ncorrect = sum(p == data_group for p in picks)
            phat, lo, hi = wilson(ncorrect, SEEDS)
            best_wrong = max((c for c in ORDER8 if c != data_group),
                             key=lambda c: np.mean(valtab[c]))
            d, p = paired_perm_test(valtab[data_group], valtab[best_wrong])
            print(f"data={data_group} frac={frac}: correct {ncorrect}/{SEEDS}  "
                  f"Wilson95 [{lo:.2f},{hi:.2f}]  picks={picks}")
            print(f"  margin true-vs-best-wrong({best_wrong}): {d:+.3f}  p={p:.4f}")
            print(f"  test NONCOMM H@1: selected {mean_ci_str(sel_test)}  "
                  f"oracle {mean_ci_str(oracle_test)}  learned-random {mean_ci_str(learned_test)}\n")


if __name__ == "__main__":
    main()
