"""Promote NATURALITY-AS-A-LOSS into the flagship KG setting -- and test it fairly, with rigor.

The project's stated core novelty is "object-indexed naturality as the learned structural constraint."
Yet the headline KG experiments (kg_run, kg_monoid, puzzle, algebra_select) train on plain atomic
cross-entropy with NO naturality term -- the mechanism is absent exactly where the wins are claimed.
fusion.py tested it once at low data and found it inert; this script gives it the fairest shot the
project never ran:

  * the regime where it *should* help: LEARNING the algebra (codes under-determined), across the full
    data spectrum, not just the collapsed low-data point;
  * a ROBUST (Welsch, redescending) naturality variant -- the eta-sweep's lesson was that a plain
    naturality penalty PROPAGATES corruption; the robust version should be the fair representative;
  * 10 seeds, tie-aware metrics, bootstrap CIs, and a paired significance test vs atomic-only.

Naturality signal = composition-consistency on the training graph's OWN observed 2-hops
(h-r1->m-r2->t, both edges seen), with test composites excluded (no leakage). We report whether
naturality (plain or robust) significantly improves order-sensitive composition at any data fraction.

Run:  python nat_kg.py --seeds 10
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

from kg_data import make_kg
from kg_models import PreNatLearnedAlgebra
from fusion import make_train_paths
from statutils import metric_arrays, mean_ci_str, paired_perm_test

ARMS = ["atomic-only", "+naturality", "+robust-nat"]


def train_arm(data, paths, steps, lr, arm, delta=1.0):
    ne, nr, n = data["n_entities"], data["n_relations"], data["n"]
    torch.manual_seed(0)  # caller seeds before; keep arms identical-init per seed
    m = PreNatLearnedAlgebra(ne, nr, n, init="random")
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    tr = data["train"]; h, r, t = tr[:, 0], tr[:, 1], tr[:, 2]
    use_paths = arm != "atomic-only" and len(paths)
    if use_paths:
        tp = torch.tensor(paths); ph, pr1, pr2, pt = tp[:, 0], tp[:, 1], tp[:, 2], tp[:, 3]
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(m.score_all(h, [r]), t) + m.assoc_loss()
        if use_paths:
            pl = F.cross_entropy(m.score_all(ph, [pr1, pr2]), pt, reduction="none")
            if arm == "+robust-nat":
                w = torch.exp(-(pl.detach() / delta) ** 2)             # Welsch redescending weights
                loss = loss + (w * pl).sum() / (w.sum() + 1e-9)
            else:
                loss = loss + pl.mean()
        loss.backward(); opt.step()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S4")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.3])
    args = ap.parse_args()

    print(f"NATURALITY-AS-A-LOSS in the KG (learned algebra), fair + rigorous  group={args.group} "
          f"seeds={args.seeds}")
    print("NONCOMM H@1 (tie-aware) by data fraction; naturality = train-path composition-consistency.\n")
    print(f"  {'frac':5s} | {'atomic-only':^22s} | {'+naturality':^22s} | {'+robust-nat':^22s} | "
          f"{'best-nat vs atomic':^20s}")
    print("  " + "-" * 100)
    for frac in args.fracs:
        vals = {a: [] for a in ARMS}; npaths = []
        for seed in range(args.seeds):
            data = make_kg(args.group, 8, frac, seed=seed)
            pq = data["pquery"]; nc = pq[:, 4].bool()
            rng = np.random.default_rng(100 + seed)
            exclude = {(int(x[0]), int(x[1]), int(x[2])) for x in pq}
            paths = make_train_paths(data["train"].numpy(), exclude, 2000, rng)
            npaths.append(len(paths))
            for arm in ARMS:
                torch.manual_seed(seed)
                m = train_arm(data, paths, args.steps, args.lr, arm)
                vals[arm].append(float(np.mean(metric_arrays(
                    m, pq[nc][:, 0], [pq[nc][:, 1], pq[nc][:, 2]], pq[nc][:, 3])[1])))
        # best naturality arm vs atomic-only, paired
        bn = "+naturality" if np.mean(vals["+naturality"]) >= np.mean(vals["+robust-nat"]) else "+robust-nat"
        d, p = paired_perm_test(vals[bn], vals["atomic-only"])
        sig = "*" if p < 0.05 else " "
        print(f"  {frac:0.2f}  | {mean_ci_str(vals['atomic-only']):^22s} | "
              f"{mean_ci_str(vals['+naturality']):^22s} | {mean_ci_str(vals['+robust-nat']):^22s} | "
              f"{bn[1:]:>11s} {d:+.3f}{sig}")
    print(f"\n  (paths/seed at the listed fracs vary; naturality can only help where 2-hops are observed.)")
    print("  HONEST READING: '*' marks a statistically significant (p<0.05, paired, n=seeds) gain from")
    print("  naturality over atomic-only. Report the verdict as it falls -- this is the project's")
    print("  central mechanism finally tested with rigor in its flagship setting.")


if __name__ == "__main__":
    main()
