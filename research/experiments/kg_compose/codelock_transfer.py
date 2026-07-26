"""Code-locking attack #2: AMORTIZED / transfer init (the one promising angle left).

The structural tricks (fix-entity, low-rank, sparsity) and naturality all failed to learn the
algebra at low data. The remaining idea: LEARN the algebra ONCE on an abundant SOURCE instance, then
reuse those frozen structure constants on a scarce TARGET instance -- learning only codes + entities.
The S4 group algebra is the same across instances (only the relation/entity labelling differs), so a
transferable gamma would mean you amortise the hard part once and lock on cheaply thereafter.

  scratch       : learn gamma + codes + entities from scratch on the target (the failing baseline).
  transfer(froz): gamma frozen at the SOURCE-learned algebra; learn only codes + entities.
  oracle(fixed) : the true group algebra (upper bound / data-efficiency target).

If transfer(frozen) >> scratch at low data and approaches oracle, the blocker is crackable by
amortisation -- learn the algebra on data-rich tasks, deploy on data-poor ones.

Run:  python codelock_transfer.py
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
from kg_models import PreNatLearnedAlgebra, build
from kg_run import train as train_fixed, metrics


def train_la(model, data, steps, lr, freeze_gamma=False):
    params = [p for n, p in model.named_parameters() if not (freeze_gamma and n == "gamma")]
    opt = torch.optim.Adam(params, lr=lr)
    tr = data["train"]; h, r, t = tr[:, 0], tr[:, 1], tr[:, 2]
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model.score_all(h, [r]), t)
        if not freeze_gamma:
            loss = loss + model.assoc_loss()
        loss.backward(); opt.step()


def noncomm_h1(model, data):
    pq = data["pquery"]; nc = pq[:, 4].bool()
    return metrics(model, pq[nc][:, 0], [pq[nc][:, 1], pq[nc][:, 2]], pq[nc][:, 3])[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.5, 0.3])
    args = ap.parse_args()

    # SOURCE: learn the algebra once on an abundant instance (full data, seed 99)
    src = make_kg(args.group, 8, 1.0, seed=99)
    torch.manual_seed(99)
    src_model = PreNatLearnedAlgebra(src["n_entities"], src["n_relations"], src["n"], init="random")
    train_la(src_model, src, args.steps, args.lr)
    gamma_star = src_model.gamma.detach().clone()
    print(f"Source algebra learned on full data: NONCOMM H@1 = {noncomm_h1(src_model, src):.3f} "
          f"(assoc resid {float(src_model.assoc_loss()):.1e})\n")

    print(f"AMORTIZED TRANSFER on {args.group} KG (NONCOMM path H@1)  seeds={args.seeds}")
    print(f"  {'frac':6s} | {'oracle (fixed)':^14s} | {'scratch':^12s} | {'transfer (frozen)':^17s}")
    print("  " + "-" * 56)
    for frac in args.fracs:
        o, sc, tr_ = [], [], []
        for seed in range(args.seeds):
            data = make_kg(args.group, 8, frac, seed=seed)
            ne, nr, n = data["n_entities"], data["n_relations"], data["n"]
            torch.manual_seed(seed)
            mo = build("PreNat", data); train_fixed(mo, data, args.steps, args.lr)
            o.append(noncomm_h1(mo, data))
            torch.manual_seed(seed)
            ms = PreNatLearnedAlgebra(ne, nr, n, init="random"); train_la(ms, data, args.steps, args.lr)
            sc.append(noncomm_h1(ms, data))
            torch.manual_seed(seed)
            mt = PreNatLearnedAlgebra(ne, nr, n, init="random")
            mt.gamma.data = gamma_star.clone()
            train_la(mt, data, args.steps, args.lr, freeze_gamma=True)
            tr_.append(noncomm_h1(mt, data))
        print(f"  {frac:0.2f}   | {np.mean(o):.3f}+/-{np.std(o):.2f}    | {np.mean(sc):.3f}+/-{np.std(sc):.2f} | "
              f"{np.mean(tr_):.3f}+/-{np.std(tr_):.2f}")

    print("\n  If transfer(frozen) >> scratch and nears oracle, the algebra is AMORTISABLE: learn it "
          "once on abundant data, freeze, and lock on cheaply at low data -- the practical crack for "
          "the blocker.")


if __name__ == "__main__":
    main()
