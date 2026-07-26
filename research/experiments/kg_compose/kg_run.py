"""Benchmark: non-abelian relation composition (atomic + path queries).

Hypotheses
----------
- Atomic link prediction: all models competitive (PreNat is not crippled on the easy task).
- Path composition, ORDER-SENSITIVE subset: abelian models (TransE/RotatE) FAIL by construction
  (they cannot tell r1.r2 from r2.r1); PreNat and RESCAL can represent it.
- PreNat vs RESCAL: with the fixed group algebra PreNat uses far fewer relation params, so it
  should match/beat the free-matrix model and degrade more gracefully as data gets scarce
  (lower --train_frac). If they tie at full data, the honest claim is parameter efficiency.

Run:  python kg_run.py                       (S4, full data)
      python kg_run.py --train_frac 0.2      (scarce data: sample-efficiency regime)
      python kg_run.py --group S5 --n_relations 12
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from kg_data import make_kg
from kg_models import build

MODELS = ["TransE", "RotatE", "RESCAL", "PreNat"]


def train(model, data, steps, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr = data["train"]
    h, r, t = tr[:, 0], tr[:, 1], tr[:, 2]
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model.score_all(h, [r]), t)
        loss.backward()
        opt.step()
    return float(loss.detach())


@torch.no_grad()
def metrics(model, h, rel_list, t):
    scores = model.score_all(h, rel_list)
    true = scores.gather(1, t.unsqueeze(1))
    rank = 1 + (scores > true).sum(1)                       # unique true tail => filtered == raw
    return (float((1.0 / rank.float()).mean()),
            float((rank == 1).float().mean()),
            float((rank <= 3).float().mean()))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S4")
    ap.add_argument("--n_relations", type=int, default=8)
    ap.add_argument("--train_frac", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.02)
    args = ap.parse_args()

    agg = {m: {k: [] for k in ("atom_mrr", "atom_h1", "path_mrr", "path_h1",
                               "nc_h1", "c_h1", "params")} for m in MODELS}
    meta = {}
    for seed in range(args.seeds):
        data = make_kg(args.group, args.n_relations, args.train_frac, seed=seed)
        meta = data
        torch.manual_seed(seed)
        pq = data["pquery"]
        ph, pr1, pr2, pt, nc = pq[:, 0], pq[:, 1], pq[:, 2], pq[:, 3], pq[:, 4]
        ta = data["test_atomic"]
        for name in MODELS:
            torch.manual_seed(seed)
            model = build(name, data)
            train(model, data, args.steps, args.lr)
            a_mrr, a_h1, _ = metrics(model, ta[:, 0], [ta[:, 1]], ta[:, 2])
            p_mrr, p_h1, _ = metrics(model, ph, [pr1, pr2], pt)
            ncm = nc.bool()
            nc_h1 = metrics(model, ph[ncm], [pr1[ncm], pr2[ncm]], pt[ncm])[1]
            c_h1 = metrics(model, ph[~ncm], [pr1[~ncm], pr2[~ncm]], pt[~ncm])[1]
            agg[name]["atom_mrr"].append(a_mrr); agg[name]["atom_h1"].append(a_h1)
            agg[name]["path_mrr"].append(p_mrr); agg[name]["path_h1"].append(p_h1)
            agg[name]["nc_h1"].append(nc_h1); agg[name]["c_h1"].append(c_h1)
            agg[name]["params"].append(n_params(model))

    print(f"Group {meta['group_name']}  entities={meta['n_entities']}  "
          f"relations={meta['n_relations']}  train_frac={args.train_frac}  "
          f"non-commuting path frac={meta['frac_noncomm']:.2f}  seeds={args.seeds}")
    print("(MRR / Hits@1, higher better; ranking over all entities)\n")
    hdr = (f"  {'model':8s} | {'atomic MRR':^12s} | {'path MRR':^12s} | "
           f"{'path H@1':^12s} | {'NONCOMM H@1':^13s} | {'comm H@1':^11s} | params")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    def ms(name, k):
        a = np.array(agg[name][k]); return a.mean(), a.std()

    for name in MODELS:
        am, asd = ms(name, "atom_mrr"); pm, psd = ms(name, "path_mrr")
        ph1m, ph1s = ms(name, "path_h1"); ncm, ncs = ms(name, "nc_h1")
        cm, _ = ms(name, "c_h1"); pr = int(ms(name, "params")[0])
        print(f"  {name:8s} | {am:5.3f}+/-{asd:4.3f} | {pm:5.3f}+/-{psd:4.3f} | "
              f"{ph1m:5.3f}+/-{ph1s:4.3f} | {ncm:5.3f}+/-{ncs:4.3f} | {cm:5.3f}     | {pr}")

    print("\nKey column = NONCOMM H@1 (order-sensitive composition). Abelian TransE/RotatE should "
          "be near-floor there; PreNat/RESCAL should be high. Compare PreNat vs RESCAL across "
          "--train_frac to test the sample-efficiency claim of the fixed group algebra.")


if __name__ == "__main__":
    main()
