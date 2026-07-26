"""Reality check: PreNat-style models on the real UMLS knowledge graph.

Real KGs do not have a clean finite-group/monoid relation algebra, so the only applicable PreNat
variant is the LEARNED-algebra one (kg_learn_algebra). We compare standard KGE baselines and the
learned-algebra model on:
  (a) standard filtered tail link prediction (MRR / Hits@k) -- is it competitive on real data?
  (b) a 2-hop composition probe (predict the tail of an observed r1->r2 path) -- does the
      algebra-composition inductive bias help where relations only APPROXIMATELY compose?

Honest expectation: standard KGEs are strong on UMLS; the interesting question is whether the
learned-algebra composition structure is competitive and whether it helps on multi-hop.

Run:  python kg_real.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from umls import load_umls
from kg_models import TransE, RotatE, RESCAL, PreNatLearnedAlgebra

MODELS = ["TransE", "RotatE", "RESCAL", "LearnedAlg"]


def make_model(name, n_ent, n_rel, d):
    if name == "TransE":
        return TransE(n_ent, n_rel, d)
    if name == "RotatE":
        return RotatE(n_ent, n_rel, d if d % 2 == 0 else d + 1)
    if name == "RESCAL":
        return RESCAL(n_ent, n_rel, d)
    if name == "LearnedAlg":
        return PreNatLearnedAlgebra(n_ent, n_rel, d, init="cyclic")
    raise ValueError(name)


def train(model, train_arr, n_ent, epochs, bs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    h = torch.tensor(train_arr[:, 0]); r = torch.tensor(train_arr[:, 1]); t = torch.tensor(train_arr[:, 2])
    n = len(train_arr)
    is_la = isinstance(model, PreNatLearnedAlgebra)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(model.score_all(h[idx], [r[idx]]), t[idx])
            if is_la:
                loss = loss + model.assoc_loss()
            loss.backward()
            opt.step()


@torch.no_grad()
def eval_filtered(model, test_arr, filt, n_ent):
    h = torch.tensor(test_arr[:, 0]); r = torch.tensor(test_arr[:, 1]); t = torch.tensor(test_arr[:, 2])
    scores = model.score_all(h, [r])                       # [N, n_ent]
    for i in range(len(test_arr)):
        hi, ri, ti = int(h[i]), int(r[i]), int(t[i])
        others = [x for x in filt[(hi, ri)] if x != ti]
        if others:
            scores[i, others] = -1e9
    true = scores.gather(1, t.unsqueeze(1))
    rank = 1 + (scores > true).sum(1)
    rr = (1.0 / rank.float())
    return (float(rr.mean()), float((rank == 1).float().mean()),
            float((rank <= 3).float().mean()), float((rank <= 10).float().mean()))


def build_path_probe(train_arr, max_q=4000, seed=0):
    rng = np.random.default_rng(seed)
    out_edges = defaultdict(list)                          # h -> list of (r, t)
    tails = defaultdict(set)                               # (h, r) -> tails
    for hh, rr, tt in train_arr:
        out_edges[int(hh)].append((int(rr), int(tt)))
        tails[(int(hh), int(rr))].add(int(tt))
    queries, pfilt = [], defaultdict(set)
    for hh, r1, m in train_arr:
        for r2, t in out_edges[int(m)]:
            queries.append((int(hh), int(r1), int(r2), int(t)))
            pfilt[(int(hh), int(r1), int(r2))].add(int(t))
    queries = np.array(queries, dtype=np.int64)
    if len(queries) > max_q:
        queries = queries[rng.choice(len(queries), max_q, replace=False)]
    return queries, pfilt


@torch.no_grad()
def eval_path(model, queries, pfilt):
    h = torch.tensor(queries[:, 0]); r1 = torch.tensor(queries[:, 1])
    r2 = torch.tensor(queries[:, 2]); t = torch.tensor(queries[:, 3])
    scores = model.score_all(h, [r1, r2])
    for i in range(len(queries)):
        others = [x for x in pfilt[(int(h[i]), int(r1[i]), int(r2[i]))] if x != int(t[i])]
        if others:
            scores[i, others] = -1e9
    true = scores.gather(1, t.unsqueeze(1))
    rank = 1 + (scores > true).sum(1)
    return float((1.0 / rank.float()).mean()), float((rank == 1).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    D = load_umls()
    print(f"UMLS: {D['n_ent']} entities, {D['n_rel']} relations, "
          f"train/valid/test={len(D['train'])}/{len(D['valid'])}/{len(D['test'])}  d={args.d}")
    probe, pfilt = build_path_probe(D["train"])
    print(f"2-hop composition probe: {len(probe)} path queries (sampled from train)\n")

    print(f"  {'model':10s} | {'MRR':^6s} {'H@1':^6s} {'H@3':^6s} {'H@10':^6s} | "
          f"{'pathMRR':^7s} {'pathH@1':^7s}")
    print("  " + "-" * 62)
    for name in MODELS:
        torch.manual_seed(args.seed)
        model = make_model(name, D["n_ent"], D["n_rel"], args.d)
        train(model, D["train"], D["n_ent"], args.epochs, args.bs, args.lr)
        mrr, h1, h3, h10 = eval_filtered(model, D["test"], D["filt"], D["n_ent"])
        pmrr, ph1 = eval_path(model, probe, pfilt)
        print(f"  {name:10s} | {mrr:.3f}  {h1:.3f}  {h3:.3f}  {h10:.3f} | {pmrr:.3f}   {ph1:.3f}")

    print("\n  (Filtered tail ranking. Honest reality check: are PreNat-style learned-algebra "
          "models competitive on a real KG, and does composition structure help on 2-hop?)")


if __name__ == "__main__":
    main()
