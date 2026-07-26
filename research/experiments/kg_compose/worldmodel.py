"""Direction A: a world model where the algebra is REAL (actions genuinely compose).

A gridworld agent on a 3x3 torus with 4 orientations (36 states). Primitive actions:
  R = rotate left        (permutation; does NOT commute with F -> non-abelian)
  F = move forward       (permutation on the torus, orientation-dependent)
  Z = reset to origin    (sends EVERY state to state 0 -> NON-invertible)
The transition dynamics are literally a non-abelian, non-invertible transformation monoid -- so
"compose action operators exactly" is not an assumption, it's the truth.

We learn action operators from SINGLE-step transitions and evaluate MULTI-step rollout accuracy as
a function of horizon. Models (all operate on a state embedding, rank the true next state):
  TransE-WM   : action = translation (additive, abelian) -> must fail non-abelian / reset.
  RESCAL-WM   : action = free matrix, compose by matmul, NO composition-consistency.
  PreNat-WM   : RESCAL + NATURALITY (2-step composition-consistency self-supervision).

Claim: enforcing composition-consistency (naturality) makes the world model degrade far slower
over the rollout horizon -- the payoff of the exact-composition prior in a domain where it holds.

Run:  python worldmodel.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from kg_models import TransE, RESCAL

W = H = 3
O = 4
N = W * H * O


def sidx(x, y, o):
    return ((y * W) + x) * O + o


def build_world():
    nxtR = np.zeros(N, int); nxtF = np.zeros(N, int); nxtZ = np.zeros(N, int)
    for x in range(W):
        for y in range(H):
            for o in range(O):
                s = sidx(x, y, o)
                nxtR[s] = sidx(x, y, (o + 1) % O)                      # rotate left
                dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][o]
                nxtF[s] = sidx((x + dx) % W, (y + dy) % H, o)          # move forward (torus)
                nxtZ[s] = sidx(0, 0, 0)                                # reset (non-invertible)
    return {0: nxtR, 1: nxtF, 2: nxtZ}                                 # action id -> transition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--max_h", type=int, default=6)
    ap.add_argument("--obs_frac", type=float, default=1.0,
                    help="fraction of single-step (state,action) transitions observed")
    args = ap.parse_args()

    trans = build_world()
    n_act = len(trans)
    # single-step transitions (s, a, s')  -- subsample to obs_frac (partial observation)
    full_atomic = [(s, a, int(trans[a][s])) for a in range(n_act) for s in range(N)]
    rng0 = np.random.default_rng(0)
    if args.obs_frac < 1.0:
        keep = rng0.random(len(full_atomic)) < args.obs_frac
        atomic = np.array([x for x, k in zip(full_atomic, keep) if k], dtype=np.int64)
    else:
        atomic = np.array(full_atomic, dtype=np.int64)
    # 2-step composition-consistency built ONLY from OBSERVED single steps (no leakage)
    obs_next = {(int(s), int(a)): int(t) for s, a, t in atomic}
    paths2 = np.array([(s, a1, a2, obs_next[(m, a2)])
                       for (s, a1), m in obs_next.items() for a2 in range(n_act)
                       if (m, a2) in obs_next], dtype=np.int64)

    def rollout_acc(model, horizon, seed):
        rng = np.random.default_rng(1000 + seed)
        nq = 1000
        s = rng.integers(0, N, nq)
        acts = rng.integers(0, n_act, (nq, horizon))
        true = s.copy()
        for k in range(horizon):
            true = np.array([int(trans[acts[i, k]][true[i]]) for i in range(nq)])
        sh = torch.tensor(s)
        rel_list = [torch.tensor(acts[:, k]) for k in range(horizon)]
        with torch.no_grad():
            scores = model.score_all(sh, rel_list)
        pred = scores.argmax(1).numpy()
        return float((pred == true).mean())

    def train(model, use_nat):
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        s, a, t = (torch.tensor(atomic[:, i]) for i in range(3))
        p2 = torch.tensor(paths2)
        for _ in range(args.steps):
            opt.zero_grad()
            loss = F.cross_entropy(model.score_all(s, [a]), t)
            if use_nat:
                idx = torch.randint(0, len(p2), (1024,))
                loss = loss + F.cross_entropy(
                    model.score_all(p2[idx, 0], [p2[idx, 1], p2[idx, 2]]), p2[idx, 3])
            loss.backward(); opt.step()

    configs = [("TransE-WM", lambda: TransE(N, n_act, N), False),
               ("RESCAL-WM", lambda: RESCAL(N, n_act, N), False),
               ("PreNat-WM", lambda: RESCAL(N, n_act, N), True)]

    print(f"Gridworld world model: {N} states, actions {{R, F, Z}} (Z=reset, non-invertible; "
          f"R,F non-commuting)  seeds={args.seeds}")
    print(f"Multi-step ROLLOUT accuracy vs horizon (rank true final state among all {N})\n")
    hs = list(range(1, args.max_h + 1))
    print("  model      | " + " ".join(f"h={h:<4d}" for h in hs))
    print("  " + "-" * (13 + 6 * len(hs)))
    for name, ctor, use_nat in configs:
        accs = {h: [] for h in hs}
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            m = ctor()
            train(m, use_nat)
            for h in hs:
                accs[h].append(rollout_acc(m, h, seed))
        row = " ".join(f"{np.mean(accs[h]):.2f} " for h in hs)
        print(f"  {name:10s} | {row}")
    print("\n  PreNat-WM (composition-consistency / naturality) should hold accuracy over the "
          "horizon where RESCAL-WM decays and TransE-WM (additive) fails early -- the value of "
          "exact composition in a domain where actions genuinely compose.")


if __name__ == "__main__":
    main()
