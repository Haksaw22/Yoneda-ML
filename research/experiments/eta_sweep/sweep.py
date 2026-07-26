"""The make-or-break experiment: soft vs hard naturality across the eta-sweep.

Pre-registered prediction
-------------------------
  - eta = 0 (exactly functorial): soft ~= hard  (hard is correctly specified; a tie
    here is PREDICTED, not a failure).
  - eta growing (near-functorial): soft pulls ahead of hard; the gap grows with eta.
  - abelian and no-nat fail held-out transport at all eta.

Pre-registered falsification: if `hard` matches `soft` across the ENTIRE sweep
(including large eta), the central thesis is dead -- it would be "hard-wired naturality
with extra steps". This script is built to let that outcome show.

Run:  python sweep.py            (defaults below)
      python sweep.py --group D4 --seeds 5 --steps 800
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from groups import GROUPS
from data import make_world, make_splits, make_observations
from train import run_one

VARIANTS = ["soft", "soft-huber", "soft-welsch", "hard", "no-nat"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S3", choices=list(GROUPS))
    ap.add_argument("--n_objects", type=int, default=120)
    ap.add_argument("--n_probes", type=int, default=8)
    ap.add_argument("--n_obs_probes", type=int, default=4)
    ap.add_argument("--d_obs", type=int, default=3)
    ap.add_argument("--etas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lambda_nat", type=float, default=1.0)
    ap.add_argument("--lambda_cycle", type=float, default=1.0)
    ap.add_argument("--nat_delta", type=float, default=0.5)
    ap.add_argument("--init_scale", type=float, default=0.1)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--plot", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    G = GROUPS[args.group]()
    assert args.d_obs < G.n, f"need d_obs < |G|={G.n} for under-determination"
    print(f"Group {G!s}  |G|={G.n}  abelian={G.is_abelian()}")
    print(f"objects={args.n_objects} probes={args.n_probes} obs_probes={args.n_obs_probes} "
          f"d_obs={args.d_obs} steps={args.steps} seeds={args.seeds}\n")

    # results[variant][eta] = list of normalised held-out MSE over seeds
    results = {v: {e: [] for e in args.etas} for v in VARIANTS}
    rows = []

    for seed in range(args.seeds):
        world = make_world(G, args.n_objects, args.n_probes, args.d_obs, seed=seed)
        splits = make_splits(world, args.n_obs_probes, seed=seed)
        for eta in args.etas:
            obs = make_observations(world, splits, eta, seed=seed)
            for v in VARIANTS:
                cfg = dict(
                    steps=args.steps, lr=args.lr,
                    lambda_nat=(0.0 if v == "no-nat" else args.lambda_nat),
                    lambda_cycle=(0.0 if v == "no-nat" else args.lambda_cycle),
                    nat_delta=args.nat_delta,
                    init_scale=args.init_scale, seed=seed,
                )
                out = run_one(v, world, splits, obs, cfg)
                results[v][eta].append(out["heldout_norm_mse"])
                rows.append(dict(
                    seed=seed, eta=eta, variant=v,
                    heldout_norm_mse=out["heldout_norm_mse"],
                    heldout_mse=out["heldout_mse"],
                    train_task=out["train_task"], train_nat=out["train_nat"],
                    n_corrupt=obs["n_corrupt"], n_test=out["n_test"],
                ))

    # ---- table ----
    def stat(v, e):
        a = np.array(results[v][e])
        return a.mean(), a.std()

    print("Held-out transport error (normalised MSE; 1.0 = predicting the mean; lower better)")
    header = "  eta  | " + " | ".join(f"{v:^15s}" for v in VARIANTS)
    print(header)
    print("-" * len(header))
    for e in args.etas:
        cells = []
        for v in VARIANTS:
            m, s = stat(v, e)
            cells.append(f"{m:6.3f} +/-{s:5.3f}")
        print(f"  {e:0.2f} | " + " | ".join(f"{c:^15s}" for c in cells))

    print("\nSoft-minus-hard gap (negative => soft better; should grow more negative with eta):")
    for e in args.etas:
        sm, _ = stat("soft", e)
        hm, _ = stat("hard", e)
        print(f"  eta={e:0.2f}:  soft-hard = {sm - hm:+.4f}")

    # verdict heuristic
    gaps = [stat("soft", e)[0] - stat("hard", e)[0] for e in args.etas]
    big_eta = [e for e in args.etas if e >= 0.3]
    if big_eta:
        soft_wins_big = np.mean([stat("soft", e)[0] - stat("hard", e)[0] for e in big_eta]) < -0.02
        tie_small = abs(stat("soft", args.etas[0])[0] - stat("hard", args.etas[0])[0]) < 0.05
        print("\nVERDICT (heuristic):")
        print(f"  tie at eta={args.etas[0]:.2f}: {tie_small}")
        print(f"  soft beats hard at large eta: {soft_wins_big}")
        if soft_wins_big and tie_small:
            print("  -> consistent with the thesis: pursue PreNat.")
        elif not soft_wins_big:
            print("  -> soft does NOT beat hard at large eta: thesis in danger (per pre-registration).")

    # robustness recovery (Part 1): does a robust L_nat recover the high-eta regime,
    # where plain soft was beaten by no-nat?
    robust_variants = [v for v in ("soft-huber", "soft-welsch") if v in results]
    if robust_variants and "no-nat" in results:
        print("\nROBUSTNESS (best robust soft vs plain soft vs no-nat; lower better):")
        for e in args.etas:
            sp = stat("soft", e)[0]
            nn_ = stat("no-nat", e)[0]
            rb = min(stat(v, e)[0] for v in robust_variants)
            tag = "   <- robust beats both" if (rb <= sp + 1e-9 and rb <= nn_ + 1e-9) else ""
            print(f"  eta={e:0.2f}:  soft={sp:.3f}  no-nat={nn_:.3f}  best-robust={rb:.3f}{tag}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out} ({len(rows)} rows).")

    if args.plot:
        _plot(args, results)


def _plot(args, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: {e})")
        return
    plt.figure(figsize=(7, 5))
    for v in VARIANTS:
        m = [np.mean(results[v][e]) for e in args.etas]
        s = [np.std(results[v][e]) for e in args.etas]
        plt.errorbar(args.etas, m, yerr=s, marker="o", capsize=3, label=v)
    plt.xlabel("eta (functoriality violation)")
    plt.ylabel("held-out transport error (normalised MSE)")
    plt.title(f"soft vs hard naturality eta-sweep  ({args.group})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out = "eta_sweep.png"
    plt.savefig(out, dpi=130)
    print(f"Wrote {out}.")


if __name__ == "__main__":
    main()
