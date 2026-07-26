"""Part 2: the LEARNED-rho model + the collapse-guard ablation.

The fixed-rho toy (sweep.py) deliberately froze rho = regular rep, so the
commutativity/subalgebra-collapse failure mode could not appear. Here rho (the R_g
matrices) is LEARNED, which re-introduces it. We ask three things:

  1. Can a learned-rho PreNat still recover non-abelian compositional transport
     (matching the fixed-rho reference)?
  2. Is L_homo (rho an algebra homomorphism) load-bearing? -> ablate it.
  3. Do L_comm (non-commutativity floor) and L_recon (Yoneda autoencoding) prevent
     collapse to an abelian / non-faithful rho? -> ablate them, especially when data
     is sparse (where L_task alone no longer pins rho).

Diagnostics reported per run:
  heldout : normalised held-out transport MSE (1.0 = predicting the mean; lower better)
  comm    : mean non-commutativity ||R_gR_h - R_hR_g|| of the learned rho (tau = floor target)
  homo_err: ||R_gR_h - R_{g.h}||^2  (0 => rho is a genuine representation)
  faith   : min distance between distinct R_g (0 => collapsed, elements indistinguishable)

Run:  python learned_rho.py                 (dense regime, eta=0)
      python learned_rho.py --n_obs_probes 3 --eta 0.0   (sparse: guards should matter more)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from groups import GROUPS
from data import make_world, make_splits, make_observations
from models import PreNatLearnedRho, build_model
from train import eval_heldout

ABLATIONS = {
    "full":          dict(homo=1.0, comm=1.0, recon=1.0),
    "no-homo":       dict(homo=0.0, comm=1.0, recon=1.0),
    "no-comm":       dict(homo=1.0, comm=0.0, recon=1.0),
    "no-recon":      dict(homo=1.0, comm=1.0, recon=0.0),
    "no-comm-recon": dict(homo=1.0, comm=0.0, recon=0.0),
}


def make_recon(splits, device):
    """Per object: predict its first observed probe from its second (Yoneda autoencoding)."""
    observed = splits["observed"]
    N = observed.shape[1]
    A_l, tgt_l, src_l = [], [], []
    for A in range(N):
        obs_i = np.where(observed[:, A])[0]
        if len(obs_i) >= 2:
            A_l.append(A); tgt_l.append(int(obs_i[0])); src_l.append(int(obs_i[1]))
    t = lambda x: torch.tensor(x, device=device, dtype=torch.long)
    return dict(A=t(A_l), tgt=t(tgt_l), src=t(src_l))


def train_learned(model, world, splits, obs, recon, cfg):
    dev = world["F"].device
    obs_mask = torch.as_tensor(splits["observed"], device=dev)
    O_obs = obs["O_obs"]
    sel = obs_mask.unsqueeze(-1)
    n_obs = float(obs_mask.sum().item())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    for _ in range(cfg["steps"]):
        opt.zero_grad()
        pred = model.predict_grid()
        L_task = (((pred - O_obs) ** 2) * sel).sum() / (n_obs * world["d_obs"])
        L_nat = model.nat_loss(splits["observed"])
        L_cycle = model.cycle_loss(splits["cycle_triples"])
        L_homo = model.homo_loss()
        L_comm = model.comm_loss()
        L_recon = model.recon_loss(recon, O_obs)
        loss = (L_task + cfg["lambda_nat"] * L_nat + cfg["lambda_cycle"] * L_cycle
                + cfg["lam"]["homo"] * L_homo + cfg["lam"]["comm"] * L_comm
                + cfg["lam"]["recon"] * L_recon)
        loss.backward()
        opt.step()
    return dict(task=float(L_task.detach()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S3", choices=list(GROUPS))
    ap.add_argument("--n_objects", type=int, default=120)
    ap.add_argument("--n_probes", type=int, default=8)
    ap.add_argument("--n_obs_probes", type=int, default=4)
    ap.add_argument("--d_obs", type=int, default=3)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--lambda_nat", type=float, default=1.0)
    ap.add_argument("--lambda_cycle", type=float, default=1.0)
    ap.add_argument("--rep_init", default="perturbed", choices=["perturbed", "random"])
    args = ap.parse_args()

    G = GROUPS[args.group]()
    print(f"Group {G!s}  rep_init={args.rep_init}  eta={args.eta}  "
          f"obs_probes={args.n_obs_probes}  steps={args.steps}  seeds={args.seeds}")
    print("(held-out: normalised transport MSE, lower better; comm: learned non-commutativity; "
          "faith: min distinct-R distance)\n")

    variants = list(ABLATIONS) + ["fixed-rho (ref)"]
    agg = {v: {"heldout": [], "comm": [], "homo": [], "faith": []} for v in variants}
    tau_ref = None

    for seed in range(args.seeds):
        world = make_world(G, args.n_objects, args.n_probes, args.d_obs, seed=seed)
        splits = make_splits(world, args.n_obs_probes, seed=seed)
        obs = make_observations(world, splits, args.eta, seed=seed)
        recon = make_recon(splits, world["F"].device)

        for name, lam in ABLATIONS.items():
            torch.manual_seed(seed)
            model = PreNatLearnedRho(world, rep_init=args.rep_init)
            tau_ref = model.comm_tau
            cfg = dict(steps=args.steps, lr=args.lr, lambda_nat=args.lambda_nat,
                       lambda_cycle=args.lambda_cycle, lam=lam)
            train_learned(model, world, splits, obs, recon, cfg)
            ev = eval_heldout(model, world, splits, obs)
            dg = model.diagnostics()
            agg[name]["heldout"].append(ev["heldout_norm_mse"])
            agg[name]["comm"].append(dg["comm_use"])
            agg[name]["homo"].append(dg["homo_err"])
            agg[name]["faith"].append(dg["faith_min"])

        # fixed-rho reference (Part-1 model, same data)
        torch.manual_seed(seed)
        ref = build_model("soft", world, dict(init_scale=0.1, nat_delta=0.5))
        opt = torch.optim.Adam(ref.parameters(), lr=0.05)
        obs_mask = torch.as_tensor(splits["observed"], device=world["F"].device)
        sel = obs_mask.unsqueeze(-1); n_obs = float(obs_mask.sum().item())
        for _ in range(600):
            opt.zero_grad()
            pred = ref.predict_grid()
            Lt = (((pred - obs["O_obs"]) ** 2) * sel).sum() / (n_obs * world["d_obs"])
            loss = Lt + ref.nat_loss(splits["observed"]) + ref.cycle_loss(splits["cycle_triples"])
            loss.backward(); opt.step()
        ev = eval_heldout(ref, world, splits, obs)
        agg["fixed-rho (ref)"]["heldout"].append(ev["heldout_norm_mse"])
        for k in ("comm", "homo", "faith"):
            agg["fixed-rho (ref)"][k].append(float("nan"))

    def ms(v, key):
        a = np.array(agg[v][key], dtype=float)
        if np.all(np.isnan(a)):
            return float("nan"), float("nan")
        return np.nanmean(a), np.nanstd(a)

    print(f"  comm floor target tau = {tau_ref:.3f}\n")
    print(f"  {'variant':16s} | {'heldout':^15s} | {'comm':^13s} | {'homo_err':^11s} | {'faith':^11s}")
    print("  " + "-" * 76)
    for v in variants:
        hm, hs = ms(v, "heldout"); cm, cs = ms(v, "comm")
        hom, _ = ms(v, "homo"); fm, _ = ms(v, "faith")
        comm_str = "    n/a    " if np.isnan(cm) else f"{cm:6.3f}"
        homo_str = "   n/a   " if np.isnan(hom) else f"{hom:.2e}"
        faith_str = "   n/a   " if np.isnan(fm) else f"{fm:6.3f}"
        print(f"  {v:16s} | {hm:6.3f} +/-{hs:5.3f} | {comm_str:^13s} | {homo_str:^11s} | {faith_str:^11s}")

    print("\nReading: 'full' should match 'fixed-rho (ref)' on heldout (learned rho recovers "
          "transport). 'no-homo' should blow up heldout (rho not a representation -> transport "
          "meaningless). Compare comm/faith of 'no-comm'/'no-comm-recon' vs 'full' to see whether "
          "the guards keep rho non-abelian/faithful; the effect is expected to grow as data gets "
          "sparser (try --n_obs_probes 3).")


if __name__ == "__main__":
    main()
