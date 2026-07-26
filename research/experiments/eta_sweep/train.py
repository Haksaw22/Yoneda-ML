"""Training loop + held-out compositional-transport evaluation.

Losses:  L = L_task + lambda_nat * L_nat + lambda_cycle * L_cycle
  L_task  : MSE on OBSERVED observations (possibly eta-corrupted).
  L_nat   : soft naturality on observed probe pairs (zero/absent for hard & no-nat).
  L_cycle : two observed transport paths to a held-out (k, A) must agree.

Held-out metric: predict each held-out (k, A) by averaging exact transports from all
observed probes of A, compare to the CLEAN target. Reported as MSE normalised by the
variance of the clean targets (1.0 == no better than predicting the mean).
"""

from __future__ import annotations

import numpy as np
import torch


def _flat_index(value):
    return torch.as_tensor(value, dtype=torch.long)


def train_model(model, world, splits, obs, cfg):
    dev = world["F"].device
    observed = splits["observed"]
    obs_mask = torch.as_tensor(observed, device=dev)            # [P, N]
    O_obs = obs["O_obs"]                                        # [P, N, d_obs]
    cycle = splits["cycle_triples"]

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sel = obs_mask.unsqueeze(-1)
    n_obs = float(obs_mask.sum().item())

    for step in range(cfg["steps"]):
        opt.zero_grad()
        pred = model.predict_grid()                            # [P, N, d_obs]
        L_task = (((pred - O_obs) ** 2) * sel).sum() / (n_obs * world["d_obs"])
        L_nat = model.nat_loss(observed)
        L_cycle = model.cycle_loss(cycle)
        loss = L_task + cfg["lambda_nat"] * L_nat + cfg["lambda_cycle"] * L_cycle
        loss.backward()
        opt.step()

    return dict(
        train_task=float(L_task.detach()),
        train_nat=float(L_nat.detach()),
        train_cycle=float(L_cycle.detach()),
    )


@torch.no_grad()
def eval_heldout(model, world, splits, obs):
    """Mean held-out transport error, normalised by clean-target variance."""
    dev = world["F"].device
    observed = splits["observed"]
    test_pairs = splits["test_pairs"]
    O_clean = obs["O_clean"]

    rows_A, rows_k, rows_i, group_id, targets = [], [], [], [], []
    for gid, (k, A) in enumerate(test_pairs):
        src = np.where(observed[:, A])[0]
        for i in src:
            rows_A.append(A); rows_k.append(k); rows_i.append(int(i)); group_id.append(gid)
        targets.append(O_clean[k, A])

    A_idx = _flat_index(rows_A).to(dev)
    k_idx = _flat_index(rows_k).to(dev)
    i_idx = _flat_index(rows_i).to(dev)
    gid = _flat_index(group_id).to(dev)
    targets = torch.stack(targets)                             # [G, d_obs]

    o_pred, _ = model.transport_predict(A_idx, k_idx, i_idx)   # [rows, d_obs]

    # average predictions per held-out pair (group)
    G = len(test_pairs)
    sums = torch.zeros(G, world["d_obs"], device=dev)
    counts = torch.zeros(G, 1, device=dev)
    sums.index_add_(0, gid, o_pred)
    counts.index_add_(0, gid, torch.ones_like(gid, dtype=sums.dtype).unsqueeze(-1))
    means = sums / counts

    mse = ((means - targets) ** 2).mean().item()
    var = targets.var(unbiased=False).item()
    norm_mse = mse / max(var, 1e-8)
    return dict(heldout_mse=mse, heldout_norm_mse=norm_mse, target_var=var, n_test=G)


def run_one(variant, world, splits, obs, cfg):
    from models import build_model
    torch.manual_seed(cfg["seed"])
    model = build_model(variant, world, cfg)
    tr = train_model(model, world, splits, obs, cfg)
    ev = eval_heldout(model, world, splits, obs)
    return {**tr, **ev}
