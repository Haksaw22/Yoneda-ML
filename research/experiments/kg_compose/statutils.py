"""Shared statistical-rigor primitives for the credible re-run of the PreNat experiments.

Three problems in the original harness this module fixes:

1. TIE-OPTIMISTIC RANKING.  The original metric is `rank = 1 + (scores > true).sum(1)` (strict
   `>`), so any entity tying the true score is counted *behind* it -- maximally generous to a model
   that produces exact ties (PreNat, whose exact composition yields distance exactly 0 on the truth
   and ties elsewhere).  We use the standard *mid-rank* convention
       rank = 1 + #{strictly better} + 0.5 * #{tied, excluding self}
   so a k-way tie at the top yields rank (k+1)/2, not 1.  This neither rewards nor punishes ties;
   it removes the structural advantage to the exact model.  (A genuinely-correct exact prediction --
   truth strictly closest -- still gets rank 1.)

2. NO UNCERTAINTY.  Everything was mean +/- std over 3 seeds.  We report bootstrap 95% confidence
   intervals (percentile method) over the resampling unit that matters (queries, or seeds).

3. NO SIGNIFICANCE / SELECTION CONFIDENCE.  "Selects the true group 3/3" gave no confidence; we
   report a Wilson score interval for the selection probability, and a paired sign-flip permutation
   test for "model A beats model B".

Everything here is deterministic given a numpy Generator seed (no Date/random global state).
"""

from __future__ import annotations

import numpy as np
import torch

TOL = 1e-6   # absolute tolerance for "tied score" (cdist distances are O(1))


# --------------------------------------------------------------------------------------------------
# tie-aware ranking metrics
# --------------------------------------------------------------------------------------------------

@torch.no_grad()
def rr_hits(scores: torch.Tensor, true: torch.Tensor):
    """Per-query reciprocal rank and hit@{1,3,10} under the MID-RANK tie convention.

    scores [B, n_ent] (higher = better), true [B] (gold entity index).
    Returns numpy arrays (rr, h1, h3, h10), each [B], so the caller can bootstrap over queries.
    """
    true_score = scores.gather(1, true.unsqueeze(1))                    # [B,1]
    better = (scores > true_score + TOL).sum(1).float()                 # strictly better
    tied = (torch.abs(scores - true_score) <= TOL).sum(1).float() - 1.0  # tied, minus self
    tied = tied.clamp(min=0.0)
    rank = 1.0 + better + 0.5 * tied                                    # mid-rank
    rr = (1.0 / rank).cpu().numpy()
    rank_np = rank.cpu().numpy()
    return rr, (rank_np <= 1.0).astype(float), (rank_np <= 3.0).astype(float), \
        (rank_np <= 10.0).astype(float)


@torch.no_grad()
def metric_arrays(model, h, rel_list, t):
    """Tie-aware per-query (rr, h1, h3, h10) arrays for a model with `.score_all`."""
    return rr_hits(model.score_all(h, rel_list), t)


# --------------------------------------------------------------------------------------------------
# bootstrap confidence intervals
# --------------------------------------------------------------------------------------------------

def boot_ci(values, n_boot=4000, alpha=0.05, seed=0, stat=np.mean):
    """Percentile bootstrap CI for `stat` over a 1-D sample (e.g. per-query metric values, or
    per-seed scalars). Returns (point, lo, hi)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(stat(v))
    if len(v) == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = stat(v[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def fmt_ci(point, lo, hi, prec=3):
    return f"{point:.{prec}f} [{lo:.{prec}f},{hi:.{prec}f}]"


def mean_ci_str(values, prec=3, **kw):
    return fmt_ci(*boot_ci(values, **kw), prec=prec)


# --------------------------------------------------------------------------------------------------
# proportion (selection-probability) interval
# --------------------------------------------------------------------------------------------------

def wilson(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (phat, lo, hi).

    For 'selected the true group 10/10', this reports the *confidence* the prose omitted:
    10/10 -> phat 1.00, lo ~0.72 (not 'certain')."""
    if n == 0:
        return float("nan"), 0.0, 1.0
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return phat, float(max(0.0, centre - half)), float(min(1.0, centre + half))


# --------------------------------------------------------------------------------------------------
# paired significance: does A beat B?
# --------------------------------------------------------------------------------------------------

def paired_perm_test(a, b, n_perm=20000, seed=0):
    """Two-sided paired sign-flip permutation test on mean(a-b) for paired samples a,b
    (e.g. per-seed or per-query metric for two models on the same items).
    Returns (mean_diff, p_value)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), float("nan")
    obs = float(d.mean())
    if np.allclose(d, 0):
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(axis=1)
    p = (np.abs(null) >= abs(obs) - 1e-12).mean()
    return obs, float(p)


def paired_diff_ci(a, b, **kw):
    """Bootstrap CI for the paired mean difference mean(a-b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return boot_ci(a - b, **kw)


if __name__ == "__main__":
    # self-test: mid-rank removes the tie-optimism; a top 3-way tie -> rank 2 -> not Hits@1.
    s = torch.tensor([[0.0, 0.0, 0.0, -1.0]])   # entities 0,1,2 tie at the top
    rr, h1, h3, h10 = rr_hits(s, torch.tensor([0]))
    print(f"3-way top tie: rr={rr[0]:.3f} (=1/2), h1={h1[0]} (=0)  [strict-> would give h1=1]")
    s2 = torch.tensor([[1.0, 0.0, 0.0, -1.0]])  # entity 0 strictly best
    rr2, h12, _, _ = rr_hits(s2, torch.tensor([0]))
    print(f"clean win:     rr={rr2[0]:.3f} (=1), h1={h12[0]} (=1)")
    print("wilson 10/10:", tuple(round(x, 3) for x in wilson(10, 10)))
    print("wilson 3/3:  ", tuple(round(x, 3) for x in wilson(3, 3)))
    a = [1, 1, 1, 0, 1]; b = [0, 1, 0, 0, 1]
    print("paired test:", paired_perm_test(a, b))
