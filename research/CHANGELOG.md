# Credibility upgrade — changelog

*An independent audit of this project found that the science was reproducible but the **framing
out-ran the evidence** in specific, fixable ways. This round fixes them: it adds statistical rigor,
de-tautologises the headline, corrects the overstated numbers, and adds experiments that **earn** the
project's own framing. Every number below is from a fresh run; honest negatives are kept. New code
lives in [`experiments/kg_compose/`](experiments/kg_compose/); the originals are unchanged so the
deltas are attributable.*

All new results: **seeds = 10** (was 3), **tie-aware mid-rank** metrics (was tie-optimistic strict
`>`), **bootstrap 95% CIs** (was mean±std over 3 points), **Wilson intervals** for selection
probability, and **paired permutation tests** for "A beats B". Primitives in
[`statutils.py`](experiments/kg_compose/statutils.py); rigor re-run in
[`rigor.py`](experiments/kg_compose/rigor.py).

---

## 1. Three methodological holes, closed

| Hole (original) | Fix |
|---|---|
| `rank = 1 + (scores > true).sum()` — strict `>`, **tie-optimistic**, flatters the exact-tie model (PreNat) | mid-rank `1 + #{>} + 0.5·#{=}` ([`statutils.rr_hits`](experiments/kg_compose/statutils.py)). *Finding: it does **not** change the headline — PreNat's 1.000 are genuine strict wins, not tie artifacts.* |
| 3 seeds, mean±std, no CIs, no significance | 10 seeds, bootstrap 95% CIs, paired permutation tests everywhere |
| "selects the true group 3/3" with no confidence | Wilson interval. *3/3 only certifies selection-prob ≥ 0.44; we now report 10-seed rates.* |

## 2. Headline claims, re-run with rigor (what held, what moved)

**Held up (now with significance):**

- **Non-abelian S4 composition.** NONCOMM H@1: PreNat **1.000 [1.000,1.000]**, RESCAL 0.825 [0.785,0.866],
  RotatE 0.375, TransE 0.035. Paired PreNat−RESCAL **p = 0.0021**. At 50% data PreNat 0.687 vs RESCAL
  0.208 (**p = 0.0021**) — the data-efficiency claim is real and significant. (*p = 0.0021 is the floor of
  a two-sided sign-flip test at n = 10 seeds — reached when all 10 paired differences share a sign — so
  read it as "as significant as 10 paired seeds allow", not a precise tail probability.*)
- **T3 monoid (non-invertible).** path NON-INV H@1: PreNat **0.884 [0.836,0.929]**, RESCAL 0.803,
  RotatE 0.409 (PreNat−RotatE **p = 0.0021**), at 1/6.9 the params.
- **Algebra selection.** Q8 selected **10/10** (Wilson [0.72,1.00]); D4 **8/10** ([0.49,0.94]) — true
  group wins val-path-MRR with a significant margin (D4 +0.072 p=0.013; Q8 +0.132 p=0.002).

**Moved (overstated → corrected):**

- **"Drift-free at any horizon" is false for the *learned* model.** At 10 seeds the learned PreNat is
  **0.86 at h=40 and collapses to 0.14 [0.04,0.33] by h=200** — its codes never lock to exact one-hot
  (gauge freedom). The exact, drift-free property belongs **only to the committed *symbolic* model**
  (1.00 at h=200, by construction). Worse, on small clean worlds (D4) **a free-matrix RESCAL locks
  exact permutations and out-drifts PreNat** — the "exact beats free on long horizons" story is
  **regime-specific**, not a law. See [`rigor.py`](experiments/kg_compose/rigor.py) §C and
  [`discover_compose.py`](experiments/kg_compose/discover_compose.py).
- **"Learning the algebra recovers oracle (0.999)" → 0.967 [0.922,0.999]** at full data (and a separate
  earlier run gave 0.926±0.10). Statistically indistinguishable from oracle (paired p=0.13), but the
  point estimate is **~0.93–0.97, not 0.999**, and it collapses to 0.13 at 50% data (oracle 0.69).
- **Identifiability "0.17 vs 0.71" was inflated by a metric mismatch** (atomic-MRR vs path-**Hits@1**).
  With **matched MRR**: D4 atomic spread 0.13 < path 0.32 (direction holds, smaller), but **Q8 reverses**
  (atomic 0.54 > path 0.29), and atomic *argmax* identifies the true group in both cases. Honest version:
  *composition gives a larger identification margin for some groups, not a categorical "atomic cannot
  identify".*
- **"Vanilla nets can't compose"** (grok) compared the MLP's held-out **atomic** MRR (0.22, near chance)
  against PreNat's **path** MRR — mismatched. The same MLP composes 2-hops at **0.74 NONCOMM H@1**. The
  honest claim is "a generic net doesn't acquire the algebra *for free* at this scale", not "can't compose".

## 3. De-tautologising the headline: discover → commit → exact

The sharpest fair criticism is that PreNat is **handed** the data-generating algebra as fixed buffers,
so "composes / drift-free" is an identity, not a capability.
[`discover_compose.py`](experiments/kg_compose/discover_compose.py) answers it with a chain that is
inferred end-to-end on **D4 with the group hidden**:

1. **Discover** the group by validation 2-hop selection (Wilson-reported);
2. **Identify** the relation→element map by discrete majority match (no gradient, no ground truth);
3. **Commit** to the symbolic model → **provably drift-free at any horizon** *iff* steps 1–2 are correct.

Result: discovery succeeds **7/10** clean and stays **6/10 under 30% observation corruption**
(committed NONCOMM H@1 ~0.73–0.80); the committed-from-the-correct-discovery model is exactly
drift-free where a wrong (C8) commitment fails (h=200: 0.78 vs 0.24, paired p = 0.007). *Exactness is thus **contingent on
correct discovery** — the real inferred work — not a free identity.*

## 4. Earning the project's framing (new evidence)

- **The structure microscope, validated as a predictive instrument.**
  [`microscope_calibration.py`](experiments/kg_compose/microscope_calibration.py) builds a controlled
  algebraic→messy spectrum and shows the **cheap** algebraicity profile predicts the **expensive**
  held-out prior-benefit: **Spearman 0.95 [0.85,0.98]** over 36 worlds, **97% sign agreement**, and
  **97% leave-one-η-out held-out sign prediction**. (The original had 3 clustered real points with null
  within-regime predictiveness; this demonstrates the instrument actually forecasts where the prior helps.)
- **LIMN's uniqueness certificate, made to do real work.**
  [`limn_hard.py`](experiments/kg_compose/limn_hard.py) removes the original's baked-in answer: the
  apex dimension is **unknown and varies** (product of 2–3 random factors) and cones are **noisy**.
  Existence/residual is ambiguous (≈4 over-complete apexes factor the data); the naive "mint the
  biggest that fits" scores **0.00**, while the uniqueness certificate recovers the true dimension at
  **1.00 [0.98,1.00]** across all noise levels (beating even Occam-smallest, which drops to 0.61 at
  σ=0.2), with the minted subspace factoring held-out cones 100%. *Honest framing: in the linear case
  this criterion is exactly a spectral-gap rank test — elementary, but it is the defensible kernel of
  "minting by universal property", now noise-robust and non-tautological.*

## 5. Naturality-as-a-loss, finally tested in the flagship setting

The project's *stated* core novelty — naturality as a learned constraint — was **absent from every
headline KG experiment** (they train plain atomic cross-entropy).
[`nat_kg.py`](experiments/kg_compose/nat_kg.py) gives it the fairest rigorous shot (learned-algebra
model, full data spectrum, plain + robust/Welsch variants, 10 seeds, CIs, paired tests).

**Verdict (S4, NONCOMM H@1, 10 seeds):** naturality is **not** load-bearing here. At full data every
arm is 1.000; at every scarcer fraction the path-consistency signal gives **no significant gain** over
atomic-only (Δ ≈ 0 to −0.03, no comparison reaches p<0.05), and the **robust/Welsch variant actively
hurts** at 0.70 data (0.199 vs 0.488) by down-weighting the very paths that carry signal.

| data frac | atomic-only | +naturality | +robust-nat |
|---|---|---|---|
| 1.00 | 1.000 [0.999,1.000] | 1.000 [1.000,1.000] | 1.000 [0.999,1.000] |
| 0.70 | 0.488 [0.335,0.661] | 0.458 [0.344,0.607] | 0.199 [0.134,0.265] |
| 0.50 | 0.126 [0.110,0.140] | 0.122 [0.105,0.140] | 0.086 [0.081,0.093] |
| 0.30 | 0.068 [0.061,0.076] | 0.064 [0.061,0.067] | 0.059 [0.054,0.064] |

This **confirms, rigorously and in the flagship setting**, the project's own earlier finding (Run 7):
naturality-as-a-loss does not manufacture signal from scarce data, and the headline KG wins come from
the **exact/discovered composition algebra**, not from a naturality training signal. The honest
consequence: the project's contribution should be framed as *"an exact, discoverable composition-algebra
prior + a validated structure microscope + a uniqueness-of-mediator certificate"* — **not** as
"naturality as the learned constraint", which the experiments do not support as load-bearing.

## 6. Baselines & the ULTRA gap (honest status)

- **The baselines are fairly run** and that is now verified: TransE/RotatE/RESCAL share PreNat's
  dimension (`d = |G|`), training budget, optimiser and scoring; RESCAL has 6–7× the parameters and is
  not crippled. The abelian models' failure on NONCOMM is **structural** (phase/translation addition is
  commutative), not under-tuning — a correct mathematical fact, demonstrated.
- **Regime-dependence, stated plainly.** A free-matrix RESCAL is *not* always the drift-prone one: on
  small/clean worlds it locks exact permutations and is itself drift-free, occasionally beating the
  learned PreNat. The durable guarantee is the *committed symbolic* model's, not the gradient model's.
- **ULTRA is still not run, and we no longer imply otherwise.** A faithful comparison needs a
  pretrained inductive GNN on real KGs at scale; the synthetic regular-G-set worlds here have
  **featureless entities**, so inductive transfer to unseen entities (ULTRA's whole point) is
  degenerate in them. The credible path is real KGs (FB15k-237/WN18RR) + the published ULTRA checkpoint
  — out of scope for this CPU-only offline pilot, and labelled as such rather than asserted as a beaten
  baseline.

## 7. Reproducibility

```bash
cd experiments/kg_compose
python statutils.py                 # self-test of the rigor primitives
python rigor.py --seeds 10          # all headline claims, tie-aware + CIs + significance
python discover_compose.py          # discover -> identify -> commit (earned, non-tautological)
python microscope_calibration.py    # the microscope as a validated predictive instrument
python nat_kg.py                    # naturality-as-a-loss in the flagship setting (fair + rigorous)
python limn_hard.py                 # uniqueness certificate on unknown-dim, noisy minting
```
Environment: Python 3.11, PyTorch 2.x (CPU). Every script is seeded and self-contained.
