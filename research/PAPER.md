# Discovering and Exploiting the Compositional-Algebraic Structure of Relational Data

*Working paper. Companion code: [experiments/](experiments/); credibility log + audit response:
[CHANGELOG.md](CHANGELOG.md); running design log: [Yoneda-NN-Design.md](Yoneda-NN-Design.md);
detailed report: [RESULTS.md](RESULTS.md). All headline numbers are 10-seed runs with tie-aware
ranking and bootstrap 95% CIs (see [`statutils.py`](experiments/kg_compose/statutils.py),
[`rigor.py`](experiments/kg_compose/rigor.py)).*

---

## Abstract

Relational models — knowledge-graph embeddings, GNNs, transformers — represent entities by their
relationships. We study a sharper, *conditional* question: **when do relations carry genuine algebraic
structure (a finite group / monoid / category), can we discover that structure from data, and what does
exploiting it buy?** We answer with **PreNat**, a model whose multi-hop composition is the exact product
of a fixed associative algebra. Our claims, and their honest scope:

1. **Where relations are algebraic, the right algebra is a powerful, data-efficient prior.** On synthetic
   non-abelian groups (S4) and non-invertible monoids (T3), PreNat composes held-out multi-hop queries at
   **1.000** / **0.884** NONCOMM-Hits@1 where abelian KGEs *structurally* cannot, matching free matrices
   at ~1/7 the learnable parameters and dominating them in the mid-data regime (50% data: **0.69 vs 0.21**,
   paired p=0.002). This is a legitimate **inductive-bias** result.
2. **The algebra is discoverable, and exact composition can be *earned* rather than handed in.** With the
   group hidden, validation selects the true group from the order-8 candidate set (**8–10/10** seeds
   depending on the group), a discrete step identifies the relation→element map, and the **committed
   symbolic model is then provably drift-free at any horizon** — contingent on correct discovery (the
   wrong algebra fails, p=0.007), robust to 30% observation noise (end-to-end pipeline 6–7/10).
3. **A relational-structure microscope, validated as a predictive instrument.** A cheap algebraicity
   profile **predicts** whether the algebraic prior will help on a new dataset across a controlled
   algebraic↔messy spectrum: Spearman **0.95 [0.85,0.98]**, 97% held-out sign prediction.
4. **A uniqueness-of-mediator certificate that mints a universal object under noise.** With the latent
   dimension unknown and cones noisy, existence/fit is ambiguous; the uniqueness certificate recovers the
   true dimension at **1.00 [0.98,1.00]** where naive fit-thresholding scores 0.00.

We are equally explicit about the **boundaries**: (a) the dramatic "1.000 / drift-free" numbers are
*exact by construction* once the algebra is known — the science is in the *discovery* and the
*conditionality*, not in the arithmetic; (b) the *learned* (gradient) model is **not** reliably
drift-free and a free-matrix model can out-compose it on small clean worlds — the advantage is
regime-specific; (c) on messy real KGs the exact prior does not help composition (relax it via a
per-relation residual); (d) **naturality as a learned loss — the project's original headline mechanism —
is *not* load-bearing**: tested rigorously in the flagship setting it gives no significant gain. The
contribution is the **discover-and-exploit-algebraic-structure** program and its honestly-mapped
envelope, not a naturality training signal and not a state-of-the-art KGE.

---

## 1. Introduction

A relational dataset is a set of triples `(head, relation, tail)`. Instead of asking "what coordinates
should each entity have?", we ask: **what is the algebraic structure of the relations, can we discover
it, and when is exploiting it worth the rigidity it imposes?** The motivating intuition is Yoneda — an
object is determined by its relationships *under composition* — but our empirical claims are about
**composition algebra**, measured, not about a naturality slogan. We make the question concrete with
PreNat (§2), then map precisely where its prior helps (§3–6) and where it does not (§7).

## 2. Method

**Algebra.** A finite algebra with structure constants `T[k,g,h] = 1 iff g∘h = k`. A relation is a code
`c ∈ ℝ^n`; its representation is `ρ(c) = Σ_k c_k R_k`; composition is the exact product
`μ(a,b) = Σ_{g,h} T[·,g,h] a_g b_h`. Because `T` is a genuine associative algebra, **associativity and
identity are algebraic identities, never losses**. The algebra may be a **group** (invertible), a
**monoid** (non-invertible: `is-a`, `causes`), or a **category** (typed, partial composition). On link
prediction the score is `−‖ρ(c_r) e_h − e_t‖`; a path query composes codes first.

**Three ways to obtain the algebra**, in increasing honesty about what is *learned*:
*fixed* (handed in — useful as an oracle/ablation, but composition is then **by construction**);
*selected* (validation model-selection over a candidate library — genuinely discovered, §4);
*learned* (structure constants trained from data with associativity enforced — works at full data,
data-hungry, §4). The *soft* variant adds a per-relation residual `D_r` whose penalty interpolates
exact↔free and whose `‖D_r‖` measures non-algebraicity (§7).

**On the tautology, stated up front.** When PreNat is *handed* the data-generating algebra, "composes
zero-shot / drift-free" is an algebraic identity, not a learned capability — a fair criticism of the
naive framing. §3 reports the handed-in numbers as an **upper bound / inductive-bias illustration**; §4
*earns* them by discovering the algebra instead.

## 3. Where the algebra is real, the prior is powerful (handed-in upper bound)

All numbers: 10 seeds, tie-aware NONCOMM-Hits@1, bootstrap 95% CI; baselines share PreNat's dimension,
budget, optimiser and scoring (RESCAL has 6.75× the params and is not crippled).

| model | S4 NONCOMM H@1 | S4 @50% data | T3 path NON-INV H@1 | rel. params |
|---|---|---|---|---|
| TransE (additive) | 0.035 [0.03,0.05] | 0.030 | 0.279 [0.24,0.32] | 1× |
| RotatE (abelian) | 0.375 [0.34,0.41] | 0.260 | 0.409 [0.39,0.43] | 0.9× |
| RESCAL (free) | 0.825 [0.79,0.87] | 0.208 | 0.803 [0.78,0.83] | 6.75× |
| **PreNat (fixed algebra)** | **1.000 [1.00,1.00]** | **0.687 [0.54,0.83]** | **0.884 [0.84,0.93]** | **1×** |

Abelian models **structurally** fail order-sensitive (non-abelian) and collapsing (non-invertible)
composition — a mathematical fact, demonstrated, not under-tuning. PreNat matches/beats free matrices at
a fraction of the learnable parameters and **dominates in the mid-data regime** (S4 50%: 0.69 vs 0.21,
paired p=0.002). *These 1.000/0.884 values are exact-by-construction given the algebra; their honest
content is parameter-efficiency, structural exclusion of the baselines, and data-efficiency — not a
discovered skill.*

**Long-horizon composition, told honestly.** The exact **symbolic** model (one-hot codes + regular
embedding) is drift-free at *any* horizon — 1.00 at h=200 — because it is a hand/discovery-committed
lookup. The **gradient-learned** PreNat is **not**: its codes never lock exactly, so on S4 it holds to
~h=100 then collapses (h=200: **0.14 [0.04,0.33]**), and on small clean worlds a free-matrix RESCAL
locks exact permutations and out-drifts it. *"Drift-free at any horizon" is a property of the exact
algebra, not of the learned model — corrected from the earlier 3-seed table.*

## 4. The algebra is discoverable — earning §3 without being handed it

[`discover_compose.py`](experiments/kg_compose/discover_compose.py): on D4 with the group **hidden**, a
three-step pipeline — **discover** (validation selection over all five order-8 groups), **identify**
(discrete relation→element match), **commit** (symbolic model) — yields exact composition *iff* discovery
is correct.

- **Selection works.** In the *dedicated* selection experiment (all five order-8 groups, 10 seeds,
  rigor §D / [`algebra_select.py`](experiments/kg_compose/algebra_select.py)) validation path-fit picks
  the true group with a significant margin: Q8 **10/10** (Wilson [0.72,1.00], +0.132 p=0.002), D4 **8/10**
  ([0.49,0.94], +0.072 p=0.013), distinguishing the two *non-abelian* groups in both directions. The
  *full* discover→identify→commit pipeline (D4 only) selects correctly **7/10 [0.40,0.89]** on clean data
  and **6/10** under 30% corruption (committed NONCOMM H@1 ~0.73–0.80). The discriminating signal is
  **compositional**: atomic link prediction separates the candidates far less (D4 atomic-MRR spread 0.13
  vs path 0.32; for Q8 the two are comparable — so "atomic *cannot* identify" is too strong; "composition
  gives a larger margin" is the honest claim).
- **Learning** the structure constants from random init reaches **0.967 [0.922,0.999]** at full data
  (statistically indistinguishable from oracle, paired p=0.13; *not* the previously-claimed 0.999), but
  collapses at 50% data (0.13 vs oracle 0.69) and is hurt by a structurally-wrong warm-start — so
  **fix or select** is the sample-efficient recipe.
- **The earned payoff:** the discovered-then-committed symbolic model is drift-free where a wrong
  commitment fails (h=200: discovered **0.78 [0.55,0.93]** end-to-end — equal to the oracle on the
  ~7/10 of seeds where discovery succeeds — vs wrong-C8 **0.24**, paired p=0.007). Exactness is thus
  contingent on correct discovery, not handed in.

## 5. The relational-structure microscope, validated as an instrument

[`microscope_calibration.py`](experiments/kg_compose/microscope_calibration.py). Beyond profiling a few
datasets, we show the **cheap** algebraicity profile **predicts** the **expensive** downstream
prior-benefit across a controlled algebraic→messy spectrum (a group action mixed with a tunable fraction
of non-algebraic random maps):

| world (messiness η) | cheap algebraicity | held-out prior-benefit |
|---|---|---|
| 0.00 (algebraic) | 1.49 [1.33,1.63] | **+0.35 [0.28,0.42]** |
| 0.25 | 1.07 [0.95,1.20] | +0.06 [−0.02,0.15] |
| 0.50 | 0.78 [0.70,0.91] | −0.10 [−0.13,−0.04] |
| 1.00 (messy) | 0.75 [0.69,0.80] | **−0.16 [−0.19,−0.12]** |

**Spearman(profile, benefit) = 0.95 [0.85,0.98]** over 36 worlds; **97%** sign agreement and **97%**
leave-one-η-out held-out sign prediction. *A quick profiling run forecasts whether the algebraic prior
is worth using — the instrument's payoff, demonstrated across a spectrum rather than 3 clustered points.*
*(Honest scope: the cheap and expensive measures compare the same constrained-vs-free pair at different
budgets/splits, so this shows a cheap proxy forecasts the expensive outcome — useful — not that an
unrelated structural fingerprint does; and the CI treats 36 correlated worlds from 6 independent η-levels,
so read it as indicative.)*
On real KGs (UMLS/Kinship/Nations) the instrument flags all as messy (~0.6) and the prior as a net
negative — consistent, and a useful caution: Kinship's relations *compose semantically* yet are **not**
algebraic *as a KG* (functionality 5.06).

## 6. Concept minting by a uniqueness-of-mediator certificate

[`limn_hard.py`](experiments/kg_compose/limn_hard.py). For a product of an **unknown, varying** number of
factors with **noisy** cones, *existence* is ambiguous — a whole range of over-complete apexes factor the
data (≈4), so "mint the biggest object that fits" scores **0.00** (it mints a weak, non-universal limit).
The **uniqueness certificate** (full-column-rank of the stacked projection — in the linear case, a
spectral-gap criterion needing no knowledge of the noise level) recovers the true dimension at **1.00
[0.98,1.00]** across noise levels, beating even an Occam-smallest rule under noise (0.61 at σ=0.2). (The
minted subspace also factors held-out cones 100%, but that is light corroboration — it follows from the
minted dimension equalling the true one, not independent generalization.) *Honest scope: this is the
elementary but defensible kernel of "minting by universal property" — now non-tautological (the answer is
inferred, not baked in) and noise-robust — not a learned concept-discovery system.*

## 7. Honest boundaries (kept, not buried)

- **Naturality-as-a-loss is not load-bearing.** Tested rigorously in the flagship KG setting
  ([`nat_kg.py`](experiments/kg_compose/nat_kg.py)): no significant gain at any data fraction, and a
  robust variant hurts at low data. The wins come from the **exact/discovered algebra**, not a naturality
  signal. We have reframed the project accordingly.
- **Messy real KGs.** UMLS/Kinship/Nations profile ~0.6 algebraic; forcing associativity hurts
  composition. **Soft associativity** (per-relation residual) recovers RESCAL-level composition while
  keeping best-in-class atomic and *measuring* per-relation algebraicity. The real-KG ceiling is
  "competitive relational model + structure profiler", **not** SOTA on FB15k-237.
- **Learning structure from scarce data is a wall** — it resists five independent fixes; **fix or select**
  the algebra, run minting on top of a reliable one.
- **Regime-dependence of the long-horizon win.** Free matrices are not always the drift-prone ones;
  state the guarantee as the *committed symbolic* model's.
- **ULTRA / scale.** The named must-beat baseline is **not run** (and we no longer imply it is): the
  synthetic worlds have featureless entities, so inductive transfer is degenerate in them; a faithful
  comparison needs real KGs + the pretrained checkpoint, out of scope for this CPU-only pilot.

## 8. Related work

Sheaf NNs / Neural Sheaf Diffusion (local agreement, no global composition); RotatE/TransE/NagE/RESCAL
(relation-as-operator; abelian or free, single-object); hypernetworks / DeepONet (operators from
descriptors, no functor laws); ULTRA (object = its relationships, implicit composition — the unrun
must-beat baseline); Categorical Deep Learning / GAIA (frame Yoneda/Kan, no gradient-trained experiment);
DreamCoder / library learning (concept discovery by compression, not by universal property). PreNat's
defensible niche: **discovering an exact, possibly non-invertible/partial composition algebra from data
and exploiting it, packaged with a *validated* structure-discovery microscope and a uniqueness-of-mediator
minting certificate.**

## 9. Conclusion

Where relations are algebraic, the right composition algebra is a powerful, data-efficient prior — and,
crucially, it is **discoverable**, so exact composition can be *earned* rather than handed in. Packaged as
a profiler it becomes a **validated** structure-discovery instrument that forecasts its own usefulness.
We have mapped the envelope honestly: the dramatic numbers are exact-by-construction, the learned model is
not magically drift-free, the original naturality mechanism is not load-bearing, and messy real KGs need
the prior relaxed. The honest map — *discover algebraic structure, exploit it where it is real, measure
where it is not* — is the result.

---

### Reproducibility

CPU-only, seeded, self-contained under [`experiments/kg_compose/`](experiments/kg_compose/). Rigor:
`statutils.py`, `rigor.py`. Discovery/earning: `discover_compose.py`. Instrument: `microscope_calibration.py`.
Mechanism test: `nat_kg.py`. Minting: `limn_hard.py`. Originals (`kg_run.py`, `kg_monoid.py`, `puzzle.py`,
`algebra_select.py`, `microscope.py`, `soft_alg.py`, `limn_mint.py`, …) are retained unchanged. Python
3.11, PyTorch 2.x (CPU).
