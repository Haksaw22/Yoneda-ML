# Discovering and exploiting the compositional-algebraic structure of relational data

This is the public record of **Yoneda NN** — a one-session (~15h) falsification-first program
that set out to make the Yoneda lemma a neural training signal (composition exact by
construction, naturality as the learned constraint). The bet's mechanism was retired by the
project's own audit rounds; the composition algebra underneath it turned out to be
load-bearing. What ships here is the corrected record: what is validated, what is retired,
and the instruments that survived.

**Read the story:** [ARTICLE.md](ARTICLE.md) — *The algebra was load-bearing* (opens with a
short summary so you know what you're getting).

## What is validated here

Every number below was re-verified by fresh re-runs before publication
([REVALIDATION.md](REVALIDATION.md)); 10 seeds, tie-aware metrics, bootstrap 95% CIs.

| claim | number | where |
|---|---|---|
| **The structure microscope**: a cheap algebraicity profile predicts whether the exact-algebra prior will help, before training anything expensive | Spearman **0.93 [0.80, 0.97]**, 97% held-out sign prediction *(controlled synthetic spectrum; 6 independent noise levels — read the CI as indicative; real KGs all profile "messy")* | [`microscope_calibration.py`](research/experiments/kg_compose/microscope_calibration.py) |
| **The uniqueness certificate**: mints a true universal object under noise where fit-based minting is ambiguous | **1.00 [0.98, 1.00]** true-dimension recovery vs naive 0.00, Occam 0.61 at σ=0.2 | [`limn_hard.py`](research/experiments/kg_compose/limn_hard.py) |
| **The exact-algebra prior** composes zero-shot where abelian KGEs structurally cannot (S4 non-commuting 2-hop) | PreNat **1.000** vs RotatE 0.375, RESCAL 0.825 at 6.75× params | [`rigor.py`](research/experiments/kg_compose/rigor.py) §A |
| …dominates free matrices in the mid-data regime | **0.687 vs 0.208** at 50% data (all 10 seeds agree) | [`rigor.py`](research/experiments/kg_compose/rigor.py) §A |
| …handles non-invertible (monoid) composition | **0.884** vs RotatE 0.409 | [`rigor.py`](research/experiments/kg_compose/rigor.py) §B |
| …and is **discoverable**: validation selection finds the true algebra | Q8 **10/10**, D4 **8/10** seeds | [`rigor.py`](research/experiments/kg_compose/rigor.py) §D |
| Discover→identify→**commit** earns drift-free long horizons (exactness is contingent on correct discovery) | h=200: **0.78** vs wrong-algebra 0.24 | [`discover_compose.py`](research/experiments/kg_compose/discover_compose.py) |
| On messy real KGs the exact prior *hurts* composition; a per-relation residual relaxes it and measures algebraicity | UMLS path 0.876 relaxed vs 0.838 forced; algebraicity 0.03 vs 0.99 | [`soft_alg.py`](research/experiments/kg_compose/soft_alg.py) |

## What is retired or open

- **Naturality-as-a-loss** — the project's original headline mechanism — is **not load-bearing**
  in the flagship setting: no significant gain at any data fraction; the robust variant hurts
  ([`nat_kg.py`](research/experiments/kg_compose/nat_kg.py)). Its narrow real window (noisy
  near-functorial data where it is the only binding signal) is mapped in the article.
- **Learning algebraic structure from scarce data is a wall**: five independent attacks failed
  ([`codelock.py`](research/experiments/kg_compose/codelock.py)). The recipe is fix-or-select —
  with a boundary drawn by a pre-registered probe run for this release: at half data,
  selection holds only for algebras with a distinctive compositional signature (Q8 9/10; D4
  collapses to 3/10), and by 30% data it degrades to chance
  ([PROBE-PREREG.md](revalidation/PROBE-PREREG.md), [ARTICLE.md §5](ARTICLE.md)).
- **ULTRA — the named must-beat baseline — was never run**; everything is CPU-toy scale
  (≤135 entities). The article names every other boundary in [§6](ARTICLE.md).

## Map

```
ARTICLE.md            the story, every number linked        ← start here
playground.ipynb      rebuilds the figures from verified numbers (CPU, pre-executed)
research/             the research record, vendored with minimal redactions (tooling
                      references removed) — otherwise as it stands, including its own
                      corrections: PAPER.md (source of truth), CHANGELOG.md (audit
                      response), RESULTS.md (historical + corrections banner),
                      Yoneda-NN-Design.md (running design log, Runs 1–10), experiments/
REVALIDATION.md       what was re-verified for publication, what deviated, and the
                      known internal inconsistencies of the historical record
revalidation/         fresh re-run logs (2026-07-22) + a pre-registered probe
data/                 parsed, verified numbers (JSON) — the figures build from these
figures/              the article's figures
```

## Reproduce

```bash
pip install torch numpy scipy matplotlib
cd research/experiments/kg_compose
python rigor.py                     # headline claims, 10 seeds, CIs (~1 h CPU)
python nat_kg.py                    # the naturality retirement (~30 min)
python microscope_calibration.py    # the microscope validation (~20 min)
python discover_compose.py          # discover → identify → commit
python limn_hard.py                 # the uniqueness certificate (~1 min)
```

Every script is seeded and self-contained; none writes files. Or just open
[playground.ipynb](playground.ipynb) — it rebuilds the figures from the verified re-run
outputs in seconds without training anything.

Licensed under the MIT License — see [LICENSE](LICENSE).
