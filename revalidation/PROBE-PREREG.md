# Pre-registration: selection-at-scarcity probe

*Written 2026-07-22, before the probe was run. This file is not edited after results exist;
the results land in `select_scarce.log` and the analysis goes in REVALIDATION.md and the
article, quoting this pre-registration.*

## Why this probe exists

The published recipe for the project's scarce-data wall is **"fix or select the algebra;
don't learn it at low data"** ([research/PAPER.md](../research/PAPER.md) §4, §7). The
pre-publication review found that the **select** branch of that recipe was never tested
where the wall bites: every selection experiment in the record
(`algebra_select.py`, `rigor.py` §D, `kg_monoid_select.py`, `discover_compose.py`) hardcodes
`train_frac=1.0`; Run 8's five failed attacks (and its transfer attack) all *learn* structure
constants at scarcity, which is a different mechanism (gauge-trapped). So the recipe's
prescription for the scarce regime is an extrapolation from full-data evidence. This probe
tests it directly. It is a cheap, decisive check before committing to publication, not a
new research line.

## Design

Adapting `rigor.py`'s `_select_once` with `train_frac` as a variable (scarcity subsamples only
the atomic training triples; the validation path-query set is generated independently of
`train_frac`, so the selection *signal set* is unchanged — scarcity hits exactly the thing
being tested: how much atomic training data selection needs).

- Data groups: D4 and Q8 (order-8 worlds, where selection was originally demonstrated).
- Fractions: 1.0 (anchor — should reproduce rigor §D), 0.5, 0.3.
- Seeds: 10. Candidates: all five groups of order 8. Plus one **learned-random** arm
  (structure constants learned from random init, as in rigor §F) per seed as the contrast.
- Recorded per (group, frac): correct-selection count with Wilson-95 CI; paired permutation
  test of true-vs-best-wrong validation margin; mean test NONCOMM H@1 of the *selected*
  algebra vs the *true* (oracle) algebra vs the learned-random arm.
- Same trainer, steps (2500), lr (0.02), metrics (tie-aware), and statistics as `rigor.py`.

## Predictions (stated before running)

1. At frac 0.5, selection **survives**: ≥7/10 correct on each group, margin p<0.05, and the
   selected model's test NONCOMM H@1 ≈ the oracle's — while learned-random collapses
   (rigor §F behaviour).
2. At frac 0.3, selection **degrades substantially** toward chance (1/5): entity
   under-determination flattens the validation path signal that selection relies on.

## Kill criterion (stated before running)

At frac 0.5, if the correct-selection Wilson-95 **lower bound ≤ 0.2** (chance) **and** the
true-vs-best-wrong margin has **p > 0.05** on both data groups, the **select branch of the
published recipe is falsified at scarcity**, and the article must say the recipe's selection
evidence is full-data-only ("fix" is the only demonstrated branch where the wall bites).

## Scope caveat (stated before running)

This tests selection-at-scarcity on the order-8 worlds where selection was demonstrated, not
on S4 where the wall was demonstrated (no order-24 candidate library exists in the code). On
S4 at 30% even the handed-in oracle collapses (design log §17.1), so no selection could work
there anyway; the informative frontier is where the oracle survives but learning fails —
which the order-8 worlds at 0.5/0.3 probe directly.

## Outcome interpretation (stated before running)

- Selection survives at 0.5 → one-sentence upgrade in the article: the recipe's select branch
  is evidence, not extrapolation (a footnote-strength confirmation, not a new claim).
- Selection fails at 0.5 → a story change the article must carry: at scarcity even choosing
  among five known candidates is unidentifiable; the wall thickens and the survivor claim
  ("the algebra is discoverable") becomes full-data-conditional.
