# KG composition: external validation of non-abelian relation composition

The first test of PreNat *beyond* the synthetic eta-sweep toy, in the standard
**knowledge-graph link-prediction / path-query** protocol (Guu et al. 2015), on a graph whose
relation algebra is a genuine **non-abelian group** so that composition is order-sensitive.

## What it tests

Train on **atomic** triples `(h, r, g_r·h)` only. Evaluate on:

1. **Atomic link prediction** (held-out atomic triples) — competitiveness sanity check.
2. **Length-2 path queries** `(h, [r1,r2], ?)`, true tail `g_{r2}·g_{r1}·h` — composition that was
   never trained directly.
3. **Order-sensitive subset** — path pairs where `g_{r2}g_{r1} ≠ g_{r1}g_{r2}`. Abelian models
   cannot tell the two orders apart, so this column is the discriminator.

Each `(h, rel-sequence)` has a unique true tail ⇒ clean ranking (filtered == raw).

## Models (same scoring `−‖compose(rel)·h − t‖`, only composition differs)

| model | relation | compose | non-abelian? | rel params (S4) |
|-------|----------|---------|--------------|-----------------|
| TransE | translation | add | no | small |
| RotatE | rotation/phase | add phase | no | small |
| RESCAL | free matrix | matmul | yes | `d²` per relation (large) |
| **PreNat** | code in fixed **group algebra** | exact `μ` (group-algebra product) | **yes** | `|G|` per relation (small) |

## The two questions

- **vs abelian (TransE/RotatE):** do they fail order-sensitive composition? (Expected: yes.)
- **vs RESCAL (free matrices):** RESCAL is *also* non-abelian-capable, so raw capability is not the
  story. The honest claim is **sample efficiency / inductive bias**: PreNat's fixed algebra has far
  fewer relation params and guarantees exact associative composition, so it should match-or-beat
  RESCAL and degrade more gracefully as data gets scarce. Test by lowering `--train_frac`.

## Run

```bash
cd experiments/kg_compose
python kg_run.py                     # main benchmark: S4 (24 entities), full data
python kg_run.py --train_frac 0.5    # sample-efficiency regime (PreNat >> RESCAL)
python kg_run.py --group S5 --n_relations 12   # larger (120 entities)

python kg_mismatch.py                # non-circularity control: wrong (abelian) algebra fails
python algebra_select.py             # algebra DISCOVERY: pick the true group from 5 candidates
python algebra_select.py --data_group Q8   # symmetry check (distinguishes the 2 non-abelian groups)

python kg_monoid.py                  # MONOID: non-invertible composition (RotatE fails; PreNat works)
python kg_monoid_select.py           # discover the domain needs a NON-INVERTIBLE algebra
```

`kg_run.py` reports MRR / Hits@1 for atomic + path, the **NONCOMM H@1** discriminator, and param
counts. `kg_mismatch.py` shows PreNat handed the wrong (abelian C8) algebra on non-abelian D4 data
fails like RotatE. `algebra_select.py` enumerates **all five groups of order 8** and selects by
validation path-query MRR — turning "PreNat wins *if* you know the group" into "PreNat *discovers*
it" (selects the true group 3/3 seeds, distinguishing even D4 from Q8).

## Results summary (S4 / order-8, 3 seeds)

- **Abelian fail non-abelian composition:** RotatE comm-H@1 0.935 vs noncomm-H@1 0.417.
- **PreNat (right algebra) is perfect & cheap:** 1.000 path H@1 at TransE param count.
- **Sample efficiency vs free matrices:** at 50% data PreNat noncomm-H@1 0.530 vs RESCAL 0.210 (1/6.75 params).
- **Conditional on the prior:** wrong (abelian) algebra → 0.493 noncomm (fails like RotatE).
- **Discovery:** validation selects the true group 3/3 (D4 *and* Q8), reaching oracle test performance.
- **Monoid step (non-invertible):** on T₃ data, PreNat path NON-INV H@1 0.875 vs RotatE 0.400; selection discovers it needs the non-invertible (monoid) algebra over all groups (3/3).

## Honest scope

- Synthetic, single transitive group action (entities = group elements). It isolates the
  *non-abelian composition* capability; it is **not** a substitute for FB15k-237/WN18RR or for a
  real ULTRA comparison — those are the next rung and require the full pretrained model.
- PreNat here fixes ρ to the regular rep (the eta-sweep Part-1 winner); the learned-ρ variant and
  its (so-far-inert) collapse guards are a separate question (see `../eta_sweep/`).
- A free-matrix RESCAL that learns clean permutation matrices *can* also compose correctly; the
  point of the `--train_frac` sweep is to show where the group-algebra prior actually pays off.
