# η-sweep: does *soft* naturality beat *hard-wired* naturality?

This is the **make-or-break minimal experiment** for the PreNat thesis (see
`../../Yoneda-NN-Design.md`). It is deliberately small (runs in ~1–2 min on CPU) and
built so that the thesis can **fail** if it deserves to.

## The thesis in one line

Make morphism composition **exact by construction** (a fixed non-abelian group
algebra), and make **object-indexed naturality the single *learned* constraint**. The
claim is that a *soft, learned* naturality penalty beats a *hard-wired* one **when the
data is approximately but not exactly functorial** — because then hard-wiring is
misspecified.

## What is being compared

All models share the **same faithful decoder** `ô = W·ρ(code)·Fᵢ`, so they have equal
power to fit any single observation. They differ *only* in how codes compose/transport:

| variant | composition | naturality | role |
|---------|-------------|------------|------|
| **soft** | exact non-abelian group algebra | **soft learned penalty** (`L_nat`,`L_cycle`) | PreNat (the proposal) |
| **hard** | exact non-abelian group algebra | **exact by construction** (one code/object, others derived) | sheaf/CENN-style hard-wired baseline |
| **no-nat** | exact non-abelian group algebra | none (`λ_nat = λ_cycle = 0`) | shows naturality is load-bearing |
| **abelian** | **additive / commutative** transport | soft | TransE/RotatE-class; should fail non-abelian transport |

## The world (`data.py`)

A non-abelian group `G` (default **S₃**, |G|=6). Each object `A` has a latent element
`a_A`; each probe `bᵢ` a fixed element `pᵢ` and value `Fᵢ`. The true hom from probe `i`
into `A` is `g(i,A) = a_A · pᵢ`, so the profile is **exactly functorial**:
`g(j,A) = g(i,A) · (pᵢ⁻¹pⱼ)`. Observations are `o(i,A) = W·P_{g(i,A)}·Fᵢ ∈ ℝ^{d_obs}` with
**`d_obs < |G|`** — a single probe is insufficient, so the model *must* fuse evidence
across probes (the work naturality does).

- **η knob**: an `η` fraction of *observed* entries have `g(i,A)` replaced by a random
  element → the data is near-functorial, not exactly functorial.
- **Held-out test**: `(k,A)` pairs whose probe `k` was never observed for `A`. Predicting
  them requires composing a seen code with a known **non-abelian** transport. Targets use
  the *clean* (uncorrupted) truth.

## Run

```bash
cd experiments/eta_sweep
python sweep.py                       # S3, 3 seeds, η ∈ {0,.1,.2,.3,.4,.5}
python sweep.py --group D4 --seeds 5 --steps 800 --plot
python sweep.py --group C6            # ABELIAN control: here abelian transport should suffice
```

Outputs a table (normalised held-out MSE, mean ± std over seeds), the soft−hard gap per
η, a heuristic verdict, `results.csv`, and (with `--plot`) `eta_sweep.png`.

## How to read it (pre-registered)

- **η = 0**: soft ≈ hard. A tie here is *predicted* — hard-wiring is correctly specified.
- **η growing**: soft should pull ahead; the **soft−hard gap should grow more negative**.
- **no-nat / abelian**: should be poor at all η (≈ predicting the mean, i.e. norm-MSE ≈ 1).

### Falsification

If **hard matches soft across the entire sweep (including large η)**, the thesis is
**dead** — it is "hard-wired naturality with extra steps." The script prints this verdict.
This is the honest point of the experiment: it is cheap and it can kill the idea.

## Scope / honest limitations of *this* scaffold

- **ρ is fixed** (the regular representation), so the *commutativity/subalgebra collapse*
  failure mode (`L_comm` in the design doc) is **out of scope here** — it only bites when
  ρ is learned. This toy tests the soft-vs-hard naturality question in isolation.
- **PreNat is given the known transports** `t_{ij}` (its categorical scaffold); the abelian
  baseline must learn its transports. That asymmetry is the legitimate inductive-bias
  difference, but it means the soft-vs-**abelian** comparison is secondary; the clean,
  fully-symmetric comparison is soft-vs-**hard** (identical except parametrisation).
- Synthetic, exact-group structure. Real validation = a non-abelian multi-hop KG split
  and beating ULTRA *on composition* (next rung in the design doc).

## Files

- `groups.py` — finite groups → exact (non-)abelian algebra (regular rep, structure constants).
- `data.py` — the η-sweep world, splits, observations.
- `models.py` — `PreNat` (soft/hard), `Abelian`; shared group-algebra ops.
- `train.py` — training loop, losses, held-out transport eval.
- `sweep.py` — runs the η-sweep across variants/seeds; table + CSV + plot + verdict.
