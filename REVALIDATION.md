# Revalidation — what was checked before publishing, and how

Before publishing any of this I did two things: an **intent check** (does the artifact match what
the project was actually meant to be?) and a **revalidation** — every number the article quotes
was re-verified by fresh re-run before publication, rather than trusted from the project's own
self-report. This document records both. Re-run logs are in [revalidation/](revalidation/),
parsed numbers in [data/](data/).

## Intent (drafted from the project's founding documents)

1. It began with a question typed into a chatbot: *"Do transformers have a sense of the Yoneda
   lemma?"* — and the follow-up that mattered: *"How would we build an NN to honestly and
   powerfully embed the Yoneda lemma?"*
2. The project was to take the *deep* content of Yoneda seriously — not "objects are their
   relationships" (every GNN already does that) but **naturality**: relationships commuting with
   composition — and make that the learned constraint, over composition that is exact by
   construction.
3. The bet's payoffs: zero-shot non-abelian compositional transport; and, as the stated
   moonshot, concept discovery by universal property (mint a new object because it plays a
   uniquely-determined relational role).
4. Falsification-first: kill criteria pre-registered in the design document before the first
   experiment ([research/Yoneda-NN-Design.md](research/Yoneda-NN-Design.md) §6).
5. Success meant *knowing, with evidence, whether naturality is a trainable, load-bearing
   signal* — not building a state-of-the-art KGE.

**Fidelity verdict: PASS.** The design document states exactly this bet (§0), the kill criteria
were written before results existed (§6), ten runs executed the program, and when the audit
rounds found the framing had outrun the evidence, the project corrected itself in writing and
retired its own headline mechanism. The article's thesis — "the mechanism died, the instruments
survived" — is faithful to what the project was meant to be.

## What was re-verified (2026-07-22)

Every number the article quotes is spot-checked by fresh re-run of the project's own seeded
scripts (CPU, same defaults), not trusted from its reports; spot-check depth was chosen to
match the article's claims. The source folder was not modified; scripts print to stdout and
write nothing.

| script | what it carries | verdict |
|---|---|---|
| `rigor.py --sections A B` | S4/T3 headline composition tables | **MATCH** (exact: PreNat 1.000 / 0.687 [0.542, 0.833] / 0.884 [0.836, 0.929]; RESCAL 0.825 / 0.208 / 0.803; RotatE 0.375 / 0.409; TransE 0.035 / 0.279) |
| `rigor.py --sections C D` | long-horizon drift correction; algebra selection | **MATCH** (exact: learned h=200 0.139 [0.039, 0.332], symbolic 1.00; D4 8/10 p=0.0127, Q8 10/10 p=0.0021) |
| `rigor.py --sections E F` | identifiability correction; learn-the-algebra correction | **MATCH** (exact: D4 spreads 0.13/0.32, Q8 reversal 0.54/0.29, atomic argmax correct; learned 0.967 [0.922, 0.999] p=0.127, collapses to 0.128 at 50%) |
| `discover_compose.py` | discover→identify→commit (earning exactness) | **MATCH** (exact: 7/10 [0.40, 0.89] clean, 6/10 at 30% corruption; h=200 discovered 0.776 [0.551, 0.926] vs wrong-C8 0.24, p=0.0074) |
| `microscope_calibration.py` | the microscope's Spearman validation | **PARTIAL** — Spearman 0.93 [0.80, 0.97] over 30 worlds vs documented 0.95 [0.85, 0.98] over 36 (the documented run used one more seed per η-level than the script's default); 97% sign agreement reproduces; per-η values within ~0.04. The article quotes the re-run and notes the original. |
| `nat_kg.py` | the naturality retirement table | **PARTIAL** — atomic-only and +naturality columns reproduce essentially exactly at every fraction; both qualitative headlines hold (no significant gain anywhere; robust hurts below full data). Deviation: robust-nat at frac 0.70 re-ran at 0.283 [0.21, 0.37] vs documented 0.199 [0.13, 0.27] — direction stable, magnitude seed-sensitive; disclosed in the article. |
| `limn_hard.py` | the uniqueness-certificate result | **MATCH** (exact) |

Additional re-runs beyond the Run-10 scripts, because the article quotes them:

- **`kg_real.py`** (UMLS reality check): conclusions reproduce exactly — learned-algebra best
  atomic (0.877 vs RotatE 0.869), composition lost to RESCAL (0.537 vs 0.889). Small
  single-seed drift vs the documented table (0.885→0.877 atomic, 0.549→0.537 path; likely
  torch-version nondeterminism); the article quotes the re-run values.
- **`soft_alg.py`** (the associativity dial): reproduces essentially exactly — UMLS path 0.876
  at zero penalty vs 0.838 forced, self-measured algebraicity 0.03 → 0.99; S4 ~1.00 throughout.
- **`microscope_real.py`** (real-KG profiles): **MATCH, exact** — Nations 3.38/0.67/−0.032,
  Kinship 5.06/0.59/−0.007, UMLS 6.44/0.60/−0.043; all three profile "messy" and the prior is
  a small net negative on all, as documented.
- **`codelock.py`** (the wall's five-attack table): **MATCH** — oracle 0.515±0.10 / 0.091 at
  50%/30% (documented 0.52/0.09); baseline 0.131, fix-entity 0.070, low-rank 0.059,
  code-sparsity 0.126, combined 0.045 — all within tolerance of the documented 3-seed values.
- **`codelock_transfer.py`** (the transfer attack): **MATCH** — source algebra learned at
  0.980 on full data; frozen transfer 0.129 ≈ scratch 0.131 at 50% (documented identically):
  even a correctly-learned algebra does not transfer to scarce data.
- **`grok.py`** (vanilla-net check, re-run 2026-07-24 when the article began quoting it):
  **MATCH** — MLP composes held-out 2-hops at 0.740 NONCOMM H@1 vs the built-in algebra's
  1.000 at full data (documented 0.74 vs 1.0); atomic MRR 0.224 (documented 0.22).

Also checked directly, without re-running:

- **Run 1–2 eta-sweep numbers** — recomputed from the on-disk per-seed CSVs
  ([results.csv](research/experiments/eta_sweep/results.csv)): the Run-2 table reproduces
  exactly (e.g. η=0.1: soft 0.309, robust-Welsch 0.185, hard 0.346, no-naturality 0.510;
  exact ties at η=0). The Run-1 table's slightly different `soft` values (e.g. 0.289) are from
  the earlier pre-robust run and are not quoted in the article. The "abelian 2.16" baseline
  appears in no CSV and the project itself flagged that baseline as weak — dropped.
- **The "one overnight session" claim** — file timestamps span 2026-06-28 14:33 → 06-29 05:41
  (~15h). The project has no git history (the repo was initialized but never committed);
  the article says so.

## The pre-registered probe

The pre-publication review before release found one hole worth closing: the published recipe for the
scarce-data wall — "**fix or select** the algebra; don't learn it at low data" — prescribes
*selection* for the scarce regime, but every selection experiment in the record runs at full
data (the five failed Run-8 attacks all *learn* structure at scarcity, a different mechanism).
So selection-at-scarcity was tested here, with predictions and a kill criterion written before
running: [revalidation/PROBE-PREREG.md](revalidation/PROBE-PREREG.md), script
[revalidation/select_scarce.py](revalidation/select_scarce.py), results in
[revalidation/select_scarce.log](revalidation/select_scarce.log),
[data/select_scarce.json](data/select_scarce.json), and the article's §5.

On timing: the pre-registration predates the run per the documents' own dated headers, but git
can't corroborate that here — the public history starts at release, and the dated working
history is retained privately.

**Outcome (2026-07-22).** Full-data anchors reproduce rigor §D exactly (D4 8/10, Q8 10/10),
so the probe is sound. The pre-registered prediction was half wrong: at 50% data selection
*held* on Q8 (9/10, margin p=0.0021, selected 0.862 ≈ oracle 0.872 while learn-from-scratch
collapsed to 0.498 — the recipe confirmed at its informative frontier) but *collapsed* on D4
(3/10, margin p=0.74). At 30% both degrade toward chance (1/10, 4/10) and on D4 the validation
signal significantly prefers a wrong abelian group (margin −0.040, p=0.034). The kill
criterion (both groups failing at 0.5) was not met. Verdict: the published recipe's "select"
branch is **group-dependent at scarcity**, and the article now says so.

## Known internal inconsistencies of the historical record (vendored as-is)

The research folder under [research/](research/) is the record as the project left it, vendored
with minimal redactions (tooling references removed) — otherwise untouched, including its own
errors. A careful reader will find these; better we name them:

- `RESULTS.md` (Run 1) still asserts "two independent adversarial audits passed" — Run 10's
  own correction #6 (in `Yoneda-NN-Design.md` §19) demoted that claim to prose-only and
  replaced it with checked-in rigor scripts. The per-seed eta-sweep CSVs are the defensible
  evidence for the Run 1–2 findings.
- Oracle-at-50%-data appears as 0.69 (`PAPER.md`, rigor §A config) and 0.52 (`RESULTS.md` §4 /
  design log §17, codelock config) — same phrase, different experimental configs, never
  reconciled in the record. The article quotes each only in its own config's context.
- Similarly PreNat-at-50% appears as 0.530 (3-seed era) and 0.687 (10-seed rigor).
- `PAPER.md` says Q8's atomic/path identification margins are "comparable" where
  `CHANGELOG.md` says Q8 *reverses* (atomic 0.54 > path 0.29); the CHANGELOG wording is the
  accurate one.
- `RESULTS.md`'s corrections banner references "Run 9a", but RESULTS.md contains no Run 9
  section (it lives in the design log §18); its abstract still says "six experimental runs"
  and its "Bottom line" still uses the pre-retirement framing — superseded by the banner at
  the top of the same file and by `PAPER.md`.
- The sub-READMEs (`experiments/kg_compose/README.md`, `experiments/eta_sweep/README.md`)
  predate the rigor round: they quote 3-seed numbers ("selects the true group 3/3",
  PreNat 0.530) and an `abelian` sweep variant that `sweep.py` no longer includes.
  `results_robust.csv` is a byte-identical duplicate of `results.csv`.
- The Run-1 table's `soft` column in the design log (§10) does not match the on-disk CSVs
  (which hold the Run-2 re-run values); the article quotes only the CSV-backed table.

## Honest notes

- Re-runs are seeded and reproduced the documented values within tolerance except where noted
  in the table above; any deviation is listed and the article quotes the re-run value.
- The historical Runs 1–9 tables in [research/RESULTS.md](research/RESULTS.md) are 3-seed,
  pre-rigor numbers; the article quotes them only where the CSVs back them, and otherwise uses
  the Run-10 corrected versions.
- Re-running the scripts regenerates `__pycache__` bytecode inside the source tree (a Python
  side effect, no content touched); the vendored copy here excludes it.
