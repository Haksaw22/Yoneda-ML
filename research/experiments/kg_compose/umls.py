"""UMLS loader (real biomedical knowledge graph) with filtered-ranking metadata.

135 entities, 46 relations, ~6.5k triples. Small enough for CPU; a standard relational
benchmark with genuine hierarchical/compositional structure (isa, location_of, ...).
Unlike the synthetic worlds, real relations do NOT form a clean finite group/monoid -- this is
the reality check for PreNat's exact-algebra assumption.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_DIR = os.path.join(DATA_ROOT, "UMLS")


def _read(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                out.append(tuple(parts))                  # (head, relation, tail)
    return out


def load_kg(name="UMLS", data_dir=None):
    """Load any KGDatasets-format dataset (UMLS / Kinship / Nations / ...) by folder name."""
    data_dir = data_dir or os.path.join(DATA_ROOT, name)
    train, valid, test = (_read(os.path.join(data_dir, f"{s}.txt"))
                          for s in ("train", "valid", "test"))
    ents = sorted({e for split in (train, valid, test) for (h, r, t) in split for e in (h, t)})
    rels = sorted({r for split in (train, valid, test) for (h, r, t) in split})
    eid = {e: i for i, e in enumerate(ents)}
    rid = {r: i for i, r in enumerate(rels)}

    def enc(split):
        return np.array([(eid[h], rid[r], eid[t]) for (h, r, t) in split], dtype=np.int64)

    tr, va, te = enc(train), enc(valid), enc(test)
    filt = defaultdict(set)                                # (h, r) -> all true tails (filtered eval)
    for arr in (tr, va, te):
        for h, r, t in arr:
            filt[(int(h), int(r))].add(int(t))
    return dict(name=name, n_ent=len(ents), n_rel=len(rels), train=tr, valid=va, test=te,
                filt=filt, ents=ents, rels=rels)


def load_umls(data_dir=DATA_DIR):
    return load_kg("UMLS", data_dir)


if __name__ == "__main__":
    for nm in ("UMLS", "Kinship", "Nations"):
        d = load_kg(nm)
        print(f"{nm}: {d['n_ent']} entities, {d['n_rel']} relations, "
              f"train/valid/test = {len(d['train'])}/{len(d['valid'])}/{len(d['test'])}")
