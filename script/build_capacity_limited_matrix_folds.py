#!/usr/bin/env python3
"""Build deterministic shared 5-fold matrix assignment with capacities 5/5/5/5/4."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ("GETRF", "GESSM", "TSTRF", "SSSSM_dense")
CAPACITIES = (5, 5, 5, 5, 4)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20260811)
    a = p.parse_args()
    source, output = Path(a.input_dir), Path(a.output)
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    counts = {}
    for target in TARGETS:
        data = pd.read_csv(source / f"{target}.model_ready_v4.csv", usecols=["matrix_id"])
        counts[target] = data.matrix_id.astype(str).value_counts().to_dict()
    matrices = sorted(set.intersection(*(set(x) for x in counts.values())))
    if len(matrices) != sum(CAPACITIES):
        raise RuntimeError(f"expected {sum(CAPACITIES)} shared matrices, found {len(matrices)}")
    raw = np.asarray([[counts[t][m] for t in TARGETS] for m in matrices], float)
    normalized = raw / np.maximum(raw.sum(axis=0), 1.0)
    order = sorted(range(len(matrices)), key=lambda i: (
        -float(normalized[i].sum()),
        hashlib.sha256(f"{a.seed}|{matrices[i]}".encode()).digest(),
    ))
    loads = np.zeros((5, len(TARGETS)), float)
    used = np.zeros(5, int)
    assignment = {}
    for index in order:
        available = [fold for fold in range(5) if used[fold] < CAPACITIES[fold]]
        fold = min(available, key=lambda candidate: (
            float(np.max(loads[candidate] + normalized[index])),
            float(np.sum(loads[candidate] + normalized[index])),
            int(used[candidate]), candidate,
        ))
        assignment[matrices[index]] = fold + 1
        loads[fold] += normalized[index]
        used[fold] += 1
    frame = pd.DataFrame({"matrix_id": matrices,
                          "cv_fold": [assignment[m] for m in matrices]})
    if sorted(frame.groupby("cv_fold").size()) != sorted(CAPACITIES):
        raise RuntimeError("capacity constraint failure")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    audit = {
        "schema": "capacity_limited_shared_matrix_5fold_v1", "seed": a.seed,
        "capacities": CAPACITIES, "targets": TARGETS,
        "fold_matrix_counts": frame.groupby("cv_fold").size().to_dict(),
        "fold_normalized_target_loads": loads.tolist(),
        "matrix_target_counts": {
            m: {target: int(counts[target][m]) for target in TARGETS} for m in matrices
        },
    }
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
