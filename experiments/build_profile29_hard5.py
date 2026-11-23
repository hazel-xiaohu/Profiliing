#!/usr/bin/env python3
"""Add four disjoint hard matrices to the 25-matrix H100 profiling set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


TARGETS = ("GETRF", "GESSM", "TSTRF", "SSSSM_dense")
BLOCK_SIZES = (2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384)
HARD_FOLDS = {
    "atmosmodd": 1,
    "language": 2,
    "rajat29": 3,
    "kkt_power": 4,
    "com-DBLP": 5,
}
KEY = [
    "task_id", "matrix_id", "kernel_type", "level", "block_row", "block_col",
    "update_k", "rank", "block_size", "process_rows", "process_cols",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base25", required=True)
    parser.add_argument("--hard4", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base_root = Path(args.base25)
    hard_root = Path(args.hard4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"output must be empty: {output}")

    expected_hard4 = set(HARD_FOLDS) - {"com-DBLP"}
    manifest = {
        "schema": "h100_profile29_hard5_v1",
        "base25": str(base_root.resolve()),
        "hard4": str(hard_root.resolve()),
        "hardware_id": "H100_SXM_80GB",
        "block_sizes": list(BLOCK_SIZES),
        "hard_matrix_folds": HARD_FOLDS,
        "targets": {},
    }
    reference_matrices = None
    for target in TARGETS:
        base = pd.read_csv(base_root / f"{target}.model_ready_v4.csv", low_memory=False)
        hard = pd.read_csv(hard_root / f"{target}.model_ready_v4.csv", low_memory=False)
        base_matrices = set(base["matrix_id"].astype(str))
        hard_matrices = set(hard["matrix_id"].astype(str))
        if len(base_matrices) != 25 or "com-DBLP" not in base_matrices:
            raise RuntimeError(f"{target}: invalid base25 matrix set")
        if hard_matrices != expected_hard4:
            raise RuntimeError(f"{target}: hard4={sorted(hard_matrices)}")
        if base_matrices & hard_matrices:
            raise RuntimeError(f"{target}: base/hard overlap")
        if set(hard["hardware_id"].astype(str)) != {"H100_SXM_80GB"}:
            raise RuntimeError(f"{target}: hard4 hardware mismatch")
        for matrix_id, part in hard.groupby(hard["matrix_id"].astype(str)):
            sizes = tuple(sorted(pd.to_numeric(part["block_size"]).astype(int).unique()))
            if sizes != BLOCK_SIZES:
                raise RuntimeError(f"{target}/{matrix_id}: incomplete sizes {sizes}")

        combined = pd.concat([base, hard], ignore_index=True, sort=False)
        if set(combined["hardware_id"].astype(str)) != {"H100_SXM_80GB"}:
            raise RuntimeError(f"{target}: mixed hardware")
        if combined.duplicated(KEY).any():
            raise RuntimeError(f"{target}: duplicate logical task keys")
        matrices = sorted(combined["matrix_id"].astype(str).unique())
        if len(matrices) != 29:
            raise RuntimeError(f"{target}: expected 29 matrices, got {len(matrices)}")
        if reference_matrices is None:
            reference_matrices = matrices
        elif matrices != reference_matrices:
            raise RuntimeError(f"{target}: matrix set differs across kernels")

        path = output / f"{target}.model_ready_v4.csv"
        combined.sort_values(
            ["matrix_id", "block_size", "level", "task_id"], kind="stable"
        ).to_csv(path, index=False)
        manifest["targets"][target] = {
            "base_rows": int(len(base)),
            "hard4_rows": int(len(hard)),
            "combined_rows": int(len(combined)),
            "matrices": matrices,
            "hard4_rows_by_matrix": {
                str(k): int(v)
                for k, v in hard["matrix_id"].astype(str).value_counts().sort_index().items()
            },
            "sha256": sha256(path),
        }

    folds = pd.read_csv(base_root / "shared_matrix_folds.csv")
    for matrix_id, fold in HARD_FOLDS.items():
        if matrix_id == "com-DBLP":
            actual = folds.loc[folds["matrix_id"].astype(str) == matrix_id, "cv_fold"]
            if len(actual) != 1 or int(actual.iloc[0]) != fold:
                raise RuntimeError("com-DBLP must remain in fold 5")
        else:
            folds.loc[len(folds)] = {"matrix_id": matrix_id, "cv_fold": fold}
    if folds["matrix_id"].astype(str).duplicated().any():
        raise RuntimeError("duplicate matrices in fold map")
    if sorted(folds["matrix_id"].astype(str)) != reference_matrices:
        raise RuntimeError("fold-map matrix set differs from training data")
    counts = folds.groupby("cv_fold").size().sort_index().to_dict()
    if counts != {1: 6, 2: 6, 3: 6, 4: 6, 5: 5}:
        raise RuntimeError(f"unexpected fold counts: {counts}")
    fold_path = output / "shared_matrix_folds.csv"
    folds.sort_values(["cv_fold", "matrix_id"], kind="stable").to_csv(fold_path, index=False)
    manifest["fold_map"] = {
        "path": str(fold_path.resolve()),
        "counts": counts,
        "sha256": sha256(fold_path),
        "frozen": True,
    }
    (output / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
