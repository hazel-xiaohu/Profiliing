#!/usr/bin/env python3
"""Replace nd24k with com-DBLP in the frozen H100 profile25 dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


TARGETS = ("GETRF", "GESSM", "TSTRF", "SSSSM_dense")
EXPECTED_BLOCK_SIZES = (2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--augmentation", required=True)
    parser.add_argument("--fold-map", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = Path(args.base)
    augmentation = Path(args.augmentation)
    fold_source = Path(args.fold_map)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"output must be empty: {output}")

    manifest = {
        "schema": "h100_profile25_nd24k_to_com_dblp_v1",
        "base_source": str(base.resolve()),
        "augmentation_source": str(augmentation.resolve()),
        "removed_matrix": "nd24k",
        "added_matrix": "com-DBLP",
        "hardware_id": "H100_SXM_80GB",
        "expected_block_sizes": list(EXPECTED_BLOCK_SIZES),
        "targets": {},
    }

    reference_matrices = None
    for target in TARGETS:
        base_path = base / f"{target}.model_ready_v4.csv"
        augmentation_path = augmentation / f"{target}.model_ready_v4.csv"
        old = pd.read_csv(base_path, low_memory=False)
        new = pd.read_csv(augmentation_path, low_memory=False)

        old_without = old[old["matrix_id"].astype(str) != "nd24k"].copy()
        added = new[new["matrix_id"].astype(str) == "com-DBLP"].copy()
        if added.empty:
            raise RuntimeError(f"{target}: com-DBLP has no rows")
        if set(added["hardware_id"].astype(str)) != {"H100_SXM_80GB"}:
            raise RuntimeError(f"{target}: unexpected hardware labels")
        if tuple(sorted(added["block_size"].astype(int).unique())) != EXPECTED_BLOCK_SIZES:
            raise RuntimeError(f"{target}: incomplete com-DBLP block-size coverage")
        key = ["matrix_id", "block_size", "task_id"]
        if added.duplicated(key).any():
            raise RuntimeError(f"{target}: duplicate com-DBLP task keys")

        combined = pd.concat([old_without, added], ignore_index=True, sort=False)
        matrices = sorted(combined["matrix_id"].astype(str).unique())
        if len(matrices) != 25 or "nd24k" in matrices or "com-DBLP" not in matrices:
            raise RuntimeError(f"{target}: invalid matrix replacement: {matrices}")
        if combined.duplicated(key).any():
            raise RuntimeError(f"{target}: duplicate combined task keys")
        if set(combined["hardware_id"].astype(str)) != {"H100_SXM_80GB"}:
            raise RuntimeError(f"{target}: mixed hardware labels")
        if reference_matrices is None:
            reference_matrices = matrices
        elif matrices != reference_matrices:
            raise RuntimeError(f"{target}: matrix set differs across kernels")

        output_path = output / f"{target}.model_ready_v4.csv"
        combined.sort_values(
            ["matrix_id", "block_size", "level", "task_id"], kind="stable"
        ).to_csv(output_path, index=False)
        manifest["targets"][target] = {
            "base_rows": int(len(old)),
            "removed_nd24k_rows": int((old["matrix_id"].astype(str) == "nd24k").sum()),
            "added_com_dblp_rows": int(len(added)),
            "combined_rows": int(len(combined)),
            "matrices": matrices,
            "output_sha256": sha256(output_path),
        }

    folds = pd.read_csv(fold_source)
    if set(folds.columns) != {"matrix_id", "cv_fold"}:
        raise RuntimeError(f"unexpected fold-map columns: {list(folds.columns)}")
    if (folds["matrix_id"].astype(str) == "com-DBLP").any():
        raise RuntimeError("com-DBLP already exists in the source fold map")
    mask = folds["matrix_id"].astype(str) == "nd24k"
    if int(mask.sum()) != 1:
        raise RuntimeError("nd24k must occur exactly once in the fold map")
    inherited_fold = int(folds.loc[mask, "cv_fold"].iloc[0])
    folds.loc[mask, "matrix_id"] = "com-DBLP"
    counts = folds.groupby("cv_fold").size().sort_index().tolist()
    if counts != [5, 5, 5, 5, 5]:
        raise RuntimeError(f"invalid fold sizes: {counts}")
    if sorted(folds["matrix_id"].astype(str)) != reference_matrices:
        raise RuntimeError("fold-map matrix set differs from model-ready matrix set")
    fold_output = output / "shared_matrix_folds.csv"
    folds.to_csv(fold_output, index=False)
    manifest["fold_map"] = {
        "source": str(fold_source.resolve()),
        "output": str(fold_output.resolve()),
        "com_dblp_inherited_fold": inherited_fold,
        "fold_counts": counts,
        "sha256": sha256(fold_output),
        "frozen": True,
    }

    (output / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
