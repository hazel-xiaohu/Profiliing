#!/usr/bin/env python3
"""Strictly append existing RTX PRO 6000 nd24k rows to profile29."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


BASE = Path(
    "/output/Profiliing/artifacts/platforms/RTX_PRO_6000_BLACKWELL/"
    "profile29_v4_2_model_ready_20260818"
)
SOURCE = Path(
    "/output/Profiliing/artifacts/platforms/RTX_PRO_6000_BLACKWELL/"
    "profiling25_only_new4_sm120_dense_n1_n8_20260814/model_ready_v4_1_new4"
)
OUTPUT = Path(
    "/output/Profiliing/artifacts/platforms/RTX_PRO_6000_BLACKWELL/"
    "profile30_add_nd24k_v1/model_ready_v4_1_profile30"
)
TARGETS = ("GETRF", "GESSM", "TSTRF", "SSSSM_dense")
KEY = [
    "task_id", "matrix_id", "kernel_type", "level", "block_row", "block_col",
    "update_k", "rank", "block_size", "process_rows", "process_cols",
]
BLOCK_SIZES = [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384]
HARDWARE = "RTX_PRO_6000_BLACKWELL"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"output must be empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema": "pro6000_profile30_add_nd24k_v1",
        "hardware_id": HARDWARE,
        "base": str(BASE),
        "nd24k_source": str(SOURCE),
        "targets": {},
    }
    frames = {}
    for target in TARGETS:
        base_path = BASE / f"{target}.model_ready_v4.csv"
        source_path = SOURCE / f"{target}.model_ready_v4.csv"
        base = pd.read_csv(base_path, low_memory=False)
        source = pd.read_csv(source_path, low_memory=False)
        nd24k = source[source.matrix_id.astype(str).eq("nd24k")].copy()
        if nd24k.empty or base.matrix_id.astype(str).eq("nd24k").any():
            raise RuntimeError(f"{target}: invalid nd24k membership")
        base_only = set(base.columns) - set(nd24k.columns)
        source_only = set(nd24k.columns) - set(base.columns)
        if base_only - {"_source_root"} or source_only:
            raise RuntimeError(
                f"{target}: schema mismatch base_only={base_only} "
                f"source_only={source_only}"
            )
        if "_source_root" in base_only:
            nd24k["_source_root"] = str(SOURCE)
        nd24k = nd24k[base.columns]
        if set(nd24k.hardware_id.astype(str)) != {HARDWARE}:
            raise RuntimeError(f"{target}: hardware mismatch")
        if not nd24k.replay_feature_schema_version.astype(str).eq("v4.1").all():
            raise RuntimeError(f"{target}: replay schema mismatch")
        sizes = sorted(pd.to_numeric(nd24k.block_size).astype(int).unique().tolist())
        if sizes != BLOCK_SIZES:
            raise RuntimeError(f"{target}: incomplete block sizes {sizes}")
        merged = pd.concat([base, nd24k], ignore_index=True)
        if merged.duplicated(KEY).any() or merged.matrix_id.nunique() != 30:
            raise RuntimeError(f"{target}: duplicate key or matrix-count mismatch")
        merged = merged.sort_values(
            ["matrix_id", "block_size", "level", "task_id"], kind="stable"
        ).reset_index(drop=True)
        output_path = OUTPUT / f"{target}.model_ready_v4.csv"
        atomic_csv(merged, output_path)
        frames[target] = merged
        audit["targets"][target] = {
            "profile29_rows": int(len(base)),
            "nd24k_rows": int(len(nd24k)),
            "profile30_rows": int(len(merged)),
            "matrices": int(merged.matrix_id.nunique()),
            "output_sha256": sha256(output_path),
        }

    strata = ["matrix_id", "block_size"]
    if not frames["GESSM"].groupby(strata).size().sort_index().equals(
        frames["TSTRF"].groupby(strata).size().sort_index()
    ):
        raise RuntimeError("GESSM/TSTRF strata counts differ")

    folds = pd.read_csv(BASE / "shared_matrix_folds.csv")
    folds = pd.concat(
        [folds, pd.DataFrame([{"matrix_id": "nd24k", "cv_fold": 5}])],
        ignore_index=True,
    ).sort_values(["cv_fold", "matrix_id"], kind="stable")
    counts = folds.groupby("cv_fold").size().to_dict()
    if counts != {1: 6, 2: 6, 3: 6, 4: 6, 5: 6}:
        raise RuntimeError(f"fold imbalance: {counts}")
    expected = set(folds.matrix_id.astype(str))
    for target, frame in frames.items():
        if set(frame.matrix_id.astype(str)) != expected:
            raise RuntimeError(f"{target}: fold membership mismatch")
    fold_path = OUTPUT / "shared_matrix_folds.csv"
    atomic_csv(folds, fold_path)
    audit["fold_counts"] = {str(key): int(value) for key, value in counts.items()}
    audit["fold_map_sha256"] = sha256(fold_path)
    (OUTPUT / "profile30_integrity.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
