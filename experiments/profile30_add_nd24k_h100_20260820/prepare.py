#!/usr/bin/env python3
"""Strictly append the correct H100 nd24k rows to frozen profile29."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


BASE = Path(
    "/output/Profiliing/artifacts/platforms/H100_SXM_80GB/"
    "profile29_hard5_v1/model_ready_v4_1_profile29"
)
SOURCE = Path(
    "/output/Profiliing/artifacts/platforms/H100_SXM_80GB/"
    "reprofile_new5_sm90_dense_n1_n8_20260811/model_ready_v4_1_old20_new5_sm90"
)
OUTPUT = Path(
    "/output/Profiliing/artifacts/platforms/H100_SXM_80GB/"
    "profile30_add_nd24k_v1/model_ready_v4_1_profile30"
)
TARGETS = ("GETRF", "GESSM", "TSTRF", "SSSSM_dense")
KEY = [
    "task_id", "matrix_id", "kernel_type", "level", "block_row", "block_col",
    "update_k", "rank", "block_size", "process_rows", "process_cols",
]
BLOCK_SIZES = [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384]


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
        raise SystemExit(f"output must be empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema": "h100_profile30_add_nd24k_v1",
        "hardware_id": "H100_SXM_80GB",
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
        nd24k = source[source["matrix_id"].astype(str).eq("nd24k")].copy()
        if nd24k.empty:
            raise RuntimeError(f"{target}: nd24k source is empty")
        if base["matrix_id"].astype(str).eq("nd24k").any():
            raise RuntimeError(f"{target}: nd24k already exists in profile29")
        if set(base.columns) != set(nd24k.columns):
            missing = sorted(set(base.columns) - set(nd24k.columns))
            extra = sorted(set(nd24k.columns) - set(base.columns))
            raise RuntimeError(f"{target}: schema mismatch missing={missing}, extra={extra}")
        nd24k = nd24k[base.columns]
        if set(nd24k["hardware_id"].astype(str)) != {"H100_SXM_80GB"}:
            raise RuntimeError(f"{target}: wrong nd24k hardware label")
        if not nd24k["replay_feature_schema_version"].eq("v4.1").all():
            raise RuntimeError(f"{target}: wrong replay schema")
        sizes = [
            int(value)
            for value in sorted(pd.to_numeric(nd24k["block_size"]).astype(int).unique())
        ]
        if sizes != BLOCK_SIZES:
            raise RuntimeError(f"{target}: incomplete block sizes {sizes}")
        merged = pd.concat([base, nd24k], ignore_index=True)
        if merged.duplicated(KEY).any():
            raise RuntimeError(f"{target}: duplicate task identity after merge")
        if merged["matrix_id"].astype(str).nunique() != 30:
            raise RuntimeError(f"{target}: expected 30 matrices")
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
            "matrices": int(merged["matrix_id"].nunique()),
            "block_sizes": sizes,
            "source_sha256": sha256(source_path),
            "output_sha256": sha256(output_path),
        }

    # GESSM and TSTRF have kernel-specific task IDs and coordinates, but the
    # frozen balancing contract requires equal counts in every matrix/size
    # stratum so neither kernel is overrepresented.
    strata = ["matrix_id", "block_size"]
    gcounts = frames["GESSM"].groupby(strata).size().sort_index()
    tcounts = frames["TSTRF"].groupby(strata).size().sort_index()
    if not gcounts.equals(tcounts):
        raise RuntimeError("GESSM/TSTRF matrix/block-size strata counts differ")

    folds = pd.read_csv(BASE / "shared_matrix_folds.csv")
    if len(folds) != 29 or folds["matrix_id"].astype(str).eq("nd24k").any():
        raise RuntimeError("invalid frozen profile29 fold map")
    folds = pd.concat(
        [folds, pd.DataFrame([{"matrix_id": "nd24k", "cv_fold": 5}])],
        ignore_index=True,
    ).sort_values(["cv_fold", "matrix_id"], kind="stable")
    counts = folds.groupby("cv_fold").size().sort_index().to_dict()
    if counts != {1: 6, 2: 6, 3: 6, 4: 6, 5: 6}:
        raise RuntimeError(f"profile30 folds are not balanced: {counts}")
    expected = set(folds["matrix_id"].astype(str))
    for target, frame in frames.items():
        if set(frame["matrix_id"].astype(str)) != expected:
            raise RuntimeError(f"{target}: matrix membership differs from fold map")
    fold_path = OUTPUT / "shared_matrix_folds.csv"
    atomic_csv(folds, fold_path)
    audit["fold_counts"] = {str(k): int(v) for k, v in counts.items()}
    audit["fold_map_sha256"] = sha256(fold_path)
    (OUTPUT / "profile30_integrity.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
