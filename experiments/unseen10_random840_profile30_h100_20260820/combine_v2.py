#!/usr/bin/env python3
"""Combine the nine retained v1 matrices with replacement oilpan labels."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


V1 = Path(
    "/output/Profiliing/artifacts/platforms/H100_SXM_80GB/"
    "unseen10_random840_profile30_v1/model_ready_v4_1"
)
ROOT = Path(
    "/output/Profiliing/artifacts/platforms/H100_SXM_80GB/"
    "unseen10_random840_profile30_v2_no_duplicate_chimera"
)
REPLACEMENT = ROOT / "replacement_oilpan/model_ready_v4_1"
OUTPUT = ROOT / "model_ready_v4_1"
TARGETS = ("GETRF", "GESSM", "TSTRF", "SSSSM_dense")
KEY = [
    "task_id", "matrix_id", "kernel_type", "level", "block_row", "block_col",
    "update_k", "rank", "block_size", "process_rows", "process_cols",
]
EXPECTED_SIZES = [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"output must be empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    expected = set(pd.read_csv(ROOT / "selected10.csv").matrix.astype(str))
    audit = {
        "schema": "unseen10_model_ready_v2_replace_uni_chimera_i5_with_oilpan",
        "removed": "uni_chimera_i5",
        "added": "oilpan",
        "sources": {"v1": str(V1), "replacement": str(REPLACEMENT)},
        "targets": {},
    }
    for target in TARGETS:
        old_path = V1 / f"{target}.model_ready_v4.csv"
        new_path = REPLACEMENT / f"{target}.model_ready_v4.csv"
        old = pd.read_csv(old_path, low_memory=False)
        old = old[~old.matrix_id.astype(str).eq("uni_chimera_i5")].copy()
        replacement = pd.read_csv(new_path, low_memory=False)
        if set(old.columns) != set(replacement.columns):
            raise RuntimeError(f"{target}: schema mismatch")
        replacement = replacement[old.columns]
        merged = pd.concat([old, replacement], ignore_index=True)
        if set(merged.matrix_id.astype(str)) != expected:
            raise RuntimeError(f"{target}: matrix membership mismatch")
        if merged.duplicated(KEY).any():
            raise RuntimeError(f"{target}: duplicate task identity")
        if set(merged.hardware_id.astype(str)) != {"H100_SXM_80GB"}:
            raise RuntimeError(f"{target}: hardware mismatch")
        if set(pd.to_numeric(merged.block_size).astype(int)) != set(EXPECTED_SIZES):
            raise RuntimeError(f"{target}: block-size coverage mismatch")
        merged = merged.sort_values(
            ["matrix_id", "block_size", "level", "task_id"], kind="stable"
        ).reset_index(drop=True)
        path = OUTPUT / f"{target}.model_ready_v4.csv"
        atomic_csv(merged, path)
        audit["targets"][target] = {
            "rows": int(len(merged)),
            "matrices": int(merged.matrix_id.nunique()),
            "oilpan_rows": int(merged.matrix_id.astype(str).eq("oilpan").sum()),
            "output_sha256": sha256(path),
        }
    (OUTPUT / "combine_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
