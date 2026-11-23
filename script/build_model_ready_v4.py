#!/usr/bin/env python3
"""Build model-ready v4 sets from the strictly accepted GPU/replay join.

GETRF, GESSM and TSTRF measurements from the dense/sparse campaign paths are
duplicate labels for the same logical task and are aggregated by their median.
SSSSM dense and sparse are distinct kernels and remain separate datasets.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_sets_v4 import BASELINE_FIELDS, FEATURES_BY_KERNEL


IDENTITY = [
    "task_id", "matrix_id", "kernel_type", "level", "block_row", "block_col",
    "update_k", "rank", "block_size", "process_rows", "process_cols",
]
STRUCTURE = [
    "replay_canonical_structure_hash_v2", "replay_feature_schema_version",
]
PLATFORM = ["hardware_id", "gpu_model"]
NON_SSSSM = ("GETRF", "GESSM", "TSTRF")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Strict merged_tasks.csv")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def label_statistics(grouped):
    stats = grouped["time_kernel_ms"].agg(
        measurement_count="size",
        label_time_ms="median",
        time_mean_ms="mean",
        time_std_ms=lambda x: float(np.std(x.to_numpy(float), ddof=0)),
        time_min_ms="min",
        time_max_ms="max",
    )
    stats["time_cv"] = (
        stats["time_std_ms"] / stats["time_mean_ms"].clip(lower=1e-12)
    )
    stats["time_relative_range"] = (
        (stats["time_max_ms"] - stats["time_min_ms"])
        / stats["label_time_ms"].clip(lower=1e-12)
    )
    return stats


def aggregate_duplicates(raw, kernel):
    features = FEATURES_BY_KERNEL[kernel]
    columns = list(dict.fromkeys(IDENTITY + features + BASELINE_FIELDS + STRUCTURE + PLATFORM))
    data = raw[raw["kernel_type"].eq(kernel)].copy()
    grouped = data.groupby(IDENTITY, dropna=False, sort=False)

    varying = grouped[columns].nunique(dropna=False).gt(1)
    bad = varying.any(axis=1)
    if bad.any():
        bad_columns = sorted(varying.columns[varying.any()].tolist())
        raise RuntimeError(
            f"{kernel}: {int(bad.sum())} duplicate groups disagree in "
            f"pre-kernel fields: {bad_columns}"
        )

    payload = [column for column in columns if column not in IDENTITY]
    result = grouped[payload].first().join(label_statistics(grouped))
    variant_times = (
        data.pivot_table(
            index=IDENTITY,
            columns="forced_ssssm_variant",
            values="time_kernel_ms",
            aggfunc="median",
        )
        .rename(columns={"dense": "dense_time_ms", "sparse": "sparse_time_ms"})
    )
    result = result.join(variant_times).reset_index()
    for column in ("dense_time_ms", "sparse_time_ms"):
        if column not in result:
            result[column] = np.nan
    result["dense_sparse_abs_diff_ms"] = (
        result["dense_time_ms"] - result["sparse_time_ms"]
    ).abs()
    result["dense_sparse_ape"] = (
        result["dense_sparse_abs_diff_ms"]
        / result["label_time_ms"].clip(lower=1e-12) * 100.0
    )
    return result, int(len(data))


def split_ssssm(raw, variant):
    features = FEATURES_BY_KERNEL[f"SSSSM_{variant}"]
    columns = list(dict.fromkeys(
        IDENTITY + ["forced_ssssm_variant"] + features + BASELINE_FIELDS + STRUCTURE + PLATFORM
    ))
    data = raw[
        raw["kernel_type"].eq("SSSSM")
        & raw["forced_ssssm_variant"].eq(variant)
    ].copy()
    keys = pd.MultiIndex.from_frame(data[IDENTITY])
    if not keys.is_unique:
        raise RuntimeError(f"SSSSM-{variant}: duplicate logical keys")
    result = data[columns].copy()
    result["measurement_count"] = 1
    result["label_time_ms"] = data["time_kernel_ms"].to_numpy()
    result["time_mean_ms"] = result["label_time_ms"]
    result["time_std_ms"] = 0.0
    result["time_min_ms"] = result["label_time_ms"]
    result["time_max_ms"] = result["label_time_ms"]
    result["time_cv"] = 0.0
    result["time_relative_range"] = 0.0
    return result


def stability(data, raw_rows):
    counts = data["measurement_count"].value_counts().sort_index()
    cv = data["time_cv"].to_numpy(float)
    if cv.size == 0:
        cv_quantiles = {}
    else:
        cv_quantiles = {
            str(q): float(np.quantile(cv, q))
            for q in (0, .5, .9, .95, .99, 1)
        }
    return {
        "raw_accepted_rows": int(raw_rows),
        "model_ready_rows": int(len(data)),
        "merged_duplicate_rows": int(raw_rows - len(data)),
        "measurement_count_distribution": {
            str(int(key)): int(value) for key, value in counts.items()
        },
        "time_cv_quantiles": cv_quantiles,
        "cv_above_5pct": int(np.sum(cv > .05)),
        "cv_above_10pct": int(np.sum(cv > .10)),
    }


def main():
    args = arguments()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    feature_columns = sorted(set().union(*FEATURES_BY_KERNEL.values()))
    read_columns = set(
        IDENTITY + STRUCTURE + feature_columns + BASELINE_FIELDS
        + PLATFORM + ["forced_ssssm_variant", "time_kernel_ms"]
    )
    raw = pd.read_csv(
        args.input, usecols=lambda column: column in read_columns,
        low_memory=False,
    )
    if "replay_feature_schema_version" not in raw or not raw["replay_feature_schema_version"].eq("v4.1").all():
        raise RuntimeError("model-ready v4 requires replay_feature_schema_version=v4.1 for every row")
    for column in PLATFORM:
        if column not in raw:
            raw[column] = "UNSPECIFIED" if column == "hardware_id" else ""
    hardware_ids = sorted(raw["hardware_id"].dropna().astype(str).unique().tolist())
    if len(hardware_ids) != 1 or hardware_ids[0] in ("", "UNSPECIFIED"):
        raise RuntimeError(f"expected exactly one specified hardware_id, got {hardware_ids}")
    raw["time_kernel_ms"] = pd.to_numeric(
        raw["time_kernel_ms"], errors="coerce"
    )
    raw = raw[
        np.isfinite(raw["time_kernel_ms"]) & (raw["time_kernel_ms"] > 0)
    ].copy()

    audit = {
        "schema": "online_safe_v4",
        "hardware_id": hardware_ids[0],
        "gpu_models": sorted(raw["gpu_model"].dropna().astype(str).unique().tolist()),
        "source": str(Path(args.input).resolve()),
        "non_ssssm_label_policy": (
            "median of structure-validated dense/sparse campaign measurements"
        ),
        "ssssm_policy": "dense and sparse retained as independent kernels",
        "datasets": {},
    }
    for kernel in NON_SSSSM:
        data, raw_rows = aggregate_duplicates(raw, kernel)
        path = output / f"{kernel}.model_ready_v4.csv"
        data.to_csv(path, index=False)
        audit["datasets"][kernel] = {
            **stability(data, raw_rows), "path": str(path.resolve())
        }

    for variant in ("dense", "sparse"):
        name = f"SSSSM_{variant}"
        data = split_ssssm(raw, variant)
        path = output / f"{name}.model_ready_v4.csv"
        data.to_csv(path, index=False)
        audit["datasets"][name] = {
            **stability(data, len(data)), "path": str(path.resolve())
        }

    (output / "model_ready_audit_v4.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
