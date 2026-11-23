#!/usr/bin/env python3
"""Deduplicate labels and train GESSM/TSTRF/SSSSM variants with 80/20 per matrix."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from feature_sets_v3 import FEATURES_BY_KERNEL


ALL_FEATURES = sorted(set().union(*FEATURES_BY_KERNEL.values()))
IDENTITY = [
    "task_id", "matrix_id", "kernel_type", "level", "block_row", "block_col",
    "update_k", "rank", "block_size", "process_rows", "process_cols",
]
TARGETS = [
    ("GESSM", None, "GESSM"),
    ("TSTRF", None, "TSTRF"),
    ("SSSSM", "dense", "SSSSM_dense"),
    ("SSSSM", "sparse", "SSSSM_sparse"),
]


def metric(y, pred):
    y = np.asarray(y, float)
    pred = np.maximum(np.asarray(pred, float), 0.0)
    ape = np.abs(pred - y) / np.maximum(np.abs(y), 1e-6) * 100.0
    return {
        "n_test": int(len(y)), "MAPE": float(ape.mean()),
        "MdAPE": float(np.median(ape)), "P90_APE": float(np.percentile(ape, 90)),
        "P95_APE": float(np.percentile(ape, 95)),
        "RMSE_ms": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
    }


def new_hgb(seed):
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=.07, max_iter=300,
        max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0,
        random_state=seed,
    )


def aggregate(raw, kernel, variant, features):
    d = raw[raw["kernel_type"].eq(kernel)].copy()
    group_key = IDENTITY.copy()
    if variant is not None:
        d = d[d["forced_ssssm_variant"].eq(variant)].copy()
        group_key.append("forced_ssssm_variant")
    const_cols = features + [
        "structure_hash", "replay_structure_hash",
        "replay_canonical_structure_hash_v2", "replay_feature_schema_version",
    ]
    # A logical task may occur in both dense/sparse campaign paths. They are
    # duplicate labels for GESSM/TSTRF, but distinct kernels for SSSSM.
    grouped = d.groupby(group_key, dropna=False, sort=False)
    varying = grouped[const_cols].nunique(dropna=False).gt(1)
    bad = varying.any(axis=1)
    if bad.any():
        bad_cols = sorted(varying.columns[varying.any()].tolist())
        raise RuntimeError(
            f"{kernel}/{variant}: {int(bad.sum())} groups disagree in {bad_cols}"
        )
    # Group-key fields such as block_size are restored by reset_index().
    payload_cols = [c for c in const_cols if c not in group_key]
    features = grouped[payload_cols].first()
    times = grouped["time_kernel_ms"].agg(
        measurement_count="size", label_time_ms="median", time_mean_ms="mean",
        time_std_ms=lambda x: float(np.std(x.to_numpy(float), ddof=0)),
        time_min_ms="min", time_max_ms="max",
    )
    out = features.join(times).reset_index()
    out["time_cv"] = out["time_std_ms"] / out["time_mean_ms"].clip(lower=1e-12)
    out["time_relative_range"] = (
        (out["time_max_ms"] - out["time_min_ms"])
        / out["label_time_ms"].clip(lower=1e-12)
    )
    return out, len(d)


def within_matrix_split(df, fraction, seed):
    rng = np.random.default_rng(seed)
    train = np.zeros(len(df), dtype=bool)
    rows = []
    for matrix, indices in df.groupby("matrix_id", sort=True).indices.items():
        indices = np.asarray(indices, dtype=int)
        shuffled = rng.permutation(indices)
        n_train = int(np.floor(len(indices) * fraction))
        n_train = min(max(n_train, 1), len(indices) - 1)
        train[shuffled[:n_train]] = True
        rows.append({
            "matrix_id": matrix, "n_total": int(len(indices)),
            "n_train": int(n_train), "n_test": int(len(indices) - n_train),
        })
    return train, ~train, pd.DataFrame(rows)


def stability(df, raw_count):
    cv = df["time_cv"].to_numpy(float)
    return {
        "raw_rows": int(raw_count),
        "logical_tasks": int(len(df)),
        "removed_duplicate_rows": int(raw_count - len(df)),
        "measurement_count_distribution": {
            str(int(k)): int(v)
            for k, v in df["measurement_count"].value_counts().sort_index().items()
        },
        "time_cv_quantiles": {
            str(q): float(np.quantile(cv, q)) for q in [0, .5, .9, .95, .99, 1]
        },
        "cv_above_5pct": int(np.sum(cv > .05)),
        "cv_above_10pct": int(np.sum(cv > .10)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--train-fraction", type=float, default=.8)
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between zero and one")
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    cols = set(
        IDENTITY + ALL_FEATURES + [
            "forced_ssssm_variant", "time_kernel_ms", "structure_hash",
            "replay_structure_hash", "replay_canonical_structure_hash_v2",
            "replay_feature_schema_version",
        ]
    )
    raw = pd.read_csv(args.input, usecols=lambda c: c in cols, low_memory=False)
    raw["time_kernel_ms"] = pd.to_numeric(raw["time_kernel_ms"], errors="coerce")
    raw = raw[
        np.isfinite(raw["time_kernel_ms"]) & (raw["time_kernel_ms"] > 0)
    ].copy()

    all_metrics = []
    all_stability = {}
    metadata = {
        "schema": "online_safe_v3", "split_mode": "within-matrix",
        "train_fraction": args.train_fraction, "seed": args.seed,
        "feature_policy": "kernel_specific_v3", "targets": {},
    }
    for target_index, (kernel, variant, name) in enumerate(TARGETS):
        print(f"[{name}] aggregate", flush=True)
        features = FEATURES_BY_KERNEL[name if name.startswith("SSSSM_") else kernel]
        d, raw_count = aggregate(raw, kernel, variant, features)
        train, test, split = within_matrix_split(
            d, args.train_fraction, args.seed + target_index
        )
        logical_key_cols = IDENTITY + (["forced_ssssm_variant"] if variant is not None else [])
        logical_keys = pd.MultiIndex.from_frame(d[logical_key_cols])
        if not logical_keys.is_unique:
            raise RuntimeError(f"{name}: duplicate full logical keys remain after aggregation")
        # Rows are split only after full-key aggregation, so no logical task can
        # occur in both partitions. task_id alone is not globally unique.
        d.to_csv(outdir / f"{name}.aggregated.csv", index=False)
        split.to_csv(outdir / f"{name}.within_matrix_split.csv", index=False)
        all_stability[name] = stability(d, raw_count)

        X = (
            d[features].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        )
        y = d["label_time_ms"].to_numpy(float)
        direct = new_hgb(args.seed + target_index)
        direct.fit(X[train], np.log1p(y[train]))
        direct_pred = np.maximum(np.expm1(direct.predict(X[test])), 0.0)
        all_metrics.append({
            "target": name, "model": "independent_HGB",
            **metric(y[test], direct_pred),
        })

        ops = d["replay_operation_count_v2"].to_numpy(float)
        byt = d["replay_base_bytes_est"].to_numpy(float)
        op_pos, byte_pos = ops[train][ops[train] > 0], byt[train][byt[train] > 0]
        scale = np.array([
            max(float(np.median(op_pos)), 1.0),
            max(float(np.median(byte_pos)), 1.0),
        ])
        Z = np.column_stack([np.ones(len(d)), ops / scale[0], byt / scale[1]])
        coef, _ = nnls(Z[train], y[train])
        baseline = np.maximum(Z @ coef, 1e-9)
        residual = np.log1p(y) - np.log1p(baseline)
        residual_model = new_hgb(args.seed + 10 + target_index)
        residual_model.fit(X[train], residual[train])
        residual_pred = np.maximum(
            np.expm1(
                np.log1p(baseline[test]) + residual_model.predict(X[test])
            ), 0.0
        )
        all_metrics.append({
            "target": name, "model": "independent_Roofline_residual_HGB",
            "baseline_intercept_ms": float(coef[0]),
            "baseline_ms_per_scaled_ops": float(coef[1]),
            "baseline_ms_per_scaled_bytes": float(coef[2]),
            **metric(y[test], residual_pred),
        })
        joblib.dump({
            "schema": "online_safe_v3", "kernel": kernel, "variant": variant,
            "features": features, "model": direct, "target_transform": "log1p",
        }, outdir / f"{name}.hgb_v3.joblib")
        joblib.dump({
            "schema": "online_safe_v3", "kernel": kernel, "variant": variant,
            "features": features, "residual_model": residual_model,
            "baseline_coef": coef, "baseline_scale": scale,
            "baseline_features": ["intercept", "operation_count_v2", "base_bytes_est"],
        }, outdir / f"{name}.roofline_residual_v3.joblib")
        pred = d.loc[test, [
            "task_id", "matrix_id", "measurement_count", "time_cv",
            "label_time_ms",
        ]].rename(columns={"label_time_ms": "actual_ms"})
        pred["hgb_ms"] = direct_pred
        pred["roofline_residual_ms"] = residual_pred
        pred.to_csv(outdir / f"{name}.predictions.csv", index=False)
        metadata["targets"][name] = {
            "raw_rows": raw_count, "logical_tasks": int(len(d)),
            "n_matrices": int(d["matrix_id"].nunique()),
            "n_train": int(train.sum()), "n_test": int(test.sum()),
            "baseline_scale": scale.tolist(), "baseline_coef": coef.tolist(),
        }
        print(pd.DataFrame(all_metrics[-2:]).to_string(index=False), flush=True)

    pd.DataFrame(all_metrics).to_csv(outdir / "model_metrics.csv", index=False)
    (outdir / "stability_report.json").write_text(
        json.dumps(all_stability, indent=2) + "\n"
    )
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
