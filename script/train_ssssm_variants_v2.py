#!/usr/bin/env python3
"""Train independent online-safe models for dense and sparse SSSSM kernels."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from feature_sets_v3 import SSSSM_FEATURES


FEATURES = SSSSM_FEATURES


def metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.maximum(np.asarray(pred, dtype=float), 0.0)
    ape = np.abs(pred - y) / np.maximum(np.abs(y), 1e-6) * 100.0
    return {
        "n_test": int(len(y)),
        "MAPE": float(np.mean(ape)),
        "MdAPE": float(np.median(ape)),
        "P90_APE": float(np.percentile(ape, 90)),
        "P95_APE": float(np.percentile(ape, 95)),
        "RMSE_ms": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
    }


def hgb(seed):
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=.07, max_iter=300,
        max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0,
        random_state=seed,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    cols = set(
        ["task_id", "matrix_id", "kernel_type", "time_kernel_ms",
         "forced_ssssm_variant"] + FEATURES
    )
    df = pd.read_csv(args.input, usecols=lambda c: c in cols, low_memory=False)
    df["time_kernel_ms"] = pd.to_numeric(df["time_kernel_ms"], errors="coerce")
    df = df[
        df["kernel_type"].eq("SSSSM")
        & np.isfinite(df["time_kernel_ms"])
        & (df["time_kernel_ms"] > 0)
    ].copy()
    split = json.loads(Path(args.splits).read_text())["SSSSM"]
    train_matrices = set(split["train_matrices"])
    test_matrices = set(split["test_matrices"])

    results = []
    metadata = {
        "schema": "online_safe_v3",
        "features": FEATURES,
        "variant_is_feature": False,
        "routing": "variant is selected by the solver before kernel launch",
        "train_matrices": sorted(train_matrices),
        "test_matrices": sorted(test_matrices),
        "models": {},
    }
    for variant_index, variant in enumerate(("dense", "sparse")):
        d = df[df["forced_ssssm_variant"].eq(variant)].copy()
        tr = d["matrix_id"].isin(train_matrices).to_numpy()
        te = d["matrix_id"].isin(test_matrices).to_numpy()
        if not tr.any() or not te.any():
            raise RuntimeError(f"{variant}: empty train or test partition")
        X = (
            d[FEATURES].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        )
        y = d["time_kernel_ms"].to_numpy(float)

        direct = hgb(args.seed + variant_index)
        direct.fit(X[tr], np.log1p(y[tr]))
        direct_pred = np.maximum(np.expm1(direct.predict(X[te])), 0.0)
        results.append({
            "variant": variant, "model": "independent_HGB",
            **metrics(y[te], direct_pred),
        })

        ops = d["replay_operation_count_v2"].to_numpy(float)
        byt = d["replay_base_bytes_est"].to_numpy(float)
        ops_positive = ops[tr][ops[tr] > 0]
        byt_positive = byt[tr][byt[tr] > 0]
        scale = np.array([
            max(float(np.median(ops_positive)), 1.0),
            max(float(np.median(byt_positive)), 1.0),
        ])
        Z = np.column_stack([np.ones(len(d)), ops / scale[0], byt / scale[1]])
        coef, _ = nnls(Z[tr], y[tr])
        baseline = np.maximum(Z @ coef, 1e-9)
        residual = np.log1p(y) - np.log1p(baseline)
        residual_model = hgb(args.seed + 10 + variant_index)
        residual_model.fit(X[tr], residual[tr])
        residual_pred = np.maximum(
            np.expm1(np.log1p(baseline[te]) + residual_model.predict(X[te])), 0.0
        )
        results.append({
            "variant": variant, "model": "independent_Roofline_residual_HGB",
            "baseline_intercept_ms": float(coef[0]),
            "baseline_ms_per_scaled_ops": float(coef[1]),
            "baseline_ms_per_scaled_bytes": float(coef[2]),
            **metrics(y[te], residual_pred),
        })

        joblib.dump({
            "schema": "online_safe_v3",
            "kernel": "SSSSM",
            "variant": variant,
            "features": FEATURES,
            "target_transform": "log1p",
            "model": direct,
        }, outdir / f"SSSSM_{variant}.hgb_v2.joblib")
        joblib.dump({
            "schema": "online_safe_v3",
            "kernel": "SSSSM",
            "variant": variant,
            "features": FEATURES,
            "residual_model": residual_model,
            "baseline_coef": coef,
            "baseline_scale": scale,
            "baseline_features": ["intercept", "operation_count_v2", "base_bytes_est"],
        }, outdir / f"SSSSM_{variant}.roofline_residual_v2.joblib")

        pred = d.loc[te, ["task_id", "matrix_id", "time_kernel_ms"]].copy()
        pred = pred.rename(columns={"time_kernel_ms": "actual_ms"})
        pred["independent_hgb_ms"] = direct_pred
        pred["independent_roofline_residual_ms"] = residual_pred
        pred.to_csv(outdir / f"SSSSM_{variant}.predictions.csv", index=False)
        metadata["models"][variant] = {
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "baseline_scale": scale.tolist(), "baseline_coef": coef.tolist(),
        }

    result_df = pd.DataFrame(results)
    result_df.to_csv(outdir / "variant_model_metrics.csv", index=False)
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
