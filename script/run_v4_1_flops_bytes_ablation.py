#!/usr/bin/env python3
"""Cross-model, matrix-grouped 5-fold ablation of operations and bytes (H100 v4.1)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from interpret.glassbox import ExplainableBoostingRegressor
except ImportError:
    ExplainableBoostingRegressor = None

TARGETS = ("GETRF", "GESSM", "TSTRF", "SSSSM_dense")
OPS = "replay_operation_count_v2"
BYTES = "replay_base_bytes_est"
SIZE = "replay_main_dimension"
SEED = 20260806

# Frozen v4.1 online-safe feature lists (identical to balanced_gt_v2 manifest).
FEATURES = {
    "GETRF": [SIZE, "replay_nnz_input_0", "replay_nnz_output_0", "replay_nnz_output_1",
              "replay_input_0_col_nnz_mean", "replay_input_0_col_nnz_max",
              "replay_input_0_col_nnz_cv", "replay_input_0_col_nnz_max_mean_ratio",
              "replay_input_0_col_empty_fraction"],
    "GESSM": [SIZE, "replay_nnz_input_0", "replay_nnz_input_1", "replay_nnz_output_0"] +
             [f"replay_input_{i}_col_{s}" for i in (0, 1) for s in
              ("nnz_mean", "nnz_max", "nnz_cv", "nnz_max_mean_ratio", "empty_fraction")],
    "TSTRF": [SIZE, "replay_nnz_input_0", "replay_nnz_input_1", "replay_nnz_output_0"] +
             [f"replay_input_{i}_row_{s}" for i in (0, 1) for s in
              ("nnz_mean", "nnz_max", "nnz_cv", "nnz_max_mean_ratio", "empty_fraction")],
    "SSSSM_dense": [SIZE, "replay_nnz_input_0", "replay_nnz_input_1", "replay_nnz_input_2"] +
             [f"replay_input_{i}_col_{s}" for i in (0, 1, 2) for s in
              ("nnz_mean", "nnz_max", "nnz_cv", "nnz_max_mean_ratio", "empty_fraction")],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--fold-map", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--skip-ebm", action="store_true")
    return p.parse_args()


def metric(y, pred):
    y = np.asarray(y, float)
    pred = np.maximum(np.asarray(pred, float), 0.0)
    ape = np.abs(pred - y) / np.maximum(np.abs(y), 1e-6) * 100.0
    return {"n_test": len(y), "MAPE": float(ape.mean()), "MdAPE": float(np.median(ape)),
            "P95_APE": float(np.percentile(ape, 95)),
            "WAPE": float(np.abs(pred-y).sum()/max(np.abs(y).sum(), 1e-12)*100),
            "RMSE_ms": float(np.sqrt(mean_squared_error(y, pred))), "R2": float(r2_score(y, pred))}


def values(d, cols):
    x = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    return np.log1p(np.maximum(x, 0.0))


def roofline(ops, byt, y, train):
    positive_ops = ops[train & (ops > 0)]
    positive_bytes = byt[train & (byt > 0)]
    scale = np.array([max(float(np.median(positive_ops)), 1.0), max(float(np.median(positive_bytes)), 1.0)])
    design = np.column_stack((np.ones(len(y)), ops/scale[0], byt/scale[1]))
    coef, _ = nnls(design[train], y[train])
    return np.maximum(design @ coef, 1e-9)


def new_model(kind, seed):
    if kind == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "HGB":
        return HistGradientBoostingRegressor(loss="squared_error", max_iter=250, learning_rate=.06,
                                             max_leaf_nodes=31, min_samples_leaf=20,
                                             l2_regularization=1.0, random_state=seed)
    if kind == "EBM":
        if ExplainableBoostingRegressor is None:
            raise RuntimeError("interpret is unavailable")
        return ExplainableBoostingRegressor(interactions=0, max_rounds=150, learning_rate=.05,
                                            min_samples_leaf=20, outer_bags=4, inner_bags=0,
                                            n_jobs=8, random_state=seed)
    raise ValueError(kind)


def main():
    a = parse_args()
    src, out = Path(a.input_dir), Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    fold_df = pd.read_csv(a.fold_map)
    fold_map = dict(zip(fold_df.matrix_id.astype(str), fold_df.cv_fold.astype(int)))
    all_rows, fold_rows, timing_rows, predictions = [], [], [], []
    manifest = {"schema": "flops_bytes_cross_model_ablation_v4.1", "seed": SEED,
                "cv": "shared matrix-grouped 5-fold", "targets": {}, "feature_groups": {}}
    wall0 = time.perf_counter()

    for target_index, target in enumerate(TARGETS):
        path = src / f"{target}.model_ready_v4.csv"
        required = set(["task_id", "matrix_id", "label_time_ms", "hardware_id", OPS, BYTES] + FEATURES[target])
        d = pd.read_csv(path, usecols=lambda c: c in required, low_memory=False)
        y = pd.to_numeric(d.label_time_ms, errors="coerce").to_numpy(float)
        valid = np.isfinite(y) & (y > 0)
        d, y = d.loc[valid].reset_index(drop=True), y[valid]
        if set(d.hardware_id.astype(str)) != {"H100_SXM_80GB"}:
            raise RuntimeError(f"{target}: hardware mismatch")
        folds = d.matrix_id.astype(str).map(fold_map)
        if folds.isna().any() or set(folds.astype(int)) != {1,2,3,4,5}:
            raise RuntimeError(f"{target}: invalid shared folds")
        fid = folds.to_numpy(int)
        ops = pd.to_numeric(d[OPS], errors="coerce").fillna(0).to_numpy(float)
        byt = pd.to_numeric(d[BYTES], errors="coerce").fillna(0).to_numpy(float)
        structure = [c for c in FEATURES[target] if c != SIZE]
        groups = {"Size_only": [SIZE], "Structure_only": structure, "FLOPs_only": [OPS],
                  "Bytes_only": [BYTES], "FLOPs_Bytes": [OPS, BYTES],
                  "All_minus_FB": FEATURES[target], "All": FEATURES[target] + [OPS, BYTES]}
        manifest["feature_groups"][target] = groups
        manifest["targets"][target] = {"rows": len(d), "matrices": int(d.matrix_id.nunique())}
        cache = {name: values(d, cols) for name, cols in groups.items()}

        specs = [(m, g) for m in ("Ridge", "HGB") for g in groups]
        if not a.skip_ebm:
            specs += [("EBM", g) for g in ("Structure_only", "FLOPs_Bytes", "All_minus_FB", "All")]
        specs += [("Analytical_Roofline", "FLOPs_Bytes"), ("Roofline_HGB_residual", "All_minus_FB")]

        for spec_index, (model_name, group_name) in enumerate(specs):
            oof = np.empty(len(y), float)
            for fold in range(1, 6):
                train, test = fid != fold, fid == fold
                t0 = time.perf_counter()
                if model_name == "Analytical_Roofline":
                    base = roofline(ops, byt, y, train)
                    fit_seconds = time.perf_counter() - t0
                    t1 = time.perf_counter(); oof[test] = base[test]; pred_seconds = time.perf_counter() - t1
                elif model_name == "Roofline_HGB_residual":
                    base = roofline(ops, byt, y, train)
                    residual = np.log1p(y[train]) - np.log1p(base[train])
                    model = new_model("HGB", SEED + target_index*1000 + fold)
                    model.fit(cache[group_name][train], residual)
                    fit_seconds = time.perf_counter() - t0
                    t1 = time.perf_counter()
                    oof[test] = np.expm1(np.log1p(base[test]) + model.predict(cache[group_name][test]))
                    pred_seconds = time.perf_counter() - t1
                else:
                    model = new_model(model_name, SEED + target_index*1000 + spec_index*10 + fold)
                    model.fit(cache[group_name][train], np.log1p(y[train]))
                    fit_seconds = time.perf_counter() - t0
                    t1 = time.perf_counter(); oof[test] = np.expm1(model.predict(cache[group_name][test])); pred_seconds = time.perf_counter() - t1
                fm = metric(y[test], oof[test])
                fold_rows.append({"target": target, "model": model_name, "feature_group": group_name,
                                  "fold": fold, **fm})
                timing_rows.append({"target": target, "model": model_name, "feature_group": group_name,
                                    "fold": fold, "n_train": int(train.sum()), "n_test": int(test.sum()),
                                    "fit_seconds": fit_seconds, "predict_seconds": pred_seconds,
                                    "predict_us_per_sample": pred_seconds/test.sum()*1e6})
            summary = {"target": target, "model": model_name, "feature_group": group_name, **metric(y, oof)}
            all_rows.append(summary)
            predictions.append(pd.DataFrame({"target": target, "task_id": d.task_id, "matrix_id": d.matrix_id,
                                             "cv_fold": fid, "model": model_name, "feature_group": group_name,
                                             "actual_ms": y, "predicted_ms": np.maximum(oof, 0)}))
            print(target, model_name, group_name, f"MAPE={summary['MAPE']:.3f}", flush=True)

    comparison = pd.DataFrame(all_rows)
    comparison.to_csv(out/"model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out/"fold_metrics.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(out/"timing.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out/"oof_predictions.csv.gz", index=False, compression="gzip")

    # Paired evidence: same folds and samples, hence direct MAPE deltas are meaningful.
    evidence = []
    for target in TARGETS:
        for model in ("Ridge", "HGB", "EBM"):
            q = comparison[(comparison.target == target) & (comparison.model == model)].set_index("feature_group")
            if {"All", "All_minus_FB"}.issubset(q.index):
                without, full = q.loc["All_minus_FB"], q.loc["All"]
                evidence.append({"target": target, "model": model,
                                 "MAPE_without_FB": without.MAPE, "MAPE_with_FB": full.MAPE,
                                 "MAPE_reduction_points": without.MAPE-full.MAPE,
                                 "relative_MAPE_reduction_pct": (without.MAPE-full.MAPE)/without.MAPE*100,
                                 "R2_without_FB": without.R2, "R2_with_FB": full.R2})
    pd.DataFrame(evidence).to_csv(out/"flops_bytes_evidence.csv", index=False)
    manifest["wall_seconds"] = time.perf_counter()-wall0
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
