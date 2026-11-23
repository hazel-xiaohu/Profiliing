#!/usr/bin/env python3
"""Train per-kernel PanguLU kernel-time models with matrix-grouped CV."""

import argparse
import csv
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold


KERNELS = ("GETRF", "GESSM", "TSTRF", "SSSSM")
FEATURES = [
    "level", "block_row", "block_col", "update_k", "rank", "block_size",
    "process_rows", "process_cols", "nnz_input_0", "nnz_input_1",
    "nnz_output", "flops", "estimated_bytes", "operational_intensity",
    "row_nnz_cv", "col_nnz_cv", "structural_factor",
]
LOG_FEATURES = {
    "nnz_input_0", "nnz_input_1", "nnz_output", "flops",
    "estimated_bytes", "operational_intensity", "row_nnz_cv",
    "col_nnz_cv", "structural_factor",
}


def arguments():
    parser = argparse.ArgumentParser(
        description="Train nonlinear kernel-time models from balanced training CSVs."
    )
    parser.add_argument("training_set", help="Directory containing four sampled CSVs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--l2", type=float, default=1.0)
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty training CSV: {path}")
    return rows


def number(row, name):
    try:
        value = float(row.get(name, "0") or 0.0)
    except ValueError:
        value = 0.0
    if not math.isfinite(value):
        value = 0.0
    if name in LOG_FEATURES:
        value = math.log1p(max(value, 0.0))
    return value


def design(rows):
    return np.asarray([[number(row, name) for name in FEATURES] for row in rows], dtype=float)


def target(rows, name):
    values = np.asarray([float(row[name]) for row in rows], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise SystemExit(f"target {name} contains non-positive or invalid values")
    return values


def model(args):
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=args.learning_rate,
        max_iter=args.max_iter, max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf, l2_regularization=args.l2,
        early_stopping=False, random_state=args.seed,
    )


def metrics(actual, predicted):
    predicted = np.maximum(predicted, 0.0)
    absolute = np.abs(predicted - actual)
    relative = absolute / np.maximum(actual, 1.0e-9)
    smape = 2.0 * absolute / np.maximum(np.abs(actual) + np.abs(predicted), 1.0e-9)
    return {
        "samples": int(actual.size),
        "mae_ms": float(np.mean(absolute)),
        "rmse_ms": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "mape_percent": float(np.mean(relative) * 100.0),
        "median_ape_percent": float(np.median(relative) * 100.0),
        "p90_ape_percent": float(np.percentile(relative, 90) * 100.0),
        "smape_percent": float(np.mean(smape) * 100.0),
        "r2": float(r2_score(actual, predicted)),
    }


def grouped_splits(rows, requested_folds):
    groups = np.asarray([row["matrix_id"] for row in rows])
    unique = np.unique(groups)
    folds = min(requested_folds, unique.size)
    if folds < 2:
        raise SystemExit("at least two distinct matrix_id values are required for grouped CV")
    splitter = GroupKFold(n_splits=folds)
    return groups, list(splitter.split(np.zeros(len(rows)), groups=groups)), folds


def fit_one(args, rows, target_name, label, output):
    x = design(rows)
    y_ms = target(rows, target_name)
    y_log = np.log1p(y_ms)
    groups, splits, folds = grouped_splits(rows, args.folds)
    predictions = np.empty_like(y_ms)
    fold_ids = np.empty(len(rows), dtype=int)
    for fold, (train, test) in enumerate(splits, 1):
        estimator = model(args)
        estimator.fit(x[train], y_log[train])
        predictions[test] = np.expm1(estimator.predict(x[test]))
        fold_ids[test] = fold
    result = metrics(y_ms, predictions)
    result.update({"model": label, "target": target_name, "folds": folds,
                   "matrices": int(np.unique(groups).size)})
    estimator = model(args)
    estimator.fit(x, y_log)
    artifact = {
        "estimator": estimator, "features": FEATURES,
        "log_features": sorted(LOG_FEATURES), "target": target_name,
        "target_transform": "log1p", "prediction_inverse": "expm1_clip_zero",
    }
    joblib.dump(artifact, output / f"{label}.joblib")
    return result, predictions, fold_ids


def write_predictions(path, rows, columns):
    fields = ["task_id", "matrix_id", "block_size", "repeat_index"] + list(columns)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            item = {name: row.get(name, "") for name in fields}
            for name, values in columns.items():
                item[name] = values[index]
            writer.writerow(item)


def main():
    args = arguments()
    source = Path(args.training_set).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for kernel in ("GETRF", "GESSM", "TSTRF"):
        rows = read_rows(source / f"{kernel}.csv")
        result, predicted, folds = fit_one(
            args, rows, "time_kernel_ms", kernel, output
        )
        actual = target(rows, "time_kernel_ms")
        all_metrics.append(result)
        write_predictions(output / f"{kernel}_cv_predictions.csv", rows,
                          {"actual_time_kernel_ms": actual,
                           "predicted_time_kernel_ms": predicted,
                           "cv_fold": folds})

    rows = read_rows(source / "SSSSM.csv")
    sparse_result, sparse_pred, folds = fit_one(
        args, rows, "sparse_time_kernel_ms", "SSSSM_sparse", output
    )
    dense_result, dense_pred, dense_folds = fit_one(
        args, rows, "dense_time_kernel_ms", "SSSSM_dense", output
    )
    if not np.array_equal(folds, dense_folds):
        raise RuntimeError("SSSSM sparse/dense CV folds differ")
    all_metrics.extend([sparse_result, dense_result])
    sparse_actual = target(rows, "sparse_time_kernel_ms")
    dense_actual = target(rows, "dense_time_kernel_ms")
    predicted_dense = dense_pred < sparse_pred
    actual_dense = dense_actual < sparse_actual
    chosen_actual = np.where(predicted_dense, dense_actual, sparse_actual)
    best_actual = np.minimum(sparse_actual, dense_actual)
    regret = (chosen_actual - best_actual) / np.maximum(best_actual, 1.0e-9)
    selector = {
        "model": "SSSSM_selector", "target": "faster_variant",
        "samples": len(rows),
        "selection_accuracy_percent": float(np.mean(predicted_dense == actual_dense) * 100.0),
        "mean_regret_percent": float(np.mean(regret) * 100.0),
        "p90_regret_percent": float(np.percentile(regret, 90) * 100.0),
    }
    all_metrics.append(selector)
    write_predictions(
        output / "SSSSM_cv_predictions.csv", rows,
        {"actual_sparse_ms": sparse_actual, "predicted_sparse_ms": sparse_pred,
         "actual_dense_ms": dense_actual, "predicted_dense_ms": dense_pred,
         "actual_faster_variant": np.where(actual_dense, "dense", "sparse"),
         "predicted_faster_variant": np.where(predicted_dense, "dense", "sparse"),
         "cv_fold": folds},
    )
    (output / "SSSSM_selector.json").write_text(
        json.dumps({"policy": "choose the lower prediction from SSSSM_sparse.joblib "
                              "and SSSSM_dense.joblib", **selector}, indent=2) + "\n"
    )
    (output / "metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
