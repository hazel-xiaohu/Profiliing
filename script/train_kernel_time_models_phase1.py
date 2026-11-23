#!/usr/bin/env python3
"""Train phase-1 weighted kernel-time and paired SSSSM ratio models."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold


BASE_FEATURES = [
    "level", "block_row", "block_col", "update_k", "rank", "block_size",
    "process_rows", "process_cols", "nnz_input_0", "nnz_input_1",
    "nnz_input_2", "nnz_output_0", "nnz_output_1",
    "flops", "estimated_bytes", "operational_intensity",
    "row_nnz_cv", "col_nnz_cv", "structural_factor",
]
DERIVED_FEATURES = ["output_density"]
FEATURES = BASE_FEATURES + DERIVED_FEATURES
LOG_FEATURES = {
    "nnz_input_0", "nnz_input_1", "nnz_input_2",
    "nnz_output_0", "nnz_output_1", "flops", "estimated_bytes",
    "operational_intensity", "row_nnz_cv", "col_nnz_cv", "structural_factor",
    "output_density",
}
CANDIDATES = [
    {"loss": "squared_error", "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0},
    {"loss": "absolute_error", "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0},
    {"loss": "squared_error", "max_leaf_nodes": 63, "min_samples_leaf": 50, "l2_regularization": 10.0},
    {"loss": "absolute_error", "max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 10.0},
]


def arguments():
    parser = argparse.ArgumentParser(description="Train phase-1 weighted models with nested grouped CV")
    parser.add_argument("training_set")
    parser.add_argument("--output", required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--epsilon-ms", type=float, default=1.0e-4)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="") as handle:
        result = list(csv.DictReader(handle))
    if not result:
        raise SystemExit(f"empty training CSV: {path}")
    missing = [name for name in BASE_FEATURES if name not in result[0]]
    if missing:
        raise SystemExit(
            f"{path}: incompatible pre-kernel feature schema; missing {missing}. "
            "Re-profile with the online-safe collector before retraining."
        )
    return result


def number(row, name):
    try:
        value = float(row.get(name, "0") or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    return value if math.isfinite(value) else 0.0


def design(rows):
    matrix = []
    for row in rows:
        raw = {name: number(row, name) for name in BASE_FEATURES}
        output_blocks = 2.0 if row.get("kernel_type") == "GETRF" else 1.0
        output_nnz = raw["nnz_output_0"] + raw["nnz_output_1"]
        raw.update({
            "output_density": output_nnz /
                              max(output_blocks * raw["block_size"] ** 2, 1.0),
        })
        matrix.append([math.log1p(max(raw[name], 0.0)) if name in LOG_FEATURES
                       else raw[name] for name in FEATURES])
    return np.asarray(matrix, dtype=float)


def target(rows, field):
    values = np.asarray([number(row, field) for row in rows], dtype=float)
    if np.any(values <= 0):
        raise SystemExit(f"non-positive target in {field}")
    return values


def weights(rows):
    result = np.asarray([max(number(row, "stratum_weight"), 0.0) for row in rows])
    if not np.any(result > 0):
        result = np.ones(len(rows))
    return result / np.mean(result)


def estimator(args, params):
    return HistGradientBoostingRegressor(
        **params, max_iter=args.max_iter, learning_rate=args.learning_rate,
        early_stopping=False, random_state=args.seed,
    )


def weighted_mean(values, weight):
    return float(np.average(values, weights=weight))


def log_error(actual, predicted, weight, epsilon):
    error = np.abs(np.log(actual + epsilon) - np.log(np.maximum(predicted, 0) + epsilon))
    return weighted_mean(error, weight)


def tune(args, x, y_train, y_ms, groups, weight, inverse):
    folds = min(args.inner_folds, np.unique(groups).size)
    if folds < 2:
        return CANDIDATES[0], []
    splitter = GroupKFold(n_splits=folds)
    records = []
    for index, params in enumerate(CANDIDATES):
        scores = []
        for train, valid in splitter.split(x, groups=groups):
            model = estimator(args, params)
            model.fit(x[train], y_train[train], sample_weight=weight[train])
            prediction = inverse(model.predict(x[valid]))
            scores.append(log_error(y_ms[valid], prediction, weight[valid], args.epsilon_ms))
        records.append({"candidate": index, "mean_abs_log_error": float(np.mean(scores)),
                        "params": params})
    best = min(records, key=lambda item: item["mean_abs_log_error"])
    return best["params"], records


def metrics(actual, predicted, weight):
    predicted = np.maximum(predicted, 0.0)
    absolute = np.abs(predicted - actual)
    relative = absolute / np.maximum(actual, 1.0e-9)
    smape = 2 * absolute / np.maximum(actual + predicted, 1.0e-9)
    return {
        "samples": int(actual.size), "mae_ms": float(np.mean(absolute)),
        "rmse_ms": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "mape_percent": float(np.mean(relative) * 100),
        "median_ape_percent": float(np.median(relative) * 100),
        "p90_ape_percent": float(np.percentile(relative, 90) * 100),
        "wape_percent": float(np.sum(absolute) / np.sum(actual) * 100),
        "smape_percent": float(np.mean(smape) * 100),
        "weighted_mape_percent": weighted_mean(relative, weight) * 100,
        "weighted_smape_percent": weighted_mean(smape, weight) * 100,
        "r2": float(r2_score(actual, predicted)),
    }


def nested_time_model(args, rows, target_field, label, output):
    x = design(rows)
    y_ms = target(rows, target_field)
    y_train = np.log(y_ms + args.epsilon_ms)
    group = np.asarray([row["matrix_id"] for row in rows])
    weight = weights(rows)
    folds = min(args.outer_folds, np.unique(group).size)
    split = GroupKFold(n_splits=folds)
    predictions = np.empty_like(y_ms)
    fold_ids = np.empty(len(rows), dtype=int)
    outer_params = []
    inverse = lambda value: np.maximum(np.exp(value) - args.epsilon_ms, 0.0)
    for fold, (train, test) in enumerate(split.split(x, groups=group), 1):
        params, tuning = tune(args, x[train], y_train[train], y_ms[train],
                              group[train], weight[train], inverse)
        model = estimator(args, params)
        model.fit(x[train], y_train[train], sample_weight=weight[train])
        predictions[test] = inverse(model.predict(x[test]))
        fold_ids[test] = fold
        outer_params.append({"fold": fold, "selected": params, "tuning": tuning})
    final_params, final_tuning = tune(args, x, y_train, y_ms, group, weight, inverse)
    final_model = estimator(args, final_params)
    final_model.fit(x, y_train, sample_weight=weight)
    artifact = {"estimator": final_model, "features": FEATURES,
                "target": target_field,
                "target_transform": f"log(time_ms + {args.epsilon_ms})",
                "prediction_inverse": f"exp(y)-{args.epsilon_ms}, clipped at zero",
                "sample_weight": "inverse selected stratum size, normalized",
                "selected_params": final_params}
    joblib.dump(artifact, output / f"{label}.joblib")
    result = metrics(y_ms, predictions, weight)
    result.update({"model": label, "target": target_field, "outer_folds": folds,
                   "matrices": int(np.unique(group).size),
                   "selected_params": final_params,
                   "outer_selected_params": outer_params,
                   "final_tuning": final_tuning})
    return result, predictions, fold_ids


def nested_ratio_model(args, rows, output):
    x = design(rows)
    sparse = target(rows, "sparse_time_kernel_ms")
    dense = target(rows, "dense_time_kernel_ms")
    ratio = np.log(dense / sparse)
    group = np.asarray([row["matrix_id"] for row in rows])
    base_weight = weights(rows)
    oracle_regret = np.abs(dense - sparse) / np.minimum(dense, sparse)
    weight = base_weight * (0.1 + np.minimum(oracle_regret, 10.0))
    weight /= np.mean(weight)
    folds = min(args.outer_folds, np.unique(group).size)
    split = GroupKFold(n_splits=folds)
    predictions = np.empty_like(ratio)
    fold_ids = np.empty(len(rows), dtype=int)
    outer_params = []

    def tune_ratio(indices):
        inner_groups = group[indices]
        inner_folds = min(args.inner_folds, np.unique(inner_groups).size)
        records = []
        for ci, params in enumerate(CANDIDATES):
            scores = []
            splitter = GroupKFold(n_splits=inner_folds)
            for train0, valid0 in splitter.split(x[indices], groups=inner_groups):
                train, valid = indices[train0], indices[valid0]
                model = estimator(args, params)
                model.fit(x[train], ratio[train], sample_weight=weight[train])
                pred = model.predict(x[valid])
                scores.append(weighted_mean(np.abs(pred - ratio[valid]), weight[valid]))
            records.append({"candidate": ci, "mean_abs_log_ratio_error": float(np.mean(scores)),
                            "params": params})
        best = min(records, key=lambda item: item["mean_abs_log_ratio_error"])
        return best["params"], records

    for fold, (train, test) in enumerate(split.split(x, groups=group), 1):
        params, tuning = tune_ratio(train)
        model = estimator(args, params)
        model.fit(x[train], ratio[train], sample_weight=weight[train])
        predictions[test] = model.predict(x[test])
        fold_ids[test] = fold
        outer_params.append({"fold": fold, "selected": params, "tuning": tuning})
    all_indices = np.arange(len(rows))
    final_params, final_tuning = tune_ratio(all_indices)
    final_model = estimator(args, final_params)
    final_model.fit(x, ratio, sample_weight=weight)
    joblib.dump({"estimator": final_model, "features": FEATURES,
                 "target": "log(dense_time/sparse_time)",
                 "selection": "dense if prediction < 0 else sparse",
                 "sample_weight": "stratum weight * clipped oracle regret",
                 "selected_params": final_params}, output / "SSSSM_ratio.joblib")
    return predictions, fold_ids, {"outer_selected_params": outer_params,
                                   "selected_params": final_params,
                                   "final_tuning": final_tuning,
                                   "mean_abs_log_ratio_error": float(np.mean(np.abs(predictions-ratio)))}


def selector_metrics(sparse, dense, choose_dense, weight, label):
    actual_dense = dense < sparse
    chosen = np.where(choose_dense, dense, sparse)
    best = np.minimum(sparse, dense)
    regret = (chosen - best) / np.maximum(best, 1.0e-9)
    return {"model": label, "samples": len(sparse),
            "selection_accuracy_percent": float(np.mean(choose_dense == actual_dense) * 100),
            "weighted_selection_accuracy_percent": weighted_mean(choose_dense == actual_dense, weight) * 100,
            "mean_regret_percent": float(np.mean(regret) * 100),
            "weighted_mean_regret_percent": weighted_mean(regret, weight) * 100,
            "p90_regret_percent": float(np.percentile(regret, 90)) * 100}


def write_predictions(path, rows, columns):
    fields = ["task_id", "matrix_id", "block_size", "repeat_index"] + list(columns)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            item = {name: row.get(name, "") for name in fields}
            for name, values in columns.items(): item[name] = values[i]
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
        result, prediction, folds = nested_time_model(
            args, rows, "time_kernel_ms", kernel, output)
        all_metrics.append(result)
        actual = target(rows, "time_kernel_ms")
        write_predictions(output / f"{kernel}_cv_predictions.csv", rows,
                          {"actual_time_kernel_ms": actual,
                           "predicted_time_kernel_ms": prediction, "cv_fold": folds})

    rows = read_rows(source / "SSSSM.csv")
    sparse_result, sparse_pred, folds = nested_time_model(
        args, rows, "sparse_time_kernel_ms", "SSSSM_sparse", output)
    dense_result, dense_pred, dense_folds = nested_time_model(
        args, rows, "dense_time_kernel_ms", "SSSSM_dense", output)
    if not np.array_equal(folds, dense_folds): raise RuntimeError("SSSSM folds differ")
    all_metrics.extend([sparse_result, dense_result])
    sparse = target(rows, "sparse_time_kernel_ms")
    dense = target(rows, "dense_time_kernel_ms")
    weight = weights(rows)
    independent = selector_metrics(sparse, dense, dense_pred < sparse_pred, weight,
                                   "SSSSM_independent_time_selector")
    ratio_pred, ratio_folds, ratio_details = nested_ratio_model(args, rows, output)
    if not np.array_equal(folds, ratio_folds): raise RuntimeError("SSSSM ratio folds differ")
    ratio_selector = selector_metrics(sparse, dense, ratio_pred < 0, weight,
                                      "SSSSM_regret_weighted_ratio_selector")
    ratio_selector["training"] = ratio_details
    all_metrics.extend([independent, ratio_selector])
    write_predictions(output / "SSSSM_cv_predictions.csv", rows, {
        "actual_sparse_ms": sparse, "predicted_sparse_ms": sparse_pred,
        "actual_dense_ms": dense, "predicted_dense_ms": dense_pred,
        "actual_log_dense_over_sparse": np.log(dense/sparse),
        "predicted_log_dense_over_sparse": ratio_pred,
        "independent_predicted_variant": np.where(dense_pred<sparse_pred,"dense","sparse"),
        "ratio_predicted_variant": np.where(ratio_pred<0,"dense","sparse"),
        "cv_fold": folds})
    (output / "metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2)+"\n")
    (output / "selector_comparison.json").write_text(
        json.dumps([independent, ratio_selector], ensure_ascii=False, indent=2)+"\n")
    print(json.dumps([{k:v for k,v in item.items() if k not in
                       ("outer_selected_params","final_tuning","training")}
                      for item in all_metrics], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
