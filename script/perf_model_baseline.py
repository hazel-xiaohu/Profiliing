#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
import random
from collections import defaultdict

import numpy as np


FEATURES = [
    "level",
    "block_row",
    "block_col",
    "update_k",
    "rank",
    "block_size",
    "process_rows",
    "process_cols",
    "nnz_input_0",
    "nnz_input_1",
    "nnz_output",
    "flops",
    "estimated_bytes",
    "operational_intensity",
    "row_nnz_cv",
    "col_nnz_cv",
    "structural_factor",
]

LOG_FEATURES = {
    "nnz_input_0",
    "nnz_input_1",
    "nnz_output",
    "flops",
    "estimated_bytes",
    "operational_intensity",
    "row_nnz_cv",
    "col_nnz_cv",
    "structural_factor",
}


def read_rows(patterns):
    rows = []
    skipped = []
    for pattern in patterns:
        paths = sorted(glob.glob(pattern))
        if not paths and os.path.isfile(pattern):
            paths = [pattern]
        for path in paths:
            with open(path, newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                if not any(name in fieldnames for name in ("time_kernel_ms", "time_total_ms", "time_ms")):
                    skipped.append(path)
                    continue
                for row in reader:
                    row["_source"] = path
                    rows.append(row)
    return rows, skipped


def value(row, name):
    raw = row.get(name, "")
    if raw in ("", None):
        return 0.0
    try:
        val = float(raw)
    except ValueError:
        return 0.0
    if name in LOG_FEATURES:
        return math.log1p(max(val, 0.0))
    return val


def feature_matrix(rows):
    return np.array([[value(row, name) for name in FEATURES] for row in rows], dtype=float)


def normalize_train_test(train_rows, test_rows):
    x_train = feature_matrix(train_rows)
    x_test = feature_matrix(test_rows)
    if x_train.size == 0:
        return x_train, x_test

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.0] = 1.0
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    train_intercept = np.ones((x_train.shape[0], 1), dtype=float)
    test_intercept = np.ones((x_test.shape[0], 1), dtype=float)
    return np.hstack([train_intercept, x_train]), np.hstack([test_intercept, x_test])


def design_matrix(rows):
    x = np.array([[value(row, name) for name in FEATURES] for row in rows], dtype=float)
    if x.size == 0:
        return x
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0.0] = 1.0
    x = (x - mean) / std
    intercept = np.ones((x.shape[0], 1), dtype=float)
    return np.hstack([intercept, x])


def fit_ridge(x, y, alpha):
    penalty = np.eye(x.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def split_rows(rows, test_fraction, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    test_size = max(1, int(round(len(rows) * test_fraction)))
    test_size = min(test_size, len(rows) - 1)
    return rows[test_size:], rows[:test_size]


def relative_errors(y_true, y_pred):
    denom = np.maximum(np.expm1(y_true), 1.0e-9)
    numer = np.abs(np.expm1(y_pred) - np.expm1(y_true))
    return numer / denom


def target_time_ms(row, target_field):
    names = (target_field,) if target_field != "time_total_ms" else ("time_total_ms", "time_ms")
    for name in names:
        raw = row.get(name, "")
        if raw in ("", None):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        if val > 0.0:
            return val
    return 0.0


def evaluate_kernel(kernel, rows, alpha, test_fraction, seed, target_field):
    if len(rows) < 4:
        return {
            "kernel": kernel,
            "samples": len(rows),
            "status": "skipped: need at least 4 samples",
        }

    train_rows, test_rows = split_rows(rows, test_fraction, seed)
    x_train, x_test = normalize_train_test(train_rows, test_rows)
    y_train = np.array([math.log1p(target_time_ms(row, target_field)) for row in train_rows], dtype=float)
    y_test = np.array([math.log1p(target_time_ms(row, target_field)) for row in test_rows], dtype=float)

    coef = fit_ridge(x_train, y_train, alpha)
    pred = x_test @ coef
    err = relative_errors(y_test, pred)

    return {
        "kernel": kernel,
        "samples": len(rows),
        "train": len(train_rows),
        "test": len(test_rows),
        "mape": float(err.mean() * 100.0),
        "p50": float(np.percentile(err, 50) * 100.0),
        "p90": float(np.percentile(err, 90) * 100.0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train a per-kernel log-linear ridge baseline from PanguLU catalog CSV files."
    )
    parser.add_argument("catalog", nargs="+", help="Catalog CSV file or glob, e.g. 'calib_catalog/*.csv'")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Held-out test fraction")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic split seed")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated random splits")
    parser.add_argument(
        "--target-field",
        choices=("time_kernel_ms", "time_total_ms"),
        default="time_kernel_ms",
        help="CSV timing field to model (default: CUDA Event kernel time)",
    )
    args = parser.parse_args()

    rows, skipped = read_rows(args.catalog)
    for path in skipped:
        print(f"warning: skipped catalog without a supported timing field: {path}")
    if not rows:
        raise SystemExit(
            "No usable catalog rows found. Rebuild PanguLU and re-run with "
            "CALIB_MODE=catalog so catalogs include the selected timing field."
        )

    by_kernel = defaultdict(list)
    for row in rows:
        if target_time_ms(row, args.target_field) > 0.0:
            kernel = row.get("kernel_type", "UNKNOWN")
            variant = row.get("kernel_variant", "default")
            by_kernel[f"{kernel}:{variant}"].append(row)

    print("kernel,samples,train,test,mape_percent,p50_percent,p90_percent,status")
    for kernel in sorted(by_kernel):
        results = [
            evaluate_kernel(kernel, by_kernel[kernel], args.alpha, args.test_fraction, args.seed + i, args.target_field)
            for i in range(max(args.repeats, 1))
        ]
        result = results[0]
        if "status" in result:
            print(f"{kernel},{result['samples']},0,0,,,,{result['status']}")
        else:
            if args.repeats > 1:
                mape = np.array([item["mape"] for item in results], dtype=float)
                p50 = np.array([item["p50"] for item in results], dtype=float)
                p90 = np.array([item["p90"] for item in results], dtype=float)
                status = (
                    f"ok; repeats={args.repeats}; "
                    f"mape_std={mape.std():.3f}; p50_std={p50.std():.3f}; p90_std={p90.std():.3f}"
                )
                result = dict(result)
                result["mape"] = float(mape.mean())
                result["p50"] = float(p50.mean())
                result["p90"] = float(p90.mean())
            else:
                status = "ok"
            print(
                f"{kernel},{result['samples']},{result['train']},{result['test']},"
                f"{result['mape']:.3f},{result['p50']:.3f},{result['p90']:.3f},{status}"
            )


if __name__ == "__main__":
    main()
