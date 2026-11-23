#!/usr/bin/env python3
"""Fit interpretable latency-aware Roofline models for PanguLU kernels."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold


SPECS = (
    ("GETRF", "GETRF.csv", "time_kernel_ms"),
    ("GESSM", "GESSM.csv", "time_kernel_ms"),
    ("TSTRF", "TSTRF.csv", "time_kernel_ms"),
    ("SSSSM_sparse", "SSSSM.csv", "sparse_time_kernel_ms"),
    ("SSSSM_dense", "SSSSM.csv", "dense_time_kernel_ms"),
)
EPS = 1.0e-12


def arguments():
    parser = argparse.ArgumentParser(description="Fit latency-aware Roofline formulas")
    parser.add_argument("training_set")
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty training CSV: {path}")
    return rows


def finite(row, field):
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def arrays(rows, target_field):
    kept, values = [], []
    for row in rows:
        f, b = finite(row, "flops"), finite(row, "estimated_bytes")
        y = finite(row, target_field)
        if None in (f, b, y) or min(f, b, y) < 0 or y == 0:
            continue
        kept.append(row)
        values.append((f / 1.0e9, b / 1.0e9, 1.0, y))
    if not values:
        raise SystemExit(f"no valid rows for {target_field}")
    data = np.asarray(values, dtype=float)
    return kept, data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def predict(parameters, flops_g, bytes_gb, launches):
    base, extra_launch, compute_tflops, bandwidth_tbps = parameters
    compute_ms = flops_g / compute_tflops
    memory_ms = bytes_gb / bandwidth_tbps
    latency_ms = base + extra_launch * np.maximum(launches - 1.0, 0.0)
    return latency_ms + np.maximum(compute_ms, memory_ms)


def fit(flops_g, bytes_gb, launches, actual):
    variable_launches = np.unique(launches).size > 1
    initial = np.asarray([
        max(float(np.percentile(actual, 10)) * 0.5, 1.0e-4),
        1.0e-3, 10.0, 1.0,
    ])
    active = np.asarray([True, variable_launches, True, True])
    fixed = initial.copy()
    if not variable_launches:
        fixed[1] = 0.0

    def expand(theta):
        result = fixed.copy()
        result[active] = np.exp(theta)
        return result

    def residual(theta):
        estimated = predict(expand(theta), flops_g, bytes_gb, launches)
        return np.log1p(estimated) - np.log1p(actual)

    result = least_squares(
        residual, np.log(initial[active]),
        bounds=(math.log(1.0e-9), math.log(1.0e6)),
        loss="soft_l1", f_scale=0.1, max_nfev=5000,
    )
    return expand(result.x), variable_launches


def metric(actual, estimated):
    absolute = np.abs(estimated - actual)
    relative = absolute / np.maximum(actual, 1.0e-9)
    return {
        "samples": int(actual.size),
        "mae_ms": float(np.mean(absolute)),
        "rmse_ms": float(np.sqrt(np.mean((estimated - actual) ** 2))),
        "mape_percent": float(np.mean(relative) * 100.0),
        "median_ape_percent": float(np.median(relative) * 100.0),
        "p90_ape_percent": float(np.percentile(relative, 90) * 100.0),
        "r2": float(r2_score(actual, estimated)),
    }


def grouped_cv(rows, flops_g, bytes_gb, launches, actual, requested):
    groups = np.asarray([row["matrix_id"] for row in rows])
    folds = min(requested, np.unique(groups).size)
    if folds < 2:
        raise SystemExit("at least two matrices are required for grouped CV")
    estimated = np.empty_like(actual)
    fold_ids = np.empty(actual.size, dtype=int)
    split = GroupKFold(n_splits=folds)
    for fold_id, (train, test) in enumerate(split.split(actual, groups=groups), 1):
        parameters, _ = fit(flops_g[train], bytes_gb[train], launches[train], actual[train])
        estimated[test] = predict(parameters, flops_g[test], bytes_gb[test], launches[test])
        fold_ids[test] = fold_id
    return estimated, fold_ids, folds


def write_predictions(path, rows, actual, estimated, folds):
    fields = ["task_id", "matrix_id", "block_size", "repeat_index",
              "actual_time_kernel_ms", "predicted_time_kernel_ms", "cv_fold"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, y, yhat, fold in zip(rows, actual, estimated, folds):
            writer.writerow({
                "task_id": row.get("task_id", ""), "matrix_id": row.get("matrix_id", ""),
                "block_size": row.get("block_size", ""),
                "repeat_index": row.get("repeat_index", ""),
                "actual_time_kernel_ms": y, "predicted_time_kernel_ms": yhat,
                "cv_fold": fold,
            })


def formula(parameters):
    base, launch, compute, bandwidth = parameters
    return (
        f"T_ms = {base:.8g} + {launch:.8g} * max(launches - 1, 0) "
        f"+ max(F_GFLOP / {compute:.8g}, B_GB / {bandwidth:.8g})"
    )


def main():
    args = arguments()
    source = Path(args.training_set).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    results = []

    for name, filename, target_field in SPECS:
        rows, flops_g, bytes_gb, launches, actual = arrays(
            read_rows(source / filename), target_field
        )
        cv_estimated, fold_ids, folds = grouped_cv(
            rows, flops_g, bytes_gb, launches, actual, args.folds
        )
        parameters, variable_launches = fit(flops_g, bytes_gb, launches, actual)
        compute_ms = flops_g / parameters[2]
        memory_ms = bytes_gb / parameters[3]
        item = {
            "model": name,
            "formula": formula(parameters),
            "units": {"T": "ms", "F": "GFLOP", "B": "GB",
                      "P_eff": "TFLOP/s", "BW_eff": "TB/s"},
            "base_latency_ms": float(parameters[0]),
            "extra_launch_latency_ms": float(parameters[1]),
            "launch_latency_fitted": variable_launches,
            "effective_compute_tflops": float(parameters[2]),
            "effective_bandwidth_tbps": float(parameters[3]),
            "compute_bound_percent": float(np.mean(compute_ms >= memory_ms) * 100.0),
            "memory_bound_percent": float(np.mean(memory_ms > compute_ms) * 100.0),
            "cv_folds": folds,
            "matrices": len({row["matrix_id"] for row in rows}),
            "cv_metrics": metric(actual, cv_estimated),
        }
        warnings = []
        if item["effective_compute_tflops"] > 1000.0:
            warnings.append("compute rate is weakly identified")
        if item["effective_bandwidth_tbps"] > 100.0:
            warnings.append("bandwidth is weakly identified; estimated_bytes is not explanatory")
        if item["cv_metrics"]["r2"] < 0.0:
            warnings.append("negative grouped-CV R2; formula is not suitable for prediction")
        item["warnings"] = warnings
        item["fit_status"] = "diagnostic_only" if warnings else "usable_preliminary"
        results.append(item)
        write_predictions(output / f"{name}_cv_predictions.csv", rows, actual,
                          cv_estimated, fold_ids)

    (output / "roofline_models.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# PanguLU 可解释 Roofline 模型", "",
        "模型形式：`T_ms = L0 + Llaunch × max(launches-1, 0) + max(F_GFLOP/P_eff, B_GB/BW_eff)`。",
        "拟合值是当前任务路径的有效吞吐，不是 GPU 硬件峰值。", "",
        "| Kernel | L0 (ms) | Llaunch (ms) | P_eff (TFLOP/s) | BW_eff (TB/s) | Compute-bound | CV R² | CV MAPE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        m = item["cv_metrics"]
        lines.append(
            f"| {item['model']} | {item['base_latency_ms']:.6g} | "
            f"{item['extra_launch_latency_ms']:.6g} | "
            f"{item['effective_compute_tflops']:.6g} | "
            f"{item['effective_bandwidth_tbps']:.6g} | "
            f"{item['compute_bound_percent']:.2f}% | {m['r2']:.4f} | "
            f"{m['mape_percent']:.2f}% |"
        )
    lines.extend(["", "## 公式", ""])
    for item in results:
        lines.extend([f"### {item['model']}", "", f"`{item['formula']}`", ""])
        if item["warnings"]:
            lines.extend([
                "警告：" + "；".join(item["warnings"]) + "。", "",
            ])
    (output / "roofline_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
