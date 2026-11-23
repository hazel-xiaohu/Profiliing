#!/usr/bin/env python3
"""Shared reproducible matrix-grouped 5-fold v3 ablation/model training."""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from feature_sets_v3 import FEATURES_BY_KERNEL

TARGETS = ("GETRF", "GESSM", "TSTRF", "SSSSM_dense", "SSSSM_sparse")
KEY = [
    "task_id", "matrix_id", "kernel_type", "level", "block_row", "block_col",
    "update_k", "rank", "block_size", "process_rows", "process_cols",
]
SIZE_NNZ = [
    "replay_main_dimension", "replay_block_rows", "replay_block_cols",
    "replay_aspect_ratio", "replay_min_dimension", "replay_max_dimension",
    "replay_nnz_input_0", "replay_nnz_input_1", "replay_nnz_input_2",
    "replay_nnz_output_0", "replay_nnz_output_1",
]
ARITHMETIC = [
    "replay_num_macs", "replay_num_divisions",
    "replay_operation_count_v2", "replay_base_bytes_est",
]
IRREGULARITY = [
    "replay_row_nnz_mean", "replay_row_nnz_max", "replay_row_nnz_cv",
    "replay_row_nnz_max_mean_ratio", "replay_row_empty_fraction",
    "replay_col_nnz_mean", "replay_col_nnz_max", "replay_col_nnz_cv",
    "replay_col_nnz_max_mean_ratio", "replay_col_empty_fraction",
    "replay_work_mean", "replay_work_max", "replay_work_cv",
    "replay_work_max_mean_ratio", "replay_active_work_fraction",
]
MEMORY = [
    "replay_density_input_0", "replay_density_input_1",
    "replay_density_input_2", "replay_value_bytes_est",
    "replay_index_bytes_est", "replay_pointer_bytes_est",
    "replay_output_write_bytes_est", "replay_dense_workspace_bytes_est",
    "replay_index_value_byte_ratio", "replay_reuse_factor_est",
]
LOCALITY = [
    "replay_block_csc_index_gap_mean", "replay_block_csc_index_gap_cv",
    "replay_block_csc_consecutive_fraction",
    "replay_derived_csr_index_gap_mean", "replay_derived_csr_index_gap_cv",
    "replay_derived_csr_consecutive_fraction",
]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def unique_in_order(values):
    return list(dict.fromkeys(values))


def ablation_sets(target):
    allowed = FEATURES_BY_KERNEL[target]
    allowed_set = set(allowed)
    stages = [
        ("A_size_nnz", SIZE_NNZ),
        ("B_arithmetic", SIZE_NNZ + ARITHMETIC),
        ("C_irregularity", SIZE_NNZ + ARITHMETIC + IRREGULARITY),
        ("D_memory", SIZE_NNZ + ARITHMETIC + IRREGULARITY + MEMORY),
        ("E_locality", SIZE_NNZ + ARITHMETIC + IRREGULARITY + MEMORY + LOCALITY),
        ("F_filtered_full", allowed),
    ]
    return {
        name: unique_in_order([f for f in values if f in allowed_set])
        for name, values in stages
    }


def stable_matrix_fold(matrix_target_counts, folds, seed):
    """Shared deterministic folds balanced across all five target data sets."""
    matrices = sorted(matrix_target_counts)
    if len(matrices) < folds:
        raise RuntimeError(
            f"matrix-grouped {folds}-fold CV needs at least {folds} matrices; "
            f"found {len(matrices)}"
        )
    raw = np.asarray([matrix_target_counts[matrix] for matrix in matrices], float)
    totals = np.maximum(raw.sum(axis=0), 1.0)
    weight = raw / totals
    order = sorted(range(len(matrices)), key=lambda index: (
        -float(weight[index].sum()),
        hashlib.sha256(f"{seed}|{matrices[index]}".encode()).digest(),
    ))
    loads = np.zeros((folds, raw.shape[1]), float)
    matrix_counts = np.zeros(folds, int)
    assignment = {}
    for index in order:
        vector = weight[index]
        fold = min(range(folds), key=lambda candidate: (
            float(np.max(loads[candidate] + vector)),
            float(np.sum(loads[candidate] + vector)),
            int(matrix_counts[candidate]), candidate,
        ))
        assignment[matrices[index]] = fold + 1
        loads[fold] += vector
        matrix_counts[fold] += 1
    return assignment


def metrics(y, prediction):
    y = np.asarray(y, float)
    prediction = np.maximum(np.asarray(prediction, float), 0.0)
    ape = np.abs(prediction - y) / np.maximum(np.abs(y), 1e-6) * 100
    return {
        "n_test": int(len(y)),
        "MAPE": float(np.mean(ape)),
        "MdAPE": float(np.median(ape)),
        "P90_APE": float(np.percentile(ape, 90)),
        "P95_APE": float(np.percentile(ape, 95)),
        "RMSE_ms": float(np.sqrt(mean_squared_error(y, prediction))),
        "R2": float(r2_score(y, prediction)),
    }


def feature_matrix(data, features):
    return (data[features].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float))


def hgb(seed):
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=.07, max_iter=300,
        max_leaf_nodes=31, min_samples_leaf=30,
        l2_regularization=1.0, random_state=seed,
    )


def ebm(seed, features):
    return ExplainableBoostingRegressor(
        feature_names=features, max_bins=128, max_interaction_bins=32,
        interactions=10, outer_bags=4, inner_bags=0, learning_rate=.04,
        max_rounds=1800, early_stopping_rounds=100, min_samples_leaf=20,
        n_jobs=4, random_state=seed,
    )


def add_timing(records, target, model, phase, seconds, fold=None,
               stage="", n_train=0, n_test=0):
    records.append({
        "target": target, "model": model, "stage": stage,
        "fold": "" if fold is None else int(fold), "phase": phase,
        "seconds": float(seconds), "n_train": int(n_train),
        "n_test": int(n_test),
    })


def timed_fit_predict(model, x, y, train, test, timing, target, label,
                      fold, stage=""):
    started = time.perf_counter()
    model.fit(x[train], y[train])
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = model.predict(x[test])
    predict_seconds = time.perf_counter() - started
    add_timing(timing, target, label, "cv_fit", fit_seconds, fold, stage,
               len(train), len(test))
    add_timing(timing, target, label, "cv_predict", predict_seconds, fold,
               stage, len(train), len(test))
    return prediction


def fit_roofline(operations, byte_count, y, train):
    positive_operations = operations[train][operations[train] > 0]
    positive_bytes = byte_count[train][byte_count[train] > 0]
    scale = np.array([
        max(float(np.median(positive_operations)), 1.0),
        max(float(np.median(positive_bytes)), 1.0),
    ])
    design = np.column_stack([
        np.ones(len(y)), operations / scale[0], byte_count / scale[1]
    ])
    coefficient, _ = nnls(design[train], y[train])
    return scale, coefficient, np.maximum(design @ coefficient, 1e-9)


def write_timing_report(output, records, total_seconds, folds):
    frame = pd.DataFrame(records)
    frame.to_csv(output / "training_timing_details_v3.csv", index=False)
    phase = (frame.groupby("phase", as_index=False)["seconds"].sum()
             .sort_values("seconds", ascending=False))
    model = (frame.groupby(["target", "model"], as_index=False)["seconds"].sum()
             .sort_values(["target", "seconds"], ascending=[True, False]))
    report = {
        "schema": "online_safe_v3",
        "cv_policy": f"shared matrix-grouped {folds}-fold",
        "wall_clock_total_seconds": float(total_seconds),
        "recorded_stage_seconds": float(frame["seconds"].sum()),
        "phase_totals": phase.to_dict("records"),
        "target_model_totals": model.to_dict("records"),
        "details_csv": "training_timing_details_v3.csv",
    }
    (output / "training_timing_report_v3.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# V3 正式训练耗时报告", "",
        f"- 交叉验证：共享 matrix-grouped {folds}-fold",
        f"- Wall-clock 总时长：{total_seconds:.3f} s",
        f"- 已记录阶段累计时长：{frame['seconds'].sum():.3f} s",
        "", "## 按阶段汇总", "",
        "| 阶段 | 秒 | 总时长占比 |", "|---|---:|---:|",
    ]
    for row in phase.itertuples(index=False):
        lines.append(f"| {row.phase} | {row.seconds:.3f} | "
                     f"{100*row.seconds/max(total_seconds, 1e-12):.2f}% |")
    lines += ["", "## 按目标与模型汇总", "",
              "| 目标 | 模型 | 秒 |", "|---|---|---:|"]
    for row in model.itertuples(index=False):
        lines.append(f"| {row.target} | {row.model} | {row.seconds:.3f} |")
    lines += ["", "逐折 fit/predict、样本数及消融阶段见 `training_timing_details_v3.csv`。", ""]
    (output / "training_timing_report_v3.md").write_text("\n".join(lines))


def main():
    total_started = time.perf_counter()
    args = arguments()
    if args.folds != 5:
        raise ValueError("formal v3 training requires --folds 5")
    source = Path(args.input_dir)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"formal training output must be empty: {output}")
    split_dir, model_dir = output / "splits", output / "models"
    prediction_dir = output / "predictions"
    for directory in (output, split_dir, model_dir, prediction_dir):
        directory.mkdir(parents=True, exist_ok=True)

    timing, ablation_results, comparison_results, fold_results = [], [], [], []
    manifest = {
        "schema": "online_safe_v3", "seed": args.seed, "cv_folds": args.folds,
        "split_policy": "shared deterministic SHA-256 matrix-grouped 5-fold",
        "identity_fields": KEY, "ssssm_shared_split": True, "targets": {},
    }

    started = time.perf_counter()
    matrix_target_counts = defaultdict(lambda: np.zeros(len(TARGETS), float))
    for target_index, target in enumerate(TARGETS):
        header = pd.read_csv(source / f"{target}.model_ready_v3.csv",
                             usecols=["matrix_id"])
        for matrix_id, count in header["matrix_id"].astype(str).value_counts().items():
            matrix_target_counts[matrix_id][target_index] = count
    fold_map = stable_matrix_fold(matrix_target_counts, args.folds, args.seed)
    pd.DataFrame(sorted(fold_map.items()), columns=["matrix_id", "cv_fold"]).to_csv(
        split_dir / "shared_matrix_folds_v3.csv", index=False)
    add_timing(timing, "ALL", "ALL", "shared_fold_assignment",
               time.perf_counter() - started)

    shared_ssssm = None
    for target_index, target in enumerate(TARGETS):
        print(f"[{target}] load and validate", flush=True)
        path = source / f"{target}.model_ready_v3.csv"
        stages = ablation_sets(target)
        required = set(KEY + ["label_time_ms", "replay_feature_schema_version"])
        required.update(stages["F_filtered_full"])
        started = time.perf_counter()
        data = pd.read_csv(path, usecols=lambda c: c in required, low_memory=False)
        data["label_time_ms"] = pd.to_numeric(data["label_time_ms"], errors="coerce")
        data = data[np.isfinite(data["label_time_ms"]) &
                    (data["label_time_ms"] > 0)].reset_index(drop=True)
        if not data["replay_feature_schema_version"].eq("v3").all():
            raise RuntimeError(f"{target}: non-v3 rows")
        data["cv_fold"] = data["matrix_id"].astype(str).map(fold_map)
        if data["cv_fold"].isna().any() or data["cv_fold"].nunique() != args.folds:
            raise RuntimeError(f"{target}: incomplete shared 5-fold assignment")
        data[KEY + ["cv_fold"]].to_csv(split_dir / f"{target}.folds.csv", index=False)
        add_timing(timing, target, "ALL", "load_validate_split",
                   time.perf_counter() - started)

        if target.startswith("SSSSM_"):
            current = data[KEY + ["cv_fold"]].sort_values(KEY).reset_index(drop=True)
            if shared_ssssm is None:
                shared_ssssm = current
            elif not current.equals(shared_ssssm):
                raise RuntimeError("SSSSM dense/sparse rows or folds differ")

        y = data["label_time_ms"].to_numpy(float)
        fold_id = data["cv_fold"].to_numpy(int)
        splits = [(np.flatnonzero(fold_id != fold), np.flatnonzero(fold_id == fold))
                  for fold in range(1, args.folds + 1)]
        full_features = stages["F_filtered_full"]
        full_x = None
        hgb_oof = None

        for stage_index, (stage, features) in enumerate(stages.items()):
            print(f"[{target}] 5-fold ablation {stage} ({len(features)} features)",
                  flush=True)
            started = time.perf_counter()
            x = feature_matrix(data, features)
            add_timing(timing, target, "HGB", "feature_matrix",
                       time.perf_counter() - started, stage=stage)
            oof = np.empty(len(y), float)
            for fold, (train, test) in enumerate(splits, 1):
                model = hgb(args.seed + target_index * 100 + stage_index * 10 + fold)
                raw = timed_fit_predict(model, x, np.log1p(y), train, test,
                                        timing, target, "HGB", fold, stage)
                oof[test] = np.maximum(np.expm1(raw), 0.0)
                fold_results.append({"target": target, "stage": stage,
                                     "model": "HGB", "fold": fold,
                                     **metrics(y[test], oof[test])})
            ablation_results.append({
                "target": target, "stage": stage, "n_features": len(features),
                "n_samples": len(y), "cv_folds": args.folds, **metrics(y, oof),
            })
            if stage == "F_filtered_full":
                full_x, hgb_oof = x, oof
                started = time.perf_counter()
                final_hgb = hgb(args.seed + target_index * 100)
                final_hgb.fit(x, np.log1p(y))
                add_timing(timing, target, "HGB", "final_fit",
                           time.perf_counter() - started, stage=stage,
                           n_train=len(y))
                started = time.perf_counter()
                joblib.dump({"schema": "online_safe_v3", "target": target,
                             "features": features, "target_transform": "log1p",
                             "cv_policy": "shared matrix-grouped 5-fold",
                             "model": final_hgb},
                            model_dir / f"{target}.hgb_v3.joblib")
                add_timing(timing, target, "HGB", "artifact_write",
                           time.perf_counter() - started, stage=stage)
        comparison_results.append({"target": target, "model": "HGB",
                                   "cv_folds": args.folds, **metrics(y, hgb_oof)})

        print(f"[{target}] EBM 5-fold", flush=True)
        ebm_oof = np.empty(len(y), float)
        for fold, (train, test) in enumerate(splits, 1):
            model = ebm(args.seed + target_index * 10 + fold, full_features)
            raw = timed_fit_predict(model, full_x, np.log1p(y), train, test,
                                    timing, target, "EBM", fold)
            ebm_oof[test] = np.maximum(np.expm1(raw), 0.0)
            fold_results.append({"target": target, "stage": "F_filtered_full",
                                 "model": "EBM", "fold": fold,
                                 **metrics(y[test], ebm_oof[test])})
        comparison_results.append({"target": target, "model": "EBM",
                                   "cv_folds": args.folds, **metrics(y, ebm_oof)})
        started = time.perf_counter()
        final_ebm = ebm(args.seed + target_index, full_features)
        final_ebm.fit(full_x, np.log1p(y))
        add_timing(timing, target, "EBM", "final_fit",
                   time.perf_counter() - started, n_train=len(y))
        started = time.perf_counter()
        joblib.dump({"schema": "online_safe_v3", "target": target,
                     "features": full_features, "target_transform": "log1p",
                     "cv_policy": "shared matrix-grouped 5-fold", "model": final_ebm},
                    model_dir / f"{target}.ebm_v3.joblib")
        add_timing(timing, target, "EBM", "artifact_write",
                   time.perf_counter() - started)

        operations = pd.to_numeric(data["replay_operation_count_v2"],
                                   errors="coerce").fillna(0).to_numpy(float)
        byte_count = pd.to_numeric(data["replay_base_bytes_est"],
                                   errors="coerce").fillna(0).to_numpy(float)
        roof_oof, residual_oof = np.empty(len(y)), np.empty(len(y))
        outer_roof = []
        for fold, (train, test) in enumerate(splits, 1):
            started = time.perf_counter()
            scale, coefficient, baseline = fit_roofline(
                operations, byte_count, y, train)
            add_timing(timing, target, "Analytical_Roofline", "cv_fit",
                       time.perf_counter() - started, fold, n_train=len(train),
                       n_test=len(test))
            started = time.perf_counter()
            roof_oof[test] = baseline[test]
            add_timing(timing, target, "Analytical_Roofline", "cv_predict",
                       time.perf_counter() - started, fold, n_train=len(train),
                       n_test=len(test))
            residual_train = np.log1p(y[train]) - np.log1p(baseline[train])
            model = hgb(args.seed + 1000 + target_index * 10 + fold)
            started = time.perf_counter()
            model.fit(full_x[train], residual_train)
            add_timing(timing, target, "Roofline_residual_HGB", "cv_fit",
                       time.perf_counter() - started, fold,
                       n_train=len(train), n_test=len(test))
            started = time.perf_counter()
            raw = model.predict(full_x[test])
            add_timing(timing, target, "Roofline_residual_HGB", "cv_predict",
                       time.perf_counter() - started, fold,
                       n_train=len(train), n_test=len(test))
            residual_oof[test] = np.maximum(np.expm1(
                np.log1p(baseline[test]) + raw), 0.0)
            outer_roof.append({"fold": fold, "scale": scale.tolist(),
                               "coefficient": coefficient.tolist()})
            fold_results.append({"target": target, "stage": "F_filtered_full",
                                 "model": "Analytical_Roofline", "fold": fold,
                                 **metrics(y[test], roof_oof[test])})
            fold_results.append({"target": target, "stage": "F_filtered_full",
                                 "model": "Roofline_residual_HGB", "fold": fold,
                                 **metrics(y[test], residual_oof[test])})
        comparison_results.append({"target": target, "model": "Analytical_Roofline",
                                   "cv_folds": args.folds, **metrics(y, roof_oof)})
        comparison_results.append({"target": target, "model": "Roofline_residual_HGB",
                                   "cv_folds": args.folds, **metrics(y, residual_oof)})

        all_indices = np.arange(len(y))
        started = time.perf_counter()
        scale, coefficient, baseline = fit_roofline(
            operations, byte_count, y, all_indices)
        add_timing(timing, target, "Analytical_Roofline", "final_fit",
                   time.perf_counter() - started, n_train=len(y))
        residual = np.log1p(y) - np.log1p(baseline)
        started = time.perf_counter()
        final_residual = hgb(args.seed + 1000 + target_index)
        final_residual.fit(full_x, residual)
        add_timing(timing, target, "Roofline_residual_HGB", "final_fit",
                   time.perf_counter() - started, n_train=len(y))
        started = time.perf_counter()
        joblib.dump({
            "schema": "online_safe_v3", "target": target,
            "features": full_features, "residual_model": final_residual,
            "baseline_features": ["intercept", "operation_count_v2",
                                  "base_bytes_est"],
            "baseline_scale": scale, "baseline_coef": coefficient,
            "outer_fold_roofline": outer_roof,
            "cv_policy": "shared matrix-grouped 5-fold",
        }, model_dir / f"{target}.roofline_residual_v3.joblib")
        add_timing(timing, target, "Roofline_residual_HGB", "artifact_write",
                   time.perf_counter() - started)

        started = time.perf_counter()
        predictions = data[KEY + ["label_time_ms", "cv_fold"]].copy()
        predictions = predictions.rename(columns={"label_time_ms": "actual_ms"})
        predictions["HGB_ms"] = hgb_oof
        predictions["EBM_ms"] = ebm_oof
        predictions["Analytical_Roofline_ms"] = roof_oof
        predictions["Roofline_residual_HGB_ms"] = residual_oof
        predictions.to_csv(prediction_dir / f"{target}.predictions.csv", index=False)
        add_timing(timing, target, "ALL", "prediction_write",
                   time.perf_counter() - started)
        manifest["targets"][target] = {
            "source": str(path.resolve()), "rows": int(len(data)),
            "matrices": int(data["matrix_id"].nunique()),
            "fold_counts": {str(k): int(v) for k, v in
                            data["cv_fold"].value_counts().sort_index().items()},
            "features": full_features,
        }
        print(json.dumps(comparison_results[-4:], indent=2), flush=True)

    started = time.perf_counter()
    pd.DataFrame(ablation_results).to_csv(output / "ablation_metrics_v3.csv",
                                          index=False)
    pd.DataFrame(comparison_results).to_csv(output / "model_comparison_v3.csv",
                                            index=False)
    pd.DataFrame(fold_results).to_csv(output / "cv_fold_metrics_v3.csv", index=False)
    (output / "experiment_manifest_v3.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (output / "ablation_feature_sets_v3.json").write_text(
        json.dumps({target: ablation_sets(target) for target in TARGETS},
                   indent=2, ensure_ascii=False) + "\n")
    add_timing(timing, "ALL", "ALL", "summary_write",
               time.perf_counter() - started)
    write_timing_report(output, timing, time.perf_counter() - total_started,
                        args.folds)
    print(f"wrote grouped 5-fold results and timing report to {output}", flush=True)


if __name__ == "__main__":
    main()
