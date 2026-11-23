#!/usr/bin/env python3
"""Aggregate duplicate GETRF measurements and train matrix-grouped experts.

All routing features are available before the target GETRF kernel launches.
The segment router is learned only from training matrices; test runtimes are
used solely for evaluation.
"""
import argparse
import json
import math
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from feature_sets_v3 import GETRF_FEATURES


BASE = [
    "replay_main_dimension", "replay_block_rows", "replay_block_cols",
    "replay_aspect_ratio", "replay_min_dimension", "replay_max_dimension",
    "replay_nnz_input_0", "replay_nnz_input_1", "replay_nnz_input_2",
    "replay_nnz_output_0", "replay_nnz_output_1",
]
ARITH = ["replay_num_macs", "replay_num_divisions", "replay_operation_count_v2"]
IRREG = [
    "replay_row_nnz_mean", "replay_row_nnz_max", "replay_row_nnz_cv",
    "replay_row_nnz_max_mean_ratio", "replay_row_empty_fraction",
    "replay_col_nnz_mean", "replay_col_nnz_max", "replay_col_nnz_cv",
    "replay_col_nnz_max_mean_ratio", "replay_col_empty_fraction",
    "replay_work_mean", "replay_work_max", "replay_work_cv",
    "replay_work_max_mean_ratio", "replay_active_work_fraction",
]
MEMORY = [
    "replay_density_input_0", "replay_density_input_1", "replay_density_input_2",
    "replay_value_bytes_est", "replay_index_bytes_est", "replay_pointer_bytes_est",
    "replay_output_write_bytes_est", "replay_dense_workspace_bytes_est",
    "replay_base_bytes_est", "replay_index_value_byte_ratio", "replay_reuse_factor_est",
]
LOCALITY = [
    "replay_block_csc_index_gap_mean", "replay_block_csc_index_gap_cv",
    "replay_block_csc_consecutive_fraction",
    "replay_derived_csr_index_gap_mean", "replay_derived_csr_index_gap_cv",
    "replay_derived_csr_consecutive_fraction",
]
PARALLEL = ["replay_active_rows", "replay_active_cols"]
LAUNCH = [
    "getrf_launch_policy_code", "getrf_grid_blocks", "getrf_block_threads",
    "getrf_warps_per_block", "getrf_rows_per_block",
    "getrf_grid_blocks_per_sm", "getrf_min_waves",
    "getrf_thread_slots", "getrf_nnz_per_thread_slot",
]
FEATURES = GETRF_FEATURES + LAUNCH
SEGMENT_NAMES = np.array(["tiny", "regular", "large"])


def metrics(y, pred):
    pred = np.maximum(np.asarray(pred, dtype=float), 0.0)
    y = np.asarray(y, dtype=float)
    ape = np.abs(pred - y) / np.maximum(np.abs(y), 1e-6) * 100.0
    return {
        "n_test": int(len(y)),
        "MAPE": float(np.mean(ape)),
        "MdAPE": float(np.median(ape)),
        "P90_APE": float(np.percentile(ape, 90)),
        "P95_APE": float(np.percentile(ape, 95)),
        "RMSE_ms": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)) if len(y) > 1 else None,
    }


def segment_id(y):
    y = np.asarray(y, dtype=float)
    return np.where(y < 1.0, 0, np.where(y < 100.0, 1, 2)).astype(np.int8)


def add_launch_features(df, sm_count):
    n = pd.to_numeric(df["replay_block_rows"], errors="coerce").fillna(0).to_numpy(np.int64)
    nnz = pd.to_numeric(df["replay_nnz_input_0"], errors="coerce").fillna(0).to_numpy(np.int64)
    avg = np.floor_divide(nnz, np.maximum(n, 1))
    policy = np.select(
        [avg <= 32, avg <= 96, avg <= 192, avg <= 384],
        [0, 1, 2, 3], default=4,
    ).astype(np.int8)
    threads = np.array([64, 64, 128, 256, 512], dtype=np.int64)[policy]
    rows_per_block = np.where(policy == 0, 2, 1)
    grid = np.ceil(n / rows_per_block).astype(np.int64)
    slots = np.maximum(grid * threads, 1)
    df["getrf_nnz_avg_floor"] = avg
    df["getrf_launch_policy_code"] = policy
    df["getrf_launch_policy"] = np.array(
        ["WarpLevel_sflu", "BlockLevel_sflu_L1", "BlockLevel_sflu_L2",
         "BlockLevel_sflu_L3", "BlockLevel_sflu_L4"]
    )[policy]
    df["getrf_grid_blocks"] = grid
    df["getrf_block_threads"] = threads
    df["getrf_warps_per_block"] = threads // 32
    df["getrf_rows_per_block"] = rows_per_block
    df["getrf_sm_count"] = int(sm_count)
    df["getrf_grid_blocks_per_sm"] = grid / float(sm_count)
    df["getrf_min_waves"] = np.ceil(grid / float(sm_count)).astype(np.int64)
    df["getrf_thread_slots"] = slots
    df["getrf_nnz_per_thread_slot"] = nnz / slots
    return df


def aggregate_getrf(input_csv, sm_count):
    identity = [
        "task_id", "matrix_id", "kernel_type", "level", "block_row", "block_col",
        "update_k", "rank", "block_size", "process_rows", "process_cols",
    ]
    const_cols = list(dict.fromkeys(
        identity + BASE + ARITH + IRREG + MEMORY + LOCALITY + PARALLEL
        + ["structure_hash", "replay_structure_hash",
           "replay_canonical_structure_hash_v2", "replay_feature_schema_version"]
    ))
    read_cols = list(dict.fromkeys(
        const_cols + ["forced_ssssm_variant", "repeat_index", "gpu_run_id", "time_kernel_ms"]
    ))
    df = pd.read_csv(input_csv, usecols=lambda c: c in set(read_cols), low_memory=False)
    df = df[df["kernel_type"].eq("GETRF")].copy()
    df["time_kernel_ms"] = pd.to_numeric(df["time_kernel_ms"], errors="coerce")
    df = df[np.isfinite(df["time_kernel_ms"]) & (df["time_kernel_ms"] > 0)].copy()
    if df.empty:
        raise RuntimeError("no valid GETRF rows")

    # The same logical GETRF task appears in dense and sparse SSSSM campaigns.
    group_key = identity
    nunique = df.groupby(group_key, dropna=False)[const_cols].nunique(dropna=False)
    varying = nunique.gt(1)
    bad = varying.any(axis=1)
    if bad.any():
        bad_cols = sorted(varying.columns[varying.any()].tolist())
        raise RuntimeError(
            f"{int(bad.sum())} duplicate groups disagree in pre-kernel features: {bad_cols}"
        )

    first = df.groupby(group_key, dropna=False, sort=False)[const_cols].first()
    times = df.groupby(group_key, dropna=False, sort=False)["time_kernel_ms"].agg(
        measurement_count="size", label_time_ms="median", time_mean_ms="mean",
        time_std_ms=lambda x: float(np.std(x.to_numpy(float), ddof=0)),
        time_min_ms="min", time_max_ms="max",
    )
    variants = (
        df.pivot_table(index=group_key, columns="forced_ssssm_variant",
                       values="time_kernel_ms", aggfunc="median")
        .rename(columns={"dense": "dense_time_ms", "sparse": "sparse_time_ms"})
    )
    out = first.join(times).join(variants).reset_index(drop=True)
    out["time_cv"] = out["time_std_ms"] / out["time_mean_ms"].clip(lower=1e-12)
    out["time_relative_range"] = (
        (out["time_max_ms"] - out["time_min_ms"]) / out["label_time_ms"].clip(lower=1e-12)
    )
    out["dense_sparse_abs_diff_ms"] = (
        out.get("dense_time_ms", np.nan) - out.get("sparse_time_ms", np.nan)
    ).abs()
    out["dense_sparse_ape"] = (
        out["dense_sparse_abs_diff_ms"] / out["label_time_ms"].clip(lower=1e-12) * 100.0
    )
    return add_launch_features(out, sm_count), len(df)


def stability_report(df, raw_rows):
    cv = df["time_cv"].to_numpy(float)
    rr = df["time_relative_range"].to_numpy(float)
    counts = df["measurement_count"].value_counts().sort_index()
    report = {
        "raw_getrf_rows": int(raw_rows),
        "aggregated_logical_tasks": int(len(df)),
        "removed_duplicate_labels": int(raw_rows - len(df)),
        "measurement_count_distribution": {str(int(k)): int(v) for k, v in counts.items()},
        "time_cv_quantiles": {
            str(q): float(np.quantile(cv, q)) for q in [0, .5, .75, .9, .95, .99, 1]
        },
        "relative_range_quantiles": {
            str(q): float(np.quantile(rr, q)) for q in [0, .5, .75, .9, .95, .99, 1]
        },
        "cv_threshold_counts": {
            str(t): int(np.sum(cv > t)) for t in [.01, .05, .10, .20, .50]
        },
        "label_aggregation": "median of duplicate dense/sparse campaign measurements",
    }
    return report


def fit_regressor(X, y, seed):
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=.07, max_iter=300,
        max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(X, np.log1p(y))
    return model


def predict_regressor(model, X):
    return np.maximum(np.expm1(model.predict(X)), 0.0)


def plot_predictions(pred_df, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"tiny": "#4C78A8", "regular": "#F58518", "large": "#54A24B"}
    for name in SEGMENT_NAMES:
        q = pred_df[pred_df["actual_segment"].eq(name)]
        axes[0].scatter(q["actual_ms"], q["segmented_soft_ms"], s=9, alpha=.45,
                        color=colors[name], label=f"{name} (n={len(q)})")
    lo = max(pred_df["actual_ms"].min(), 1e-4)
    hi = max(pred_df["actual_ms"].max(), pred_df["segmented_soft_ms"].max())
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0].set(xscale="log", yscale="log", xlabel="Measured kernel time (ms)",
                ylabel="Predicted kernel time (ms)", title="Test-set prediction")
    axes[0].legend(fontsize=8)
    pred_df.boxplot(column="soft_ape", by="actual_segment", ax=axes[1], grid=False)
    axes[1].set_yscale("log")
    axes[1].set(xlabel="Actual runtime segment", ylabel="Absolute percentage error",
                title="Error by actual segment")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument(
        "--split-mode", choices=["matrix-holdout", "within-matrix"],
        default="matrix-holdout",
        help="matrix-holdout uses --splits; within-matrix samples 80/20 inside every matrix",
    )
    ap.add_argument("--train-fraction", type=float, default=0.8)
    ap.add_argument("--sm-count", type=int, default=132)
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    df, raw_rows = aggregate_getrf(args.input, args.sm_count)
    df.to_csv(outdir / "getrf_aggregated.csv", index=False)
    stability = stability_report(df, raw_rows)
    (outdir / "stability_report.json").write_text(json.dumps(stability, indent=2) + "\n")
    df[[
        "task_id", "matrix_id", "block_size", "measurement_count",
        "dense_time_ms", "sparse_time_ms", "label_time_ms", "time_cv",
        "time_relative_range", "dense_sparse_ape",
    ]].sort_values("time_cv", ascending=False).to_csv(
        outdir / "repeat_stability.csv", index=False
    )

    if args.split_mode == "matrix-holdout":
        split = json.loads(Path(args.splits).read_text())["GETRF"]
        train_set, test_set = set(split["train_matrices"]), set(split["test_matrices"])
        unknown = set(df["matrix_id"].astype(str)) - train_set - test_set
        if unknown:
            raise RuntimeError(f"matrices absent from supplied split: {sorted(unknown)}")
        train_mask = df["matrix_id"].astype(str).isin(train_set).to_numpy()
        test_mask = df["matrix_id"].astype(str).isin(test_set).to_numpy()
    else:
        if not 0.0 < args.train_fraction < 1.0:
            raise ValueError("--train-fraction must be between zero and one")
        rng = np.random.default_rng(args.seed)
        train_mask = np.zeros(len(df), dtype=bool)
        for _, indices in df.groupby("matrix_id", sort=True).indices.items():
            indices = np.asarray(indices, dtype=int)
            shuffled = rng.permutation(indices)
            n_train = int(np.floor(len(indices) * args.train_fraction))
            n_train = min(max(n_train, 1), len(indices) - 1)
            train_mask[shuffled[:n_train]] = True
        test_mask = ~train_mask
        split_rows = []
        for matrix, indices in df.groupby("matrix_id", sort=True).indices.items():
            indices = np.asarray(indices, dtype=int)
            split_rows.append({
                "matrix_id": matrix, "n_total": int(len(indices)),
                "n_train": int(train_mask[indices].sum()),
                "n_test": int(test_mask[indices].sum()),
            })
        pd.DataFrame(split_rows).to_csv(outdir / "within_matrix_split.csv", index=False)

    X = (
        df[FEATURES].apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float64)
    )
    y = df["label_time_ms"].to_numpy(np.float64)
    yseg = segment_id(y)
    Xtr, Xte, ytr, yte = X[train_mask], X[test_mask], y[train_mask], y[test_mask]
    strn, ste = yseg[train_mask], yseg[test_mask]

    classifier = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=.07, max_iter=300, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, random_state=args.seed,
    )
    classifier.fit(Xtr, strn)
    no_launch_count = len(FEATURES) - len(LAUNCH)
    no_launch_single = fit_regressor(Xtr[:, :no_launch_count], ytr, args.seed)
    single = fit_regressor(Xtr, ytr, args.seed)
    experts = {}
    expert_pred = np.empty((len(Xte), 3), dtype=float)
    for s, name in enumerate(SEGMENT_NAMES):
        mask = strn == s
        if mask.sum() == 0:
            raise RuntimeError(f"training split has no {name} samples")
        experts[name] = fit_regressor(Xtr[mask], ytr[mask], args.seed + s + 1)
        expert_pred[:, s] = predict_regressor(experts[name], Xte)

    pred_no_launch = predict_regressor(no_launch_single, Xte[:, :no_launch_count])
    pred_single = predict_regressor(single, Xte)
    probabilities = classifier.predict_proba(Xte)
    # sklearn may omit a class; map probabilities back to the fixed class order.
    full_prob = np.zeros((len(Xte), 3), dtype=float)
    full_prob[:, classifier.classes_.astype(int)] = probabilities
    predicted_segment = np.argmax(full_prob, axis=1)
    pred_hard = expert_pred[np.arange(len(Xte)), predicted_segment]
    # Bound by the interval selected by the pre-kernel router, not the test label.
    pred_hard_bounded = pred_hard.copy()
    pred_hard_bounded[predicted_segment == 0] = np.clip(
        pred_hard_bounded[predicted_segment == 0], 0.0, 1.0
    )
    pred_hard_bounded[predicted_segment == 1] = np.clip(
        pred_hard_bounded[predicted_segment == 1], 1.0, 100.0
    )
    pred_hard_bounded[predicted_segment == 2] = np.maximum(
        pred_hard_bounded[predicted_segment == 2], 100.0
    )
    pred_soft = np.sum(full_prob * expert_pred, axis=1)
    # Diagnostic ceiling only; true-segment routing is not deployable.
    pred_oracle = expert_pred[np.arange(len(Xte)), ste]

    rows = []
    for name, pred in [
        ("single_hgb_no_launch", pred_no_launch),
        ("single_hgb_augmented", pred_single),
        ("segmented_hard_router", pred_hard),
        ("segmented_hard_bounded", pred_hard_bounded),
        ("segmented_soft_router", pred_soft),
        ("segmented_oracle_diagnostic", pred_oracle),
    ]:
        rows.append({"scope": "all", "model": name, **metrics(yte, pred)})
        for s, seg_name in enumerate(SEGMENT_NAMES):
            m = ste == s
            if m.any():
                rows.append({
                    "scope": f"actual_{seg_name}", "model": name,
                    **metrics(yte[m], pred[m]),
                })
    pd.DataFrame(rows).to_csv(outdir / "matrix_grouped_metrics.csv", index=False)

    cm = confusion_matrix(ste, predicted_segment, labels=[0, 1, 2])
    pd.DataFrame(cm, index=SEGMENT_NAMES, columns=SEGMENT_NAMES).to_csv(
        outdir / "segment_confusion_matrix.csv"
    )
    classifier_report = {
        "accuracy": float(np.mean(predicted_segment == ste)),
        "train_segment_counts": {
            name: int(np.sum(strn == s)) for s, name in enumerate(SEGMENT_NAMES)
        },
        "test_segment_counts": {
            name: int(np.sum(ste == s)) for s, name in enumerate(SEGMENT_NAMES)
        },
        "thresholds_ms": {"tiny": "<1", "regular": "[1,100)", "large": ">=100"},
        "routing_rule": "pre-kernel classifier trained only on training matrices",
    }
    (outdir / "segment_router_report.json").write_text(
        json.dumps(classifier_report, indent=2) + "\n"
    )

    test = df.loc[test_mask, [
        "task_id", "matrix_id", "block_size", "measurement_count",
        "time_cv", "getrf_launch_policy", "getrf_grid_blocks",
        "getrf_block_threads", "getrf_min_waves",
    ]].reset_index(drop=True)
    test["actual_ms"] = yte
    test["actual_segment"] = SEGMENT_NAMES[ste]
    test["predicted_segment"] = SEGMENT_NAMES[predicted_segment]
    for s, name in enumerate(SEGMENT_NAMES):
        test[f"p_{name}"] = full_prob[:, s]
    test["single_hgb_no_launch_ms"] = pred_no_launch
    test["single_hgb_ms"] = pred_single
    test["segmented_hard_ms"] = pred_hard
    test["segmented_hard_bounded_ms"] = pred_hard_bounded
    test["segmented_soft_ms"] = pred_soft
    test["soft_ape"] = np.abs(pred_soft - yte) / np.maximum(yte, 1e-6) * 100
    test.to_csv(outdir / "predictions.csv", index=False)
    plot_predictions(test, outdir / "prediction_diagnostics.png")

    bundle = {
        "schema": "getrf_online_safe_v2_segmented",
        "features": FEATURES,
        "sm_count": args.sm_count,
        "segment_thresholds_ms": [1.0, 100.0],
        "classifier": classifier,
        "single_regressor_no_launch": no_launch_single,
        "single_regressor": single,
        "experts": experts,
        "target_transform": "log1p",
        "launch_policy_source": "src/pangulu_cuda.cu:pangulu_getrf_cuda_kernel",
    }
    joblib.dump(bundle, outdir / "getrf_segmented_v2.joblib")
    metadata = {
        "input": str(Path(args.input).resolve()),
        "split_file": (
            str(Path(args.splits).resolve())
            if args.split_mode == "matrix-holdout" else None
        ),
        "split_mode": args.split_mode,
        "train_fraction": args.train_fraction if args.split_mode == "within-matrix" else None,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "features": FEATURES,
        "sm_count": args.sm_count,
        "wave_definition": "ceil(grid_blocks / sm_count), a lower-bound scheduling wave count",
        "label_definition": "median runtime after dense/sparse campaign duplicate aggregation",
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "output": str(outdir), "raw_rows": raw_rows, "logical_tasks": len(df),
        "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
