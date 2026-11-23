#!/usr/bin/env python3
"""Compare plain and per-matrix-balanced direct log-time XGBoost on profile29."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


CONTRACT = Path("/output/Partition/script/train_export_direct_log_xgboost_abcd.py")
MODES = ("plain", "matrix_balanced")


def load_contract():
    spec = importlib.util.spec_from_file_location("profile29_abcd_contract", CONTRACT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_target(root: Path, target: str, contract, hardware_id: str):
    required = {
        "task_id", "matrix_id", "block_size", "label_time_ms", "hardware_id",
        *contract.FEATURES[target],
    }
    for mean, cv in contract.DERIVED[target].values():
        required.update((mean, cv))
    data = pd.read_csv(
        root / f"{target}.model_ready_v4.csv",
        usecols=lambda column: column in required,
        low_memory=False,
    )
    for new, (mean, cv) in contract.DERIVED[target].items():
        data[new] = (
            pd.to_numeric(data[mean], errors="coerce").fillna(0)
            * pd.to_numeric(data[cv], errors="coerce").fillna(0)
        )
    labels = pd.to_numeric(data["label_time_ms"], errors="coerce").to_numpy(float)
    valid = np.isfinite(labels) & (labels > 0)
    data = data.loc[valid].reset_index(drop=True)
    labels = labels[valid]
    if set(data["hardware_id"].astype(str)) != {hardware_id}:
        raise RuntimeError(f"{target}: hardware mismatch")
    return data, labels, contract.values(data, contract.FEATURES[target])


def sample_weights(data: pd.DataFrame, mode: str) -> np.ndarray | None:
    if mode == "plain":
        return None
    matrix = data["matrix_id"].astype(str)
    counts = matrix.value_counts()
    weights = matrix.map(lambda item: 1.0 / counts[item]).to_numpy(float)
    return weights / weights.mean()


def metric_with_tail(contract, labels, predictions, tail_mask):
    result = contract.metric(labels, predictions)
    squared = np.square(np.asarray(labels) - np.asarray(predictions))
    count = max(1, int(np.ceil(len(squared) * 0.01)))
    result["top1pct_sse_fraction"] = float(
        np.sort(squared)[-count:].sum() / max(squared.sum(), 1e-30)
    )
    if np.any(tail_mask):
        tail = contract.metric(np.asarray(labels)[tail_mask], np.asarray(predictions)[tail_mask])
        result.update({f"tail_{key}": value for key, value in tail.items()})
    return result


def pair_metrics(frame: pd.DataFrame):
    actual = frame["actual_ms"].to_numpy(float)
    predicted = frame["predicted_ms"].to_numpy(float)
    correct = 0
    total = 0
    weighted_correct = 0.0
    weighted_total = 0.0
    for left in range(len(frame)):
        for right in range(left + 1, len(frame)):
            delta_actual = actual[left] - actual[right]
            if delta_actual == 0:
                continue
            delta_predicted = predicted[left] - predicted[right]
            weight = abs(delta_actual)
            hit = np.sign(delta_actual) == np.sign(delta_predicted)
            correct += int(hit)
            total += 1
            weighted_correct += weight * int(hit)
            weighted_total += weight
    return (
        correct / max(total, 1),
        weighted_correct / max(weighted_total, 1e-30),
    )


def scheme_metrics(tasks: pd.DataFrame, mode: str, contract):
    schemes = tasks.groupby(
        ["matrix_id", "block_size"], as_index=False
    ).agg(
        actual_ms=("actual_ms", "sum"),
        predicted_ms=("predicted_ms", "sum"),
        tasks=("task_id", "size"),
    )
    selections = []
    for matrix_id, group in schemes.groupby("matrix_id", sort=True):
        oracle = group.loc[group["actual_ms"].idxmin()]
        selected = group.loc[group["predicted_ms"].idxmin()]
        pairwise, weighted_pairwise = pair_metrics(group)
        selections.append({
            "mode": mode,
            "matrix_id": matrix_id,
            "oracle_block_size": int(oracle["block_size"]),
            "selected_block_size": int(selected["block_size"]),
            "oracle_actual_ms": float(oracle["actual_ms"]),
            "selected_actual_ms": float(selected["actual_ms"]),
            "regret_percent": float((selected["actual_ms"] / oracle["actual_ms"] - 1) * 100),
            "spearman": float(group["actual_ms"].rank().corr(group["predicted_ms"].rank())),
            "pairwise_accuracy": pairwise,
            "weighted_pairwise_accuracy": weighted_pairwise,
        })
    selection = pd.DataFrame(selections)
    summary = {
        "mode": mode,
        **contract.metric(schemes["actual_ms"], schemes["predicted_ms"]),
        "matrices": int(len(selection)),
        "oracle_accuracy": float(
            (selection["oracle_block_size"] == selection["selected_block_size"]).mean()
        ),
        "mean_pairwise_accuracy": float(selection["pairwise_accuracy"].mean()),
        "mean_weighted_pairwise_accuracy": float(
            selection["weighted_pairwise_accuracy"].mean()
        ),
        "mean_spearman": float(selection["spearman"].mean()),
        "mean_regret_percent": float(selection["regret_percent"].mean()),
        "median_regret_percent": float(selection["regret_percent"].median()),
        "p95_regret_percent": float(selection["regret_percent"].quantile(0.95)),
    }
    schemes.insert(0, "mode", mode)
    return schemes, selection, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--fold-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hardware-id", default="H100_SXM_80GB")
    parser.add_argument(
        "--expected-fold-counts", default="6,6,6,6,5",
        help="Comma-separated matrix counts for folds 1..5.",
    )
    args = parser.parse_args()
    source = Path(args.input_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"output must be empty: {output}")
    (output / "deploy_models").mkdir()
    contract = load_contract()
    fold_frame = pd.read_csv(args.fold_map)
    fold_map = dict(zip(
        fold_frame["matrix_id"].astype(str),
        fold_frame["cv_fold"].astype(int),
    ))
    expected_values = [int(value) for value in args.expected_fold_counts.split(",")]
    if len(expected_values) != 5 or any(value <= 0 for value in expected_values):
        raise RuntimeError("--expected-fold-counts must contain five positive integers")
    expected_counts = {fold: expected_values[fold - 1] for fold in range(1, 6)}
    actual_counts = fold_frame.groupby("cv_fold").size().sort_index().to_dict()
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"expected frozen fold sizes {expected_counts}, got {actual_counts}"
        )

    started = time.time()
    task_rows = []
    matrix_rows = []
    prediction_frames = []
    deploy_manifests = {}
    deploy_models = {mode: {} for mode in MODES}
    loaded = {
        target: load_target(source, target, contract, args.hardware_id)
        for target in contract.TARGETS
    }

    for mode_index, mode in enumerate(MODES):
        for target_index, target in enumerate(contract.TARGETS):
            data, labels, features = loaded[target]
            folds = data["matrix_id"].astype(str).map(fold_map)
            if folds.isna().any() or set(folds.astype(int)) != {1, 2, 3, 4, 5}:
                raise RuntimeError(f"{target}: invalid fold coverage")
            folds = folds.to_numpy(int)
            weights = sample_weights(data, mode)
            predictions = np.empty(len(labels), dtype=float)
            tail_mask = np.zeros(len(labels), dtype=bool)
            for fold in range(1, 6):
                train = folds != fold
                test = folds == fold
                threshold = float(np.quantile(labels[train], 0.95))
                tail_mask[test] = labels[test] >= threshold
                model = contract.XGBRegressor(
                    **contract.PARAMS,
                    random_state=contract.SEED + mode_index * 10000 + target_index * 1000 + fold,
                )
                fit_kwargs = {"verbose": False}
                if weights is not None:
                    fit_kwargs["sample_weight"] = weights[train]
                model.fit(features[train], np.log1p(labels[train]), **fit_kwargs)
                predictions[test] = np.maximum(np.expm1(model.predict(features[test])), 0)

            task_rows.append({
                "mode": mode,
                "target": target,
                **metric_with_tail(contract, labels, predictions, tail_mask),
            })
            frame = pd.DataFrame({
                "mode": mode,
                "target": target,
                "task_id": data["task_id"].astype(str),
                "matrix_id": data["matrix_id"].astype(str),
                "block_size": pd.to_numeric(data["block_size"]).astype(int),
                "cv_fold": folds,
                "is_train_fold_q95_tail": tail_mask,
                "actual_ms": labels,
                "predicted_ms": predictions,
            })
            prediction_frames.append(frame)
            for matrix_id, part in frame.groupby("matrix_id", sort=True):
                matrix_rows.append({
                    "mode": mode,
                    "target": target,
                    "matrix_id": matrix_id,
                    **contract.metric(part["actual_ms"], part["predicted_ms"]),
                })

            full_model = contract.XGBRegressor(
                **contract.PARAMS,
                random_state=contract.SEED + mode_index * 10000 + target_index * 1000,
            )
            fit_kwargs = {"verbose": False}
            if weights is not None:
                fit_kwargs["sample_weight"] = weights
            full_model.fit(features, np.log1p(labels), **fit_kwargs)
            model_path = output / "deploy_models" / f"{target}.{mode}.direct_log_xgboost_ABCD.json"
            full_model.save_model(model_path)
            deploy_models[mode][target] = full_model
            deploy_manifests.setdefault(mode, {})[target] = {
                "rows": int(len(labels)),
                "matrices": int(data["matrix_id"].nunique()),
                "features": contract.FEATURES[target],
                "model": str(model_path.resolve()),
                "weight_formula": (
                    "none" if mode == "plain"
                    else "w_i=(1/n_tasks_of_same_matrix), normalized to mean one within kernel"
                ),
            }

    predictions = pd.concat(prediction_frames, ignore_index=True)
    scheme_frames = []
    selection_frames = []
    scheme_rows = []
    for mode in MODES:
        schemes, selections, summary = scheme_metrics(
            predictions[predictions["mode"] == mode], mode, contract
        )
        scheme_frames.append(schemes)
        selection_frames.append(selections)
        scheme_rows.append(summary)

    pd.DataFrame(task_rows).to_csv(output / "task_oof_metrics.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(output / "matrix_oof_metrics.csv", index=False)
    predictions.to_csv(output / "task_oof_predictions.csv.gz", index=False, compression="gzip")
    pd.concat(scheme_frames, ignore_index=True).to_csv(
        output / "scheme_oof_predictions.csv", index=False
    )
    pd.concat(selection_frames, ignore_index=True).to_csv(
        output / "scheme_oof_selection.csv", index=False
    )
    pd.DataFrame(scheme_rows).to_csv(output / "scheme_oof_metrics.csv", index=False)

    for mode in MODES:
        header = output / "deploy_models" / f"pangulu_partition_xgb_v4_2_ABCD_{mode}.h"
        export_manifest = {
            "schema": "profile_direct_log_xgboost_ABCD_deploy_v1",
            "mode": mode,
            "hardware_id": args.hardware_id,
            "target": "log1p(label_time_ms)",
            "prediction": "expm1(raw_prediction)",
            "input_dir": str(source.resolve()),
            "targets": deploy_manifests[mode],
        }
        contract.export_header(deploy_models[mode], header, export_manifest)
        export_manifest["header"] = str(header.resolve())
        (output / "deploy_models" / f"deploy_manifest_{mode}.json").write_text(
            json.dumps(export_manifest, indent=2) + "\n"
        )

    manifest = {
        "schema": "profile_plain_vs_matrix_balanced_abcd_v1",
        "hardware_id": args.hardware_id,
        "input_dir": str(source.resolve()),
        "fold_map": str(Path(args.fold_map).resolve()),
        "fold_counts": {str(k): int(v) for k, v in fold_frame.groupby("cv_fold").size().items()},
        "features": "A+B+C+D frozen v4.2",
        "target": "log1p(label_time_ms)",
        "xgboost_params": contract.PARAMS,
        "modes": MODES,
        "tail_definition": "test task time >= 95th percentile of that fold training labels, per kernel",
        "wall_seconds": time.time() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(task_rows).to_string(index=False), flush=True)
    print(pd.DataFrame(scheme_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
