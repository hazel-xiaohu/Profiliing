#!/usr/bin/env python3
"""Profile29 plain XGBoost cumulative A--E ablation with frozen matrix folds."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


CONTRACT = Path("/output/Partition/script/train_export_direct_log_xgboost_abcd.py")
STAGES = ("A", "A+B", "A+B+C", "A+B+C+D", "A+B+C+D+E")


def load_contract():
    spec = importlib.util.spec_from_file_location("profile29_ablation_contract", CONTRACT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {CONTRACT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def feature_groups(target: str) -> dict[str, list[str]]:
    axis = "row" if target == "TSTRF" else "col"
    inputs = {"GETRF": 1, "GESSM": 2, "TSTRF": 2, "SSSSM_dense": 3}[target]
    return {
        "A": ["replay_main_dimension"] + [f"replay_nnz_input_{i}" for i in range(inputs)],
        "B": ["replay_operation_count_v2"],
        "C": [f"replay_input_{i}_{axis}_nnz_max" for i in range(inputs)],
        "D": (
            [f"replay_input_{i}_{axis}_nnz_mean" for i in range(inputs)]
            + [f"replay_input_{i}_{axis}_nnz_std_derived" for i in range(inputs)]
        ),
        "E": [f"replay_input_{i}_{axis}_empty_fraction" for i in range(inputs)],
    }


def load_target(root: Path, target: str, contract):
    groups = feature_groups(target)
    features = [column for key in "ABCDE" for column in groups[key]]
    required = {"task_id", "matrix_id", "block_size", "label_time_ms", "hardware_id", *features}
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
    if set(data["hardware_id"].astype(str)) != {"H100_SXM_80GB"}:
        raise RuntimeError(f"{target}: hardware mismatch")
    return data, labels, groups


def pair_metrics(frame: pd.DataFrame) -> tuple[float, float]:
    actual = frame["actual_ms"].to_numpy(float)
    predicted = frame["predicted_ms"].to_numpy(float)
    correct = total = 0
    weighted_correct = weighted_total = 0.0
    for left in range(len(frame)):
        for right in range(left + 1, len(frame)):
            delta = actual[left] - actual[right]
            if delta == 0:
                continue
            hit = np.sign(delta) == np.sign(predicted[left] - predicted[right])
            weight = abs(delta)
            correct += int(hit)
            total += 1
            weighted_correct += weight * int(hit)
            weighted_total += weight
    return correct / max(total, 1), weighted_correct / max(weighted_total, 1e-30)


def scheme_metrics(tasks: pd.DataFrame, stage: str, contract):
    schemes = tasks.groupby(["matrix_id", "block_size"], as_index=False).agg(
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
            "stage": stage,
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
        "stage": stage,
        **contract.metric(schemes["actual_ms"], schemes["predicted_ms"]),
        "matrices": int(len(selection)),
        "oracle_accuracy": float((selection["oracle_block_size"] == selection["selected_block_size"]).mean()),
        "mean_pairwise_accuracy": float(selection["pairwise_accuracy"].mean()),
        "mean_weighted_pairwise_accuracy": float(selection["weighted_pairwise_accuracy"].mean()),
        "mean_spearman": float(selection["spearman"].mean()),
        "mean_regret_percent": float(selection["regret_percent"].mean()),
        "median_regret_percent": float(selection["regret_percent"].median()),
        "p95_regret_percent": float(selection["regret_percent"].quantile(0.95)),
    }
    schemes.insert(0, "stage", stage)
    return schemes, selection, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--fold-map", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"output must be empty: {output}")

    contract = load_contract()
    fold_frame = pd.read_csv(args.fold_map)
    fold_counts = fold_frame.groupby("cv_fold").size().sort_index().to_dict()
    if fold_counts != {1: 6, 2: 6, 3: 6, 4: 6, 5: 5}:
        raise RuntimeError(f"unexpected profile29 fold sizes: {fold_counts}")
    fold_map = dict(zip(fold_frame["matrix_id"].astype(str), fold_frame["cv_fold"].astype(int)))

    started = time.time()
    task_rows = []
    fold_rows = []
    matrix_rows = []
    prediction_frames = []
    timing_rows = []
    loaded = {target: load_target(source, target, contract) for target in contract.TARGETS}

    for target_index, target in enumerate(contract.TARGETS):
        data, labels, groups = loaded[target]
        folds = data["matrix_id"].astype(str).map(fold_map)
        if folds.isna().any() or set(folds.astype(int)) != {1, 2, 3, 4, 5}:
            raise RuntimeError(f"{target}: invalid fold coverage")
        folds = folds.to_numpy(int)
        cumulative: list[str] = []
        for stage_index, stage in enumerate(STAGES):
            cumulative += groups[chr(ord("A") + stage_index)]
            features = contract.values(data, cumulative)
            predictions = np.empty(len(labels), dtype=float)
            stage_started = time.time()
            for fold in range(1, 6):
                train = folds != fold
                test = folds == fold
                model = contract.XGBRegressor(
                    **contract.PARAMS,
                    random_state=contract.SEED + target_index * 1000 + fold,
                )
                fit_started = time.time()
                model.fit(features[train], np.log1p(labels[train]), verbose=False)
                fit_seconds = time.time() - fit_started
                predict_started = time.time()
                fold_prediction = np.maximum(np.expm1(model.predict(features[test])), 0)
                predict_seconds = time.time() - predict_started
                predictions[test] = fold_prediction
                fold_rows.append({
                    "target": target,
                    "stage": stage,
                    "fold": fold,
                    "features": len(cumulative),
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                    **contract.metric(labels[test], fold_prediction),
                })
                timing_rows.append({
                    "target": target,
                    "stage": stage,
                    "fold": fold,
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                })

            task_rows.append({
                "target": target,
                "stage": stage,
                "features": len(cumulative),
                **contract.metric(labels, predictions),
                "stage_wall_seconds": time.time() - stage_started,
            })
            frame = pd.DataFrame({
                "stage": stage,
                "target": target,
                "task_id": data["task_id"].astype(str),
                "matrix_id": data["matrix_id"].astype(str),
                "block_size": pd.to_numeric(data["block_size"]).astype(int),
                "cv_fold": folds,
                "actual_ms": labels,
                "predicted_ms": predictions,
            })
            prediction_frames.append(frame)
            for matrix_id, part in frame.groupby("matrix_id", sort=True):
                matrix_rows.append({
                    "target": target,
                    "stage": stage,
                    "matrix_id": matrix_id,
                    **contract.metric(part["actual_ms"], part["predicted_ms"]),
                })
            print(target, stage, task_rows[-1], flush=True)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    scheme_frames = []
    selection_frames = []
    scheme_rows = []
    for stage in STAGES:
        schemes, selections, summary = scheme_metrics(
            predictions[predictions["stage"] == stage], stage, contract
        )
        scheme_frames.append(schemes)
        selection_frames.append(selections)
        scheme_rows.append(summary)

    task_metrics = pd.DataFrame(task_rows)
    matrix_metrics = pd.DataFrame(matrix_rows)
    macro = matrix_metrics.groupby(["target", "stage"], as_index=False).agg(
        matrices=("matrix_id", "nunique"),
        macro_MAPE=("MAPE", "mean"),
        macro_MdAPE=("MdAPE", "mean"),
        macro_P95_APE=("P95_APE", "mean"),
        macro_R2_raw=("R2_raw", "mean"),
        macro_R2_log=("R2_log", "mean"),
    )
    task_metrics.to_csv(output / "task_oof_ablation_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_oof_ablation_metrics.csv", index=False)
    matrix_metrics.to_csv(output / "matrix_oof_ablation_metrics.csv", index=False)
    macro.to_csv(output / "matrix_macro_oof_ablation_metrics.csv", index=False)
    predictions.to_csv(output / "task_oof_ablation_predictions.csv.gz", index=False, compression="gzip")
    pd.concat(scheme_frames, ignore_index=True).to_csv(output / "scheme_oof_ablation_predictions.csv", index=False)
    pd.concat(selection_frames, ignore_index=True).to_csv(output / "scheme_oof_ablation_selection.csv", index=False)
    pd.DataFrame(scheme_rows).to_csv(output / "scheme_oof_ablation_metrics.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(output / "timing.csv", index=False)

    manifest = {
        "schema": "profile29_plain_matrix5fold_cumulative_ABCDE_ablation_v1",
        "input_dir": str(source.resolve()),
        "fold_map": str(Path(args.fold_map).resolve()),
        "fold_counts": {str(key): int(value) for key, value in fold_counts.items()},
        "weighting": "plain (no sample weights)",
        "target": "log1p(label_time_ms)",
        "feature_transform": "log1p(max(feature,0))",
        "stages": STAGES,
        "feature_groups": {target: feature_groups(target) for target in contract.TARGETS},
        "xgboost_params": contract.PARAMS,
        "seed_rule": "20260811 + target_index*1000 + fold; identical random stream across stages",
        "wall_seconds": time.time() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\nTASK\n", task_metrics.to_string(index=False), flush=True)
    print("\nSCHEME\n", pd.DataFrame(scheme_rows).to_string(index=False), flush=True)
    print("wrote", output, flush=True)


if __name__ == "__main__":
    main()
