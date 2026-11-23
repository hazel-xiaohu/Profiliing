#!/usr/bin/env python3
"""Compare three frozen v4.2 A+B+C+D XGBoost objectives.

The profiling matrices are used for matrix-grouped five-fold OOF evaluation
and full-data deployment fitting.  The unseen matrices are used only once,
after every model choice (including weighted-log alpha) has been frozen.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


CONTRACT = Path("/output/Partition/script/train_export_direct_log_xgboost_abcd.py")


def load_contract():
    spec = importlib.util.spec_from_file_location("v42_contract", CONTRACT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_target(src: Path, target: str, contract, hardware_id: str):
    required = {
        "task_id", "matrix_id", "block_size", "label_time_ms", "hardware_id",
        *contract.FEATURES[target],
    }
    for mean, cv in contract.DERIVED[target].values():
        required |= {mean, cv}
    path = src / f"{target}.model_ready_v4.csv"
    data = pd.read_csv(path, usecols=lambda col: col in required, low_memory=False)
    for new, (mean, cv) in contract.DERIVED[target].items():
        data[new] = (
            pd.to_numeric(data[mean], errors="coerce").fillna(0.0)
            * pd.to_numeric(data[cv], errors="coerce").fillna(0.0)
        )
    y = pd.to_numeric(data.label_time_ms, errors="coerce").to_numpy(float)
    keep = np.isfinite(y) & (y > 0.0)
    data = data.loc[keep].reset_index(drop=True)
    y = y[keep]
    observed_hardware = set(data.hardware_id.astype(str))
    if observed_hardware != {hardware_id}:
        raise RuntimeError(f"{target}: hardware mismatch {observed_hardware}")
    raw = (
        data[contract.FEATURES[target]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(float)
    )
    return data, y, raw


def x_values(raw: np.ndarray, transform: str):
    if transform == "raw":
        return raw
    if transform == "log1p":
        return np.log1p(np.maximum(raw, 0.0))
    raise ValueError(transform)


def scheme_denominators(frames, alpha: float):
    labels = pd.concat(
        [
            pd.DataFrame({
                "matrix_id": d.matrix_id.astype(str),
                "block_size": pd.to_numeric(d.block_size).astype(int),
                "value": np.power(y, alpha),
            })
            for d, y, _ in frames.values()
        ],
        ignore_index=True,
    )
    return labels.groupby(["matrix_id", "block_size"]).value.sum()


def scheme_weights(data, y, denominator, alpha: float):
    keys = pd.MultiIndex.from_arrays(
        [data.matrix_id.astype(str), pd.to_numeric(data.block_size).astype(int)]
    )
    weights = np.power(y, alpha) / denominator.reindex(keys).to_numpy(float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise RuntimeError("invalid scheme-normalized sample weights")
    return weights / weights.mean()


def pair_accuracy(group):
    actual = group.actual_ms.to_numpy(float)
    predicted = group.predicted_ms.to_numpy(float)
    correct = total = 0
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            if actual[i] == actual[j]:
                continue
            correct += int(
                np.sign(actual[i] - actual[j])
                == np.sign(predicted[i] - predicted[j])
            )
            total += 1
    return correct / max(total, 1)


def aggregate_schemes(tasks, contract, scope: str, objective: str):
    schemes = tasks.groupby(["matrix_id", "block_size"], as_index=False).agg(
        actual_ms=("actual_ms", "sum"),
        predicted_ms=("predicted_ms", "sum"),
        tasks=("task_id", "size"),
    )
    selections = []
    for matrix_id, group in schemes.groupby("matrix_id", sort=True):
        oracle = group.loc[group.actual_ms.idxmin()]
        selected = group.loc[group.predicted_ms.idxmin()]
        selections.append({
            "objective": objective,
            "scope": scope,
            "matrix_id": matrix_id,
            "oracle_block_size": int(oracle.block_size),
            "predicted_best_block_size": int(selected.block_size),
            "oracle_actual_ms": float(oracle.actual_ms),
            "selected_actual_ms": float(selected.actual_ms),
            "regret_percent": float((selected.actual_ms / oracle.actual_ms - 1.0) * 100.0),
            "spearman": float(group.actual_ms.rank().corr(group.predicted_ms.rank())),
            "pairwise_accuracy": pair_accuracy(group),
        })
    selected = pd.DataFrame(selections)
    summary = {
        "objective": objective,
        "scope": scope,
        **contract.metric(schemes.actual_ms, schemes.predicted_ms),
        "matrices": int(len(selected)),
        "oracle_accuracy": float(
            (selected.oracle_block_size == selected.predicted_best_block_size).mean()
        ),
        "mean_spearman": float(selected.spearman.mean()),
        "mean_pairwise_accuracy": float(selected.pairwise_accuracy.mean()),
        "mean_regret": float(selected.regret_percent.mean()),
        "median_regret": float(selected.regret_percent.median()),
        "p95_regret": float(selected.regret_percent.quantile(0.95)),
    }
    schemes.insert(0, "objective", objective)
    schemes.insert(1, "scope", scope)
    return schemes, selected, summary


def objective_spec(name: str, alpha: float | None = None):
    if name == "log_time":
        return {"feature_transform": "log1p", "target": "log1p", "alpha": None}
    if name == "raw_time":
        return {"feature_transform": "raw", "target": "raw", "alpha": None}
    if name == "weighted_log_time":
        return {"feature_transform": "log1p", "target": "log1p", "alpha": alpha}
    raise ValueError(name)


def fit_predict(model, spec, train_x, train_y, test_x, weights=None):
    fit_y = np.log1p(train_y) if spec["target"] == "log1p" else train_y
    model.fit(train_x, fit_y, sample_weight=weights, verbose=False)
    prediction = model.predict(test_x)
    if spec["target"] == "log1p":
        prediction = np.expm1(prediction)
    return np.maximum(prediction, 0.0)


def grouped_oof(
    objective, spec, frames, fold_map, denominators, contract, seed_offset=0
):
    task_frames = []
    task_metrics = []
    for target_index, target in enumerate(contract.TARGETS):
        data, y, raw = frames[target]
        fold = data.matrix_id.astype(str).map(fold_map)
        if fold.isna().any() or set(fold.astype(int)) != {1, 2, 3, 4, 5}:
            raise RuntimeError(f"{target}: incomplete five-fold map")
        fold = fold.to_numpy(int)
        x = x_values(raw, spec["feature_transform"])
        all_weights = None
        if spec["alpha"] is not None:
            all_weights = scheme_weights(data, y, denominators, spec["alpha"])
        predicted = np.empty(len(y), float)
        for fold_id in range(1, 6):
            train = fold != fold_id
            test = fold == fold_id
            model = contract.XGBRegressor(
                **contract.PARAMS,
                random_state=contract.SEED + seed_offset + target_index * 1000 + fold_id,
            )
            predicted[test] = fit_predict(
                model, spec, x[train], y[train], x[test],
                None if all_weights is None else all_weights[train],
            )
        task_frames.append(pd.DataFrame({
            "objective": objective,
            "target": target,
            "task_id": data.task_id.astype(str),
            "matrix_id": data.matrix_id.astype(str),
            "block_size": pd.to_numeric(data.block_size).astype(int),
            "actual_ms": y,
            "predicted_ms": predicted,
        }))
        task_metrics.append({
            "objective": objective,
            "scope": "profiling25_grouped_5fold_oof",
            "target": target,
            **contract.metric(y, predicted),
        })
    tasks = pd.concat(task_frames, ignore_index=True)
    schemes, selections, summary = aggregate_schemes(
        tasks, contract, "profiling25_grouped_5fold_oof", objective
    )
    return tasks, pd.DataFrame(task_metrics), schemes, selections, summary


def fit_full_and_unseen(
    objective, spec, train_frames, unseen_frames, denominators, contract, model_dir,
    seed_offset=0,
):
    task_frames = []
    task_metrics = []
    for target_index, target in enumerate(contract.TARGETS):
        train, train_y, train_raw = train_frames[target]
        unseen, unseen_y, unseen_raw = unseen_frames[target]
        overlap = set(train.matrix_id.astype(str)) & set(unseen.matrix_id.astype(str))
        if overlap:
            raise RuntimeError(f"{target}: profiling/unseen overlap {sorted(overlap)}")
        train_x = x_values(train_raw, spec["feature_transform"])
        unseen_x = x_values(unseen_raw, spec["feature_transform"])
        weights = None
        if spec["alpha"] is not None:
            weights = scheme_weights(train, train_y, denominators, spec["alpha"])
        model = contract.XGBRegressor(
            **contract.PARAMS,
            random_state=contract.SEED + 50000 + seed_offset + target_index * 1000,
        )
        predicted = fit_predict(
            model, spec, train_x, train_y, unseen_x, weights
        )
        model.save_model(model_dir / f"{target}.{objective}.json")
        task_frames.append(pd.DataFrame({
            "objective": objective,
            "target": target,
            "task_id": unseen.task_id.astype(str),
            "matrix_id": unseen.matrix_id.astype(str),
            "block_size": pd.to_numeric(unseen.block_size).astype(int),
            "actual_ms": unseen_y,
            "predicted_ms": predicted,
        }))
        task_metrics.append({
            "objective": objective,
            "scope": "unseen_com-DBLP_rajat29",
            "target": target,
            **contract.metric(unseen_y, predicted),
        })
    tasks = pd.concat(task_frames, ignore_index=True)
    schemes, selections, summary = aggregate_schemes(
        tasks, contract, "unseen_com-DBLP_rajat29", objective
    )
    return tasks, pd.DataFrame(task_metrics), schemes, selections, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--unseen-dir", required=True)
    parser.add_argument("--fold-map", required=True)
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weighted-alphas", default="0.5,1.0")
    args = parser.parse_args()

    started = time.time()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    model_dir.mkdir()
    contract = load_contract()
    train_dir = Path(args.train_dir)
    unseen_dir = Path(args.unseen_dir)
    train_frames = {
        target: load_target(train_dir, target, contract, args.hardware_id)
        for target in contract.TARGETS
    }
    unseen_frames = {
        target: load_target(unseen_dir, target, contract, args.hardware_id)
        for target in contract.TARGETS
    }
    fold_table = pd.read_csv(args.fold_map)
    fold_map = dict(zip(fold_table.matrix_id.astype(str), fold_table.cv_fold.astype(int)))

    all_oof = []
    weighted_candidates = []
    candidate_outputs = {}
    for index, alpha in enumerate(float(x) for x in args.weighted_alphas.split(",")):
        objective = f"weighted_log_time_alpha_{alpha:g}"
        spec = objective_spec("weighted_log_time", alpha)
        denominator = scheme_denominators(train_frames, alpha)
        result = grouped_oof(
            objective, spec, train_frames, fold_map, denominator, contract,
            seed_offset=100000 + index * 10000,
        )
        candidate_outputs[alpha] = result
        weighted_candidates.append({"alpha": alpha, **result[-1]})
        print("OOF", objective, result[-1], flush=True)
    weighted_selection = pd.DataFrame(weighted_candidates).sort_values(
        ["mean_regret", "p95_regret", "MAPE", "alpha"]
    )
    selected_alpha = float(weighted_selection.iloc[0].alpha)

    objective_specs = [
        ("log_time", objective_spec("log_time"), None),
        ("raw_time", objective_spec("raw_time"), None),
        (
            f"weighted_log_time_alpha_{selected_alpha:g}",
            objective_spec("weighted_log_time", selected_alpha),
            scheme_denominators(train_frames, selected_alpha),
        ),
    ]
    oof_tasks = []
    oof_task_metrics = []
    oof_schemes = []
    oof_selections = []
    oof_summaries = []
    unseen_tasks = []
    unseen_task_metrics = []
    unseen_schemes = []
    unseen_selections = []
    unseen_summaries = []
    for index, (objective, spec, denominator) in enumerate(objective_specs):
        if objective.startswith("weighted_log_time"):
            oof_result = candidate_outputs[selected_alpha]
        else:
            oof_result = grouped_oof(
                objective, spec, train_frames, fold_map, denominator, contract,
                seed_offset=index * 10000,
            )
        oof_tasks.append(oof_result[0])
        oof_task_metrics.append(oof_result[1])
        oof_schemes.append(oof_result[2])
        oof_selections.append(oof_result[3])
        oof_summaries.append(oof_result[4])
        print("OOF SELECTED", objective, oof_result[4], flush=True)

        unseen_result = fit_full_and_unseen(
            objective, spec, train_frames, unseen_frames, denominator, contract,
            model_dir, seed_offset=index * 10000,
        )
        unseen_tasks.append(unseen_result[0])
        unseen_task_metrics.append(unseen_result[1])
        unseen_schemes.append(unseen_result[2])
        unseen_selections.append(unseen_result[3])
        unseen_summaries.append(unseen_result[4])
        print("UNSEEN", objective, unseen_result[4], flush=True)

    pd.concat(oof_tasks, ignore_index=True).to_csv(
        output / "oof_task_predictions.csv.gz", index=False, compression="gzip"
    )
    pd.concat(oof_task_metrics, ignore_index=True).to_csv(
        output / "oof_task_metrics.csv", index=False
    )
    pd.concat(oof_schemes, ignore_index=True).to_csv(
        output / "oof_scheme_predictions.csv", index=False
    )
    pd.concat(oof_selections, ignore_index=True).to_csv(
        output / "oof_scheme_selection.csv", index=False
    )
    pd.DataFrame(oof_summaries).to_csv(output / "oof_scheme_summary.csv", index=False)
    weighted_selection.to_csv(output / "weighted_alpha_selection.csv", index=False)

    pd.concat(unseen_tasks, ignore_index=True).to_csv(
        output / "unseen_task_predictions.csv.gz", index=False, compression="gzip"
    )
    pd.concat(unseen_task_metrics, ignore_index=True).to_csv(
        output / "unseen_task_metrics.csv", index=False
    )
    pd.concat(unseen_schemes, ignore_index=True).to_csv(
        output / "unseen_scheme_predictions.csv", index=False
    )
    pd.concat(unseen_selections, ignore_index=True).to_csv(
        output / "unseen_scheme_selection.csv", index=False
    )
    pd.DataFrame(unseen_summaries).to_csv(
        output / "unseen_scheme_summary.csv", index=False
    )

    manifest = {
        "schema": "xgboost_v4.2_ABCD_three_objectives_grouped_oof_unseen_v1",
        "hardware_id": args.hardware_id,
        "features": contract.FEATURES,
        "xgboost_params": contract.PARAMS,
        "objectives": {
            "log_time": "log1p(T)=XGB(log1p(features)); unweighted",
            "raw_time": "T=XGB(raw features); unweighted",
            "weighted_log_time": (
                "log1p(T)=XGB(log1p(features)); scheme-normalized T^alpha weights"
            ),
        },
        "weighted_alpha_candidates": [float(x) for x in args.weighted_alphas.split(",")],
        "selected_alpha": selected_alpha,
        "alpha_selection_rule": "minimum profiling25 OOF mean scheme regret; ties by P95 regret, scheme MAPE, alpha",
        "unseen_used_for_feature_model_or_alpha_selection": False,
        "train_dir": str(train_dir.resolve()),
        "unseen_dir": str(unseen_dir.resolve()),
        "fold_map": str(Path(args.fold_map).resolve()),
        "wall_seconds": time.time() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("SELECTED_ALPHA", selected_alpha, flush=True)
    print("wrote", output, flush=True)


if __name__ == "__main__":
    main()
