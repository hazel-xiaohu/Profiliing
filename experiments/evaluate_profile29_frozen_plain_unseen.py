#!/usr/bin/env python3
"""Evaluate the frozen profile29 plain direct-log XGBoost models without refitting."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


TRAINING_CONTRACT = Path(
    "/output/Partition/script/train_export_direct_log_xgboost_abcd.py"
)


def load_training_contract():
    # Only constants and feature-construction helpers are needed. Avoid making
    # the evaluator depend on the Python xgboost package or its version.
    if "xgboost" not in sys.modules:
        stub = types.ModuleType("xgboost")
        stub.XGBRegressor = object
        sys.modules["xgboost"] = stub
    spec = importlib.util.spec_from_file_location(
        "profile29_abcd_contract", TRAINING_CONTRACT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def predict_frozen_json(path: Path, features: np.ndarray) -> np.ndarray:
    """Vectorized inference matching the generated C evaluator."""
    model = json.loads(path.read_text())
    learner = model["learner"]
    base_score = learner["learner_model_param"]["base_score"].strip("[]")
    prediction = np.full(len(features), float(base_score), dtype=np.float64)
    features = np.asarray(features, dtype=np.float32)
    rows = np.arange(len(features))
    trees = learner["gradient_booster"]["model"]["trees"]
    for tree in trees:
        left = np.asarray(tree["left_children"], dtype=np.int32)
        right = np.asarray(tree["right_children"], dtype=np.int32)
        split_feature = np.asarray(tree["split_indices"], dtype=np.int32)
        threshold = np.asarray(tree["split_conditions"], dtype=np.float32)
        default_left = np.asarray(tree["default_left"], dtype=np.int8)
        node = np.zeros(len(features), dtype=np.int32)
        active = np.ones(len(features), dtype=bool)
        while active.any():
            active_rows = rows[active]
            active_nodes = node[active]
            is_leaf = left[active_nodes] < 0
            if is_leaf.any():
                prediction[active_rows[is_leaf]] += threshold[
                    active_nodes[is_leaf]
                ].astype(np.float64)
            if (~is_leaf).any():
                branch_rows = active_rows[~is_leaf]
                branch_nodes = active_nodes[~is_leaf]
                values = features[branch_rows, split_feature[branch_nodes]]
                go_left = np.where(
                    np.isnan(values),
                    default_left[branch_nodes] != 0,
                    values < threshold[branch_nodes],
                )
                node[branch_rows] = np.where(
                    go_left, left[branch_nodes], right[branch_nodes]
                )
            active[active_rows[is_leaf]] = False
    return prediction


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--fold-map", required=True)
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hardware-id", default="H100_SXM_80GB")
    return parser.parse_args()


def pair_metrics(frame: pd.DataFrame) -> tuple[float, float]:
    actual = frame["actual_ms"].to_numpy(float)
    predicted = frame["predicted_ms"].to_numpy(float)
    correct = total = 0
    weighted_correct = weighted_total = 0.0
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


def main():
    args = parse_args()
    source = Path(args.input_dir).resolve()
    training = Path(args.training_dir).resolve()
    folds = Path(args.fold_map).resolve()
    frozen = Path(args.frozen_dir).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    contract = load_training_contract()
    freeze_manifest_path = frozen / "FREEZE_MANIFEST.json"
    deploy_manifest_path = frozen / "deploy_manifest_plain.json"
    if freeze_manifest_path.is_file():
        freeze_manifest = json.loads(freeze_manifest_path.read_text())
        if freeze_manifest.get("status") != "candidate_not_deployed":
            raise RuntimeError("unexpected frozen candidate status")
        for name, expected in freeze_manifest["files_sha256"].items():
            actual = sha256(frozen / name)
            if actual != expected:
                raise RuntimeError(f"frozen file hash mismatch: {name}")
        model_manifest_path = freeze_manifest_path
        model_manifest = freeze_manifest
    elif deploy_manifest_path.is_file():
        model_manifest = json.loads(deploy_manifest_path.read_text())
        if model_manifest.get("mode") != "plain":
            raise RuntimeError("deployment manifest is not the plain model")
        if model_manifest.get("hardware_id") != args.hardware_id:
            raise RuntimeError("deployment manifest hardware mismatch")
        model_manifest_path = deploy_manifest_path
    else:
        raise RuntimeError("missing frozen/deployment model manifest")

    development = set(pd.read_csv(folds).matrix_id.astype(str))
    task_frames = []
    kernel_metrics = []
    matrix_metrics = []
    matrix_block_metrics = []
    range_rows = []
    manifest = {
        "schema": "frozen_plain_unseen_evaluation_v2",
        "no_refit": True,
        "no_threshold_tuning": True,
        "target": "log1p(kernel_time_ms)",
        "inverse": "maximum(expm1(raw_prediction), 0)",
        "feature_set": "v4.2 A+B+C+D",
        "frozen_dir": str(frozen),
        "model_manifest": str(model_manifest_path),
        "model_manifest_sha256": sha256(model_manifest_path),
        "fold_map": str(folds),
        "fold_map_sha256": sha256(folds),
        "targets": {},
    }

    for target in contract.TARGETS:
        raw_features = contract.FEATURES[target]
        required = {
            "task_id",
            "matrix_id",
            "block_size",
            "label_time_ms",
            "hardware_id",
            *raw_features,
        }
        for mean_name, cv_name in contract.DERIVED[target].values():
            required.update((mean_name, cv_name))

        unseen = pd.read_csv(
            source / f"{target}.model_ready_v4.csv",
            usecols=lambda name: name in required,
            low_memory=False,
        )
        train = pd.read_csv(
            training / f"{target}.model_ready_v4.csv",
            usecols=lambda name: name in required,
            low_memory=False,
        )
        for frame in (unseen, train):
            for new_name, (mean_name, cv_name) in contract.DERIVED[target].items():
                frame[new_name] = (
                    pd.to_numeric(frame[mean_name], errors="coerce").fillna(0)
                    * pd.to_numeric(frame[cv_name], errors="coerce").fillna(0)
                )

        external = set(unseen.matrix_id.astype(str))
        overlap = sorted(development & external)
        if overlap:
            raise RuntimeError(f"{target}: development/unseen overlap: {overlap}")
        hardware = sorted(set(unseen.hardware_id.astype(str)))
        if hardware != [args.hardware_id]:
            raise RuntimeError(f"{target}: unexpected hardware: {hardware}")

        y = pd.to_numeric(unseen.label_time_ms, errors="coerce").to_numpy(float)
        valid = np.isfinite(y) & (y > 0)
        unseen = unseen.loc[valid].reset_index(drop=True)
        y = y[valid]

        model_name = f"{target}.plain.direct_log_xgboost_ABCD.json"
        model_path = frozen / model_name
        x = contract.values(unseen, raw_features)
        prediction = np.maximum(
            np.expm1(predict_frozen_json(model_path, x)), 0
        )
        tasks = pd.DataFrame(
            {
                "target": target,
                "task_id": unseen.task_id.astype(str),
                "matrix_id": unseen.matrix_id.astype(str),
                "block_size": pd.to_numeric(unseen.block_size),
                "actual_ms": y,
                "predicted_ms": prediction,
            }
        )
        task_frames.append(tasks)
        kernel_metrics.append({"target": target, **contract.metric(y, prediction)})
        for matrix_id, part in tasks.groupby("matrix_id", sort=True):
            matrix_metrics.append(
                {
                    "target": target,
                    "matrix_id": matrix_id,
                    **contract.metric(part.actual_ms, part.predicted_ms),
                }
            )
        for (matrix_id, block_size), part in tasks.groupby(
            ["matrix_id", "block_size"], sort=True
        ):
            matrix_block_metrics.append(
                {
                    "target": target,
                    "matrix_id": matrix_id,
                    "block_size": int(block_size),
                    **contract.metric(part.actual_ms, part.predicted_ms),
                }
            )

        for feature in raw_features:
            train_values = pd.to_numeric(train[feature], errors="coerce").to_numpy(float)
            train_values = train_values[np.isfinite(train_values)]
            lower = float(train_values.min())
            upper = float(train_values.max())
            for matrix_id, part in unseen.groupby("matrix_id", sort=True):
                values = pd.to_numeric(part[feature], errors="coerce").to_numpy(float)
                finite = np.isfinite(values)
                outside = finite & ((values < lower) | (values > upper))
                range_rows.append(
                    {
                        "target": target,
                        "matrix_id": matrix_id,
                        "feature": feature,
                        "train_min": lower,
                        "train_max": upper,
                        "unseen_min": float(np.nanmin(values)),
                        "unseen_max": float(np.nanmax(values)),
                        "outside_count": int(outside.sum()),
                        "finite_count": int(finite.sum()),
                        "outside_fraction": float(outside.sum() / max(finite.sum(), 1)),
                    }
                )

        manifest["targets"][target] = {
            "rows": len(tasks),
            "matrices": sorted(external),
            "development_overlap": overlap,
            "hardware_ids": hardware,
            "features": raw_features,
            "model": str(model_path),
            "model_sha256": sha256(model_path),
        }

    tasks = pd.concat(task_frames, ignore_index=True)
    tasks.to_csv(
        output / "task_predictions.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(kernel_metrics).to_csv(
        output / "task_metrics_by_kernel.csv", index=False
    )
    pd.DataFrame(matrix_metrics).to_csv(
        output / "task_metrics_by_kernel_matrix.csv", index=False
    )
    pd.DataFrame(matrix_block_metrics).to_csv(
        output / "task_metrics_by_kernel_matrix_block.csv", index=False
    )
    pd.DataFrame(range_rows).to_csv(output / "feature_range_audit.csv", index=False)

    schemes = (
        tasks.groupby(["matrix_id", "block_size"], as_index=False)
        .agg(
            actual_ms=("actual_ms", "sum"),
            predicted_ms=("predicted_ms", "sum"),
            tasks=("task_id", "size"),
        )
        .sort_values(["matrix_id", "block_size"])
    )
    schemes["APE"] = (
        np.abs(schemes.predicted_ms - schemes.actual_ms)
        / schemes.actual_ms
        * 100.0
    )
    schemes.to_csv(output / "scheme_predictions.csv", index=False)
    selections = []
    for matrix_id, group in schemes.groupby("matrix_id", sort=True):
        oracle = group.loc[group.actual_ms.idxmin()]
        selected = group.loc[group.predicted_ms.idxmin()]
        pairwise, weighted_pairwise = pair_metrics(group)
        selections.append(
            {
                "matrix_id": matrix_id,
                "oracle_block_size": int(oracle.block_size),
                "selected_block_size": int(selected.block_size),
                "oracle_actual_ms": float(oracle.actual_ms),
                "selected_actual_ms": float(selected.actual_ms),
                "selected_predicted_ms": float(selected.predicted_ms),
                "regret_percent": float(
                    (selected.actual_ms / oracle.actual_ms - 1.0) * 100.0
                ),
                "spearman": float(
                    group.actual_ms.rank().corr(group.predicted_ms.rank())
                ),
                "pairwise_accuracy": pairwise,
                "weighted_pairwise_accuracy": weighted_pairwise,
            }
        )
    selection = pd.DataFrame(selections)
    selection.to_csv(output / "scheme_selection.csv", index=False)
    scheme_summary = {
        "scope": "all",
        **contract.metric(schemes.actual_ms, schemes.predicted_ms),
        "matrices": int(len(selection)),
        "oracle_accuracy": float(
            (selection.oracle_block_size == selection.selected_block_size).mean()
        ),
        "mean_pairwise_accuracy": float(selection.pairwise_accuracy.mean()),
        "mean_weighted_pairwise_accuracy": float(
            selection.weighted_pairwise_accuracy.mean()
        ),
        "mean_spearman": float(selection.spearman.mean()),
        "mean_regret_percent": float(selection.regret_percent.mean()),
        "median_regret_percent": float(selection.regret_percent.median()),
        "p95_regret_percent": float(selection.regret_percent.quantile(0.95)),
        "max_regret_percent": float(selection.regret_percent.max()),
    }
    pd.DataFrame([scheme_summary]).to_csv(
        output / "scheme_metrics.csv", index=False
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
