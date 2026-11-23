#!/usr/bin/env python3
"""Train direct raw-time XGBoost on profiling25 and evaluate unseen matrices.

The feature contract and XGBoost hyperparameters are imported from the frozen
v4.2 A+B+C+D direct-log model.  The only modeling change is the target:

    direct-log: log1p(time_ms) = f(log1p(features))
    direct-raw:          time_ms = f(log1p(features))

The unseen input is never used for fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


CONTRACT = Path("/output/Partition/script/train_export_direct_log_xgboost_abcd.py")


def load_contract():
    spec = importlib.util.spec_from_file_location("v42_abcd_contract", CONTRACT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_target(src: Path, target: str, contract, feature_transform: str):
    required = {
        "task_id", "matrix_id", "block_size", "label_time_ms", "hardware_id",
        *contract.FEATURES[target],
    }
    for mean, cv in contract.DERIVED[target].values():
        required |= {mean, cv}
    data = pd.read_csv(
        src / f"{target}.model_ready_v4.csv",
        usecols=lambda column: column in required,
        low_memory=False,
    )
    for new, (mean, cv) in contract.DERIVED[target].items():
        data[new] = (
            pd.to_numeric(data[mean], errors="coerce").fillna(0)
            * pd.to_numeric(data[cv], errors="coerce").fillna(0)
        )
    labels = pd.to_numeric(data.label_time_ms, errors="coerce").to_numpy(float)
    valid = np.isfinite(labels) & (labels > 0)
    data = data.loc[valid].reset_index(drop=True)
    labels = labels[valid]
    hardware = set(data.hardware_id.astype(str))
    if hardware != {"H100_SXM_80GB"}:
        raise RuntimeError(f"{target}: hardware mismatch: {hardware}")
    raw = (
        data[contract.FEATURES[target]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(float)
    )
    if feature_transform == "log1p":
        features = np.log1p(np.maximum(raw, 0.0))
    elif feature_transform == "raw":
        features = raw
    else:
        raise ValueError(feature_transform)
    return data, labels, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--unseen-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--feature-transform", choices=("log1p", "raw"), default="log1p",
        help="Input feature preprocessing; the prediction target is always raw time_ms.",
    )
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    unseen_dir = Path(args.unseen_dir)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    model_dir.mkdir()

    contract = load_contract()
    started = time.time()
    task_frames = []
    task_summary = []
    per_matrix = []
    per_block = []
    manifest = {
        "schema": "direct_raw_xgboost_v4.2_ABCD_unseen_v1",
        "hardware_id": "H100_SXM_80GB",
        "target": "label_time_ms",
        "prediction": "max(0, raw_prediction)",
        "feature_transform": (
            "log1p(max(raw_feature, 0))" if args.feature_transform == "log1p"
            else "none (raw numeric features)"
        ),
        "training_contract": str(CONTRACT),
        "train_dir": str(train_dir.resolve()),
        "unseen_dir": str(unseen_dir.resolve()),
        "xgboost_params": contract.PARAMS,
        "unseen_used_for_fit": False,
        "targets": {},
    }

    for target_index, target in enumerate(contract.TARGETS):
        train, train_y, train_x = load_target(
            train_dir, target, contract, args.feature_transform
        )
        unseen, unseen_y, unseen_x = load_target(
            unseen_dir, target, contract, args.feature_transform
        )
        overlap = sorted(set(train.matrix_id.astype(str)) & set(unseen.matrix_id.astype(str)))
        if overlap:
            raise RuntimeError(f"{target}: train/unseen matrix overlap: {overlap}")

        model = contract.XGBRegressor(
            **contract.PARAMS,
            random_state=contract.SEED + target_index * 1000,
        )
        fit_started = time.time()
        model.fit(train_x, train_y, verbose=False)
        fit_seconds = time.time() - fit_started
        model_path = model_dir / f"{target}.direct_raw_xgboost_ABCD.json"
        model.save_model(model_path)

        prediction = np.maximum(model.predict(unseen_x), 0.0)
        frame = pd.DataFrame({
            "target": target,
            "task_id": unseen.task_id.astype(str),
            "matrix_id": unseen.matrix_id.astype(str),
            "block_size": pd.to_numeric(unseen.block_size).astype(int),
            "actual_ms": unseen_y,
            "predicted_ms": prediction,
        })
        task_frames.append(frame)
        task_summary.append({"target": target, **contract.metric(unseen_y, prediction)})
        for matrix_id, group in frame.groupby("matrix_id", sort=True):
            per_matrix.append({
                "target": target,
                "matrix_id": matrix_id,
                **contract.metric(group.actual_ms, group.predicted_ms),
            })
        for (matrix_id, block_size), group in frame.groupby(
            ["matrix_id", "block_size"], sort=True
        ):
            per_block.append({
                "target": target,
                "matrix_id": matrix_id,
                "block_size": int(block_size),
                **contract.metric(group.actual_ms, group.predicted_ms),
            })
        manifest["targets"][target] = {
            "train_rows": int(len(train)),
            "train_matrices": int(train.matrix_id.astype(str).nunique()),
            "unseen_rows": int(len(unseen)),
            "unseen_matrices": sorted(unseen.matrix_id.astype(str).unique()),
            "features": contract.FEATURES[target],
            "fit_seconds": fit_seconds,
            "model": str(model_path.resolve()),
            "model_sha256": sha256(model_path),
        }
        print(target, task_summary[-1], flush=True)

    tasks = pd.concat(task_frames, ignore_index=True)
    tasks.to_csv(output / "task_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(task_summary).to_csv(output / "task_metrics_by_kernel.csv", index=False)
    pd.DataFrame(per_matrix).to_csv(output / "task_metrics_by_kernel_matrix.csv", index=False)
    pd.DataFrame(per_block).to_csv(output / "task_metrics_by_kernel_matrix_block.csv", index=False)

    schemes = tasks.groupby(["matrix_id", "block_size"], as_index=False).agg(
        actual_ms=("actual_ms", "sum"),
        predicted_ms=("predicted_ms", "sum"),
        tasks=("task_id", "size"),
    )
    schemes.to_csv(output / "scheme_predictions.csv", index=False)
    scheme_metrics = [{"scope": "all", **contract.metric(schemes.actual_ms, schemes.predicted_ms)}]
    selections = []
    for matrix_id, group in schemes.groupby("matrix_id", sort=True):
        scheme_metrics.append({
            "scope": matrix_id,
            **contract.metric(group.actual_ms, group.predicted_ms),
        })
        oracle = group.loc[group.actual_ms.idxmin()]
        selected = group.loc[group.predicted_ms.idxmin()]
        selections.append({
            "matrix_id": matrix_id,
            "oracle_block_size": int(oracle.block_size),
            "oracle_actual_ms": float(oracle.actual_ms),
            "predicted_best_block_size": int(selected.block_size),
            "selected_actual_ms": float(selected.actual_ms),
            "selected_predicted_ms": float(selected.predicted_ms),
            "regret_percent": float((selected.actual_ms / oracle.actual_ms - 1) * 100),
            "spearman": float(group.actual_ms.rank().corr(group.predicted_ms.rank())),
        })
    pd.DataFrame(scheme_metrics).to_csv(output / "scheme_metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(output / "scheme_selection.csv", index=False)
    manifest["wall_seconds"] = time.time() - started
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", output, flush=True)


if __name__ == "__main__":
    main()
