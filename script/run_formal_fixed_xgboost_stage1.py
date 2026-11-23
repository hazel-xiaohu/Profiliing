#!/usr/bin/env python3
"""Formal stage 1 with one fixed XGBoost configuration on both platforms/kernels."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from compare_h100_rtx_xgboost_matched_80_20 import KEY, TARGETS, read_target
from run_v4_1_flops_bytes_ablation import metric, values
from run_v4_2_paper_order_max_first_ablation import groups

SEED = 20260811
STAGES = ("A", "A+B", "A+B+C", "A+B+C+D")
PLATFORMS = ("H100", "RTX_PRO_6000")
FIXED_PARAMS = dict(
    objective="reg:squarederror", n_estimators=500, learning_rate=.05,
    max_depth=6, min_child_weight=20, subsample=.8, colsample_bytree=.8,
    reg_alpha=0., reg_lambda=1., gamma=0., tree_method="hist", max_bin=256,
    n_jobs=8,
)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h100", required=True); p.add_argument("--rtx", required=True)
    p.add_argument("--fold-map", required=True); p.add_argument("--output", required=True)
    return p.parse_args()


def matrix_metrics(frame):
    rows = []
    for matrix_id, part in frame.groupby("matrix_id", sort=True):
        rows.append({"matrix_id": matrix_id, **metric(part.actual_ms, part.predicted_ms)})
    table = pd.DataFrame(rows)
    return table, {"macro_MAPE": float(table.MAPE.mean()),
                   "macro_MdAPE": float(table.MdAPE.mean()),
                   "worst_matrix_MAPE": float(table.MAPE.max())}


def new_model(target_index, fold=0):
    return XGBRegressor(**FIXED_PARAMS, random_state=SEED + target_index * 100 + fold)


def main():
    a = arguments(); out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output must be empty: {out}")
    (out / "frozen_evaluation_models").mkdir(parents=True)
    shutil.copy2(a.fold_map, out / "shared_matrix_folds.csv")
    fold_frame = pd.read_csv(a.fold_map)
    counts = fold_frame.groupby("cv_fold").size().sort_index().tolist()
    if counts != [5, 5, 5, 5, 4]:
        raise RuntimeError(f"required fold capacities 5/5/5/5/4, got {counts}")
    fold_map = dict(zip(fold_frame.matrix_id.astype(str), fold_frame.cv_fold.astype(int)))
    roots = {"H100": Path(a.h100), "RTX_PRO_6000": Path(a.rtx)}
    summaries, fold_rows, matrix_rows, predictions = [], [], [], []
    manifest = {
        "schema": "formal_fixed_xgboost_stage1_v1", "seed": SEED,
        "protocol": "shared matrix-grouped five-fold; no hyperparameter tuning; no early stopping",
        "fixed_xgboost_params": FIXED_PARAMS, "feature_set": "A+B+C+D",
        "fold_capacities": counts, "platforms": {},
    }
    started = time.perf_counter()
    for target_index, target in enumerate(TARGETS):
        feature_groups, derived = groups(target)
        cumulative, stage_features = [], {}
        for index, stage in enumerate(STAGES):
            cumulative += feature_groups[chr(ord("A") + index)]
            stage_features[stage] = list(cumulative)
        features = stage_features["A+B+C+D"]
        h = read_target(roots["H100"], target, features, derived)
        r = read_target(roots["RTX_PRO_6000"], target, features, derived)
        paired = h.merge(r, on=KEY, suffixes=("_h100", "_rtx"), validate="one_to_one")
        paired["cv_fold"] = paired.matrix_id.astype(str).map(fold_map)
        if paired.cv_fold.isna().any() or paired.cv_fold.nunique() != 5:
            raise RuntimeError(f"{target}: invalid fold coverage")
        for feature in features:
            left = pd.to_numeric(paired[f"{feature}_h100"], errors="coerce").fillna(0)
            right = pd.to_numeric(paired[f"{feature}_rtx"], errors="coerce").fillna(0)
            if not np.allclose(left, right, rtol=1e-12, atol=1e-12):
                raise RuntimeError(f"{target}: feature mismatch: {feature}")
        canonical = paired.rename(columns={f"{f}_h100": f for f in features})
        fid = paired.cv_fold.to_numpy(int)
        for platform in PLATFORMS:
            ycol = "label_time_ms_h100" if platform == "H100" else "label_time_ms_rtx"
            y = paired[ycol].to_numpy(float)
            manifest["platforms"].setdefault(platform, {})[target] = {
                "features": features, "rows": len(y), "matrices": 24,
                "fixed_xgboost_params": FIXED_PARAMS,
            }
            for stage in STAGES:
                fs = stage_features[stage]
                x = pd.DataFrame(values(canonical, fs), columns=fs)
                oof = np.empty(len(y))
                for fold in range(1, 6):
                    train, test = fid != fold, fid == fold
                    model = new_model(target_index, fold)
                    model.fit(x.loc[train], np.log1p(y[train]), verbose=False)
                    oof[test] = np.maximum(np.expm1(model.predict(x.loc[test])), 0)
                    fold_rows.append({"target": target, "platform": platform,
                        "stage": stage, "fold": fold, "features": len(fs),
                        **metric(y[test], oof[test])})
                frame = pd.DataFrame({"matrix_id": paired.matrix_id,
                                      "actual_ms": y, "predicted_ms": oof})
                per_matrix, macro = matrix_metrics(frame)
                summaries.append({"target": target, "platform": platform,
                    "stage": stage, "features": len(fs), **metric(y, oof), **macro})
                if stage == "A+B+C+D":
                    pred = pd.DataFrame({"target": target, "platform": platform,
                        "task_id": paired.task_id, "matrix_id": paired.matrix_id,
                        "cv_fold": fid, "actual_ms": y, "predicted_ms": oof})
                    predictions.append(pred)
                    per_matrix.insert(0, "platform", platform)
                    per_matrix.insert(0, "target", target)
                    matrix_rows.append(per_matrix)
                    full = new_model(target_index)
                    full.fit(x, np.log1p(y), verbose=False)
                    full.save_model(out / "frozen_evaluation_models" / f"{target}.{platform}.json")
                    manifest["platforms"][platform][target]["oof_metrics"] = {
                        **metric(y, oof), **macro,
                    }
                    print(target, platform, f"OOF_MAPE={metric(y, oof)['MAPE']:.3f}", flush=True)
    pd.DataFrame(summaries).to_csv(out / "ablation_and_fixed_oof_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    pd.concat(matrix_rows, ignore_index=True).to_csv(out / "per_matrix_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        out / "oof_predictions.csv.gz", index=False, compression="gzip")
    manifest["wall_seconds"] = time.perf_counter() - started
    (out / "frozen_stage1_configuration.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
