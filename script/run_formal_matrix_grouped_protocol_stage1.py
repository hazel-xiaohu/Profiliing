#!/usr/bin/env python3
"""Formal stage 1: shared matrix-grouped CV, nested tuning, frozen eval models."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from compare_h100_rtx_xgboost_matched_80_20 import KEY, PARAMS, TARGETS, read_target
from run_v4_1_flops_bytes_ablation import metric, values
from run_v4_2_paper_order_max_first_ablation import groups

SEED = 20260807
DEPTHS = (4, 6, 8)
MAX_ESTIMATORS = 6000
EARLY_STOPPING_ROUNDS = 100
STAGES = ("A", "A+B", "A+B+C", "A+B+C+D")
PLATFORMS = ("H100", "RTX_PRO_6000")


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h100", required=True)
    p.add_argument("--rtx", required=True)
    p.add_argument("--fold-map", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def macro_metrics(frame):
    rows = []
    for matrix_id, part in frame.groupby("matrix_id", sort=True):
        rows.append({"matrix_id": matrix_id, **metric(part.actual_ms, part.predicted_ms)})
    result = pd.DataFrame(rows)
    return result, {
        "macro_MAPE": float(result.MAPE.mean()),
        "macro_MdAPE": float(result.MdAPE.mean()),
        "worst_matrix_MAPE": float(result.MAPE.max()),
    }


def model(depth, trees, target_index, fold=0, early=False):
    params = dict(PARAMS)
    params.update(max_depth=depth, n_estimators=trees, eval_metric="rmse")
    if early:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return XGBRegressor(**params, random_state=SEED + target_index * 100 + fold)


def main():
    a = args()
    roots = {"H100": Path(a.h100), "RTX_PRO_6000": Path(a.rtx)}
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output must be empty: {out}")
    (out / "frozen_evaluation_models").mkdir(parents=True)
    fold_source = Path(a.fold_map)
    fold_frame = pd.read_csv(fold_source)
    if set(fold_frame.columns) != {"matrix_id", "cv_fold"}:
        raise RuntimeError("fold map must contain exactly matrix_id,cv_fold")
    if fold_frame.matrix_id.duplicated().any() or set(fold_frame.cv_fold) != set(range(1, 6)):
        raise RuntimeError("invalid five-fold matrix map")
    shutil.copy2(fold_source, out / "shared_matrix_folds.csv")
    fold_map = dict(zip(fold_frame.matrix_id.astype(str), fold_frame.cv_fold.astype(int)))
    ablation_summary, ablation_folds = [], []
    tuning_trials, tuned_folds, matrix_rows, prediction_rows = [], [], [], []
    frozen = {
        "schema": "formal_matrix_grouped_protocol_stage1_v1", "seed": SEED,
        "feature_set": "A+B+C+D", "depth_candidates": DEPTHS,
        "max_estimators": MAX_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "selection": "minimum mean inner-matrix validation MAPE; median selected tree count",
        "outer_cv": "shared matrix-grouped five-fold",
        "inner_validation": "the next matrix fold cyclically; never the outer test fold",
        "platforms": {},
    }
    started = time.perf_counter()
    for target_index, target in enumerate(TARGETS):
        feature_groups, derived = groups(target)
        cumulative, stage_features = [], {}
        for stage_index, stage in enumerate(STAGES):
            cumulative += feature_groups[chr(ord("A") + stage_index)]
            stage_features[stage] = list(cumulative)
        full_features = stage_features["A+B+C+D"]
        h = read_target(roots["H100"], target, full_features, derived)
        r = read_target(roots["RTX_PRO_6000"], target, full_features, derived)
        paired = h.merge(r, on=KEY, suffixes=("_h100", "_rtx"), validate="one_to_one")
        paired["cv_fold"] = paired.matrix_id.astype(str).map(fold_map)
        if paired.cv_fold.isna().any() or paired.cv_fold.nunique() != 5:
            raise RuntimeError(f"{target}: fold coverage failure")
        for feature in full_features:
            left = pd.to_numeric(paired[f"{feature}_h100"], errors="coerce").fillna(0)
            right = pd.to_numeric(paired[f"{feature}_rtx"], errors="coerce").fillna(0)
            if not np.allclose(left, right, rtol=1e-12, atol=1e-12):
                raise RuntimeError(f"{target}: cross-platform feature mismatch: {feature}")
        canonical = paired.rename(columns={f"{f}_h100": f for f in full_features})
        fid = paired.cv_fold.to_numpy(int)
        for platform in PLATFORMS:
            ycol = "label_time_ms_h100" if platform == "H100" else "label_time_ms_rtx"
            y = paired[ycol].to_numpy(float)
            # Fixed-model cumulative feature ablation.
            for stage_index, stage in enumerate(STAGES):
                fs = stage_features[stage]
                x = pd.DataFrame(values(canonical, fs), columns=fs)
                oof = np.empty(len(y))
                for fold in range(1, 6):
                    train, test = fid != fold, fid == fold
                    m = model(8, 500, target_index, fold)
                    m.fit(x.loc[train], np.log1p(y[train]), verbose=False)
                    oof[test] = np.maximum(np.expm1(m.predict(x.loc[test])), 0)
                    ablation_folds.append({
                        "target": target, "platform": platform, "stage": stage,
                        "fold": fold, "features": len(fs), **metric(y[test], oof[test]),
                    })
                pf = pd.DataFrame({"matrix_id": paired.matrix_id, "actual_ms": y,
                                   "predicted_ms": oof})
                _, macro = macro_metrics(pf)
                ablation_summary.append({
                    "target": target, "platform": platform, "stage": stage,
                    "features": len(fs), **metric(y, oof), **macro,
                })
            # Nested grouped CV for depth/tree selection on A-D.
            x = pd.DataFrame(values(canonical, full_features), columns=full_features)
            tuned_oof = np.empty(len(y))
            platform_trials = []
            for outer_fold in range(1, 6):
                inner_fold = outer_fold % 5 + 1
                inner_train = (fid != outer_fold) & (fid != inner_fold)
                inner_valid = fid == inner_fold
                candidates = []
                for depth in DEPTHS:
                    m = model(depth, MAX_ESTIMATORS, target_index, outer_fold, early=True)
                    began = time.perf_counter()
                    m.fit(x.loc[inner_train], np.log1p(y[inner_train]),
                          eval_set=[(x.loc[inner_valid], np.log1p(y[inner_valid]))],
                          verbose=False)
                    pred = np.maximum(np.expm1(m.predict(x.loc[inner_valid])), 0)
                    row = {
                        "target": target, "platform": platform,
                        "outer_fold": outer_fold, "inner_validation_fold": inner_fold,
                        "max_depth": depth, "selected_trees": int(m.best_iteration) + 1,
                        "fit_seconds": time.perf_counter() - began,
                        **metric(y[inner_valid], pred),
                    }
                    tuning_trials.append(row); platform_trials.append(row); candidates.append(row)
                best = min(candidates, key=lambda z: (z["MAPE"], z["max_depth"], z["selected_trees"]))
                outer_train, outer_test = fid != outer_fold, fid == outer_fold
                final_fold = model(best["max_depth"], best["selected_trees"], target_index, outer_fold)
                final_fold.fit(x.loc[outer_train], np.log1p(y[outer_train]), verbose=False)
                tuned_oof[outer_test] = np.maximum(np.expm1(final_fold.predict(x.loc[outer_test])), 0)
                tuned_folds.append({
                    "target": target, "platform": platform, "outer_fold": outer_fold,
                    "selected_depth": best["max_depth"], "selected_trees": best["selected_trees"],
                    **metric(y[outer_test], tuned_oof[outer_test]),
                })
            trial_frame = pd.DataFrame(platform_trials)
            depth_summary = trial_frame.groupby("max_depth", as_index=False).agg(
                mean_inner_MAPE=("MAPE", "mean"), median_trees=("selected_trees", "median"))
            chosen = depth_summary.sort_values(["mean_inner_MAPE", "max_depth"]).iloc[0]
            chosen_depth, chosen_trees = int(chosen.max_depth), int(round(chosen.median_trees))
            pred_frame = pd.DataFrame({
                "target": target, "platform": platform, "task_id": paired.task_id,
                "matrix_id": paired.matrix_id, "cv_fold": fid,
                "actual_ms": y, "predicted_ms": tuned_oof,
            })
            prediction_rows.append(pred_frame)
            per_matrix, macro = macro_metrics(pred_frame)
            per_matrix.insert(0, "platform", platform); per_matrix.insert(0, "target", target)
            matrix_rows.append(per_matrix)
            frozen_model = model(chosen_depth, chosen_trees, target_index)
            frozen_model.fit(x, np.log1p(y), verbose=False)
            frozen_model.save_model(out / "frozen_evaluation_models" / f"{target}.{platform}.json")
            frozen["platforms"].setdefault(platform, {})[target] = {
                "features": full_features, "common_rows": len(y),
                "selected_depth": chosen_depth, "selected_trees": chosen_trees,
                "mean_inner_MAPE_by_depth": {
                    str(int(z.max_depth)): float(z.mean_inner_MAPE)
                    for z in depth_summary.itertuples()
                },
                "nested_oof_metrics": {**metric(y, tuned_oof), **macro},
            }
            print(target, platform, "depth", chosen_depth, "trees", chosen_trees,
                  f"nested_OOF_MAPE={metric(y, tuned_oof)['MAPE']:.3f}", flush=True)
    pd.DataFrame(ablation_summary).to_csv(out / "ablation_oof_summary.csv", index=False)
    pd.DataFrame(ablation_folds).to_csv(out / "ablation_fold_metrics.csv", index=False)
    pd.DataFrame(tuning_trials).to_csv(out / "nested_inner_trials.csv", index=False)
    pd.DataFrame(tuned_folds).to_csv(out / "nested_outer_fold_metrics.csv", index=False)
    pd.concat(matrix_rows, ignore_index=True).to_csv(out / "nested_per_matrix_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        out / "nested_oof_predictions.csv.gz", index=False, compression="gzip")
    frozen["wall_seconds"] = time.perf_counter() - started
    (out / "frozen_stage1_configuration.json").write_text(json.dumps(frozen, indent=2) + "\n")


if __name__ == "__main__":
    main()
