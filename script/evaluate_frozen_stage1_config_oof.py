#!/usr/bin/env python3
"""Evaluate each frozen stage-1 hyperparameter configuration by shared grouped OOF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from compare_h100_rtx_xgboost_matched_80_20 import KEY, PARAMS, TARGETS, read_target
from run_v4_1_flops_bytes_ablation import metric, values
from run_v4_2_paper_order_max_first_ablation import groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h100", required=True); p.add_argument("--rtx", required=True)
    p.add_argument("--stage1", required=True)
    a = p.parse_args(); stage1 = Path(a.stage1)
    configuration = json.loads((stage1 / "frozen_stage1_configuration.json").read_text())
    fold = pd.read_csv(stage1 / "shared_matrix_folds.csv")
    fold_map = dict(zip(fold.matrix_id.astype(str), fold.cv_fold.astype(int)))
    roots = {"H100": Path(a.h100), "RTX_PRO_6000": Path(a.rtx)}
    summaries, folds, matrices, predictions = [], [], [], []
    for target_index, target in enumerate(TARGETS):
        for platform, root in roots.items():
            setting = configuration["platforms"][platform][target]
            features = setting["features"]; _, derived = groups(target)
            data = read_target(root, target, features, derived)
            data["cv_fold"] = data.matrix_id.astype(str).map(fold_map)
            if data.cv_fold.isna().any() or data.cv_fold.nunique() != 5:
                raise RuntimeError(f"{target}/{platform}: fold failure")
            x = pd.DataFrame(values(data, features), columns=features)
            y = data.label_time_ms.to_numpy(float); fid = data.cv_fold.to_numpy(int)
            oof = np.empty(len(data))
            for cv_fold in range(1, 6):
                train, test = fid != cv_fold, fid == cv_fold
                params = dict(PARAMS)
                params.update(max_depth=int(setting["selected_depth"]),
                              n_estimators=int(setting["selected_trees"]))
                model = XGBRegressor(**params,
                    random_state=configuration["seed"] + target_index * 100 + cv_fold)
                model.fit(x.loc[train], np.log1p(y[train]), verbose=False)
                oof[test] = np.maximum(np.expm1(model.predict(x.loc[test])), 0)
                folds.append({"target": target, "platform": platform, "fold": cv_fold,
                              **metric(y[test], oof[test])})
            pred = pd.DataFrame({"target": target, "platform": platform,
                "task_id": data.task_id, "matrix_id": data.matrix_id,
                "cv_fold": fid, "actual_ms": y, "predicted_ms": oof})
            predictions.append(pred); per_matrix = []
            for matrix_id, part in pred.groupby("matrix_id", sort=True):
                per_matrix.append({"target": target, "platform": platform,
                    "matrix_id": matrix_id, **metric(part.actual_ms, part.predicted_ms)})
            matrix_frame = pd.DataFrame(per_matrix); matrices.append(matrix_frame)
            summaries.append({"target": target, "platform": platform,
                "max_depth": setting["selected_depth"], "n_estimators": setting["selected_trees"],
                **metric(y, oof), "macro_MAPE": float(matrix_frame.MAPE.mean()),
                "macro_MdAPE": float(matrix_frame.MdAPE.mean()),
                "worst_matrix_MAPE": float(matrix_frame.MAPE.max())})
            print(target, platform, f"MAPE={summaries[-1]['MAPE']:.3f}", flush=True)
    pd.DataFrame(summaries).to_csv(stage1 / "frozen_config_oof_metrics.csv", index=False)
    pd.DataFrame(folds).to_csv(stage1 / "frozen_config_fold_metrics.csv", index=False)
    pd.concat(matrices, ignore_index=True).to_csv(stage1 / "frozen_config_per_matrix_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        stage1 / "frozen_config_oof_predictions.csv.gz", index=False, compression="gzip")


if __name__ == "__main__": main()
