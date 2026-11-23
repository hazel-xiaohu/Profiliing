#!/usr/bin/env python3
"""Tune only XGBoost depth and tree count on the matched H100/RTX split."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from compare_h100_rtx_xgboost_matched_80_20 import KEY, PARAMS, TARGETS, read_target
from run_v4_1_flops_bytes_ablation import metric, values
from run_v4_2_paper_order_max_first_ablation import groups

SEED = 20260807
DEPTHS = (4, 6, 8)
MAX_ESTIMATORS = 6000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h100", required=True)
    parser.add_argument("--rtx", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    roots = {"H100": Path(args.h100), "RTX_PRO_6000": Path(args.rtx)}
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output must be empty: {output}")
    (output / "models").mkdir(parents=True)
    membership = pd.read_csv(args.split, low_memory=False)
    trials, finals = [], []
    manifest = {
        "schema": "matched_80_20_depth_early_stopping_v1", "seed": SEED,
        "outer_test": "frozen common-task 20%; never used for tuning",
        "inner_validation": "20% of the frozen outer-training membership",
        "depths": DEPTHS, "max_estimators": MAX_ESTIMATORS,
        "early_stopping_rounds": 100, "selection_metric": "inner-validation MAPE",
        "early_stopping_metric": "RMSE on log1p(time)", "targets": {},
    }
    started = time.perf_counter()
    for target_index, target in enumerate(TARGETS):
        feature_groups, derived = groups(target)
        features = sum((feature_groups[name] for name in "ABCD"), [])
        split = membership[membership.target.eq(target)][KEY + ["membership"]]
        manifest["targets"][target] = {"features": features, "platforms": {}}
        for platform, root in roots.items():
            data = read_target(root, target, features, derived).merge(
                split, on=KEY, how="inner", validate="one_to_one",
            ).sort_values(KEY).reset_index(drop=True)
            outer_train = np.flatnonzero(data.membership.eq("train").to_numpy())
            outer_test = np.flatnonzero(data.membership.eq("test").to_numpy())
            inner_train, inner_valid = train_test_split(
                outer_train, test_size=.2, random_state=SEED + 1, shuffle=True,
            )
            x = pd.DataFrame(values(data, features), columns=features)
            y = data.label_time_ms.to_numpy(float)
            candidates = []
            for depth in DEPTHS:
                params = dict(PARAMS)
                params.update(n_estimators=MAX_ESTIMATORS, max_depth=depth,
                              early_stopping_rounds=100, eval_metric="rmse")
                model = XGBRegressor(**params, random_state=SEED + target_index * 10)
                began = time.perf_counter()
                model.fit(x.iloc[inner_train], np.log1p(y[inner_train]),
                          eval_set=[(x.iloc[inner_valid], np.log1p(y[inner_valid]))],
                          verbose=False)
                pred = np.maximum(np.expm1(model.predict(x.iloc[inner_valid])), 0)
                row = {
                    "target": target, "platform": platform, "max_depth": depth,
                    "best_iteration": int(model.best_iteration),
                    "selected_trees": int(model.best_iteration) + 1,
                    "fit_seconds": time.perf_counter() - began,
                    **metric(y[inner_valid], pred),
                }
                trials.append(row); candidates.append(row)
                print(target, platform, depth, row["selected_trees"],
                      f"valid_MAPE={row['MAPE']:.4f}", flush=True)
            best = min(candidates, key=lambda row: (row["MAPE"], row["max_depth"], row["selected_trees"]))
            params = dict(PARAMS)
            params.update(n_estimators=best["selected_trees"], max_depth=best["max_depth"])
            final = XGBRegressor(**params, random_state=SEED + target_index * 10)
            began = time.perf_counter()
            final.fit(x.iloc[outer_train], np.log1p(y[outer_train]), verbose=False)
            pred = np.maximum(np.expm1(final.predict(x.iloc[outer_test])), 0)
            final_row = {
                "target": target, "platform": platform,
                "selected_depth": best["max_depth"],
                "selected_trees": best["selected_trees"],
                "inner_validation_MAPE": best["MAPE"],
                "n_outer_train": len(outer_train), "n_outer_test": len(outer_test),
                "fit_seconds": time.perf_counter() - began,
                **metric(y[outer_test], pred),
            }
            finals.append(final_row)
            final.save_model(output / "models" / f"{target}.{platform}.json")
            manifest["targets"][target]["platforms"][platform] = final_row
    pd.DataFrame(trials).to_csv(output / "inner_validation_trials.csv", index=False)
    pd.DataFrame(finals).to_csv(output / "outer_test_metrics.csv", index=False)
    manifest["wall_seconds"] = time.perf_counter() - started
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
