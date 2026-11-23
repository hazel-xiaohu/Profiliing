#!/usr/bin/env python3
"""Matched-task H100/RTX XGBoost comparison with an identical 80:20 split."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from run_v4_1_flops_bytes_ablation import metric, values
from run_v4_2_paper_order_max_first_ablation import groups


TARGETS = ("GETRF", "GESSM", "TSTRF", "SSSSM_dense")
SEED = 20260807
KEY = [
    "task_id", "matrix_id", "kernel_type", "level", "block_row", "block_col",
    "update_k", "rank", "block_size", "process_rows", "process_cols",
]
PARAMS = dict(
    objective="reg:squarederror", n_estimators=500, learning_rate=.05,
    max_depth=8, min_child_weight=20, subsample=.8, colsample_bytree=.8,
    reg_alpha=0., reg_lambda=1., gamma=0., tree_method="hist", max_bin=256,
    n_jobs=8,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h100", required=True)
    parser.add_argument("--rtx", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_target(root: Path, target: str, features: list[str], derived):
    raw = sorted(set(features + [x for pair in derived.values() for x in pair]))
    required = set(KEY + ["label_time_ms"] + raw)
    data = pd.read_csv(
        root / f"{target}.model_ready_v4.csv",
        usecols=lambda column: column in required,
        low_memory=False,
    )
    for new, (mean, cv) in derived.items():
        data[new] = (
            pd.to_numeric(data[mean], errors="coerce").fillna(0)
            * pd.to_numeric(data[cv], errors="coerce").fillna(0)
        )
    data["label_time_ms"] = pd.to_numeric(data["label_time_ms"], errors="coerce")
    data = data[np.isfinite(data.label_time_ms) & data.label_time_ms.gt(0)].copy()
    if data.duplicated(KEY).any():
        raise RuntimeError(f"{target}: duplicate task keys")
    return data[KEY + ["label_time_ms"] + features]


def normalized_importance(model, kind, features):
    score = model.get_booster().get_score(importance_type=kind)
    result = np.array([score.get(name, 0.0) for name in features], dtype=float)
    total = result.sum()
    return result / total if total > 0 else result


def predict_ms(model, matrix):
    return np.maximum(np.expm1(model.predict(matrix)), 0.0)


def main():
    args = arguments()
    hroot, rroot, output = Path(args.h100), Path(args.rtx), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output must be empty: {output}")
    (output / "models").mkdir(parents=True)
    started = time.perf_counter()
    metrics, contributions, coefficients = [], [], []
    prediction_frames, response_frames, split_frames, ratio_summaries = [], [], [], []
    manifest = {
        "schema": "matched_h100_rtx_xgboost_80_20_v1",
        "seed": SEED,
        "split": "identical train/test membership after sorting the full common task key",
        "xgboost_version": xgb.__version__,
        "xgboost_params": PARAMS,
        "primary_features": "A+B+C+D",
        "auxiliary_model": "Ridge(alpha=1) on standardized A-D features and log1p(time)",
        "targets": {},
    }

    for target_index, target in enumerate(TARGETS):
        feature_groups, derived = groups(target)
        features = sum((feature_groups[name] for name in "ABCD"), [])
        h100 = read_target(hroot, target, features, derived)
        rtx = read_target(rroot, target, features, derived)
        paired = h100.merge(
            rtx, on=KEY, how="inner", suffixes=("_h100", "_rtx"),
            validate="one_to_one", sort=False,
        ).sort_values(KEY).reset_index(drop=True)
        if paired.empty:
            raise RuntimeError(f"{target}: no common tasks")
        mismatches = {}
        for feature in features:
            left = pd.to_numeric(paired[f"{feature}_h100"], errors="coerce").fillna(0).to_numpy(float)
            right = pd.to_numeric(paired[f"{feature}_rtx"], errors="coerce").fillna(0).to_numpy(float)
            count = int(np.sum(~np.isclose(left, right, rtol=1e-12, atol=1e-12)))
            if count:
                mismatches[feature] = count
        if mismatches:
            raise RuntimeError(f"{target}: cross-hardware feature mismatch: {mismatches}")
        canonical = paired.rename(columns={f"{f}_h100": f for f in features})
        xmat = values(canonical, features)
        xframe = pd.DataFrame(xmat, columns=features)
        indices = np.arange(len(paired))
        train, test = train_test_split(
            indices, test_size=.2, random_state=SEED, shuffle=True,
        )
        membership = np.full(len(paired), "train", dtype=object)
        membership[test] = "test"
        split_frame = paired[KEY].copy()
        split_frame.insert(0, "target", target)
        split_frame["membership"] = membership
        split_frames.append(split_frame)
        platform_models = {}
        platform_predictions = {}
        for platform_index, platform in enumerate(("H100", "RTX_PRO_6000")):
            label_column = "label_time_ms_h100" if platform == "H100" else "label_time_ms_rtx"
            y = paired[label_column].to_numpy(float)
            model = XGBRegressor(
                **PARAMS, random_state=SEED + target_index * 10,
            )
            fit_started = time.perf_counter()
            model.fit(xframe.iloc[train], np.log1p(y[train]), verbose=False)
            fit_seconds = time.perf_counter() - fit_started
            prediction = predict_ms(model, xframe.iloc[test])
            platform_models[platform] = model
            platform_predictions[platform] = prediction
            row = {
                "target": target, "platform": platform,
                "features": len(features), "n_common": len(paired),
                "n_train": len(train), "n_test": len(test),
                "fit_seconds": fit_seconds, **metric(y[test], prediction),
            }
            metrics.append(row)
            model.save_model(output / "models" / f"{target}.{platform}.json")
            booster = model.get_booster()
            shap = booster.predict(
                xgb.DMatrix(xframe.iloc[test]), pred_contribs=True,
            )[:, :-1]
            shap_abs = np.mean(np.abs(shap), axis=0)
            shap_norm = shap_abs / max(shap_abs.sum(), 1e-30)
            gain = normalized_importance(model, "gain", features)
            cover = normalized_importance(model, "cover", features)
            weight = normalized_importance(model, "weight", features)
            for index, feature in enumerate(features):
                contributions.append({
                    "target": target, "platform": platform, "feature": feature,
                    "gain_normalized": gain[index], "cover_normalized": cover[index],
                    "weight_normalized": weight[index],
                    "mean_abs_shap": shap_abs[index], "shap_normalized": shap_norm[index],
                })
            scaler = StandardScaler().fit(xmat[train])
            ridge = Ridge(alpha=1.0).fit(
                scaler.transform(xmat[train]), np.log1p(y[train]),
            )
            for feature, coefficient in zip(features, ridge.coef_):
                coefficients.append({
                    "target": target, "platform": platform, "feature": feature,
                    "standardized_log_time_coefficient": coefficient,
                })
        prediction_frame = paired.loc[test, KEY].reset_index(drop=True)
        prediction_frame.insert(0, "target", target)
        prediction_frame["actual_h100_ms"] = paired.loc[test, "label_time_ms_h100"].to_numpy()
        prediction_frame["actual_rtx_ms"] = paired.loc[test, "label_time_ms_rtx"].to_numpy()
        prediction_frame["predicted_h100_ms"] = platform_predictions["H100"]
        prediction_frame["predicted_rtx_ms"] = platform_predictions["RTX_PRO_6000"]
        prediction_frame["actual_rtx_over_h100"] = (
            prediction_frame.actual_rtx_ms / prediction_frame.actual_h100_ms
        )
        prediction_frame["predicted_rtx_over_h100"] = (
            prediction_frame.predicted_rtx_ms
            / prediction_frame.predicted_h100_ms.clip(lower=1e-12)
        )
        prediction_frames.append(prediction_frame)
        for kind in ("actual", "predicted"):
            ratio = prediction_frame[f"{kind}_rtx_over_h100"].to_numpy(float)
            ratio_summaries.append({
                "target": target, "ratio": f"{kind}_rtx_over_h100",
                **{f"q{int(q*100):02d}": float(np.quantile(ratio, q))
                   for q in (0, .05, .25, .5, .75, .95, 1)},
                "mean": float(np.mean(ratio)),
            })
        sample = test if len(test) <= 5000 else np.random.default_rng(SEED).choice(
            test, 5000, replace=False,
        )
        for feature_index, feature in enumerate(features):
            grid = np.unique(np.quantile(xmat[train, feature_index], np.linspace(.05, .95, 19)))
            for value in grid:
                modified = xmat[sample].copy()
                modified[:, feature_index] = value
                modified_frame = pd.DataFrame(modified, columns=features)
                hp = predict_ms(platform_models["H100"], modified_frame)
                rp = predict_ms(platform_models["RTX_PRO_6000"], modified_frame)
                ratio = rp / np.maximum(hp, 1e-12)
                response_frames.append(pd.DataFrame([{
                    "target": target, "feature": feature, "feature_value": value,
                    "sample_count": len(sample),
                    "h100_mean_ms": np.mean(hp), "h100_median_ms": np.median(hp),
                    "h100_q25_ms": np.quantile(hp, .25), "h100_q75_ms": np.quantile(hp, .75),
                    "rtx_mean_ms": np.mean(rp), "rtx_median_ms": np.median(rp),
                    "rtx_q25_ms": np.quantile(rp, .25), "rtx_q75_ms": np.quantile(rp, .75),
                    "ratio_mean": np.mean(ratio), "ratio_median": np.median(ratio),
                }]))
        manifest["targets"][target] = {
            "h100_rows": len(h100), "rtx_rows": len(rtx),
            "common_rows": len(paired), "n_train": len(train), "n_test": len(test),
            "feature_count": len(features), "features": features,
            "feature_mismatches": mismatches,
        }
        print(f"{target}: common={len(paired)} train={len(train)} test={len(test)}", flush=True)

    metric_frame = pd.DataFrame(metrics)
    contribution_frame = pd.DataFrame(contributions)
    coefficient_frame = pd.DataFrame(coefficients)
    metric_frame.to_csv(output / "matched_test_metrics.csv", index=False)
    contribution_frame.to_csv(output / "feature_contributions.csv", index=False)
    coefficient_frame.to_csv(output / "auxiliary_ridge_coefficients.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output / "matched_test_predictions.csv.gz", index=False, compression="gzip",
    )
    pd.concat(response_frames, ignore_index=True).to_csv(
        output / "response_curves.csv.gz", index=False, compression="gzip",
    )
    pd.concat(split_frames, ignore_index=True).to_csv(
        output / "common_task_split.csv.gz", index=False, compression="gzip",
    )
    pd.DataFrame(ratio_summaries).to_csv(output / "prediction_ratio_summary.csv", index=False)
    coefficient_pivot = coefficient_frame.pivot(
        index=["target", "feature"], columns="platform",
        values="standardized_log_time_coefficient",
    ).reset_index()
    coefficient_pivot["RTX_minus_H100"] = coefficient_pivot.RTX_PRO_6000 - coefficient_pivot.H100
    coefficient_pivot.to_csv(output / "auxiliary_ridge_coefficient_comparison.csv", index=False)
    contribution_pivot = contribution_frame.pivot(
        index=["target", "feature"], columns="platform", values="shap_normalized",
    ).reset_index()
    contribution_pivot["RTX_minus_H100"] = contribution_pivot.RTX_PRO_6000 - contribution_pivot.H100
    contribution_pivot.to_csv(output / "shap_contribution_comparison.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=False)
    for axis, target in zip(axes.flat, TARGETS):
        subset = metric_frame[metric_frame.target.eq(target)]
        axis.bar(subset.platform, subset.MAPE)
        axis.set_title(target.replace("_", "-"))
        axis.set_ylabel("Test MAPE (%)")
        axis.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(output / "matched_test_mape.pdf")
    fig.savefig(output / "matched_test_mape.png", dpi=300)
    plt.close(fig)
    manifest["wall_seconds"] = time.perf_counter() - started
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(metric_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
