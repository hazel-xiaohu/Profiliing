#!/usr/bin/env python3
"""Audit and train pre-kernel task overhead models with shared matrix folds."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

TARGETS = ("GETRF", "GESSM", "TSTRF", "SSSSM_dense", "SSSSM_sparse")
KEY = ["task_id", "matrix_id", "kernel_type", "level", "block_row",
       "block_col", "update_k", "rank", "block_size", "process_rows",
       "process_cols"]
FEATURES = ["replay_main_dimension", "replay_block_rows", "replay_block_cols",
            "replay_operation_count_v2", "replay_base_bytes_est",
            "replay_aspect_ratio", "replay_min_dimension", "replay_max_dimension"]


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--fold-map", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20260803)
    return p.parse_args()


def model(seed):
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=.05, max_iter=160,
        max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=.1,
        random_state=seed)


def metrics(y, p):
    y, p = np.asarray(y, float), np.maximum(np.asarray(p, float), 0)
    ape = np.abs(p-y) / np.maximum(y, 1e-6) * 100
    return {
        "rows": int(len(y)), "MAPE_percent": float(ape.mean()),
        "MdAPE_percent": float(np.median(ape)),
        "P95_APE_percent": float(np.percentile(ape, 95)),
        "WAPE_percent": float(np.abs(p-y).sum()/np.maximum(y.sum(), 1e-12)*100),
        "R2": float(r2_score(y, p)),
        "Spearman": float(spearmanr(y, p).statistic),
    }


def main():
    a = args(); out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output must be empty: {out}")
    (out / "models").mkdir(parents=True)
    (out / "predictions").mkdir()
    cols = KEY + FEATURES + ["forced_ssssm_variant", "time_total_ms",
                              "time_kernel_ms", "hardware_id"]
    raw = pd.read_csv(a.input, usecols=lambda c: c in set(cols), low_memory=False)
    if "hardware_id" not in raw:
        raw["hardware_id"] = "H100_SXM_80GB"
    for c in FEATURES + ["time_total_ms", "time_kernel_ms"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw[np.isfinite(raw.time_total_ms) & np.isfinite(raw.time_kernel_ms)
              & (raw.time_total_ms > 0) & (raw.time_kernel_ms > 0)].copy()
    raw["overhead_ms"] = np.maximum(0, raw.time_total_ms-raw.time_kernel_ms)
    raw["target"] = np.where(raw.kernel_type.eq("SSSSM"),
                             "SSSSM_"+raw.forced_ssssm_variant.astype(str),
                             raw.kernel_type)
    folds = pd.read_csv(a.fold_map)
    fold_map = dict(zip(folds.matrix_id.astype(str), folds.cv_fold.astype(int)))
    summaries, fold_rows, stability = [], [], []

    for target_index, target in enumerate(TARGETS):
        d = raw[raw.target.eq(target)].copy()
        if not target.startswith("SSSSM_"):
            repeated = d.groupby(KEY, dropna=False).overhead_ms.agg(
                ["count", "mean", "std", "min", "max"]).reset_index()
            repeated["cv"] = repeated["std"] / repeated["mean"].clip(lower=1e-12)
            stability.append({
                "target": target, "logical_tasks": int(len(repeated)),
                "median_repeat_cv": float(repeated.cv.median()),
                "P90_repeat_cv": float(repeated.cv.quantile(.90)),
                "P95_repeat_cv": float(repeated.cv.quantile(.95)),
                "fraction_cv_above_50pct": float((repeated.cv > .5).mean()),
            })
            aggregation = {"overhead_ms": "median", "time_total_ms": "median",
                           "time_kernel_ms": "median", "hardware_id": "first"}
            aggregation.update({f: "first" for f in FEATURES})
            d = d.groupby(KEY, as_index=False, dropna=False).agg(aggregation)
        d["cv_fold"] = d.matrix_id.astype(str).map(fold_map)
        if d.cv_fold.isna().any():
            raise RuntimeError(f"{target}: matrix missing from shared fold map")
        x = d[FEATURES].to_numpy(float); y = d.overhead_ms.to_numpy(float)
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y) & (y >= 0)
        d, x, y = d.loc[valid].reset_index(drop=True), x[valid], y[valid]
        fold_id = d.cv_fold.to_numpy(int)
        pred, median_pred = np.empty(len(y)), np.empty(len(y))
        for fold in sorted(set(fold_id)):
            tr, te = fold_id != fold, fold_id == fold
            m = model(a.seed + target_index*10 + fold)
            m.fit(x[tr], np.log1p(y[tr]))
            pred[te] = np.maximum(0, np.expm1(m.predict(x[te])))
            median_pred[te] = np.median(y[tr])
            fold_rows.append({"target": target, "fold": int(fold),
                              "model": "HGB_log1p", **metrics(y[te], pred[te])})
        hgb_metric, base_metric = metrics(y, pred), metrics(y, median_pred)
        summaries.append({"target": target, "model": "HGB_log1p", **hgb_metric})
        summaries.append({"target": target, "model": "train_median", **base_metric})
        final = model(a.seed + target_index); final.fit(x, np.log1p(y))
        joblib.dump({"schema": "task_overhead_v1", "target": target,
                     "definition": "max(0,time_total_ms-time_kernel_ms)",
                     "features": FEATURES, "transform": "log1p", "model": final},
                    out / "models" / f"{target}.overhead_hgb_v1.joblib")
        pd.DataFrame({"task_id": d.task_id, "matrix_id": d.matrix_id,
                      "block_size": d.block_size, "cv_fold": fold_id,
                      "actual_overhead_ms": y, "predicted_overhead_ms": pred,
                      "median_baseline_ms": median_pred}).to_csv(
            out / "predictions" / f"{target}.oof.csv", index=False)

    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(stability).to_csv(out / "repeat_stability.csv", index=False)
    decision = {}
    for target in TARGETS:
        h = summary[(summary.target == target) & (summary.model == "HGB_log1p")].iloc[0]
        b = summary[(summary.target == target) & (summary.model == "train_median")].iloc[0]
        decision[target] = {
            "hgb_WAPE_percent": float(h.WAPE_percent),
            "median_WAPE_percent": float(b.WAPE_percent),
            "relative_WAPE_reduction": float(1-h.WAPE_percent/b.WAPE_percent),
            "hgb_Spearman": float(h.Spearman), "hgb_R2": float(h.R2),
            "stable_enough_for_candidate_evaluation": bool(
                h.WAPE_percent < b.WAPE_percent and h.Spearman >= .7 and h.R2 >= .5),
        }
    manifest = {"schema": "task_overhead_model_v1", "input": str(Path(a.input).resolve()),
                "fold_map": str(Path(a.fold_map).resolve()), "seed": a.seed,
                "features": FEATURES, "label": "max(0,total-kernel)",
                "decision_rule": "HGB WAPE < median WAPE, Spearman >=0.7, R2 >=0.5",
                "targets": decision,
                "deployment_status": "offline diagnostic; not integrated into Partition"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(summary.to_string(index=False)); print(json.dumps(decision, indent=2))


if __name__ == "__main__": main()
