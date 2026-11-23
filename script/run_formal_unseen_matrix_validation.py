#!/usr/bin/env python3
"""Stage 2: evaluate frozen models on strictly unseen matrices without refitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from compare_h100_rtx_xgboost_matched_80_20 import TARGETS, read_target
from run_v4_1_flops_bytes_ablation import metric, values
from run_v4_2_paper_order_max_first_ablation import groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, help="External model-ready v4.1 directory")
    p.add_argument("--platform", required=True, choices=("H100", "RTX_PRO_6000"))
    p.add_argument("--stage1", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    source, stage1, output = Path(a.input_dir), Path(a.stage1), Path(a.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output must be empty: {output}")
    output.mkdir(parents=True)
    configuration = json.loads((stage1 / "frozen_stage1_configuration.json").read_text())
    development = set(pd.read_csv(stage1 / "shared_matrix_folds.csv").matrix_id.astype(str))
    summaries, matrices, predictions = [], [], []
    manifest = {
        "schema": "formal_unseen_matrix_validation_v1", "platform": a.platform,
        "development_matrices": sorted(development), "targets": {},
        "rules": ["no fitting", "no early stopping", "zero development-matrix overlap"],
    }
    for target in TARGETS:
        setting = configuration["platforms"][a.platform][target]
        features = setting["features"]
        _, derived = groups(target)
        data = read_target(source, target, features, derived)
        external = set(data.matrix_id.astype(str))
        overlap = sorted(development & external)
        if overlap:
            raise RuntimeError(f"{target}: external/development overlap: {overlap}")
        model = XGBRegressor()
        model.load_model(stage1 / "frozen_evaluation_models" / f"{target}.{a.platform}.json")
        x = pd.DataFrame(values(data, features), columns=features)
        y = data.label_time_ms.to_numpy(float)
        prediction = np.maximum(np.expm1(model.predict(x)), 0)
        frame = pd.DataFrame({
            "target": target, "task_id": data.task_id,
            "matrix_id": data.matrix_id, "actual_ms": y, "predicted_ms": prediction,
        })
        predictions.append(frame)
        per_matrix = []
        for matrix_id, part in frame.groupby("matrix_id", sort=True):
            per_matrix.append({"target": target, "matrix_id": matrix_id,
                               **metric(part.actual_ms, part.predicted_ms)})
        matrix_frame = pd.DataFrame(per_matrix)
        matrices.append(matrix_frame)
        summaries.append({
            "target": target, "external_matrices": len(external),
            **metric(y, prediction),
            "macro_MAPE": float(matrix_frame.MAPE.mean()),
            "macro_MdAPE": float(matrix_frame.MdAPE.mean()),
            "worst_matrix_MAPE": float(matrix_frame.MAPE.max()),
        })
        manifest["targets"][target] = {
            "rows": len(data), "matrices": sorted(external), "overlap": overlap,
            "features": features, "model": f"{target}.{a.platform}.json",
        }
    pd.DataFrame(summaries).to_csv(output / "unseen_task_and_macro_metrics.csv", index=False)
    pd.concat(matrices, ignore_index=True).to_csv(output / "unseen_per_matrix_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output / "unseen_predictions.csv.gz", index=False, compression="gzip")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
