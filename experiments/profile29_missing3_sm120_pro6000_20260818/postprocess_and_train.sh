#!/usr/bin/env bash
set -euo pipefail

repo=/output/Profiliing
root="$repo/artifacts/platforms/RTX_PRO_6000_BLACKWELL/profile29_missing3_sm120_dense_n1_n8_20260818"
campaign="$root/campaign"
replay="$root/feature_replay_v4_1_reused"
matrix_list="$repo/experiments/profile29_missing3_sm120_pro6000_20260818/matrices.txt"
strict="$root/strict_merged_v4_1"
new3="$root/model_ready_v4_1_new3"
profile29="$repo/artifacts/platforms/RTX_PRO_6000_BLACKWELL/profile29_v4_2_model_ready_20260818"
training="$repo/artifacts/platforms/RTX_PRO_6000_BLACKWELL/profile29_plain_vs_matrix_balanced_abcd_20260818"
folds="$repo/artifacts/platforms/H100_SXM_80GB/profile29_hard5_v1/model_ready_v4_1_profile29/shared_matrix_folds.csv"

success_count=$(find "$campaign/raw" -name _SUCCESS.json | wc -l)
if [[ "$success_count" -ne 24 ]]; then
  echo "expected 24 successful GPU configurations, found $success_count" >&2
  exit 2
fi

if [[ ! -f "$strict/merge_audit.json" ]]; then
  python3 "$repo/script/merge_feature_replay_gpu.py" \
    --campaign "$campaign" \
    --replay "$replay" \
    --matrix-list "$matrix_list" \
    --output "$strict"
fi

if [[ ! -f "$new3/model_ready_audit_v4.json" ]]; then
  python3 "$repo/script/build_model_ready_v4.py" \
    --input "$strict/merged_tasks.csv" \
    --output "$new3"
fi

if [[ ! -f "$profile29/combine_audit.json" ]]; then
  python3 "$repo/script/combine_model_ready_v4.py" \
    --inputs \
      "$repo/artifacts/platforms/RTX_PRO_6000_BLACKWELL/profiling25_v4_2_model_ready_20260814" \
      "$repo/artifacts/platforms/RTX_PRO_6000_BLACKWELL/unseen_com_dblp_rajat29_v1/model_ready_v4_1" \
      "$new3" \
    --output "$profile29" \
    --hardware-id RTX_PRO_6000_BLACKWELL \
    --block-sizes 2048 4096 6144 8192 10240 12288 14336 16384 \
    --targets GETRF GESSM TSTRF SSSSM_dense \
    --exclude-matrices nd24k \
    --schema-version v4.1
  cp "$folds" "$profile29/shared_matrix_folds.csv"
fi

python3 - "$profile29" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
folds = pd.read_csv(root / "shared_matrix_folds.csv")
expected = set(folds.matrix_id.astype(str))
if len(expected) != 29 or folds.groupby("cv_fold").size().sort_index().to_dict() != {1: 6, 2: 6, 3: 6, 4: 6, 5: 5}:
    raise SystemExit("invalid frozen profile29 fold map")
audit = {"hardware_id": "RTX_PRO_6000_BLACKWELL", "fold_map_frozen": True, "targets": {}}
for target in ("GETRF", "GESSM", "TSTRF", "SSSSM_dense"):
    path = root / f"{target}.model_ready_v4.csv"
    data = pd.read_csv(path, usecols=["matrix_id", "hardware_id", "block_size"], low_memory=False)
    found = set(data.matrix_id.astype(str))
    if found != expected:
        raise SystemExit(f"{target}: matrix membership mismatch")
    if set(data.hardware_id.astype(str)) != {"RTX_PRO_6000_BLACKWELL"}:
        raise SystemExit(f"{target}: hardware mismatch")
    sizes = sorted(pd.to_numeric(data.block_size).astype(int).unique())
    if sizes != [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384]:
        raise SystemExit(f"{target}: block-size coverage mismatch")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    audit["targets"][target] = {"rows": len(data), "matrices": len(found), "sha256": digest}
if audit["targets"]["GESSM"]["rows"] != audit["targets"]["TSTRF"]["rows"]:
    raise SystemExit("GESSM/TSTRF row counts differ")
(root / "profile29_integrity.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
PY

if [[ ! -f "$training/manifest.json" ]]; then
  python3 "$repo/experiments/train_profile29_plain_vs_matrix_balanced_abcd.py" \
    --input-dir "$profile29" \
    --fold-map "$profile29/shared_matrix_folds.csv" \
    --hardware-id RTX_PRO_6000_BLACKWELL \
    --output "$training"
fi

echo "plain deployment header: $training/deploy_models/pangulu_partition_xgb_v4_2_ABCD_plain.h"
