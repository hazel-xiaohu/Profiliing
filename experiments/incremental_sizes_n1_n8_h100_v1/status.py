#!/usr/bin/env python3
import csv
import json
from collections import Counter
from pathlib import Path

root = Path("/output/Profiliing/artifacts/platforms/H100_SXM_80GB/incremental_sizes_n1_n8_v1")
campaign = root / "campaign"
success = list(campaign.glob("raw/**/_SUCCESS.json"))
failure = list(campaign.glob("raw/**/_FAILURE.json"))
by_size, by_variant, by_matrix = Counter(), Counter(), Counter()
for path in success:
    item = json.loads(path.read_text())
    by_size[int(item["block_size"])] += 1
    by_variant[item["forced_ssssm_variant"]] += 1
    by_matrix[Path(item["matrix"]).stem] += 1
print(f"successful={len(success)}/192 failed_markers={len(failure)}")
print("by_size", dict(sorted(by_size.items())))
print("by_variant", dict(sorted(by_variant.items())))
print("complete_matrices", sum(v == 8 for v in by_matrix.values()), "/24")
if failure:
    reasons = Counter()
    for path in failure:
        item = json.loads(path.read_text())
        reasons.update(item.get("errors", []))
    print("failure_reasons", dict(reasons))
pid = root / "runner.pid"
print("pid", pid.read_text().strip() if pid.exists() else "not-started")
