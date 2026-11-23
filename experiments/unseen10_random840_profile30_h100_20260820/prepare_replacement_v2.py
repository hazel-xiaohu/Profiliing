#!/usr/bin/env python3
"""Replace uni_chimera_i5 using the next eligible item in the frozen order."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
V1 = Path(
    "/output/Profiliing/artifacts/platforms/H100_SXM_80GB/"
    "unseen10_random840_profile30_v1"
)
V2 = Path(
    "/output/Profiliing/artifacts/platforms/H100_SXM_80GB/"
    "unseen10_random840_profile30_v2_no_duplicate_chimera"
)
EXCLUDED = "uni_chimera_i5"


def load_selection_module():
    path = HERE / "prepare_selection.py"
    spec = importlib.util.spec_from_file_location("unseen10_selection_v1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    selection = load_selection_module()
    V2.mkdir(parents=True, exist_ok=True)
    matrix_dir = V2 / "matrices"
    matrix_dir.mkdir(exist_ok=True)

    old = rows(V1 / "selected10.csv")
    retained = [row for row in old if row["matrix"] != EXCLUDED]
    if len(retained) != 9:
        raise RuntimeError("v1 does not contain exactly one excluded matrix")

    cohort = rows(selection.COHORT)
    training = {row["matrix_id"] for row in rows(selection.FOLDS)}
    rng = random.Random(selection.SEED)
    initial = rng.sample(cohort, 10)
    remaining = [row for row in cohort if row not in initial]
    sequence = initial + rng.sample(remaining, len(remaining))
    order = {row["matrix"]: index for index, row in enumerate(sequence, 1)}
    old_max_order = max(order[row["matrix"]] for row in old)
    chosen_names = {row["matrix"] for row in retained}
    ssstats = selection.load_ssstats(selection.SSSTATS)

    attempts = []
    replacement = None
    for candidate_order, meta in enumerate(sequence, 1):
        if candidate_order <= old_max_order:
            continue
        name, group = meta["matrix"], meta["group"]
        if name in chosen_names or name in training:
            continue
        stats = ssstats.get((group, name), {})
        if stats.get("ssstats_is_binary") == 1:
            attempts.append({
                "candidate_order": candidate_order, **meta, **stats,
                "status": "REJECTED", "reason": "SuiteSparse isBinary=1",
            })
            continue
        target = matrix_dir / f"{name}.mtx"
        try:
            url = selection.download(group, name, target)
            info = selection.inspect(target)
            if info["field"] not in {"real", "integer"}:
                raise ValueError(f"unsupported field={info['field']}")
            if info["nrows_header"] != info["ncols_header"]:
                raise ValueError("rectangular matrix")
            if info["nrows_header"] != int(meta["nrows"]):
                raise ValueError("catalog/header dimension mismatch")
            replacement = {
                "candidate_order": candidate_order,
                "initial_draw": False,
                **meta, **stats, **info,
                "origin": "suitesparse_download",
                "source_path": str(target.resolve()),
                "matrix_path": str(target.resolve()),
                "download_url": url,
                "sha256": sha256(target),
                "status": "SELECTED",
                "reason": "next eligible item after excluding uni_chimera_i5",
            }
            attempts.append(replacement)
            break
        except Exception as error:
            target.unlink(missing_ok=True)
            attempts.append({
                "candidate_order": candidate_order, **meta, **stats,
                "status": "REJECTED", "reason": str(error),
            })
    if replacement is None:
        raise RuntimeError("no eligible replacement found")

    final = retained + [replacement]
    final.sort(key=lambda row: order.get(row["matrix"], row["candidate_order"]))
    if set(row["matrix"] for row in final) & training:
        raise RuntimeError("training/unseen overlap after replacement")

    fields = list(dict.fromkeys(key for row in final for key in row))
    with (V2 / "selected10.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(final)
    fields = list(dict.fromkeys(key for row in attempts for key in row))
    with (V2 / "replacement_attempts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(attempts)
    (V2 / "matrix_list.txt").write_text(
        "".join(f"{row['matrix_path']}\n" for row in final)
    )
    (V2 / "replacement_matrix_list.txt").write_text(
        f"{replacement['matrix_path']}\n"
    )
    audit = {
        "schema": "unseen10_random840_selection_v2_no_duplicate_chimera",
        "seed": selection.SEED,
        "base_selection": str(V1 / "selected10.csv"),
        "excluded_matrix": EXCLUDED,
        "exclusion_reason": "near-duplicate scenario/group with uni_chimera_i4",
        "replacement_protocol":
            "next eligible candidate in the already frozen random permutation",
        "replacement": replacement["matrix"],
        "replacement_group": replacement["group"],
        "replacement_candidate_order": replacement["candidate_order"],
        "selected": [row["matrix"] for row in final],
        "training_overlap": [],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (V2 / "selection_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
