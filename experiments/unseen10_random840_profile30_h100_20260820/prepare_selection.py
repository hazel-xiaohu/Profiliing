#!/usr/bin/env python3
"""Reproducibly select ten numeric-real unseen matrices from generality840."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


SEED = 20260820
COHORT = Path(
    "/output/Partition/artifacts/generality840_backward_exact_rtx_pro6000_cuda12_8_v1/"
    "run_manifest_with_catalog.csv"
)
FOLDS = Path(
    "/output/Profiliing/artifacts/platforms/H100_SXM_80GB/profile30_add_nd24k_v1/"
    "model_ready_v4_1_profile30/shared_matrix_folds.csv"
)
SSSTATS = Path(
    "/output/Partition/artifacts/generality856_greedy_backward_exact_abcd_h100_20260812/"
    "ssstats.csv"
)
ROOT = Path(
    "/output/Profiliing/artifacts/platforms/H100_SXM_80GB/"
    "unseen10_random840_profile30_v1"
)
REUSE_ROOTS = (
    Path("/output/Partition/artifacts/greedy_speedup_ge1p4_by_fixed_nb_h100_v1/expansion/matrices"),
    Path("/output/Partition/test/exhaustic"),
    Path("/output/LBLU-main/test/exhaustic"),
    Path("/output/LBLU-main/matrix/benchmark_taco"),
)
BASE_URL = "http://sparse-files.engr.tamu.edu/MM"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def load_ssstats(path: Path) -> dict[tuple[str, str], dict]:
    """Read SuiteSparse ssstats.csv (first two lines are metadata, no header)."""
    result: dict[tuple[str, str], dict] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        next(stream)
        next(stream)
        for values in csv.reader(stream):
            if len(values) < 13:
                continue
            result[(values[0], values[1])] = {
                "ssstats_is_real": int(values[5]),
                "ssstats_is_binary": int(values[6]),
            }
    return result


def inspect(path: Path) -> dict:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        banner = stream.readline().strip().split()
        if len(banner) != 5 or banner[:3] != ["%%MatrixMarket", "matrix", "coordinate"]:
            raise ValueError("not Matrix Market coordinate format")
        field, symmetry = banner[3].lower(), banner[4].lower()
        for line in stream:
            if line.lstrip().startswith("%"):
                continue
            dims = line.split()
            if len(dims) < 3:
                break
            nrows, ncols, nnz = map(int, dims[:3])
            return {
                "field": field, "symmetry": symmetry,
                "nrows_header": nrows, "ncols_header": ncols, "nnz_header": nnz,
            }
    raise ValueError("missing Matrix Market dimensions")


def local_source(name: str) -> Path | None:
    for root in REUSE_ROOTS:
        path = root / f"{name}.mtx"
        if path.is_file() and path.stat().st_size > 0:
            return path.resolve()
    return None


def download(group: str, name: str, destination: Path) -> str:
    url = f"{BASE_URL}/{quote(group, safe='')}/{quote(name, safe='')}.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, 4):
        try:
            request = Request(url, headers={"User-Agent": "PanguLU-unseen-validation/1.0"})
            with tempfile.TemporaryDirectory(prefix="pangulu-unseen-") as temporary:
                archive_path = Path(temporary) / f"{name}.tar.gz"
                with urlopen(request, timeout=90) as response, archive_path.open("wb") as sink:
                    shutil.copyfileobj(response, sink, length=1 << 20)
                with tarfile.open(archive_path, "r:gz") as archive:
                    candidates = [
                        member for member in archive.getmembers()
                        if member.isfile() and Path(member.name).name == f"{name}.mtx"
                    ]
                    if len(candidates) != 1:
                        raise RuntimeError(f"archive contains {len(candidates)} matching members")
                    source = archive.extractfile(candidates[0])
                    if source is None:
                        raise RuntimeError("cannot extract matrix member")
                    staged = destination.with_suffix(".mtx.part")
                    with source, staged.open("wb") as sink:
                        shutil.copyfileobj(source, sink, length=1 << 20)
                    os.replace(staged, destination)
            return url
        except Exception as error:
            last_error = error
            destination.with_suffix(".mtx.part").unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"download failed: {last_error}")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    matrix_dir = ROOT / "matrices"
    matrix_dir.mkdir(exist_ok=True)
    cohort = rows(COHORT)
    ssstats = load_ssstats(SSSTATS)
    training = {row["matrix_id"] for row in rows(FOLDS)}
    overlap = sorted({row["matrix"] for row in cohort} & training)
    if len(cohort) != 840 or overlap:
        raise RuntimeError(f"invalid cohort or training overlap: n={len(cohort)}, overlap={overlap}")

    rng = random.Random(SEED)
    initial = rng.sample(cohort, 10)
    remaining = [row for row in cohort if row not in initial]
    sequence = initial + rng.sample(remaining, len(remaining))
    selected, attempts = [], []
    for order, meta in enumerate(sequence, 1):
        if len(selected) == 10:
            break
        name, group = meta["matrix"], meta["group"]
        target = matrix_dir / f"{name}.mtx"
        source = local_source(name)
        origin = "local_reuse" if source else "suitesparse_download"
        url = ""
        try:
            stats = ssstats.get((group, name), {})
            if stats.get("ssstats_is_binary") == 1:
                raise ValueError("SuiteSparse isBinary=1 (pattern/binary is unsupported)")
            if source:
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(source)
            elif not target.is_file() or target.stat().st_size == 0:
                url = download(group, name, target)
            info = inspect(target)
            if info["field"] not in {"real", "integer"}:
                raise ValueError(f"unsupported Matrix Market field={info['field']}")
            if info["nrows_header"] != info["ncols_header"]:
                raise ValueError("rectangular matrix")
            if info["nrows_header"] != int(meta["nrows"]):
                raise ValueError("catalog/header dimension mismatch")
            record = {
                "candidate_order": order,
                "initial_draw": order <= 10,
                **meta,
                **stats,
                **info,
                "origin": origin,
                "source_path": str(source or target.resolve()),
                "matrix_path": str(target.resolve()),
                "download_url": url,
                "sha256": sha256(target),
                "status": "SELECTED",
                "reason": "",
            }
            selected.append(record)
        except Exception as error:
            stats = ssstats.get((group, name), {})
            info = {}
            try:
                info = inspect(target)
            except Exception:
                pass
            record = {
                "candidate_order": order,
                "initial_draw": order <= 10,
                **meta,
                **stats,
                **info,
                "origin": origin,
                "source_path": str(source or ""),
                "matrix_path": str(target),
                "download_url": url,
                "sha256": sha256(target) if target.is_file() else "",
                "status": "REJECTED",
                "reason": str(error),
            }
            if not source:
                target.unlink(missing_ok=True)
        attempts.append(record)

    if len(selected) != 10:
        raise RuntimeError(f"only {len(selected)} eligible matrices selected")
    fields = list(dict.fromkeys(key for record in attempts for key in record))
    with (ROOT / "selection_attempts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(attempts)
    fields = list(dict.fromkeys(key for record in selected for key in record))
    with (ROOT / "selected10.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(selected)
    (ROOT / "matrix_list.txt").write_text(
        "".join(f"{record['matrix_path']}\n" for record in selected)
    )
    audit = {
        "schema": "unseen10_random840_selection_v1",
        "seed": SEED,
        "cohort": str(COHORT),
        "cohort_size": len(cohort),
        "training_fold_map": str(FOLDS),
        "training_matrices": len(training),
        "training_overlap": overlap,
        "sampling": "initial random.sample(k=10), then same-RNG replacement order",
        "eligibility": "Matrix Market coordinate, real/integer numeric, square",
        "attempted": len(attempts),
        "selected": [record["matrix"] for record in selected],
        "rejected": [
            {"matrix": record["matrix"], "reason": record["reason"]}
            for record in attempts if record["status"] == "REJECTED"
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "selection_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
