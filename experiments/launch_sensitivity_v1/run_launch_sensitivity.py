#!/usr/bin/env python3
"""Run paired, dense-SSSSM launch-policy sensitivity campaigns.

Each configuration must point at a separately built executable.  The script uses
run_profiling_suite.py without changing it, restricts its module-level variant
list to dense, and collects all four raw catalogs itself (the legacy aggregator
selects shared kernels from the sparse run).
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

KERNELS = ("GETRF", "GESSM", "TSTRF", "SSSSM")
META_FIELDS = ["launch_config_id", "repeat_index", "profiling_run_id",
               "matrix_path", "source_catalog", "binary_sha256"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", required=True, help="JSON mapping config id to binary and metadata")
    p.add_argument("--matrix-list", required=True, help="one local .mtx path per line")
    p.add_argument("--output", required=True)
    p.add_argument("--block-sizes", type=int, nargs="+", default=[2048, 4096, 6144, 8192, 10240])
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--np", type=int, default=1)
    p.add_argument("--mpirun", default="/openbayes/home/mpich-install/bin/mpirun")
    p.add_argument("--seed", type=int, default=20260731)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_runner(repo):
    path = repo / "script" / "run_profiling_suite.py"
    spec = importlib.util.spec_from_file_location("pangulu_campaign_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.VARIANTS = ("dense",)
    return module


def collect_raw(config_root, config_id, binary_hash):
    output = config_root / "measurements"
    output.mkdir(parents=True, exist_ok=True)
    rows = {k: [] for k in KERNELS}
    for marker in sorted((config_root / "campaign" / "raw").glob("**/_SUCCESS.json")):
        data = json.loads(marker.read_text())
        for kernel in KERNELS:
            for name in data.get("catalogs", {}).get(kernel, []):
                source = Path(name)
                with source.open(newline="") as f:
                    for row in csv.DictReader(f):
                        if kernel == "SSSSM" and row.get("kernel_variant") != "dense":
                            continue
                        row.update({
                            "launch_config_id": config_id,
                            "repeat_index": data["repeat_index"],
                            "profiling_run_id": data["run_id"],
                            "matrix_path": data["matrix"],
                            "source_catalog": str(source),
                            "binary_sha256": binary_hash,
                        })
                        rows[kernel].append(row)
    for kernel, values in rows.items():
        fields = list(values[0]) if values else META_FIELDS
        with (output / f"{kernel}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(values)
    return {k: len(v) for k, v in rows.items()}


def main():
    a = parse_args()
    repo = Path(__file__).resolve().parents[2]
    configs_path = Path(a.configs).expanduser().resolve()
    configs = json.loads(configs_path.read_text())
    if "baseline" not in configs:
        raise SystemExit("configs JSON must contain a baseline entry")
    root = Path(a.output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    runner = load_runner(repo)
    summaries = {}
    for config_id, config in configs.items():
        binary = Path(config["binary"]).expanduser().resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise SystemExit(f"binary is not executable for {config_id}: {binary}")
        config_root = root / "configs" / config_id
        campaign = config_root / "campaign"
        argv = ["run_profiling_suite.py", "--matrix-list", str(Path(a.matrix_list).resolve()),
                "--binary", str(binary), "--output", str(campaign),
                "--block-sizes", *map(str, a.block_sizes), "--repeats", str(a.repeats),
                "--warmups", str(a.warmups), "--timeout", str(a.timeout), "--np", str(a.np),
                "--mpirun", a.mpirun, "--seed", str(a.seed), "--allow-missing-residual"]
        if a.resume: argv.append("--resume")
        if a.dry_run: argv.append("--dry-run")
        old_argv = sys.argv
        try:
            sys.argv = argv
            runner.main()
        finally:
            sys.argv = old_argv
        if not a.dry_run:
            digest = sha256(binary)
            counts = collect_raw(config_root, config_id, digest)
            summaries[config_id] = {"binary": str(binary), "binary_sha256": digest,
                                    "counts": counts, **config}
    if not a.dry_run:
        manifest = {"schema": "launch_sensitivity_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "configs_source": str(configs_path), "matrix_list": str(Path(a.matrix_list).resolve()),
                    "block_sizes": a.block_sizes, "repeats": a.repeats, "warmups": a.warmups,
                    "seed": a.seed, "configs": summaries}
        (root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({k: v["counts"] for k, v in summaries.items()}, indent=2))

if __name__ == "__main__":
    main()
