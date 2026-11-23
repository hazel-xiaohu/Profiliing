#!/usr/bin/env python3
"""Replay PanguLU task generation on CPU and emit pre-kernel feature catalogs."""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

KERNELS = ("GETRF", "GESSM", "TSTRF", "SSSSM")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("matrices", nargs="*", help="Matrix Market files")
    p.add_argument("--matrix-list", action="append", default=[],
                   help="Text file containing one Matrix Market path per line")
    p.add_argument("--suitesparse-csv", action="append", default=[],
                   help="CSV containing Group and Name columns")
    p.add_argument("--matrix-cache", default="matrix/MM")
    p.add_argument("--block-sizes", nargs="+", type=int, required=True)
    p.add_argument("--binary", default="build/feature_replay/PanguLUFeatureReplay")
    p.add_argument("--output", default="artifacts/feature_replay")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--schema-version", default="v4", choices=("v2", "v3", "v4", "v4.1"),
                   help="Required feature_schema_version in emitted catalogs")
    return p.parse_args()


def candidates(args):
    paths = [Path(x).expanduser().resolve() for x in args.matrices]
    for filename in args.matrix_list:
        source = Path(filename).expanduser().resolve()
        for raw in source.read_text().splitlines():
            value = raw.split("#", 1)[0].strip()
            if value:
                candidate = Path(value).expanduser()
                paths.append((candidate if candidate.is_absolute() else source.parent / candidate).resolve())
    cache = Path(args.matrix_cache).expanduser().resolve()
    for filename in args.suitesparse_csv:
        with Path(filename).open(newline="", encoding="utf-8-sig") as src:
            reader = csv.DictReader(src)
            names = {x.lower(): x for x in (reader.fieldnames or [])}
            if "group" not in names or "name" not in names:
                raise SystemExit(f"{filename}: expected Group and Name columns")
            for row in reader:
                group, name = row[names["group"]].strip(), row[names["name"]].strip()
                paths.append((cache / group / name / f"{name}.mtx").resolve())
    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            if not path.is_file():
                raise SystemExit(f"matrix not found: {path}")
            unique.append(path)
    if not unique:
        raise SystemExit("no matrices selected")
    return unique


def slug(path):
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:8]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem) + "-" + digest


def main():
    args = parse_args()
    binary = Path(args.binary).expanduser().resolve()
    if not binary.is_file():
        raise SystemExit(f"replay binary not found: {binary}; run script/build_feature_replay.sh first")
    root = Path(args.output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    failures = []
    for matrix in candidates(args):
        for block in args.block_sizes:
            if block <= 0:
                raise SystemExit("block sizes must be positive")
            run_dir = root / slug(matrix) / f"bs_{block}"
            success = run_dir / "_SUCCESS.json"
            if args.resume and success.is_file():
                print(f"[skip] {matrix.name} block={block}")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            catalog = run_dir / "catalog"
            catalog.mkdir(exist_ok=True)
            env = os.environ.copy()
            env["CALIB_MODE"] = "feature_replay"
            env["CALIB_CATALOG_DIR"] = str(catalog)
            cmd = [str(binary), "-F", str(matrix), "-O", str(block)]
            print(f"[replay] {matrix.name} block={block}", flush=True)
            try:
                proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
                output, code = proc.stdout, proc.returncode
            except subprocess.TimeoutExpired as exc:
                output = exc.stdout or ""
                if isinstance(output, bytes): output = output.decode(errors="replace")
                code = 124
            (run_dir / "run.log").write_text(output)
            files = {k: list(catalog.glob(f"*.{k}.catalog.csv")) for k in KERNELS}
            valid = code == 0 and all(len(v) == 1 for v in files.values())
            if valid:
                for kernel, matches in files.items():
                    with matches[0].open(newline="") as src:
                        rows = list(csv.DictReader(src))
                    if any(row.get("feature_schema_version") != args.schema_version or
                           row.get("profile_scope") != "feature_replay_pre_kernel" or
                           float(row.get("time_ms", "nan")) != -1.0 for row in rows):
                        valid = False
                        output += f"\ninvalid replay rows in {kernel}\n"
                        break
            manifest = {"matrix": str(matrix), "block_size": block, "schema_version": args.schema_version, "command": cmd,
                        "returncode": code, "catalogs": {k: [str(x) for x in v] for k,v in files.items()}}
            failed_marker = run_dir / "_FAILED.json"
            marker = success if valid else failed_marker
            marker.write_text(json.dumps(manifest, indent=2) + "\n")
            if valid:
                failed_marker.unlink(missing_ok=True)
            else:
                success.unlink(missing_ok=True)
                failures.append(f"{matrix} block={block}")
    if failures:
        raise SystemExit("feature replay failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
