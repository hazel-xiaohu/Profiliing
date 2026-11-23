#!/usr/bin/env python3
"""Run a reproducible PanguLU profiling campaign and aggregate four CSVs."""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


KERNELS = ("GETRF", "GESSM", "TSTRF", "SSSSM")
VARIANTS = ("sparse", "dense")
RESIDUAL_RE = re.compile(
    r"error\s+is\s+([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan))",
    re.IGNORECASE,
)
BASE_FIELDS = (
    "task_id,matrix_id,kernel_type,kernel_variant,level,block_row,block_col,"
    "update_k,rank,block_size,process_rows,process_cols,nnz_input_0,nnz_input_1,"
    "nnz_input_2,nnz_output_0,nnz_output_1,flops,estimated_bytes,"
    "operational_intensity,row_nnz_cv,col_nnz_cv,"
    "structural_factor,structure_hash,time_ms,profile_scope,gpu_kernel_name,"
    "time_total_ms,time_kernel_ms,time_memcpy_h2d_ms,time_memcpy_d2h_ms,"
    "time_memcpy_d2d_ms,time_preprocess_ms,time_overhead_ms,num_kernel_launches,"
    "num_memcpy_h2d,num_memcpy_d2h,num_memcpy_d2d,bytes_h2d,bytes_d2h,bytes_d2d,"
    "preprocess_required,input_residency_flags"
).split(",")
V3_FEATURE_FIELDS = (
    "feature_schema_version,block_rows,block_cols,aspect_ratio,min_dimension,"
    "max_dimension,main_dimension,density_input_0,density_input_1,density_input_2,"
    "num_macs,num_divisions,operation_count_v2,row_nnz_mean,row_nnz_max,"
    "row_nnz_max_mean_ratio,row_empty_fraction,col_nnz_mean,col_nnz_max,"
    "col_nnz_max_mean_ratio,col_empty_fraction,"
    "input_0_col_nnz_mean,input_0_col_nnz_max,input_0_col_nnz_cv,input_0_col_nnz_max_mean_ratio,input_0_col_empty_fraction,"
    "input_1_col_nnz_mean,input_1_col_nnz_max,input_1_col_nnz_cv,input_1_col_nnz_max_mean_ratio,input_1_col_empty_fraction,"
    "input_2_col_nnz_mean,input_2_col_nnz_max,input_2_col_nnz_cv,input_2_col_nnz_max_mean_ratio,input_2_col_empty_fraction,"
    "input_0_row_nnz_mean,input_0_row_nnz_max,input_0_row_nnz_cv,input_0_row_nnz_max_mean_ratio,input_0_row_empty_fraction,"
    "input_1_row_nnz_mean,input_1_row_nnz_max,input_1_row_nnz_cv,input_1_row_nnz_max_mean_ratio,input_1_row_empty_fraction,"
    "input_2_row_nnz_mean,input_2_row_nnz_max,input_2_row_nnz_cv,input_2_row_nnz_max_mean_ratio,input_2_row_empty_fraction,"
    "work_mean,work_max,work_cv,"
    "work_max_mean_ratio,active_work_fraction,value_bytes_est,index_bytes_est,"
    "pointer_bytes_est,output_write_bytes_est,dense_workspace_bytes_est,"
    "base_bytes_est,index_value_byte_ratio,reuse_factor_est,"
    "block_csc_index_gap_mean,block_csc_index_gap_cv,"
    "block_csc_consecutive_fraction,derived_csr_index_gap_mean,"
    "derived_csr_index_gap_cv,derived_csr_consecutive_fraction,active_rows,"
    "active_cols,canonical_structure_hash_v2,ssssm_bin_1,ssssm_bin_2,ssssm_bin_3,"
    "ssssm_bin_4,ssssm_bin_5,ssssm_bin_6,ssssm_bin_7,ssssm_bin_8,ssssm_bin_9,"
    "ssssm_bin_10,ssssm_nonempty_bins,ssssm_max_bin_fraction,"
    "ssssm_sparse_launches_est"
).split(",")
TIME_FIELD_INDEX = BASE_FIELDS.index("time_ms")
FIELDS = (BASE_FIELDS[:TIME_FIELD_INDEX] + V3_FEATURE_FIELDS
          + BASE_FIELDS[TIME_FIELD_INDEX:])
PROVENANCE = [
    "profiling_run_id", "repeat_index", "forced_ssssm_variant",
    "matrix_path", "source_catalog", "hardware_id", "gpu_model",
]

# The model is hardware/toolchain specific. PTX ISA versions are used here as
# an inspectable binary-side proxy for the CUDA compiler release.
PLATFORM_BINARY_CONTRACTS = {
    "H100_SXM_80GB": {
        "gpu_arch": "sm_90", "ptx_version": "8.1", "cuda_release": "12.1",
    },
    "RTX_PRO_6000_BLACKWELL": {
        "gpu_arch": "sm_120", "ptx_version": "8.7", "cuda_release": "12.8",
    },
}


def arguments():
    p = argparse.ArgumentParser(
        description="Profile PanguLU and create GETRF/GESSM/TSTRF/SSSSM CSVs."
    )
    p.add_argument("matrices", nargs="*", help="Matrix paths or glob patterns")
    p.add_argument("--matrix-list", action="append", default=[],
                   help="One matrix path per line; # begins a comment")
    p.add_argument("--suitesparse-csv", action="append", default=[],
                   help="SuiteSparse CSV with Group and Name columns")
    p.add_argument("--matrix-cache", default="matrix/MM",
                   help="SuiteSparse Matrix Market cache directory")
    p.add_argument("--download-base-url",
                   default="http://sparse-files.engr.tamu.edu/MM")
    p.add_argument("--download-retries", type=int, default=3)
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--binary", default="./test/PanguLU")
    p.add_argument("--block-sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS),
                   help="Kernel variants to execute; dense alone avoids duplicate non-SSSSM labels")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--output", help="Campaign output directory")
    p.add_argument("--np", type=int, default=1)
    p.add_argument("--mpirun", default="/openbayes/home/mpich-install/bin/mpirun")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument(
        "--max-residual", type=float, default=None,
        help=("Optional limit for finite residuals; inf/nan are recorded but ignored, "
              "and an omitted limit accepts every finite residual"),
    )
    p.add_argument("--allow-missing-residual", action="store_true")
    p.add_argument("--seed", type=int, default=20260727)
    p.add_argument("--pause", type=float, default=0.0)
    p.add_argument("--hardware-id", default="UNSPECIFIED",
                   help="Stable platform label, e.g. H100_SXM_80GB")
    p.add_argument("--gpu-model", default="",
                   help="Exact GPU model string recorded in campaign provenance")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_cuda_binary(binary):
    candidates = (
        Path("/output/toolchains/cuda-12.8/bin/cuobjdump"),
        Path("/usr/local/cuda/bin/cuobjdump"),
    )
    tool = next((path for path in candidates if path.is_file()), None)
    if tool is None:
        found = shutil.which("cuobjdump")
        tool = Path(found) if found else None
    if tool is None:
        raise SystemExit("cuobjdump is required to verify the profiling binary")
    result = subprocess.run(
        [str(tool), "--dump-ptx", str(binary)], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"failed to inspect profiling binary with {tool}: {result.stderr.strip()}"
        )
    versions = sorted(set(re.findall(r"^\.version\s+(\S+)", result.stdout, re.MULTILINE)))
    targets = sorted(set(re.findall(r"^\.target\s+(sm_\d+)", result.stdout, re.MULTILINE)))
    if not versions or not targets:
        raise SystemExit(
            f"cannot determine embedded PTX version/target for profiling binary: {binary}"
        )
    return {
        "binary_sha256": sha256_file(binary),
        "cuobjdump": str(tool),
        "embedded_ptx_versions": versions,
        "embedded_gpu_targets": targets,
    }


def validate_binary_contract(binary, hardware_id):
    provenance = inspect_cuda_binary(binary)
    expected = PLATFORM_BINARY_CONTRACTS.get(hardware_id)
    provenance["hardware_id"] = hardware_id
    provenance["expected_contract"] = expected
    if expected is None:
        return provenance
    failures = []
    if expected["gpu_arch"] not in provenance["embedded_gpu_targets"]:
        failures.append(
            f"target {expected['gpu_arch']} not in {provenance['embedded_gpu_targets']}"
        )
    if expected["ptx_version"] not in provenance["embedded_ptx_versions"]:
        failures.append(
            f"PTX {expected['ptx_version']} not in {provenance['embedded_ptx_versions']}"
        )
    if failures:
        raise SystemExit(
            f"binary/toolchain contract violation for {hardware_id}: "
            + "; ".join(failures)
            + f". Required CUDA {expected['cuda_release']} + {expected['gpu_arch']}."
        )
    return provenance


def suitesparse_rows(csv_files):
    rows, seen = [], set()
    for csv_name in csv_files:
        source = Path(csv_name).expanduser().resolve()
        with source.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            normalized = {name.lower(): name for name in (reader.fieldnames or [])}
            if "group" not in normalized or "name" not in normalized:
                raise SystemExit(f"{source}: expected Group and Name columns")
            for line, row in enumerate(reader, 2):
                group = row[normalized["group"]].strip()
                name = row[normalized["name"]].strip()
                if not group or not name:
                    raise SystemExit(f"{source}:{line}: empty Group or Name")
                key = (group, name)
                if key not in seen:
                    seen.add(key)
                    rows.append((group, name))
    return rows


def download_archive_member(url, matrix_name, destination, retries):
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "PanguLU-profiling/1.0"})
            with tempfile.TemporaryDirectory(prefix="pangulu-ssget-") as temporary:
                archive_path = Path(temporary) / f"{matrix_name}.tar.gz"
                with urlopen(request, timeout=120) as response, archive_path.open("wb") as dst:
                    shutil.copyfileobj(response, dst, length=1024 * 1024)
                with tarfile.open(archive_path, "r:gz") as archive:
                    candidates = [member for member in archive.getmembers()
                                  if member.isfile()
                                  and Path(member.name).name == f"{matrix_name}.mtx"]
                    if len(candidates) != 1:
                        raise RuntimeError(
                            f"archive has {len(candidates)} matching {matrix_name}.mtx files"
                        )
                    source = archive.extractfile(candidates[0])
                    if source is None:
                        raise RuntimeError("cannot read Matrix Market member")
                    staged = destination.with_suffix(destination.suffix + ".part")
                    with source, staged.open("wb") as dst:
                        shutil.copyfileobj(source, dst, length=1024 * 1024)
                    staged.replace(destination)
            return
        except Exception as exc:
            last_error = exc
            destination.with_suffix(destination.suffix + ".part").unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"download failed after {retries} attempts: {last_error}")


def download_suitesparse(csv_files, cache_root, base_url, retries):
    result = []
    for index, (group, name) in enumerate(suitesparse_rows(csv_files), 1):
        destination = cache_root / group / name / f"{name}.mtx"
        if destination.is_file() and destination.stat().st_size > 0:
            print(f"[download {index}] cached {group}/{name}: {destination}")
        else:
            url = "/".join((base_url.rstrip("/"), quote(group, safe=""),
                            quote(name, safe="") + ".tar.gz"))
            print(f"[download {index}] {group}/{name}: {url}", flush=True)
            download_archive_member(url, name, destination, retries)
        try:
            shape = matrix_shape(destination)
        except ValueError as exc:
            raise SystemExit(f"downloaded {group}/{name} is invalid: {exc}") from exc
        print(f"  validated n={shape['nrows']} nnz={shape['nnz']}")
        result.append(destination.resolve())
    return result

def matrix_paths(patterns, list_files):
    items = []
    for pattern in patterns:
        found = glob.glob(pattern)
        items.extend(found or [pattern])
    for name in list_files:
        source = Path(name).expanduser().resolve()
        for raw in source.read_text().splitlines():
            value = raw.split("#", 1)[0].strip()
            if not value:
                continue
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = source.parent / candidate
            found = glob.glob(str(candidate))
            items.extend(found or [str(candidate)])
    result, seen = [], set()
    for item in items:
        path = Path(item).expanduser().resolve()
        if path in seen:
            continue
        if not path.is_file():
            raise SystemExit(f"matrix not found: {path}")
        seen.add(path)
        result.append(path)
    if not result:
        raise SystemExit("no matrices selected")
    return result


def matrix_shape(path):
    with path.open("r", errors="replace") as src:
        banner = src.readline().strip().split()
        if len(banner) != 5 or banner[:3] != ["%%MatrixMarket", "matrix", "coordinate"]:
            raise ValueError("expected Matrix Market coordinate format")
        numeric_type = banner[3].lower()
        if numeric_type not in {"real", "integer"}:
            raise ValueError(
                "only real or integer numeric matrices are accepted; complex and pattern are unsupported"
            )
        for line in src:
            if line.lstrip().startswith("%"):
                continue
            dims = line.split()
            if len(dims) >= 3:
                rows, cols, nnz = map(int, dims[:3])
                if rows != cols:
                    raise ValueError(f"rectangular matrix {rows}x{cols}")
                return {"nrows": rows, "ncols": cols, "nnz": nnz}
            break
    raise ValueError("missing matrix dimensions")


def slug(path):
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:8]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem) + "-" + digest


def command(args, binary, matrix, block):
    base = [str(binary), "-F", str(matrix), "-O", str(block)]
    return base if args.np == 1 else [args.mpirun, "-np", str(args.np)] + base


def residual(text):
    matches = list(RESIDUAL_RE.finditer(text))
    if not matches:
        return None
    token = matches[-1].group(1)
    value = float(token)
    # Store non-finite results as JSON-safe strings. They count as present but
    # intentionally do not participate in finite residual limit checks.
    return value if math.isfinite(value) else token.lower()


def residual_exceeds_limit(value, limit):
    return (isinstance(value, (int, float)) and math.isfinite(value)
            and limit is not None and value > limit)


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def invoke(cmd, env, log, timeout):
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              timeout=timeout, check=False)
        text, code, timed_out = proc.stdout, proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        code, timed_out = 124, True
    elapsed = time.monotonic() - started
    log.write_text(text)
    return code, timed_out, elapsed, text


def validate(catalog_dir, variant, result_residual, args,
             allow_empty_offdiagonal=False):
    errors, catalogs = [], {}
    if result_residual is None and not args.allow_missing_residual:
        errors.append("solver output has no residual")
    elif residual_exceeds_limit(result_residual, args.max_residual):
        errors.append(f"residual {result_residual:.6g} exceeds {args.max_residual:.6g}")
    for kernel in KERNELS:
        paths = sorted(catalog_dir.glob(f"*.{kernel}.catalog.csv"))
        if not paths:
            errors.append(f"missing {kernel} catalog")
            continue
        catalogs[kernel] = []
        for path in paths:
            with path.open(newline="") as src:
                reader = csv.DictReader(src)
                if reader.fieldnames not in (BASE_FIELDS, FIELDS):
                    errors.append(f"{path.name}: schema mismatch")
                    continue
                rows = list(reader)
            if not rows and not (allow_empty_offdiagonal and kernel != "GETRF"):
                errors.append(f"{path.name}: empty catalog")
                continue
            for line, row in enumerate(rows, 2):
                if row["kernel_type"] != kernel:
                    errors.append(f"{path.name}:{line}: wrong kernel type")
                if kernel == "SSSSM" and row["kernel_variant"] != variant:
                    errors.append(f"{path.name}:{line}: wrong SSSSM variant")
                try:
                    total = float(row["time_total_ms"])
                    device = float(row["time_kernel_ms"])
                    launches = int(row["num_kernel_launches"])
                except ValueError:
                    errors.append(f"{path.name}:{line}: invalid timing")
                    continue
                if min(total, device, launches) <= 0:
                    errors.append(f"{path.name}:{line}: non-positive timing")
                if total + 1.0e-6 < device:
                    errors.append(f"{path.name}:{line}: kernel time exceeds total")
            catalogs[kernel].append(str(path))
    return errors, catalogs


def warmup(args, binary, root, matrix, block, variant):
    directory = root / "warmups" / slug(matrix) / f"bs_{block}" / variant
    marker = directory / "_SUCCESS.json"
    if args.resume and marker.is_file():
        return
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(1, args.warmups + 1):
        env = os.environ.copy()
        env.pop("CALIB_MODE", None)
        env.pop("CALIB_CATALOG_DIR", None)
        env["PANGULU_SSSSM_KERNEL"] = variant
        code, timed_out, _, text = invoke(
            command(args, binary, matrix, block), env,
            directory / f"warmup_{index:03d}.log", args.timeout,
        )
        value = residual(text)
        if code or timed_out or (value is None and not args.allow_missing_residual):
            raise RuntimeError(f"warmup failed; see {directory}")
        if residual_exceeds_limit(value, args.max_residual):
            raise RuntimeError(f"warmup residual {value} exceeds limit; see {directory}")
    write_json(marker, {"status": "success", "warmups": args.warmups})


def measured_run(args, binary, root, matrix, shape, block, variant, repeat):
    directory = root / "raw" / slug(matrix) / f"bs_{block}" / variant / f"repeat_{repeat:03d}"
    success, failure = directory / "_SUCCESS.json", directory / "_FAILURE.json"
    if args.resume and success.is_file():
        return json.loads(success.read_text())
    catalog_dir = directory / "catalog"
    if args.resume and failure.is_file() and catalog_dir.is_dir():
        old = json.loads(failure.read_text())
        errors, catalogs = validate(
            catalog_dir, variant, old.get("residual"), args,
            allow_empty_offdiagonal=(block >= shape["nrows"]),
        )
        if not errors and old.get("returncode") == 0:
            old.update({"status": "success", "errors": [], "catalogs": catalogs,
                        "hardware_id": args.hardware_id, "gpu_model": args.gpu_model})
            write_json(success, old)
            failure.unlink()
            return old
    if directory.exists():
        shutil.rmtree(directory)
    catalog_dir.mkdir(parents=True)
    env = os.environ.copy()
    env.update({"CALIB_MODE": "catalog", "CALIB_CATALOG_DIR": str(catalog_dir),
                "PANGULU_SSSSM_KERNEL": variant})
    cmd = command(args, binary, matrix, block)
    code, timed_out, elapsed, text = invoke(cmd, env, directory / "run.log", args.timeout)
    value = residual(text)
    errors, catalogs = ([f"process exited with {code}"], {}) if code else validate(
        catalog_dir, variant, value, args,
        allow_empty_offdiagonal=(block >= shape["nrows"]),
    )
    data = {
        "status": "failed" if errors else "success",
        "run_id": f"{slug(matrix)}-bs{block}-{variant}-r{repeat:03d}",
        "matrix": str(matrix), "shape": shape, "block_size": block,
        "forced_ssssm_variant": variant, "repeat_index": repeat,
        "command": cmd, "returncode": code, "timed_out": timed_out,
        "duration_seconds": elapsed, "residual": value, "errors": errors,
        "hardware_id": args.hardware_id, "gpu_model": args.gpu_model,
        "catalogs": catalogs, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(failure if errors else success, data)
    return data


def write_aggregate(markers, target):
    target.mkdir(parents=True, exist_ok=True)
    handles = {kernel: (target / f".{kernel}.tmp").open("w", newline="") for kernel in KERNELS}
    writers = {kernel: csv.DictWriter(handles[kernel], fieldnames=FIELDS + PROVENANCE)
               for kernel in KERNELS}
    counts = {kernel: 0 for kernel in KERNELS}
    for writer in writers.values():
        writer.writeheader()
    marker_variants = {json.loads(marker.read_text())["forced_ssssm_variant"]
                       for marker in markers}
    canonical_common_variant = "sparse" if "sparse" in marker_variants else "dense"
    try:
        for marker in markers:
            data = json.loads(marker.read_text())
            variant = data["forced_ssssm_variant"]
            for kernel in KERNELS:
                if kernel != "SSSSM" and variant != canonical_common_variant:
                    continue
                for source_name in data["catalogs"].get(kernel, []):
                    source = Path(source_name)
                    with source.open(newline="") as src:
                        for row in csv.DictReader(src):
                            row.update({"profiling_run_id": data["run_id"],
                                        "repeat_index": data["repeat_index"],
                                        "forced_ssssm_variant": variant,
                                        "matrix_path": data["matrix"],
                                        "source_catalog": str(source),
                                        "hardware_id": data.get("hardware_id", "UNSPECIFIED"),
                                        "gpu_model": data.get("gpu_model", "")})
                            writers[kernel].writerow(row)
                            counts[kernel] += 1
    finally:
        for handle in handles.values():
            handle.close()
    for kernel in KERNELS:
        (target / f".{kernel}.tmp").replace(target / f"{kernel}.csv")
    return counts


def aggregate(root):
    markers = sorted((root / "raw").glob("**/_SUCCESS.json"))
    target = root / "aggregated"
    target.mkdir(parents=True, exist_ok=True)
    for kernel in KERNELS:
        (target / f"{kernel}.csv").unlink(missing_ok=True)
    by_matrix = {}
    for marker in markers:
        data = json.loads(marker.read_text())
        matrix_key = slug(Path(data["matrix"]))
        by_matrix.setdefault(matrix_key, []).append(marker)

    matrix_counts = {}
    for matrix_key, matrix_markers in sorted(by_matrix.items()):
        matrix_target = target / matrix_key
        matrix_counts[matrix_key] = write_aggregate(matrix_markers, matrix_target)
        first = json.loads(matrix_markers[0].read_text())
        write_json(matrix_target / "matrix.json",
                   {"matrix": first["matrix"], "shape": first["shape"]})

    total_counts = write_aggregate(markers, target / "all_matrices")
    return total_counts, matrix_counts, len(markers)


def status_csv(root, results):
    fields = ["status", "run_id", "matrix", "block_size", "forced_ssssm_variant",
              "repeat_index", "returncode", "timed_out", "duration_seconds",
              "residual", "errors"]
    with (root / "run_status.csv").open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for item in results:
            row = {field: item.get(field, "") for field in fields}
            row["errors"] = "; ".join(item.get("errors", []))
            writer.writerow(row)


def main():
    args = arguments()
    if args.repeats < 1 or args.warmups < 0 or args.np < 1:
        raise SystemExit("invalid repeats, warmups, or MPI process count")
    if args.download_retries < 1:
        raise SystemExit("--download-retries must be positive")
    if any(value <= 0 for value in args.block_sizes):
        raise SystemExit("block sizes must be positive")
    binary = Path(args.binary).expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit(f"binary is not executable: {binary}")
    binary_provenance = validate_binary_contract(binary, args.hardware_id)
    print(
        "binary_contract="
        f"hardware={args.hardware_id} "
        f"targets={binary_provenance['embedded_gpu_targets']} "
        f"ptx={binary_provenance['embedded_ptx_versions']} "
        f"sha256={binary_provenance['binary_sha256']}"
    )
    cache_root = Path(args.matrix_cache).expanduser().resolve()
    downloaded = download_suitesparse(
        args.suitesparse_csv, cache_root, args.download_base_url, args.download_retries
    )
    if args.download_only:
        print(f"downloaded_or_cached={len(downloaded)} cache={cache_root}")
        return
    if not args.output:
        raise SystemExit("--output is required unless --download-only is used")
    matrices = matrix_paths(args.matrices + [str(path) for path in downloaded],
                            args.matrix_list)
    shapes = {}
    for matrix in matrices:
        try:
            shapes[str(matrix)] = matrix_shape(matrix)
        except ValueError as exc:
            raise SystemExit(f"{matrix}: {exc}") from exc
    root = Path(args.output).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise SystemExit(f"output is not empty; choose another or use --resume: {root}")
    selected_variants = tuple(dict.fromkeys(args.variants))
    groups = [(matrix, block, variant) for matrix in matrices
              for block in sorted(set(args.block_sizes)) for variant in selected_variants]
    random.Random(args.seed).shuffle(groups)
    print(f"matrices={len(matrices)} groups={len(groups)} measured_runs={len(groups)*args.repeats}")
    print(f"output={root}")
    if args.dry_run:
        for matrix, block, variant in groups:
            print(f"{matrix} block={block} variant={variant} repeats={args.repeats}")
        return
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "campaign.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "binary": str(binary),
        "binary_provenance": binary_provenance,
        "suitesparse_csv": [str(Path(name).expanduser().resolve())
                            for name in args.suitesparse_csv],
        "matrix_cache": str(cache_root),
        "download_base_url": args.download_base_url,
        "matrices": [{"path": str(path), **shapes[str(path)]} for path in matrices],
        "block_sizes": sorted(set(args.block_sizes)), "variants": list(selected_variants),
        "repeats": args.repeats, "warmups": args.warmups, "np": args.np,
        "seed": args.seed, "max_residual": args.max_residual,
    })
    results, failures = [], 0
    for group_index, (matrix, block, variant) in enumerate(groups, 1):
        print(f"[group {group_index}/{len(groups)}] {matrix.name} block={block} variant={variant}")
        try:
            if args.warmups:
                warmup(args, binary, root, matrix, block, variant)
        except RuntimeError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            failures += args.repeats
            continue
        for repeat in range(1, args.repeats + 1):
            data = measured_run(args, binary, root, matrix, shapes[str(matrix)],
                                block, variant, repeat)
            results.append(data)
            print(f"  repeat={repeat}/{args.repeats} status={data['status']} "
                  f"residual={data['residual']} duration={data['duration_seconds']:.3f}s")
            failures += data["status"] != "success"
            if args.pause:
                time.sleep(args.pause)
    status_csv(root, results)
    counts, matrix_counts, successes = aggregate(root)
    print(f"successful_runs={successes} failures={failures}")
    print(f"matrix_directories={len(matrix_counts)}")
    for kernel in KERNELS:
        print(f"{kernel}.csv rows={counts[kernel]}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
