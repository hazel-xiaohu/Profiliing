#!/usr/bin/env python3
"""Build balanced PanguLU training sets with paired SSSSM measurements."""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


KERNELS = ("GETRF", "GESSM", "TSTRF")
QUOTAS = {"GETRF": 80, "GESSM": 160, "TSTRF": 160, "SSSSM": 320}
PROVENANCE = ["profiling_run_id", "repeat_index", "forced_ssssm_variant",
              "matrix_path", "source_catalog"]
SAMPLE_FIELDS = ["sample_index", "sampling_stratum", "sampled_with_replacement"]
PROFILE_FIELDS = [
    "profile_scope", "gpu_kernel_name", "time_ms", "time_total_ms",
    "time_kernel_ms", "time_memcpy_h2d_ms", "time_memcpy_d2h_ms",
    "time_memcpy_d2d_ms", "time_preprocess_ms", "time_overhead_ms",
    "num_kernel_launches", "num_memcpy_h2d", "num_memcpy_d2h",
    "num_memcpy_d2d", "bytes_h2d", "bytes_d2h", "bytes_d2d",
    "preprocess_required", "input_residency_flags",
]
PAIR_FIELDS = [
    "repeat_index", "matrix_path",
    "sparse_time_kernel_ms", "dense_time_kernel_ms",
    "sparse_time_total_ms", "dense_time_total_ms",
    "sparse_num_kernel_launches", "dense_num_kernel_launches",
    "sparse_gpu_kernel_name", "dense_gpu_kernel_name",
    "faster_variant", "dense_over_sparse_kernel_time",
    "sparse_profiling_run_id", "dense_profiling_run_id",
    "sparse_source_catalog", "dense_source_catalog",
]


def arguments():
    parser = argparse.ArgumentParser(
        description="Sample each matrix/block-size stratum and pair identical SSSSM tasks."
    )
    parser.add_argument("campaign")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--getrf", type=int, default=80)
    parser.add_argument("--gessm", type=int, default=160)
    parser.add_argument("--tstrf", type=int, default=160)
    parser.add_argument("--ssssm", type=int, default=320,
                        help="Number of identical sparse/dense task pairs per stratum")
    parser.add_argument("--with-replacement", action="store_true")
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument(
        "--drop-short-strata", action="store_true",
        help="Keep only strata that satisfy every requested quota",
    )
    return parser.parse_args()


def stable_rng(seed, *parts):
    text = "|".join([str(seed)] + [str(part) for part in parts])
    return random.Random(int(hashlib.sha256(text.encode()).hexdigest()[:16], 16))


def load_campaign(root):
    path = root / "campaign.json"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    data = json.loads(path.read_text())
    matrices = [item["path"] for item in data.get("matrices", [])]
    blocks = [int(item) for item in data.get("block_sizes", [])]
    if not matrices or not blocks:
        raise SystemExit("campaign.json has no matrices or block sizes")
    return data, matrices, blocks


def positive_time(row):
    try:
        return float(row["time_kernel_ms"]) > 0.0
    except (KeyError, TypeError, ValueError):
        return False


def load_rows(root):
    standard = {kernel: defaultdict(list) for kernel in KERNELS}
    ssssm = defaultdict(list)
    headers = {}
    completed = defaultdict(set)
    markers = sorted((root / "raw").glob("**/_SUCCESS.json"))
    for marker in markers:
        meta = json.loads(marker.read_text())
        matrix, block = meta["matrix"], int(meta["block_size"])
        variant = meta["forced_ssssm_variant"]
        completed[(matrix, block)].add(variant)
        for kernel in KERNELS + ("SSSSM",):
            if kernel != "SSSSM" and variant != "sparse":
                continue
            for source_name in meta.get("catalogs", {}).get(kernel, []):
                source = Path(source_name)
                with source.open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    headers.setdefault(kernel, reader.fieldnames or [])
                    for row in reader:
                        if not positive_time(row):
                            continue
                        row.update({"profiling_run_id": meta["run_id"],
                                    "repeat_index": str(meta["repeat_index"]),
                                    "forced_ssssm_variant": variant,
                                    "matrix_path": matrix,
                                    "source_catalog": str(source)})
                        if kernel == "SSSSM":
                            ssssm[(matrix, block, variant)].append(row)
                        else:
                            standard[kernel][(matrix, block)].append(row)
    return standard, ssssm, headers, completed, len(markers)


def pair_key(row):
    return (row["task_id"], row["repeat_index"], row.get("rank", "0"))


def paired_pool(ssssm, matrix, block):
    sparse = {pair_key(row): row for row in ssssm[(matrix, block, "sparse")]}
    dense = {pair_key(row): row for row in ssssm[(matrix, block, "dense")]}
    return [(sparse[key], dense[key]) for key in sorted(sparse.keys() & dense.keys())]


def choose(rows, quota, rng, replacement):
    if len(rows) >= quota:
        return [(row, False) for row in rng.sample(rows, quota)]
    if not rows or not replacement:
        return []
    result = [(row, False) for row in rows]
    result.extend((rng.choice(rows), True) for _ in range(quota - len(rows)))
    rng.shuffle(result)
    return result


def write_availability(path, records):
    fields = ["kernel", "matrix_path", "block_size", "available", "requested", "shortfall"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_standard(output, kernel, header, pools, strata, quota, seed, replacement):
    fields = list(header)
    for name in PROVENANCE + SAMPLE_FIELDS:
        if name not in fields:
            fields.append(name)
    total = replacements = 0
    with (output / f"{kernel}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for matrix, block in strata:
            selected = choose(pools[(matrix, block)], quota,
                              stable_rng(seed, kernel, matrix, block), replacement)
            label = f"{Path(matrix).stem}|bs={block}"
            for row, repeated in selected:
                total += 1
                replacements += int(repeated)
                item = dict(row)
                item.update({"sample_index": str(total), "sampling_stratum": label,
                             "sampled_with_replacement": "1" if repeated else "0"})
                writer.writerow(item)
    return total, replacements


def write_ssssm(output, header, ssssm, strata, quota, seed, replacement):
    common = [name for name in header
              if name not in PROFILE_FIELDS + PROVENANCE + ["kernel_variant"]]
    fields = common + PAIR_FIELDS + SAMPLE_FIELDS
    total = replacements = 0
    with (output / "SSSSM.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for matrix, block in strata:
            pairs = paired_pool(ssssm, matrix, block)
            selected = choose(pairs, quota,
                              stable_rng(seed, "SSSSM", matrix, block), replacement)
            label = f"{Path(matrix).stem}|bs={block}|paired"
            for (sparse, dense), repeated in selected:
                total += 1
                replacements += int(repeated)
                sparse_time = float(sparse["time_kernel_ms"])
                dense_time = float(dense["time_kernel_ms"])
                item = {name: sparse.get(name, "") for name in common}
                item.update({
                    "repeat_index": sparse["repeat_index"],
                    "matrix_path": sparse["matrix_path"],
                    "sparse_time_kernel_ms": sparse["time_kernel_ms"],
                    "dense_time_kernel_ms": dense["time_kernel_ms"],
                    "sparse_time_total_ms": sparse["time_total_ms"],
                    "dense_time_total_ms": dense["time_total_ms"],
                    "sparse_num_kernel_launches": sparse["num_kernel_launches"],
                    "dense_num_kernel_launches": dense["num_kernel_launches"],
                    "sparse_gpu_kernel_name": sparse["gpu_kernel_name"],
                    "dense_gpu_kernel_name": dense["gpu_kernel_name"],
                    "faster_variant": "sparse" if sparse_time <= dense_time else "dense",
                    "dense_over_sparse_kernel_time": f"{dense_time / sparse_time:.12g}",
                    "sparse_profiling_run_id": sparse["profiling_run_id"],
                    "dense_profiling_run_id": dense["profiling_run_id"],
                    "sparse_source_catalog": sparse["source_catalog"],
                    "dense_source_catalog": dense["source_catalog"],
                    "sample_index": str(total), "sampling_stratum": label,
                    "sampled_with_replacement": "1" if repeated else "0",
                })
                writer.writerow(item)
    return total, replacements


def main():
    args = arguments()
    quotas = {"GETRF": args.getrf, "GESSM": args.gessm,
              "TSTRF": args.tstrf, "SSSSM": args.ssssm}
    if any(value <= 0 for value in quotas.values()):
        raise SystemExit("all quotas must be positive")
    root = Path(args.campaign).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    campaign, matrices, blocks = load_campaign(root)
    standard, ssssm, headers, completed, marker_count = load_rows(root)
    strata = [(matrix, block) for matrix in matrices for block in blocks]
    if args.completed_only:
        strata = [item for item in strata if completed[item] >= {"sparse", "dense"}]
    if not strata:
        raise SystemExit("no complete matrix/block-size strata")

    availability = []
    for matrix, block in strata:
        for kernel in KERNELS:
            count = len(standard[kernel][(matrix, block)])
            availability.append({"kernel": kernel, "matrix_path": matrix,
                                 "block_size": block, "available": count,
                                 "requested": quotas[kernel],
                                 "shortfall": max(quotas[kernel] - count, 0)})
        count = len(paired_pool(ssssm, matrix, block))
        availability.append({"kernel": "SSSSM_paired", "matrix_path": matrix,
                             "block_size": block, "available": count,
                             "requested": quotas["SSSSM"],
                             "shortfall": max(quotas["SSSSM"] - count, 0)})
    write_availability(output / "sampling_availability.csv", availability)
    if args.drop_short_strata:
        eligibility = defaultdict(list)
        for row in availability:
            eligibility[(row["matrix_path"], row["block_size"])].append(
                row["shortfall"] == 0
            )
        strata = [
            item for item in strata
            if eligibility[item] and all(eligibility[item])
        ]
        if not strata:
            raise SystemExit(
                "no matrix/block-size stratum satisfies every requested quota; "
                f"see {output / 'sampling_availability.csv'}"
            )
        selected = set(strata)
        availability = [
            row for row in availability
            if (row["matrix_path"], row["block_size"]) in selected
        ]
        print(f"selected {len(strata)} strata that satisfy every quota")
    zero = [row for row in availability if row["available"] == 0]
    short = [row for row in availability if row["shortfall"] > 0]
    if zero or (short and not args.with_replacement):
        print(f"sampling blocked: zero_pools={len(zero)} short_strata={len(short)}; "
              f"see {output / 'sampling_availability.csv'}")
        raise SystemExit(2)

    totals, repeated = {}, {}
    for kernel in KERNELS:
        totals[kernel], repeated[kernel] = write_standard(
            output, kernel, headers[kernel], standard[kernel], strata,
            quotas[kernel], args.seed, args.with_replacement)
    totals["SSSSM"], repeated["SSSSM"] = write_ssssm(
        output, headers["SSSSM"], ssssm, strata, quotas["SSSSM"],
        args.seed, args.with_replacement)
    manifest = {
        "source_campaign": str(root), "seed": args.seed,
        "matrix_count": len({matrix for matrix, _ in strata}),
        "block_sizes": sorted({block for _, block in strata}),
        "strata_count": len(strata), "successful_run_markers": marker_count,
        "quota_per_matrix_block_size": quotas,
        "ssssm_pair_definition": "same task_id + repeat_index + rank",
        "with_replacement": args.with_replacement,
        "completed_only": args.completed_only,
        "drop_short_strata": args.drop_short_strata,
        "sample_counts": totals, "replacement_counts": repeated,
        "campaign_created_at_utc": campaign.get("created_at_utc"),
    }
    (output / "sampling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
