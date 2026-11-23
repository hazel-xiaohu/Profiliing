#!/usr/bin/env python3
"""Build phase-1 weighted datasets from every completed sparse/dense stratum."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import build_kernel_training_sets as base


def arguments():
    parser = argparse.ArgumentParser(description="Build all-complete-strata phase-1 datasets")
    parser.add_argument("campaign")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--getrf-cap", type=int, default=80)
    parser.add_argument("--gessm-cap", type=int, default=160)
    parser.add_argument("--tstrf-cap", type=int, default=160)
    parser.add_argument("--ssssm-cap", type=int, default=320)
    return parser.parse_args()


def select(pool, cap, seed, *key):
    if len(pool) <= cap:
        return list(pool)
    return base.stable_rng(seed, *key).sample(pool, cap)


def write_standard(output, kernel, header, pools, strata, cap, seed):
    fields = list(header)
    extra = base.PROVENANCE + base.SAMPLE_FIELDS + ["stratum_weight"]
    for name in extra:
        if name not in fields:
            fields.append(name)
    total = 0
    with (output / f"{kernel}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for matrix, block in strata:
            chosen = select(pools[(matrix, block)], cap, seed, kernel, matrix, block)
            if not chosen:
                continue
            label = f"{Path(matrix).stem}|bs={block}"
            weight = 1.0 / len(chosen)
            for row in chosen:
                total += 1
                item = dict(row)
                item.update({"sample_index": str(total), "sampling_stratum": label,
                             "sampled_with_replacement": "0",
                             "stratum_weight": f"{weight:.17g}"})
                writer.writerow(item)
    return total


def write_ssssm(output, header, pools, strata, cap, seed):
    common = [name for name in header
              if name not in base.PROFILE_FIELDS + base.PROVENANCE + ["kernel_variant"]]
    fields = common + base.PAIR_FIELDS + base.SAMPLE_FIELDS + ["stratum_weight"]
    total = 0
    with (output / "SSSSM.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for matrix, block in strata:
            pairs = base.paired_pool(pools, matrix, block)
            chosen = select(pairs, cap, seed, "SSSSM", matrix, block)
            if not chosen:
                continue
            label = f"{Path(matrix).stem}|bs={block}|paired"
            weight = 1.0 / len(chosen)
            for sparse, dense in chosen:
                total += 1
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
                    "sampled_with_replacement": "0",
                    "stratum_weight": f"{weight:.17g}",
                })
                writer.writerow(item)
    return total


def main():
    args = arguments()
    root = Path(args.campaign).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    campaign, matrices, blocks = base.load_campaign(root)
    standard, ssssm, headers, completed, marker_count = base.load_rows(root)
    strata = [(matrix, block) for matrix in matrices for block in blocks
              if completed[(matrix, block)] >= {"sparse", "dense"}]
    if not strata:
        raise SystemExit("no completed sparse/dense strata")
    caps = {"GETRF": args.getrf_cap, "GESSM": args.gessm_cap,
            "TSTRF": args.tstrf_cap, "SSSSM": args.ssssm_cap}
    counts = {}
    for kernel in base.KERNELS:
        counts[kernel] = write_standard(output, kernel, headers[kernel],
                                        standard[kernel], strata, caps[kernel], args.seed)
    counts["SSSSM"] = write_ssssm(output, headers["SSSSM"], ssssm, strata,
                                   caps["SSSSM"], args.seed)
    block_counts = Counter(block for _, block in strata)
    manifest = {
        "purpose": "phase1_all_complete_strata_weighted_no_replacement",
        "source_campaign": str(root), "seed": args.seed,
        "successful_run_markers": marker_count,
        "matrix_count": len({matrix for matrix, _ in strata}),
        "strata_count": len(strata),
        "strata_by_block_size": dict(sorted(block_counts.items())),
        "caps_per_stratum": caps, "sample_counts": counts,
        "sampling": "all available rows up to cap; no replacement",
        "weighting": "each matrix/block stratum has total weight 1 per kernel",
        "ssssm_pair_definition": "same task_id + repeat_index + rank",
    }
    (output / "sampling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
