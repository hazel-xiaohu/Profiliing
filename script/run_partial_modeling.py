#!/usr/bin/env python3
"""Build and train an exploratory model from completed profiling strata."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Sample only completed, quota-sufficient matrix/block strata and "
            "train exploratory PanguLU kernel-time models."
        )
    )
    parser.add_argument(
        "campaign", nargs="?", default="profiling_suite_001",
        help="Profiling campaign directory (default: profiling_suite_001)",
    )
    parser.add_argument(
        "--output-root", default="partial_modeling",
        help="Parent directory for timestamped training/model outputs",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--getrf", type=int, default=80)
    parser.add_argument("--gessm", type=int, default=160)
    parser.add_argument("--tstrf", type=int, default=160)
    parser.add_argument("--ssssm", type=int, default=320)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--tag", help="Output tag; defaults to the current UTC timestamp"
    )
    parser.add_argument(
        "--purpose", default="exploratory_partial_model",
        help="Purpose label stored in summary.json",
    )
    return parser.parse_args()


def run(command):
    print("+", " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, check=True)


def main():
    args = arguments()
    script_dir = Path(__file__).resolve().parent
    campaign = Path(args.campaign).expanduser().resolve()
    if not (campaign / "campaign.json").is_file():
        raise SystemExit(f"missing campaign metadata: {campaign / 'campaign.json'}")

    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path(args.output_root).expanduser().resolve() / tag
    if run_root.exists():
        raise SystemExit(f"output already exists: {run_root}")
    training = run_root / "training_set"
    models = run_root / "models"
    roofline = run_root / "roofline_models"
    figures = run_root / "roofline_figures"

    run([
        sys.executable, str(script_dir / "build_kernel_training_sets.py"),
        str(campaign), "--output", str(training),
        "--seed", str(args.seed),
        "--getrf", str(args.getrf),
        "--gessm", str(args.gessm),
        "--tstrf", str(args.tstrf),
        "--ssssm", str(args.ssssm),
        "--completed-only", "--drop-short-strata",
    ])
    run([
        sys.executable, str(script_dir / "train_kernel_time_models.py"),
        str(training), "--output", str(models),
        "--folds", str(args.folds), "--seed", str(args.seed),
    ])
    run([
        sys.executable, str(script_dir / "train_roofline_models.py"),
        str(training), "--output", str(roofline),
        "--folds", str(args.folds),
    ])
    run([
        sys.executable, str(script_dir / "plot_roofline_models.py"),
        str(training), str(roofline), "--output", str(figures),
    ])

    manifest = json.loads((training / "sampling_manifest.json").read_text())
    metrics = json.loads((models / "metrics.json").read_text())
    summary = {
        "status": "complete",
        "purpose": args.purpose,
        "source_campaign": str(campaign),
        "training_set": str(training),
        "models": str(models),
        "roofline_models": str(roofline),
        "roofline_figures": str(figures),
        "successful_run_markers": manifest["successful_run_markers"],
        "matrix_count": manifest["matrix_count"],
        "strata_count": manifest["strata_count"],
        "sample_counts": manifest["sample_counts"],
        "metrics": metrics,
        "roofline_metrics": json.loads(
            (roofline / "roofline_models.json").read_text()
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"\nPartial model complete: {run_root}")
    print(f"Summary: {run_root / 'summary.json'}")


if __name__ == "__main__":
    main()
