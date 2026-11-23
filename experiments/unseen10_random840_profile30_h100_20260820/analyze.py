#!/usr/bin/env python3
"""Create the reproducible profile30 -> unseen10 H100 error report."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(
    "/output/Profiliing/artifacts/platforms/H100_SXM_80GB/"
    "unseen10_random840_profile30_v1"
)
EVAL = ROOT / "evaluation_profile30_plain_abcd_v2"
OOF = Path(
    "/output/Profiliing/artifacts/platforms/H100_SXM_80GB/"
    "profile30_add_nd24k_v1/plain_vs_matrix_balanced_abcd_matrix5fold"
)
TARGETS = ("GETRF", "GESSM", "TSTRF", "SSSSM_dense")


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    out = EVAL / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(ROOT / "selected10.csv")
    tasks = pd.read_csv(EVAL / "task_predictions.csv.gz", low_memory=False)
    task_metrics = pd.read_csv(EVAL / "task_metrics_by_kernel.csv")
    by_matrix = pd.read_csv(EVAL / "task_metrics_by_kernel_matrix.csv")
    schemes = pd.read_csv(EVAL / "scheme_predictions.csv")
    scheme_metrics = pd.read_csv(EVAL / "scheme_metrics.csv").iloc[0]
    scheme_selection = pd.read_csv(EVAL / "scheme_selection.csv")
    ranges = pd.read_csv(EVAL / "feature_range_audit.csv")
    oof = pd.read_csv(OOF / "task_oof_metrics.csv")
    oof = oof[oof["mode"].eq("plain")].copy()
    status = pd.read_csv(ROOT / "campaign" / "run_status.csv")

    macro = (
        by_matrix.groupby("target", as_index=False)[
            ["MAPE", "MdAPE", "P95_APE", "R2_raw", "R2_log"]
        ]
        .mean()
        .rename(columns={name: f"macro_{name}" for name in
                         ["MAPE", "MdAPE", "P95_APE", "R2_raw", "R2_log"]})
    )
    macro.to_csv(out / "task_macro_metrics_by_kernel.csv", index=False)

    comparison = oof[[
        "target", "MAPE", "MdAPE", "P95_APE", "R2_raw", "R2_log"
    ]].merge(
        task_metrics, on="target", suffixes=("_profile30_OOF", "_unseen10")
    )
    comparison.to_csv(out / "profile30_oof_vs_unseen10.csv", index=False)

    range_summary = (
        ranges.groupby("target", as_index=False)
        .agg(outside_count=("outside_count", "sum"),
             finite_count=("finite_count", "sum"))
    )
    range_summary["outside_fraction"] = (
        range_summary.outside_count / range_summary.finite_count
    )
    range_summary.to_csv(out / "feature_range_summary.csv", index=False)

    # Diagnostic only: this does not change the primary ten-matrix result.
    no_outlier = scheme_selection[~scheme_selection.matrix_id.eq("uni_chimera_i5")]
    diagnostic = pd.DataFrame([{
        "scope": "primary_all_10",
        "matrices": len(scheme_selection),
        "oracle_accuracy": (scheme_selection.oracle_block_size ==
                            scheme_selection.selected_block_size).mean(),
        "mean_regret_percent": scheme_selection.regret_percent.mean(),
        "median_regret_percent": scheme_selection.regret_percent.median(),
        "mean_pairwise_accuracy": scheme_selection.pairwise_accuracy.mean(),
        "mean_weighted_pairwise_accuracy":
            scheme_selection.weighted_pairwise_accuracy.mean(),
    }, {
        "scope": "diagnostic_excluding_uni_chimera_i5",
        "matrices": len(no_outlier),
        "oracle_accuracy": (no_outlier.oracle_block_size ==
                            no_outlier.selected_block_size).mean(),
        "mean_regret_percent": no_outlier.regret_percent.mean(),
        "median_regret_percent": no_outlier.regret_percent.median(),
        "mean_pairwise_accuracy": no_outlier.pairwise_accuracy.mean(),
        "mean_weighted_pairwise_accuracy":
            no_outlier.weighted_pairwise_accuracy.mean(),
    }])
    diagnostic.to_csv(out / "scheme_outlier_diagnostic.csv", index=False)

    # Task-level predicted-versus-measured plot. Use all points; there are only
    # 24,838 accepted tasks.
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.2))
    for ax, target in zip(axes.flat, TARGETS):
        part = tasks[tasks.target.eq(target)]
        ax.scatter(part.actual_ms, part.predicted_ms, s=5, alpha=0.25,
                   linewidths=0)
        lower = max(min(part.actual_ms.min(), part.predicted_ms.min()), 1e-6)
        upper = max(part.actual_ms.max(), part.predicted_ms.max())
        ax.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(target.replace("_dense", "-dense"))
        ax.set_xlabel("Measured kernel time (ms)")
        ax.set_ylabel("Predicted kernel time (ms)")
        ax.grid(True, which="both", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out / "task_predicted_vs_measured_loglog.pdf", bbox_inches="tight")
    fig.savefig(out / "task_predicted_vs_measured_loglog.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.3))
    for matrix_id, part in schemes.groupby("matrix_id", sort=True):
        ax.scatter(part.actual_ms, part.predicted_ms, s=24, alpha=0.75,
                   label=matrix_id)
    lower = max(min(schemes.actual_ms.min(), schemes.predicted_ms.min()), 1e-6)
    upper = max(schemes.actual_ms.max(), schemes.predicted_ms.max())
    ax.plot([lower, upper], [lower, upper], "k--", linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Measured summed kernel time (ms)")
    ax.set_ylabel("Predicted summed kernel time (ms)")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "scheme_predicted_vs_measured_loglog.pdf", bbox_inches="tight")
    fig.savefig(out / "scheme_predicted_vs_measured_loglog.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ordered = scheme_selection.sort_values("regret_percent")
    colors = ["#c44e52" if value > 0 else "#4c72b0"
              for value in ordered.regret_percent]
    ax.bar(ordered.matrix_id, ordered.regret_percent, color=colors)
    ax.set_ylabel("Selected-vs-oracle regret (%)")
    ax.tick_params(axis="x", rotation=55, labelsize=8)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "scheme_selection_regret.pdf", bbox_inches="tight")
    fig.savefig(out / "scheme_selection_regret.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)

    task_rows = "\n".join(
        f"| {row.target.replace('_dense', '-dense')} | {int(row.n):,} | "
        f"{fmt(row.MAPE)} | {fmt(row.MdAPE)} | {fmt(row.P95_APE)} | "
        f"{fmt(row.R2_raw, 3)} | {fmt(row.R2_log, 3)} |"
        for row in task_metrics.itertuples()
    )
    selected_rows = "\n".join(
        f"| {row.matrix_id} | {int(row.oracle_block_size)} | "
        f"{int(row.selected_block_size)} | {fmt(row.regret_percent)} | "
        f"{fmt(row.pairwise_accuracy * 100)} | "
        f"{fmt(row.weighted_pairwise_accuracy * 100)} |"
        for row in scheme_selection.itertuples()
    )
    report = f"""# H100 profile30 frozen-model unseen10 evaluation

## Protocol

- Cohort: 840 generality matrices; fixed random seed `20260820`.
- Eligibility: square numeric real/integer Matrix Market coordinate data.
- Sampling: one fixed random permutation; the first ten eligible matrices were used.
- Development data: 30 profiling matrices; membership overlap with unseen10 is zero.
- GPU/binary: NVIDIA H100 80GB HBM3; CUDA 12.1; native `sm_90` binary.
- Blocks: 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384.
- Measurement: dense SSSSM, one measurement per matrix/block size, no warmup,
  no residual threshold.
- Model: frozen profile30 plain direct-log XGBoost v4.2, A+B+C+D.
- Target: `log1p(time_kernel_ms)`; no refit, feature selection, or tuning on unseen10.
- Strict join: 24,838/24,838 tasks accepted; zero structure-hash/NNZ mismatch.

## Task-level result

| Kernel | Tasks | MAPE (%) | MdAPE (%) | P95 APE (%) | R2 raw | R2 log |
|---|---:|---:|---:|---:|---:|---:|
{task_rows}

The task-level result is at least as good as the profile30 grouped-fivefold MAPE
for all four kernels. Raw-space R2 is lower because a few very long tasks dominate
the squared error; log-space R2 remains at least 0.991 (rounded).

## Scheme-level summed-kernel-cost result

The scheme value is the sum of measured or predicted kernel times for all tasks
at one matrix/block size. It is not the end-to-end numerical wall time.

- MAPE: {fmt(scheme_metrics.MAPE)}%; MdAPE: {fmt(scheme_metrics.MdAPE)}%;
  P95 APE: {fmt(scheme_metrics.P95_APE)}%.
- R2 raw: {fmt(scheme_metrics.R2_raw, 3)}; R2 log:
  {fmt(scheme_metrics.R2_log, 3)}.
- Pairwise ranking accuracy: {fmt(scheme_metrics.mean_pairwise_accuracy * 100)}%;
  absolute-gap-weighted accuracy:
  {fmt(scheme_metrics.mean_weighted_pairwise_accuracy * 100)}%.
- Predicted best NB equals the measured summed-kernel-cost oracle for
  {int(scheme_metrics.oracle_accuracy * 10)}/10 matrices.
- Median selection regret: {fmt(scheme_metrics.median_regret_percent)}%; mean:
  {fmt(scheme_metrics.mean_regret_percent)}%; maximum:
  {fmt(scheme_metrics.max_regret_percent)}%.

| Matrix | Oracle NB | Selected NB | Regret (%) | Pairwise (%) | Weighted pairwise (%) |
|---|---:|---:|---:|---:|---:|
{selected_rows}

## Main finding

The frozen model generalizes well at the individual-task and aggregate-time
levels, but the candidate selector has one serious tail failure:
`uni_chimera_i5`. Its selected NB=16384 contains a GETRF task with about
2.92e12 operations, above the profile30 training maximum of about 1.81e12.
That task is measured at about 66.69 s but predicted far below it. Consequently,
the model selects NB=16384 instead of the measured NB=2048 oracle and incurs
177.29% regret. This primary result must not be removed. Excluding that matrix
only as a diagnostic, mean regret is {fmt(no_outlier.regret_percent.mean())}%.

Feature-range violations are rare overall, but their concentration in the most
expensive `uni_chimera_i5` tasks shows why a count-based out-of-range rate alone
is insufficient. A deployment guard should check the maximum operation count
and NNZ of candidate tasks and conservatively reject extrapolative candidates.

## Timing and provenance

- Successful matrix/block runs: {int((status.status == 'success').sum())}/80.
- Sum of runner durations: {fmt(status.duration_seconds.sum() / 60, 1)} minutes
  (not a wall-clock timestamp difference).
- Selection: `../selected10.csv`; strict audit:
  `../strict_merged_v4_1/merge_audit.json`; frozen inference manifest:
  `../evaluation_profile30_plain_abcd_v2/manifest.json`.
"""
    (out / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
