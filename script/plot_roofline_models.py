#!/usr/bin/env python3
"""Visualize grouped-CV predictions and fitted PanguLU Roofline models."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

SPECS = (
    ("GETRF", "GETRF.csv", "time_kernel_ms"),
    ("GESSM", "GESSM.csv", "time_kernel_ms"),
    ("TSTRF", "TSTRF.csv", "time_kernel_ms"),
    ("SSSSM_sparse", "SSSSM.csv", "sparse_time_kernel_ms"),
    ("SSSSM_dense", "SSSSM.csv", "dense_time_kernel_ms"),
)
COLORS = {1024: "#4477AA", 2048: "#66CCEE", 4096: "#228833",
          8192: "#CCBB44", 16384: "#EE6677"}


def arguments():
    parser = argparse.ArgumentParser(description="Plot PanguLU Roofline diagnostics")
    parser.add_argument("training_set")
    parser.add_argument("roofline_models")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, output, stem, dpi):
    fig.savefig(output / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def panel_title(ax, model):
    status = model["fit_status"]
    ax.set_title(f"{model['model']}  [{status}]", fontsize=10,
                 color="#177245" if status == "usable_preliminary" else "#A23B3B")


def main():
    args = arguments()
    training = Path(args.training_set).expanduser().resolve()
    model_dir = Path(args.roofline_models).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    models = {item["model"]: item for item in json.loads(
        (model_dir / "roofline_models.json").read_text()
    )}

    fig_fit, axes_fit = plt.subplots(2, 3, figsize=(14, 8.5))
    fig_roof, axes_roof = plt.subplots(2, 3, figsize=(14, 8.5))
    used_blocks = set()

    for index, (name, training_file, target) in enumerate(SPECS):
        ax_fit = axes_fit.flat[index]
        ax_roof = axes_roof.flat[index]
        model = models[name]
        source = rows(training / training_file)
        prediction = rows(model_dir / f"{name}_cv_predictions.csv")
        if len(source) != len(prediction):
            raise SystemExit(f"row mismatch for {name}: {len(source)} != {len(prediction)}")

        actual = np.asarray([float(row["actual_time_kernel_ms"]) for row in prediction])
        estimated = np.asarray([float(row["predicted_time_kernel_ms"]) for row in prediction])
        flops = np.asarray([float(row["flops"]) for row in source])
        byte_count = np.asarray([float(row["estimated_bytes"]) for row in source])
        blocks = np.asarray([int(row["block_size"]) for row in source])

        positive = (actual > 0) & (estimated > 0)
        for block in sorted(set(blocks[positive])):
            mask = positive & (blocks == block)
            used_blocks.add(block)
            ax_fit.scatter(actual[mask], estimated[mask], s=8, alpha=0.28,
                           color=COLORS.get(block, "#999999"), rasterized=True)
        low = min(actual[positive].min(), estimated[positive].min())
        high = max(actual[positive].max(), estimated[positive].max())
        ax_fit.plot([low, high], [low, high], "--", color="black", lw=1, label="ideal")
        ax_fit.set_xscale("log"); ax_fit.set_yscale("log")
        ax_fit.set_xlabel("Measured kernel time (ms)")
        ax_fit.set_ylabel("Grouped-CV prediction (ms)")
        ax_fit.grid(True, which="both", alpha=0.2)
        panel_title(ax_fit, model)
        metric = model["cv_metrics"]
        ax_fit.text(0.04, 0.96, f"R2={metric['r2']:.3f}\nMAPE={metric['mape_percent']:.1f}%",
                    transform=ax_fit.transAxes, va="top", fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})

        valid = (actual > 0) & (flops > 0) & (byte_count > 0)
        intensity = flops[valid] / byte_count[valid]
        achieved = (flops[valid] / 1.0e9) / actual[valid]
        valid_blocks = blocks[valid]
        for block in sorted(set(valid_blocks)):
            mask = valid_blocks == block
            ax_roof.scatter(intensity[mask], achieved[mask], s=8, alpha=0.28,
                            color=COLORS.get(block, "#999999"), rasterized=True)
        xmin = max(float(np.percentile(intensity, 0.5)) / 2.0, 1.0e-8)
        xmax = float(np.percentile(intensity, 99.5)) * 2.0
        xline = np.logspace(np.log10(xmin), np.log10(xmax), 300)
        compute = model["effective_compute_tflops"]
        bandwidth = model["effective_bandwidth_tbps"]
        ax_roof.plot(xline, np.minimum(compute, bandwidth * xline), color="black", lw=1.8,
                     label="fitted asymptotic roof")
        ridge = compute / bandwidth
        if xmin <= ridge <= xmax:
            ax_roof.axvline(ridge, color="#777777", ls=":", lw=1)
        ax_roof.set_xscale("log"); ax_roof.set_yscale("log")
        ax_roof.set_xlabel("Operational intensity (FLOP/byte)")
        ax_roof.set_ylabel("Achieved throughput (TFLOP/s)")
        ax_roof.grid(True, which="both", alpha=0.2)
        panel_title(ax_roof, model)
        ax_roof.text(0.04, 0.96, f"P_eff={compute:.3g} TF/s\nBW_eff={bandwidth:.3g} TB/s",
                     transform=ax_roof.transAxes, va="top", fontsize=8,
                     bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})

    for axes in (axes_fit, axes_roof):
        axes.flat[-1].axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS.get(block, "#999999"),
                          label=f"block={block}") for block in sorted(used_blocks)]
    for fig, title in ((fig_fit, "PanguLU Roofline model: grouped-CV diagnostics"),
                       (fig_roof, "PanguLU measured throughput and fitted asymptotic roofs")):
        fig.suptitle(title, fontsize=15)
        fig.legend(handles=handles, loc="lower center", ncol=max(len(handles), 1), frameon=False)
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(fig_fit, output, "roofline_cv_measured_vs_predicted", args.dpi)
    save(fig_roof, output, "roofline_operational_intensity", args.dpi)

    report = """# Roofline 可视化

## 分组交叉验证拟合

![Measured versus predicted](roofline_cv_measured_vs_predicted.png)

虚线为理想预测。坐标使用对数尺度；偏离虚线表示跨矩阵预测误差。

## 算术强度与有效吞吐

![Operational intensity](roofline_operational_intensity.png)

黑线是拟合的渐近 Roofline：`min(P_eff, BW_eff × operational_intensity)`。
散点是实测任务吞吐；延迟项会使小任务落在渐近屋顶以下。
`diagnostic_only` 表示参数不可辨识或分组交叉验证不足，不应当作物理峰值。
"""
    (output / "README.md").write_text(report)
    print(f"wrote Roofline figures to {output}")


if __name__ == "__main__":
    main()
