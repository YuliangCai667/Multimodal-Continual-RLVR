#!/usr/bin/env python3
"""Create the frozen CTIR T4-60 performance and mechanism figures."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/ctir_t4_60"
EVAL = EXP / "eval"
FIGURES = EXP / "figures"
TASKS = ("MedBookVQA", "Navigation", "We-Math2", "Puzzle")
TASK_LABELS = {
    "MedBookVQA": "MedBookVQA",
    "Navigation": "Navigation",
    "We-Math2": "We-Math2",
    "Puzzle": "Puzzle (new task)",
}
CTIR_COLOR = "#0072B2"
BASELINE_COLOR = "#D55E00"
HARM_COLOR = "#009E73"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_performance() -> tuple[dict[str, dict[int, float]], dict[str, dict[int, float]]]:
    ctir: dict[str, dict[int, float]] = {task: {} for task in TASKS}
    baseline: dict[str, dict[int, float]] = {task: {} for task in TASKS}
    with (EVAL / "performance.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            task = row["eval_task"]
            step = int(row["local_step"])
            ctir[task][step] = float(row["ctir_score"])
            if row["historical_baseline_score"]:
                baseline[task][step] = float(row["historical_baseline_score"])

    # Both methods share the exact T3-final checkpoint at local step 0.
    for task in TASKS:
        baseline[task][0] = ctir[task][0]
    return ctir, baseline


def paired_step60() -> list[dict]:
    rows = []
    for task in TASKS:
        ctir_rows = json.loads((EVAL / f"step60/{task}/evaluation_results.json").read_text())
        base_rows = json.loads(
            (ROOT / f"analysis/dynamics/raw_eval/T4_step060/{task}/evaluation_results.json").read_text()
        )

        def key(row: dict) -> tuple[str, str, str]:
            return str(row["question_id"]), str(row["question"]), str(row["ground_truth"])

        ctir_by_key = {key(row): float(row["correct"]) for row in ctir_rows}
        base_by_key = {key(row): float(row["correct"]) for row in base_rows}
        if ctir_by_key.keys() != base_by_key.keys():
            raise RuntimeError(f"Step-60 example sets differ for {task}")
        differences = np.asarray(
            [ctir_by_key[item] - base_by_key[item] for item in ctir_by_key], dtype=np.float64
        )
        rows.append(
            {
                "eval_task": task,
                "total_examples": len(differences),
                "ctir_score": float(np.mean(list(ctir_by_key.values()))),
                "historical_baseline_score": float(np.mean(list(base_by_key.values()))),
                "delta_points": float(differences.mean() * 100.0),
                "improved_examples": int(np.sum(differences > 0)),
                "worsened_examples": int(np.sum(differences < 0)),
                "unchanged_examples": int(np.sum(differences == 0)),
                "net_correct_equivalent": float(differences.sum()),
            }
        )
    return rows


def mechanism_summary(step_rows: list[dict], tangent_rows: list[dict]) -> dict:
    valid = [
        row
        for row in step_rows
        if row["raw_update_fro_norm"] > 0 and math.isfinite(row["new_descent_ratio"])
    ]
    harmful = [row for row in valid if row["raw_old_harm"] > 0]
    rotated = [row for row in valid if row["chosen_beta"] > 0]
    raw_harm_sum = sum(row["raw_old_harm"] for row in harmful)
    ctir_harm_sum = sum(row["ctir_old_harm"] for row in harmful)
    final_tangent = [row for row in tangent_rows if row["step"] == 55]
    alignments = np.asarray(
        [
            (row["Q_L_alignment_vs_step0"] + row["Q_R_alignment_vs_step0"]) / 2.0
            for row in final_tangent
        ]
    )
    probe_loss = {
        step: next(row["navigation_probe_loss"] for row in tangent_rows if row["step"] == step)
        for step in (0, 55)
    }
    exact_spectrum = read_jsonl(EXP / "correctness/spectrum/correctness.jsonl")

    return {
        "logged_steps": len(step_rows),
        "nonzero_update_steps": len(valid),
        "zero_lr_steps": [row["local_step"] for row in step_rows if row["raw_update_fro_norm"] == 0],
        "raw_harm_positive_steps": len(harmful),
        "raw_harm_positive_fraction": len(harmful) / len(valid),
        "rotated_steps": len(rotated),
        "rotated_fraction": len(rotated) / len(valid),
        "beta_counts": {str(beta): count for beta, count in sorted(Counter(row["chosen_beta"] for row in valid).items())},
        "harmful_step_raw_harm_sum": raw_harm_sum,
        "harmful_step_ctir_harm_sum": ctir_harm_sum,
        "harmful_step_harm_reduction_fraction": 1.0 - ctir_harm_sum / raw_harm_sum,
        "harmful_steps_reduced": sum(row["old_harm_reduction"] > 0 for row in harmful),
        "harmful_steps_flipped_nonharmful": sum(row["ctir_old_harm"] <= 0 for row in harmful),
        "new_descent_ratio_min": min(row["new_descent_ratio"] for row in valid),
        "new_descent_ratio_mean": float(np.mean([row["new_descent_ratio"] for row in valid])),
        "rotated_new_descent_ratio_mean": float(
            np.mean([row["new_descent_ratio"] for row in rotated])
        ),
        "aggregate_new_descent_ratio": sum(row["ctir_new_descent"] for row in valid)
        / sum(row["raw_new_descent"] for row in valid),
        "new_descent_constraint_passed_steps": sum(row["new_descent_ratio"] >= 0.90 for row in valid),
        "max_online_orthogonal_certificate": max(row["max_spectrum_error"] for row in valid),
        "max_frobenius_ratio_error": max(abs(row["frob_ratio"] - 1.0) for row in valid),
        "exact_fp64_spectrum_relative_errors": [row["spectrum_relative_error"] for row in exact_spectrum],
        "tangent_step55_alignment_mean": float(alignments.mean()),
        "tangent_step55_alignment_p10": float(np.quantile(alignments, 0.10)),
        "tangent_step55_alignment_min": float(alignments.min()),
        "navigation_probe_loss_step0": probe_loss[0],
        "navigation_probe_loss_step55": probe_loss[55],
        "navigation_probe_loss_relative_change": probe_loss[55] / probe_loss[0] - 1.0,
    }


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_performance(ctir: dict, baseline: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharex=True)
    for ax, task in zip(axes.flat, TASKS):
        steps = sorted(ctir[task])
        historical_steps = sorted(baseline[task])
        ax.plot(
            steps,
            [100.0 * ctir[task][step] for step in steps],
            color=CTIR_COLOR,
            marker="o",
            linewidth=2.3,
            label="Dynamic CTIR",
        )
        ax.plot(
            historical_steps,
            [100.0 * baseline[task][step] for step in historical_steps],
            color=BASELINE_COLOR,
            marker="s",
            linestyle="--",
            linewidth=2.0,
            label="Historical vanilla (sampled)",
        )
        delta = 100.0 * (ctir[task][60] - baseline[task][60])
        ax.annotate(
            f"step 60: {delta:+.2f} pp",
            xy=(60, 100.0 * ctir[task][60]),
            xytext=(-8, 14 if delta >= 0 else -24),
            textcoords="offset points",
            ha="right",
            color=CTIR_COLOR,
            fontsize=9,
            fontweight="bold",
        )
        values = [100.0 * score for score in ctir[task].values()] + [
            100.0 * score for score in baseline[task].values()
        ]
        margin = max(2.5, (max(values) - min(values)) * 0.20)
        ax.set_ylim(min(values) - margin, max(values) + margin)
        ax.set_title(TASK_LABELS[task], fontweight="bold")
        ax.set_ylabel("Official accuracy (%)")
        ax.grid(alpha=0.25)
        ax.set_xticks(range(0, 61, 10))
    for ax in axes[-1]:
        ax.set_xlabel("T4 Puzzle optimizer step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("CTIR T4 first-60 official evaluation", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.018,
        "Historical vanilla is available only at shared step 0 and steps 30/60; dashed segments are visual guides.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.89))
    save_figure(fig, "performance_curves.png")


def plot_tradeoff(ctir: dict, baseline: dict) -> None:
    points = {}
    for step in (30, 60):
        points[step] = (
            100.0 * (baseline["Puzzle"][step] - ctir["Puzzle"][step]),
            100.0 * (ctir["Navigation"][step] - baseline["Navigation"][step]),
        )
    fig, ax = plt.subplots(figsize=(8.4, 6.5))
    xmin, xmax = -1.5, 12.5
    ymin, ymax = -4.0, 7.0
    ax.fill_betweenx([5.0, ymax], xmin, 3.0, color=HARM_COLOR, alpha=0.12, label="Predeclared go region")
    ax.axvline(3.0, color="#666666", linestyle=":", linewidth=1.5)
    ax.axhline(5.0, color="#666666", linestyle=":", linewidth=1.5)
    ax.axvline(0.0, color="#BBBBBB", linewidth=1.0)
    ax.axhline(0.0, color="#BBBBBB", linewidth=1.0)
    ax.annotate("", xy=points[60], xytext=points[30], arrowprops={"arrowstyle": "->", "color": "#777777"})
    for step, (x_value, y_value) in points.items():
        color = CTIR_COLOR if step == 60 else "#56B4E9"
        ax.scatter(x_value, y_value, s=120, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(
            f"step {step}\nNav {y_value:+.2f} pp\nPuzzle cost {x_value:.2f} pp",
            (x_value, y_value),
            xytext=(10, 10 if step == 60 else -48),
            textcoords="offset points",
            fontsize=10,
        )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Puzzle cost vs historical vanilla (percentage points; lower is better)")
    ax.set_ylabel("Navigation recovery vs historical vanilla (percentage points; higher is better)")
    ax.set_title("Retention–plasticity trade-off", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(frameon=True, facecolor="white", framealpha=1.0, edgecolor="#DDDDDD", loc="center right")
    fig.tight_layout()
    save_figure(fig, "nav_recovery_vs_puzzle_cost.png")


def plot_old_harm(step_rows: list[dict], summary: dict) -> None:
    valid = [row for row in step_rows if row["raw_update_fro_norm"] > 0]
    steps = np.asarray([row["local_step"] for row in valid])
    raw = np.asarray([row["raw_old_harm"] for row in valid]) * 1000.0
    ctir = np.asarray([row["ctir_old_harm"] for row in valid]) * 1000.0
    fig, ax = plt.subplots(figsize=(12.2, 5.8))
    ax.plot(steps, raw, color=BASELINE_COLOR, linewidth=1.8, marker="o", markersize=3.5, label="Raw optimizer update")
    ax.plot(steps, ctir, color=CTIR_COLOR, linewidth=1.8, marker="o", markersize=3.5, label="After CTIR")
    ax.fill_between(steps, ctir, raw, where=raw > ctir, color=HARM_COLOR, alpha=0.18, label="Predicted harm removed")
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(r"Navigation first-order harm proxy ($\times 10^{-3}$)")
    ax.set_title("Online Navigation harm proxy", fontsize=15, fontweight="bold")
    ax.text(
        0.99,
        0.96,
        f"raw-harmful steps: {summary['raw_harm_positive_steps']}/{summary['nonzero_update_steps']}\n"
        f"summed harmful proxy reduced: {100*summary['harmful_step_harm_reduction_fraction']:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#CCCCCC"},
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    save_figure(fig, "old_harm_by_step.png")


def plot_beta(step_rows: list[dict], summary: dict) -> None:
    valid = [
        row
        for row in step_rows
        if row["raw_update_fro_norm"] > 0 and math.isfinite(row["new_descent_ratio"])
    ]
    steps = [row["local_step"] for row in valid]
    betas = [row["chosen_beta"] for row in valid]
    ratios = [row["new_descent_ratio"] for row in valid]
    fig, ax = plt.subplots(figsize=(12.2, 5.8))
    ax.bar(steps, betas, width=0.78, color=CTIR_COLOR, alpha=0.82, label="Chosen beta")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Chosen beta")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", alpha=0.2)
    other = ax.twinx()
    other.plot(steps, ratios, color=BASELINE_COLOR, linewidth=1.6, marker=".", label="Puzzle descent ratio")
    other.axhline(0.90, color="#333333", linestyle="--", linewidth=1.2, label="0.90 constraint")
    other.set_ylim(0.88, 1.01)
    other.set_ylabel("CTIR / raw Puzzle descent proxy")
    ax.set_title("Dynamic rotation choices and one-step plasticity constraint", fontsize=15, fontweight="bold")
    counts = ", ".join(f"β={beta}: {count}" for beta, count in summary["beta_counts"].items())
    ax.text(0.01, 0.97, counts, transform=ax.transAxes, va="top", fontsize=9)
    handles_a, labels_a = ax.get_legend_handles_labels()
    handles_b, labels_b = other.get_legend_handles_labels()
    ax.legend(handles_a + handles_b, labels_a + labels_b, frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    save_figure(fig, "beta_by_step.png")


def plot_tangent(tangent_rows: list[dict]) -> None:
    steps = sorted({row["step"] for row in tangent_rows})
    mean_alignment = []
    p10_alignment = []
    min_alignment = []
    losses = []
    for step in steps:
        current = [row for row in tangent_rows if row["step"] == step]
        values = np.asarray(
            [
                (row["Q_L_alignment_vs_step0"] + row["Q_R_alignment_vs_step0"]) / 2.0
                for row in current
            ]
        )
        mean_alignment.append(values.mean())
        p10_alignment.append(np.quantile(values, 0.10))
        min_alignment.append(values.min())
        losses.append(current[0]["navigation_probe_loss"])
    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.plot(steps, mean_alignment, color=CTIR_COLOR, marker="o", linewidth=2.2, label="Mean alignment")
    ax.plot(steps, p10_alignment, color="#56B4E9", marker="o", linestyle="--", linewidth=1.8, label="10th percentile")
    ax.plot(steps, min_alignment, color="#999999", marker=".", linestyle=":", linewidth=1.6, label="Minimum matrix")
    ax.set_ylim(0.82, 1.01)
    ax.set_xlabel("Tangent refresh step")
    ax.set_ylabel("Mean of left/right subspace alignment vs step 0")
    ax.grid(alpha=0.2)
    other = ax.twinx()
    other.plot(steps, losses, color=BASELINE_COLOR, marker="s", linewidth=1.8, label="Navigation probe NLL")
    other.set_ylabel("Teacher-forced Navigation full-CoT NLL")
    ax.set_title("Current Navigation tangent drift", fontsize=15, fontweight="bold")
    handles_a, labels_a = ax.get_legend_handles_labels()
    handles_b, labels_b = other.get_legend_handles_labels()
    ax.legend(handles_a + handles_b, labels_a + labels_b, frameon=False, loc="lower left")
    fig.tight_layout()
    save_figure(fig, "tangent_rotation.png")


def plot_spectrum(step_rows: list[dict]) -> None:
    valid = [row for row in step_rows if row["raw_update_fro_norm"] > 0]
    rotated = [row for row in valid if row["chosen_beta"] > 0]
    exact = read_jsonl(EXP / "correctness/spectrum/correctness.jsonl")
    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.semilogy(
        [row["local_step"] for row in rotated],
        [row["max_spectrum_error"] for row in rotated],
        color=CTIR_COLOR,
        marker="o",
        linewidth=1.8,
        label="Online max orthogonal-map certificate",
    )
    ax.semilogy(
        [row["local_step"] for row in valid],
        [max(abs(row["frob_ratio"] - 1.0), 1e-16) for row in valid],
        color=HARM_COLOR,
        marker=".",
        linewidth=1.4,
        label="Online |Frobenius ratio − 1|",
    )
    ax.scatter(
        [1.7, 2.3],
        [row["spectrum_relative_error"] for row in exact],
        marker="*",
        s=180,
        color=BASELINE_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
        label="Correctness run: exact FP64 full-SVD error",
    )
    ax.axhline(1e-5, color="#333333", linestyle="--", linewidth=1.2, label="Pre-run pass gate")
    ax.set_ylim(1e-12, 2e-5)
    ax.set_xlim(0, 61)
    ax.set_xlabel("Optimizer step (stars are the separate step-2 correctness run)")
    ax.set_ylabel("Relative numerical error (log scale)")
    ax.set_title("Full-delta isospectral preservation", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.01,
        0.03,
        "Online values are orthogonality + Frobenius certificates, not repeated full SVDs.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    save_figure(fig, "spectrum_error.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )
    ctir, baseline = load_performance()
    step_rows = read_jsonl(EXP / "logs/step_metrics.jsonl")
    tangent_rows = read_jsonl(EXP / "logs/tangent_refresh.jsonl")
    paired = paired_step60()
    mechanism = mechanism_summary(step_rows, tangent_rows)

    with (EVAL / "step60_paired_differences.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    (EVAL / "analysis_summary.json").write_text(
        json.dumps({"step60_paired": paired, "mechanism": mechanism}, indent=2) + "\n"
    )

    plot_performance(ctir, baseline)
    plot_tradeoff(ctir, baseline)
    plot_old_harm(step_rows, mechanism)
    plot_beta(step_rows, mechanism)
    plot_tangent(tangent_rows)
    plot_spectrum(step_rows)
    print(f"Wrote six figures to {FIGURES}")
    print(f"Wrote analysis tables to {EVAL}")


if __name__ == "__main__":
    main()
