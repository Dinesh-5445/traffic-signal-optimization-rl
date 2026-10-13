"""
evaluation/plot_phase4.py — Final benchmark visualizations for Phase 4.

Reads results/phase4/raw_episodes.csv and results/phase4/timing_stats.csv
and generates all comparison plots required for the V2 final report.

Usage:
    python evaluation/plot_phase4.py
    python evaluation/plot_phase4.py --out-dir results/phase4/plots
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────────────────────
CONTROLLER_COLORS = {
    "fixed_time":   "#4C72B0",
    "random":       "#DD8452",
    "rl_marl_gnn":  "#55A868",
}
CONTROLLER_LABELS = {
    "fixed_time":  "Fixed-Time",
    "random":      "Random",
    "rl_marl_gnn": "RL (EXP-06)",
}
SCENARIOS = ["balanced", "ns_heavy", "ew_heavy", "rush_hour", "congested", "random"]

METRIC_LABELS = {
    "mean_queue": "Mean Queue Length (vehicles/intersection)",
    "mean_wait":  "Mean Waiting Time (s/intersection)",
    "total_flow": "Completed Trips (vehicles)",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(raw_path: str, timing_path: str):
    df = pd.read_csv(raw_path)
    # Cast numeric columns
    for col in ["mean_queue", "mean_wait", "total_flow", "steps"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["mean_queue", "mean_wait"])

    timing_df = None
    if os.path.exists(timing_path):
        timing_df = pd.read_csv(timing_path)
    return df, timing_df


def make_bar_chart(df, metric, ylabel, out_path, title=None):
    """Grouped bar chart: x=scenario, groups=controller."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(SCENARIOS))
    n_ctrl = 3
    width = 0.22
    offsets = np.linspace(-(n_ctrl - 1) / 2, (n_ctrl - 1) / 2, n_ctrl) * width

    controllers = ["fixed_time", "random", "rl_marl_gnn"]
    for i, ctrl in enumerate(controllers):
        means, stds = [], []
        for scen in SCENARIOS:
            sub = df[(df["scenario"] == scen) & (df["controller"] == ctrl)][metric].dropna()
            means.append(sub.mean() if len(sub) > 0 else 0)
            stds.append(sub.std() if len(sub) > 1 else 0)
        bars = ax.bar(x + offsets[i], means, width,
                      label=CONTROLLER_LABELS[ctrl],
                      color=CONTROLLER_COLORS[ctrl],
                      yerr=stds, capsize=4, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in SCENARIOS], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title or f"{ylabel} by Scenario", fontsize=12, pad=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def make_improvement_chart(df, out_path):
    """RL improvement % vs Fixed-Time for queue and wait."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(SCENARIOS))
    width = 0.35

    for ax, metric, ylabel in zip(
        axes,
        ["mean_queue", "mean_wait"],
        ["Queue Improvement vs Fixed-Time (%)", "Wait Improvement vs Fixed-Time (%)"]
    ):
        rl_pcts, rand_pcts = [], []
        for scen in SCENARIOS:
            ft  = df[(df["scenario"] == scen) & (df["controller"] == "fixed_time")][metric].mean()
            rl  = df[(df["scenario"] == scen) & (df["controller"] == "rl_marl_gnn")][metric].mean()
            rnd = df[(df["scenario"] == scen) & (df["controller"] == "random")][metric].mean()
            if ft and ft > 0:
                rl_pcts.append((ft - rl) / ft * 100)
                rand_pcts.append((ft - rnd) / ft * 100)
            else:
                rl_pcts.append(0); rand_pcts.append(0)

        ax.bar(x - width / 2, rl_pcts,  width, label="RL (EXP-06)", color=CONTROLLER_COLORS["rl_marl_gnn"], alpha=0.88)
        ax.bar(x + width / 2, rand_pcts, width, label="Random",       color=CONTROLLER_COLORS["random"],     alpha=0.88)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in SCENARIOS], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, pad=8)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Improvement vs Fixed-Time Baseline (positive = better than Fixed-Time)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def make_timing_chart(timing_df, out_path):
    """Box plot of RL green phase durations per intersection."""
    if timing_df is None or timing_df.empty:
        print("  Skipping timing chart — no data.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    intersections = sorted(timing_df["intersection"].unique())
    data = [timing_df[timing_df["intersection"] == tl]["mean_green"].dropna().values
            for tl in intersections]
    bp = ax.boxplot(data, labels=intersections, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(CONTROLLER_COLORS["rl_marl_gnn"])
        patch.set_alpha(0.7)
    ax.axhline(10, color="red",    linestyle="--", linewidth=1, label="Min green (10s)")
    ax.axhline(60, color="orange", linestyle="--", linewidth=1, label="Max green (60s)")
    ax.set_xlabel("Intersection", fontsize=11)
    ax.set_ylabel("Mean Green Duration per Episode (s)", fontsize=11)
    ax.set_title("RL Signal Green Durations by Intersection (across scenarios)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def make_aggregate_comparison(df, out_path):
    """Aggregate mean across all scenarios for queue, wait, flow."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = [("mean_queue", "Mean Queue\n(vehicles/intersection)"),
               ("mean_wait",  "Mean Wait\n(s/intersection)"),
               ("total_flow", "Completed Trips\n(vehicles/episode)")]
    controllers = ["fixed_time", "random", "rl_marl_gnn"]

    for ax, (metric, ylabel) in zip(axes, metrics):
        means = [df[df["controller"] == c][metric].mean() for c in controllers]
        stds  = [df[df["controller"] == c][metric].std()  for c in controllers]
        colors = [CONTROLLER_COLORS[c] for c in controllers]
        labels = [CONTROLLER_LABELS[c] for c in controllers]
        bars = ax.bar(labels, means, color=colors, alpha=0.88, yerr=stds, capsize=5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel.split("\n")[0], fontsize=11, pad=6)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        # Add value labels
        for bar, mean in zip(bars, means):
            if mean == mean:  # not nan
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f"{mean:.1f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Aggregate Comparison — All Scenarios (Mean ± Std)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def print_results_table(df):
    """Print a concise summary table."""
    print("\n" + "=" * 100)
    print("  FINAL BENCHMARK RESULTS TABLE")
    print("=" * 100)
    print(f"{'Scenario':<12} {'Controller':<14} {'Queue Mean':>10} {'Queue Std':>10} "
          f"{'Wait Mean':>10} {'Wait Std':>10} {'Trips Mean':>10} {'N':>4}")
    print("-" * 100)
    for scen in SCENARIOS:
        for ctrl in ["fixed_time", "random", "rl_marl_gnn"]:
            sub = df[(df["scenario"] == scen) & (df["controller"] == ctrl)]
            valid = sub.dropna(subset=["mean_queue"])
            q_mean = valid["mean_queue"].mean()
            q_std  = valid["mean_queue"].std()
            w_mean = valid["mean_wait"].mean()
            w_std  = valid["mean_wait"].std()
            f_mean = valid["total_flow"].mean()
            n      = len(valid)
            print(f"{scen:<12} {ctrl:<14} {q_mean:>10.3f} {q_std:>10.3f} "
                  f"{w_mean:>10.3f} {w_std:>10.3f} {f_mean:>10.1f} {n:>4}")
        print()
    print("=" * 100)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw",     default="results/phase4/raw_episodes.csv")
    parser.add_argument("--timing",  default="results/phase4/timing_stats.csv")
    parser.add_argument("--out-dir", default="results/phase4/plots")
    args = parser.parse_args()

    if not os.path.exists(args.raw):
        print(f"ERROR: {args.raw} not found. Run phase4_benchmark.py first.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    df, timing_df = load_data(args.raw, args.timing)

    print(f"\nLoaded {len(df)} episode rows, {df['controller'].nunique()} controllers, "
          f"{df['scenario'].nunique()} scenarios.")

    # ── Generate plots ────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    make_bar_chart(df, "mean_queue", "Mean Queue Length (vehicles/intersection)",
                   os.path.join(args.out_dir, "queue_by_scenario.png"),
                   "Mean Queue Length per Intersection by Scenario")

    make_bar_chart(df, "mean_wait", "Mean Waiting Time (s/intersection)",
                   os.path.join(args.out_dir, "wait_by_scenario.png"),
                   "Mean Waiting Time per Intersection by Scenario")

    make_bar_chart(df, "total_flow", "Completed Trips (vehicles/episode)",
                   os.path.join(args.out_dir, "flow_by_scenario.png"),
                   "Completed Trips (Throughput) by Scenario")

    make_improvement_chart(df, os.path.join(args.out_dir, "improvement_vs_fixed.png"))

    make_timing_chart(timing_df, os.path.join(args.out_dir, "rl_timing_distribution.png"))

    make_aggregate_comparison(df, os.path.join(args.out_dir, "aggregate_comparison.png"))

    print_results_table(df)
    print(f"\nAll plots saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
