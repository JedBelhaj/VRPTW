import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("benchmark_stage") == "tabu"]


def _as_float(row, key, fallback=""):
    value = row.get(key, "")
    if not value and fallback:
        value = row.get(fallback, "")
    return float(value)


def generate_comparative_plot(csv_path):
    rows = _load_rows(csv_path)
    if not rows:
        raise ValueError(f"No tabu benchmark rows found in {csv_path}")

    methods = [row["method"] for row in rows]
    init_distance = np.array([_as_float(row, "init_distance", "distance") for row in rows], dtype=float)
    best_distance = np.array([_as_float(row, "tabu_distance", "distance") for row in rows], dtype=float)
    total_time = np.array([_as_float(row, "total_elapsed_seconds", "elapsed_seconds") for row in rows], dtype=float)
    best_routes = np.array([int(float(row["routes_used"])) for row in rows], dtype=float)
    savings_pct = np.where(init_distance > 0, (init_distance - best_distance) / init_distance * 100.0, 0.0)

    x = np.arange(len(methods))
    bar_w = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.bar(x - bar_w / 2, init_distance, width=bar_w, label="Init distance", color="#8da0cb")
    ax.bar(x + bar_w / 2, best_distance, width=bar_w, label="Best distance", color="#fc8d62")
    ax.set_title("Distance Comparison")
    ax.set_ylabel("Distance")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.legend()

    ax = axes[0, 1]
    ax.bar(x, savings_pct, width=0.6, color="#66c2a5")
    ax.set_title("Relative Improvement")
    ax.set_ylabel("Improvement (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)

    ax = axes[1, 0]
    ax.bar(x, total_time, width=0.6, color="#e78ac3")
    ax.set_title("Total Runtime")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)

    ax = axes[1, 1]
    ax.bar(x, best_routes, width=0.6, color="#a6d854")
    ax.set_title("Best Solution Route Count")
    ax.set_ylabel("Routes")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)

    instance = rows[0]["instance"]
    fig.suptitle(f"Benchmark Comparative Plot | {instance}")

    output_file = csv_path.with_name(csv_path.stem + "_comparative.png")
    fig.savefig(output_file, dpi=170)
    plt.close(fig)
    print(output_file)


def main():
    parser = argparse.ArgumentParser(description="Generate comparative benchmark plot by method.")
    parser.add_argument("--csv", required=True, help="Path to benchmark CSV")
    args = parser.parse_args()

    generate_comparative_plot(Path(args.csv))


if __name__ == "__main__":
    main()

