import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

# Allow running this file directly: `python utils/plot_benchmark_maps.py ...`
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.parser import parse_instance


def _load_rows(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _route_colors(count):
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(count)]


def _plot_solution(ax, problem, routes, title):
    depot = problem.customers[problem.depot_id]
    customer_points_x = [problem.customers[cid].x for cid in problem.customer_ids]
    customer_points_y = [problem.customers[cid].y for cid in problem.customer_ids]

    ax.scatter(customer_points_x, customer_points_y, s=12, c="#444444", alpha=0.45)
    ax.scatter([depot.x], [depot.y], s=70, c="#d62728", marker="s")

    colors = _route_colors(len(routes))
    for idx, route in enumerate(routes):
        xs = [problem.customers[node].x for node in route]
        ys = [problem.customers[node].y for node in route]
        ax.plot(xs, ys, color=colors[idx], linewidth=1.8, alpha=0.95)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _distance_from_row(row, key, fallback_key):
    val = row.get(key, "")
    if val:
        return val
    return row.get(fallback_key, "")


def generate_maps(csv_path):
    rows = _load_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows in CSV: {csv_path}")

    instance = rows[0]["instance"]
    problem = parse_instance(instance)

    out_dir = csv_path.parent / (csv_path.stem + "_maps")
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        method = row["method"]
        init_solution = json.loads(row["init_solution"])
        best_solution = json.loads(row["best_solution"])

        init_dist = _distance_from_row(row, "init_distance", "distance")
        best_dist = _distance_from_row(row, "tabu_distance", "distance")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        _plot_solution(axes[0], problem, init_solution, f"{method} - init (dist={init_dist})")
        _plot_solution(axes[1], problem, best_solution, f"{method} - best (dist={best_dist})")

        fig.suptitle(f"{instance} | {method}")

        output_file = out_dir / f"{method}.png"
        fig.savefig(output_file, dpi=160)
        plt.close(fig)

    print(out_dir)


def main():
    parser = argparse.ArgumentParser(description="Plot init and best benchmark solutions as maps.")
    parser.add_argument("--csv", required=True, help="Path to benchmark CSV file")
    args = parser.parse_args()

    generate_maps(Path(args.csv))


if __name__ == "__main__":
    main()

