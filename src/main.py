from typing import List
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from heuristics.tabu_search import print_solution as tabu_print_solution
from heuristics.tabu_search import run_tabu_from_method as tabu_run_tabu_from_method
from init.methods import get_initial_methods
from utils.checker import evaluate_solution, maybe_repair_to_vehicle_limit
from utils.parser import parse_instance
from utils.pyvrp_validator import compare_solution_with_pyvrp


ALL_METHODS = ["greedy", "solomon", "clarke_wright", "random", "sweep"]

# Hardcoded settings
INSTANCE = "C201"
SEED = 0
# init method only testing
INIT_METHOD = "skip"  # "skip" to skip "all" or one of: greedy, solomon, clarke_wright, random, sweep
RUN_DATASET_METHOD_AVERAGE = False
RUN_INIT_BENCHMARK = False
RUN_TABU_BENCHMARK = False
TABU_BENCHMARK_PRINT_ITERATIONS = True

APPLY_FLEET_REPAIR = False # repair vehicle number
USE_PYVRP_VALIDATOR = True # cross check
PYVRP_DISTANCE_TOLERANCE = 1e-2

RUN_TABU = True
TABU_START_METHOD = "solomon"
TABU_ITERATIONS = 100
TABU_TENURE = 15 # how long does a move stay tabu
TABU_ASPIRATION = True # Allows tabu moves if they improve the best solution.
TABU_DIVERSIFICATION_INTERVAL = 35 # If no improvement for this many iterations, diversify by resetting to a different initial solution.
TABU_INTENSIFICATION_INTERVAL = 15 # Every 15 iterations → focus search around best solutions found.
TABU_PER_OPERATOR_MOVES = 80
TABU_OPERATORS = ["relocate", "swap", "two_opt_intra", "two_opt_inter", "or_opt", "cross_exchange"]
TABU_EXTRA_VERBOSE = False
ENABLE_IMPROVEMENT_OPERATOR = True
IMPROVEMENT_INTERVAL = 30 # run on exact iteration multiples: 30, 60, 90, ...
IMPROVEMENT_REGRET_K = 2


def _get_problem(instance: str):
    return parse_instance(instance)


def _get_methods(seed: int):
    return get_initial_methods(seed=seed)


def _serialize_routes(routes: List[List[int]]) -> str:
    return json.dumps(routes, separators=(",", ":"))


def _instance_group(instance_name: str) -> str:
    if instance_name.startswith("RC"):
        return "RC"
    if instance_name.startswith("R"):
        return "R"
    if instance_name.startswith("C"):
        return "C"
    return "OTHER"


def _list_archive_instances() -> List[str]:
    archive_dir = Path(__file__).resolve().parents[1] / "Archive"
    names: List[str] = []
    for path in sorted(archive_dir.glob("*.txt")):
        stem = path.stem
        if stem.startswith("__"):
            continue
        if _instance_group(stem) in ("R", "C", "RC"):
            names.append(stem)
    return names


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_benchmark_csv(output_csv: str, rows: List[dict]):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames: List[str] = []
        if rows:
            # Combined benchmark rows may come from init and tabu stages with different keys.
            # Build a stable union of keys to avoid DictWriter field mismatch errors.
            seen = set()
            for row in rows:
                for key in row.keys():
                    if key not in seen:
                        seen.add(key)
                        fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _run_single_initial(problem, methods, method_name: str):
    start = time.perf_counter()
    routes = methods[method_name](problem)
    routes, repair_note = maybe_repair_to_vehicle_limit(problem, routes, apply_fleet_repair=APPLY_FLEET_REPAIR)
    feasible, distance, message = evaluate_solution(problem, routes)
    elapsed = time.perf_counter() - start
    if repair_note:
        message = f"{message}{repair_note}"
    message = f"{message} | time={elapsed:.3f}s"
    tabu_print_solution(method_name, routes, distance, feasible, message)


def _benchmark_single_initial(problem, methods, method_name: str):
    start = time.perf_counter()
    routes = methods[method_name](problem)
    routes, repair_note = maybe_repair_to_vehicle_limit(problem, routes, apply_fleet_repair=APPLY_FLEET_REPAIR)
    feasible, distance, message = evaluate_solution(problem, routes)
    elapsed = time.perf_counter() - start
    if repair_note:
        message = f"{message}{repair_note}"

    pyvrp_check = compare_solution_with_pyvrp(
        problem,
        routes,
        baseline_feasible=feasible,
        baseline_distance=distance,
        distance_tolerance=PYVRP_DISTANCE_TOLERANCE,
    ) if USE_PYVRP_VALIDATOR else {
        "pyvrp_status": "disabled",
        "pyvrp_feasible": None,
        "pyvrp_distance": None,
        "pyvrp_distance_delta": None,
        "pyvrp_matches": None,
        "pyvrp_message": "disabled",
    }

    return {
        "instance": problem.name,
        "seed": SEED,
        "benchmark_stage": "init",
        "method": method_name,
        "feasible": feasible,
        "distance": distance,
        "elapsed_seconds": elapsed,
        "routes_used": len(routes),
        "vehicles_available": problem.vehicle_count,
        "capacity": problem.capacity,
        "message": message,
        "init_solution": _serialize_routes(routes),
        "best_solution": _serialize_routes(routes),
        **pyvrp_check,
    }


def run_init_benchmark(instance: str, seed: int):
    problem = _get_problem(instance)
    methods = _get_methods(seed)
    selected = [name for name in ALL_METHODS if name in methods]

    if not selected:
        raise ValueError("No initial methods available for benchmark")

    print(f"Running init benchmark on instance {problem.name} with methods: {', '.join(selected)}")

    rows = []
    for name in selected:
        row = _benchmark_single_initial(problem, methods, name)
        rows.append(row)
        print(
            f"- {name:14s} feasible={row['feasible']} dist={row['distance']:.3f} "
            f"time={row['elapsed_seconds']:.3f}s pyvrp={row['pyvrp_status']}"
        )

    return rows


def run_tabu_benchmark(instance: str, seed: int, print_iterations: bool = False):
    problem = _get_problem(instance)
    methods = _get_methods(seed)
    selected = [name for name in ALL_METHODS if name in methods]

    if not selected:
        raise ValueError("No initial methods available for tabu benchmark")

    print(f"Running tabu benchmark on instance {instance} with starts: {', '.join(selected)}")

    rows = []
    for name in selected:
        row = tabu_run_tabu_from_method(
            instance=instance,
            seed=seed,
            method=name,
            iterations=TABU_ITERATIONS,
            tabu_tenure=TABU_TENURE,
            aspiration=TABU_ASPIRATION,
            diversification_interval=TABU_DIVERSIFICATION_INTERVAL,
            intensification_interval=TABU_INTENSIFICATION_INTERVAL,
            per_operator_moves=TABU_PER_OPERATOR_MOVES,
            enabled_operators=TABU_OPERATORS,
            apply_fleet_repair=APPLY_FLEET_REPAIR,
            extra_verbose=TABU_EXTRA_VERBOSE,
            print_iterations=print_iterations,
            enable_improvement_operator=ENABLE_IMPROVEMENT_OPERATOR,
            improvement_interval=IMPROVEMENT_INTERVAL,
            regret_k=IMPROVEMENT_REGRET_K,
        )

        init_routes = row.pop("init_routes", None)
        final_routes = row.pop("best_routes", row.pop("routes"))
        pyvrp_check = compare_solution_with_pyvrp(
            problem,
            final_routes,
            baseline_feasible=row["tabu_feasible"],
            baseline_distance=row["tabu_distance"],
            distance_tolerance=PYVRP_DISTANCE_TOLERANCE,
        ) if USE_PYVRP_VALIDATOR else {
            "pyvrp_status": "disabled",
            "pyvrp_feasible": None,
            "pyvrp_distance": None,
            "pyvrp_distance_delta": None,
            "pyvrp_matches": None,
            "pyvrp_message": "disabled",
        }
        row.update(pyvrp_check)
        row["instance"] = problem.name
        row["seed"] = seed
        row["benchmark_stage"] = "tabu"
        row["init_solution"] = _serialize_routes(init_routes if init_routes is not None else [])
        row["best_solution"] = _serialize_routes(final_routes)
        rows.append(row)
        print(
            f"- {name:14s} init={row['init_distance']:.3f} -> tabu={row['tabu_distance']:.3f} "
            f"time={row['total_elapsed_seconds']:.3f}s pyvrp={row['pyvrp_status']}"
        )

    return rows


def run_combined_benchmark(instance: str, seed: int, output_csv: str, print_iterations: bool = False):
    rows: List[dict] = []

    if RUN_INIT_BENCHMARK:
        rows.extend(run_init_benchmark(instance=instance, seed=seed))

    if RUN_TABU_BENCHMARK:
        rows.extend(run_tabu_benchmark(instance=instance, seed=seed, print_iterations=print_iterations))

    if not rows:
        print("No benchmark rows produced.")
        return

    _write_benchmark_csv(output_csv, rows)
    print(f"Combined benchmark CSV saved to: {Path(output_csv)}")


def run_dataset_method_average(seed: int, output_prefix: str):
    instances = _list_archive_instances()
    if not instances:
        raise ValueError("No R/C/RC instances found in Archive")

    print(f"Running all-instances benchmark for {len(instances)} instances...")
    detailed_rows: List[dict] = []

    for idx, instance_name in enumerate(instances, start=1):
        print(f"[{idx}/{len(instances)}] {instance_name}")
        rows = run_tabu_benchmark(instance=instance_name, seed=seed, print_iterations=False)
        for row in rows:
            detailed_rows.append(
                {
                    "instance": row["instance"],
                    "group": _instance_group(row["instance"]),
                    "method": row["method"],
                    "seed": row["seed"],
                    "init_distance": float(row["init_distance"]),
                    "best_distance": float(row["tabu_distance"]),
                    "total_elapsed_seconds": float(row["total_elapsed_seconds"]),
                    "routes_used": int(row["routes_used"]),
                    "feasible": bool(row["tabu_feasible"]),
                }
            )

    if not detailed_rows:
        raise ValueError("No benchmark rows produced during all-instances run")

    summary_map: dict = {}
    for row in detailed_rows:
        for group_name in (row["group"], "ALL"):
            key = (group_name, row["method"])
            if key not in summary_map:
                summary_map[key] = {
                    "group": group_name,
                    "method": row["method"],
                    "runs": 0,
                    "feasible_runs": 0,
                    "sum_init_distance": 0.0,
                    "sum_best_distance": 0.0,
                    "sum_time": 0.0,
                    "sum_routes": 0.0,
                }
            agg = summary_map[key]
            agg["runs"] += 1
            agg["feasible_runs"] += 1 if row["feasible"] else 0
            agg["sum_init_distance"] += row["init_distance"]
            agg["sum_best_distance"] += row["best_distance"]
            agg["sum_time"] += row["total_elapsed_seconds"]
            agg["sum_routes"] += row["routes_used"]

    summary_rows: List[dict] = []
    for (_, _), agg in summary_map.items():
        runs = agg["runs"]
        summary_rows.append(
            {
                "group": agg["group"],
                "method": agg["method"],
                "runs": runs,
                "feasible_rate": agg["feasible_runs"] / runs,
                "avg_init_distance": agg["sum_init_distance"] / runs,
                "avg_best_distance": agg["sum_best_distance"] / runs,
                "avg_total_time_sec": agg["sum_time"] / runs,
                "avg_routes_used": agg["sum_routes"] / runs,
            }
        )

    group_order = {"R": 0, "C": 1, "RC": 2, "ALL": 3}
    summary_rows.sort(key=lambda r: (group_order.get(r["group"], 99), r["avg_best_distance"]))

    output_base = Path(output_prefix)
    detailed_path = output_base.with_name(output_base.name + "_detailed.csv")
    summary_path = output_base.with_name(output_base.name + "_summary.csv")

    _write_csv(
        detailed_path,
        detailed_rows,
        fieldnames=[
            "instance",
            "group",
            "method",
            "seed",
            "init_distance",
            "best_distance",
            "total_elapsed_seconds",
            "routes_used",
            "feasible",
        ],
    )
    _write_csv(
        summary_path,
        summary_rows,
        fieldnames=[
            "group",
            "method",
            "runs",
            "feasible_rate",
            "avg_init_distance",
            "avg_best_distance",
            "avg_total_time_sec",
            "avg_routes_used",
        ],
    )

    print(f"Detailed per-instance results: {detailed_path}")
    print(f"Average by method/group: {summary_path}")

    print("Best method per group (lowest avg_best_distance):")
    seen_groups = []
    for row in summary_rows:
        group = row["group"]
        if group in seen_groups:
            continue
        seen_groups.append(group)
        print(
            f"- {group}: {row['method']} "
            f"(avg_best_distance={row['avg_best_distance']:.3f}, avg_time={row['avg_total_time_sec']:.3f}s)"
        )


def run_initial_solutions(instance: str, seed: int, method: str):
    if method == "skip":
        print("Skipping initial solution generation.")
        return

    problem = _get_problem(instance)
    methods = _get_methods(seed)

    if method == "all":
        selected = [name for name in ALL_METHODS if name in methods]
    else:
        if method not in methods:
            raise ValueError(f"Unknown init method: {method}")
        selected = [method]

    print(f"Instance: {problem.name}")
    print(f"Vehicles available: {problem.vehicle_count}")
    print(f"Capacity: {problem.capacity}")

    for name in selected:
        _run_single_initial(problem, methods, name)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_output_csv = f"results/benchmark_{INSTANCE}_{timestamp}.csv"

    if RUN_DATASET_METHOD_AVERAGE:
        run_dataset_method_average(seed=SEED, output_prefix=f"results/method_average_{timestamp}")
        return

    if RUN_INIT_BENCHMARK or RUN_TABU_BENCHMARK:
        run_combined_benchmark(
            instance=INSTANCE,
            seed=SEED,
            output_csv=benchmark_output_csv,
            print_iterations=TABU_BENCHMARK_PRINT_ITERATIONS,
        )

    run_initial_solutions(instance=INSTANCE, seed=SEED, method=INIT_METHOD)

    if RUN_TABU and not RUN_TABU_BENCHMARK:
        tabu_run_tabu_from_method(
            instance=INSTANCE,
            seed=SEED,
            method=TABU_START_METHOD,
            iterations=TABU_ITERATIONS,
            tabu_tenure=TABU_TENURE,
            aspiration=TABU_ASPIRATION,
            diversification_interval=TABU_DIVERSIFICATION_INTERVAL,
            intensification_interval=TABU_INTENSIFICATION_INTERVAL,
            per_operator_moves=TABU_PER_OPERATOR_MOVES,
            enabled_operators=TABU_OPERATORS,
            apply_fleet_repair=APPLY_FLEET_REPAIR,
            enable_improvement_operator=ENABLE_IMPROVEMENT_OPERATOR,
            improvement_interval=IMPROVEMENT_INTERVAL,
            regret_k=IMPROVEMENT_REGRET_K,
        )


if __name__ == "__main__":
    main()
